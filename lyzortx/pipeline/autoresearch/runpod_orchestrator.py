"""RunPod pod lifecycle orchestrator for AUTORESEARCH experiments.

Replaces inline workflow bash with testable Python. Each subcommand corresponds
to one workflow step; the workflow YAML calls them sequentially and threads
outputs via a shared runtime directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.autoresearch.runpod_bundle import (
    BUNDLE_FILENAME,
    BUNDLE_MANIFEST_FILENAME,
    RUNPOD_CONTAINER_DISK_GB,
    RUNPOD_DEFAULT_EXPERIMENT_TIMEOUT_SECONDS,
    RUNPOD_DEFAULT_REMOTE_ROOT,
    RUNPOD_LOCKED_GPU_COUNT,
    RUNPOD_LOCKED_GPU_TYPE_ID,
    RUNPOD_LOCKED_IMAGE_NAME,
    RUNPOD_SUPPORT_PUBLIC_IP,
    RUNPOD_VOLUME_GB,
)

LOGGER = logging.getLogger(__name__)

RUNPOD_API_BASE = "https://rest.runpod.io/v1"
VOLUME_MOUNT_PATH = "/workspace"
SSH_PORT_SPEC = "22/tcp"
POD_POLL_INTERVAL_SECONDS = 10
POD_POLL_MAX_ATTEMPTS = 60

CANDIDATE_BUNDLE_FILES = (
    "ar07_baseline_summary.json",
    "ar07_inner_val_predictions.csv",
)


# ---------------------------------------------------------------------------
# Pure data builders (testable, no side effects)
# ---------------------------------------------------------------------------


def build_create_pod_payload(
    *,
    ssh_public_key: str,
    pod_name: str,
    gpu_type_id: str = RUNPOD_LOCKED_GPU_TYPE_ID,
    gpu_count: int = RUNPOD_LOCKED_GPU_COUNT,
    image_name: str = RUNPOD_LOCKED_IMAGE_NAME,
    container_disk_gb: int = RUNPOD_CONTAINER_DISK_GB,
    volume_gb: int = RUNPOD_VOLUME_GB,
    support_public_ip: bool = RUNPOD_SUPPORT_PUBLIC_IP,
) -> dict[str, Any]:
    return {
        "cloudType": "COMMUNITY",
        "gpuCount": gpu_count,
        "gpuTypeId": gpu_type_id,
        "containerDiskInGb": container_disk_gb,
        "volumeInGb": volume_gb,
        "volumeMountPath": VOLUME_MOUNT_PATH,
        "dockerArgs": "",
        "imageName": image_name,
        "name": pod_name,
        "ports": SSH_PORT_SPEC,
        "supportPublicIp": support_public_ip,
        "env": {"SSH_PUBLIC_KEY": ssh_public_key},
    }


def build_experiment_metadata(
    *,
    github_repository: str = "",
    github_ref_name: str = "",
    github_sha: str = "",
    github_run_id: str = "",
    github_run_attempt: str = "",
    bundle_source: str = "",
    experiment_command: str = "",
) -> dict[str, str]:
    return {
        "github_repository": github_repository,
        "github_ref_name": github_ref_name,
        "github_sha": github_sha,
        "github_run_id": github_run_id,
        "github_run_attempt": github_run_attempt,
        "bundle_source": bundle_source,
        "experiment_command": experiment_command,
    }


def build_pod_execution_metadata(
    *,
    pod_id: str,
    public_ip: str,
    ssh_port: str,
    experiment_exit_code: int,
    experiment_command: str,
) -> dict[str, Any]:
    return {
        "pod_id": pod_id,
        "public_ip": public_ip,
        "ssh_port": ssh_port,
        "experiment_exit_code": experiment_exit_code,
        "locked_gpu_type_id": RUNPOD_LOCKED_GPU_TYPE_ID,
        "locked_image_name": RUNPOD_LOCKED_IMAGE_NAME,
        "container_disk_gb": RUNPOD_CONTAINER_DISK_GB,
        "volume_gb": RUNPOD_VOLUME_GB,
        "experiment_command": experiment_command,
    }


def parse_create_pod_response(response_body: dict[str, Any]) -> str:
    pod_id = response_body.get("id")
    if not pod_id:
        raise ValueError(f"RunPod pod creation did not return an id: {json.dumps(response_body)}")
    return str(pod_id)


@dataclass(frozen=True)
class PodSSHEndpoint:
    public_ip: str
    ssh_port: str


def parse_pod_status(response_body: dict[str, Any]) -> Optional[PodSSHEndpoint]:
    desired_status = response_body.get("desiredStatus", "")
    public_ip = response_body.get("publicIp", "")
    port_mappings = response_body.get("portMappings") or {}
    ssh_port = port_mappings.get("22", "")
    if desired_status == "RUNNING" and public_ip and ssh_port:
        return PodSSHEndpoint(public_ip=str(public_ip), ssh_port=str(ssh_port))
    return None


def build_remote_experiment_script(
    *,
    remote_root: str,
    experiment_command: str,
    experiment_timeout_seconds: int = RUNPOD_DEFAULT_EXPERIMENT_TIMEOUT_SECONDS,
) -> str:
    return f"""\
set -euo pipefail
export MAMBA_ROOT_PREFIX=/opt/micromamba
export PATH=/opt/micromamba/bin:$PATH
if ! command -v micromamba >/dev/null 2>&1; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba
fi
cd {remote_root}
micromamba create -y -n phage_env -f environment.yml
timeout {experiment_timeout_seconds} bash -lc "{experiment_command}" 2>&1 | tee runpod_experiment.log
exit ${{PIPESTATUS[0]}}
"""


# ---------------------------------------------------------------------------
# API / subprocess helpers (thin wrappers, side-effecting)
# ---------------------------------------------------------------------------


def _runpod_api_request(
    *,
    method: str,
    path: str,
    api_key: str,
    payload: Optional[dict[str, Any]] = None,
) -> tuple[int, dict[str, Any]]:
    url = f"{RUNPOD_API_BASE}{path}"
    cmd = [
        "curl",
        "--silent",
        "--show-error",
        "--output",
        "/dev/stdout",
        "--write-out",
        "\n%{http_code}",
        "--request",
        method,
        "--url",
        url,
        "--header",
        f"Authorization: Bearer {api_key}",
    ]
    if payload is not None:
        cmd += ["--header", "Content-Type: application/json", "--data", json.dumps(payload)]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    lines = result.stdout.rsplit("\n", 1)
    body_text = lines[0] if len(lines) > 1 else ""
    http_code_text = lines[-1].strip()

    try:
        http_code = int(http_code_text)
    except ValueError:
        raise RuntimeError(f"Failed to parse HTTP status from curl: {result.stdout!r}")

    try:
        body = json.loads(body_text) if body_text.strip() else {}
    except json.JSONDecodeError:
        body = {"_raw": body_text}

    return http_code, body


def create_pod(*, api_key: str, payload: dict[str, Any]) -> str:
    LOGGER.info("Creating RunPod pod: %s", payload.get("name", "unnamed"))
    http_code, body = _runpod_api_request(method="POST", path="/pods", api_key=api_key, payload=payload)
    LOGGER.info("RunPod create pod HTTP %d: %s", http_code, json.dumps(body))
    if http_code < 200 or http_code >= 300:
        raise RuntimeError(f"RunPod pod creation failed with HTTP {http_code}: {json.dumps(body)}")
    return parse_create_pod_response(body)


def wait_for_ssh(
    *,
    api_key: str,
    pod_id: str,
    poll_interval: int = POD_POLL_INTERVAL_SECONDS,
    max_attempts: int = POD_POLL_MAX_ATTEMPTS,
) -> PodSSHEndpoint:
    LOGGER.info("Waiting for pod %s SSH endpoint (max %ds)", pod_id, poll_interval * max_attempts)
    for attempt in range(1, max_attempts + 1):
        http_code, body = _runpod_api_request(method="GET", path=f"/pods/{pod_id}", api_key=api_key)
        endpoint = parse_pod_status(body)
        if endpoint:
            LOGGER.info(
                "Pod %s SSH ready at %s:%s (attempt %d)", pod_id, endpoint.public_ip, endpoint.ssh_port, attempt
            )
            return endpoint
        time.sleep(poll_interval)
    raise TimeoutError(
        f"Pod {pod_id} did not expose SSH within {poll_interval * max_attempts}s. Last status: {json.dumps(body)}"
    )


def delete_pod(*, api_key: str, pod_id: str) -> None:
    LOGGER.info("Deleting RunPod pod %s", pod_id)
    http_code, body = _runpod_api_request(method="DELETE", path=f"/pods/{pod_id}", api_key=api_key)
    LOGGER.info("RunPod delete pod HTTP %d: %s", http_code, json.dumps(body))


def _ssh_cmd(*, key_path: Path, host: str, port: str) -> list[str]:
    return [
        "ssh",
        "-i",
        str(key_path),
        "-o",
        "StrictHostKeyChecking=no",
        "-p",
        port,
        f"root@{host}",
    ]


def _scp_cmd(*, key_path: Path, host: str, port: str) -> list[str]:
    return [
        "scp",
        "-i",
        str(key_path),
        "-O",
        "-o",
        "StrictHostKeyChecking=no",
        "-P",
        port,
    ]


def sync_bundle(
    *,
    key_path: Path,
    endpoint: PodSSHEndpoint,
    staged_bundle_dir: Path,
    metadata_path: Path,
    remote_root: str,
) -> None:
    base_ssh = _ssh_cmd(key_path=key_path, host=endpoint.public_ip, port=endpoint.ssh_port)
    base_scp = _scp_cmd(key_path=key_path, host=endpoint.public_ip, port=endpoint.ssh_port)

    LOGGER.info("Creating remote directory %s", remote_root)
    subprocess.run([*base_ssh, f"mkdir -p {remote_root}"], check=True)

    bundle_tgz = staged_bundle_dir / BUNDLE_FILENAME
    bundle_manifest = staged_bundle_dir / BUNDLE_MANIFEST_FILENAME
    remote_dest = f"root@{endpoint.public_ip}:{remote_root}/"

    LOGGER.info("Uploading bundle to %s", remote_dest)
    subprocess.run([*base_scp, str(bundle_tgz), str(bundle_manifest), str(metadata_path), remote_dest], check=True)

    LOGGER.info("Extracting bundle on pod")
    subprocess.run([*base_ssh, f"cd {remote_root} && tar -xzf {BUNDLE_FILENAME}"], check=True)


def run_experiment(
    *,
    key_path: Path,
    endpoint: PodSSHEndpoint,
    remote_root: str,
    experiment_command: str,
    experiment_timeout_seconds: int = RUNPOD_DEFAULT_EXPERIMENT_TIMEOUT_SECONDS,
) -> int:
    script = build_remote_experiment_script(
        remote_root=remote_root,
        experiment_command=experiment_command,
        experiment_timeout_seconds=experiment_timeout_seconds,
    )
    base_ssh = _ssh_cmd(key_path=key_path, host=endpoint.public_ip, port=endpoint.ssh_port)
    LOGGER.info("Running experiment on pod: %s", experiment_command)
    result = subprocess.run([*base_ssh, "bash -s"], input=script, text=True, check=False)
    LOGGER.info("Experiment exited with code %d", result.returncode)
    return result.returncode


def collect_candidate(
    *,
    key_path: Path,
    endpoint: PodSSHEndpoint,
    remote_root: str,
    output_dir: Path,
    pod_id: str,
    experiment_exit_code: int,
    experiment_command: str,
) -> Path:
    base_ssh = _ssh_cmd(key_path=key_path, host=endpoint.public_ip, port=endpoint.ssh_port)
    base_scp = _scp_cmd(key_path=key_path, host=endpoint.public_ip, port=endpoint.ssh_port)
    output_subdir = RUNPOD_DEFAULT_REMOTE_ROOT / "lyzortx/generated_outputs/autoresearch/train_runs/runpod_candidate"

    collect_script = f"""\
set -euo pipefail
cd {remote_root}
mkdir -p candidate_bundle
cp lyzortx/autoresearch/train.py candidate_bundle/train.py
cp local_run_metadata.json candidate_bundle/local_run_metadata.json
cp {BUNDLE_MANIFEST_FILENAME} candidate_bundle/{BUNDLE_MANIFEST_FILENAME}
[ -f runpod_experiment.log ] && cp runpod_experiment.log candidate_bundle/runpod_experiment.log
for f in {" ".join(CANDIDATE_BUNDLE_FILES)}; do
  src="{output_subdir}/$f"
  [ -f "$src" ] && cp "$src" candidate_bundle/
done
tar -czf candidate_bundle.tgz -C candidate_bundle .
"""
    LOGGER.info("Collecting candidate artifacts from pod")
    subprocess.run([*base_ssh, "bash -s"], input=collect_script, text=True, check=True)

    execution_metadata = build_pod_execution_metadata(
        pod_id=pod_id,
        public_ip=endpoint.public_ip,
        ssh_port=endpoint.ssh_port,
        experiment_exit_code=experiment_exit_code,
        experiment_command=experiment_command,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    local_tgz = output_dir / "candidate_bundle.tgz"
    remote_tgz = f"root@{endpoint.public_ip}:{remote_root}/candidate_bundle.tgz"
    subprocess.run([*base_scp, remote_tgz, str(local_tgz)], check=True)

    metadata_path = output_dir / "runpod_execution_metadata.json"
    metadata_path.write_text(json.dumps(execution_metadata, indent=2), encoding="utf-8")

    import tarfile

    with tarfile.open(local_tgz, "r:gz") as archive:
        archive.extractall(path=output_dir, filter="data")

    LOGGER.info("Candidate artifacts collected in %s", output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def cmd_create_pod(args: argparse.Namespace) -> int:
    ssh_public_key = args.ssh_public_key_path.read_text(encoding="utf-8").strip()
    payload = build_create_pod_payload(ssh_public_key=ssh_public_key, pod_name=args.pod_name)
    _write_json(args.runtime_dir / "create_pod_payload.json", payload)

    pod_id = create_pod(api_key=args.api_key, payload=payload)
    (args.runtime_dir / "pod_id").write_text(pod_id, encoding="utf-8")
    LOGGER.info("Pod created: %s", pod_id)
    return 0


def cmd_wait_for_ssh(args: argparse.Namespace) -> int:
    pod_id = (args.runtime_dir / "pod_id").read_text(encoding="utf-8").strip()
    endpoint = wait_for_ssh(api_key=args.api_key, pod_id=pod_id)
    _write_json(
        args.runtime_dir / "ssh_endpoint.json", {"public_ip": endpoint.public_ip, "ssh_port": endpoint.ssh_port}
    )
    return 0


def cmd_sync_bundle(args: argparse.Namespace) -> int:
    endpoint_data = json.loads((args.runtime_dir / "ssh_endpoint.json").read_text(encoding="utf-8"))
    endpoint = PodSSHEndpoint(**endpoint_data)
    sync_bundle(
        key_path=args.ssh_key_path,
        endpoint=endpoint,
        staged_bundle_dir=args.staged_bundle_dir,
        metadata_path=args.runtime_dir / "local_run_metadata.json",
        remote_root=str(RUNPOD_DEFAULT_REMOTE_ROOT),
    )
    return 0


def cmd_run_experiment(args: argparse.Namespace) -> int:
    endpoint_data = json.loads((args.runtime_dir / "ssh_endpoint.json").read_text(encoding="utf-8"))
    endpoint = PodSSHEndpoint(**endpoint_data)
    exit_code = run_experiment(
        key_path=args.ssh_key_path,
        endpoint=endpoint,
        remote_root=str(RUNPOD_DEFAULT_REMOTE_ROOT),
        experiment_command=args.experiment_command,
    )
    (args.runtime_dir / "experiment_exit_code").write_text(str(exit_code), encoding="utf-8")
    return exit_code


def cmd_collect_candidate(args: argparse.Namespace) -> int:
    endpoint_data = json.loads((args.runtime_dir / "ssh_endpoint.json").read_text(encoding="utf-8"))
    endpoint = PodSSHEndpoint(**endpoint_data)
    pod_id = (args.runtime_dir / "pod_id").read_text(encoding="utf-8").strip()
    exit_code_text = (args.runtime_dir / "experiment_exit_code").read_text(encoding="utf-8").strip()
    collect_candidate(
        key_path=args.ssh_key_path,
        endpoint=endpoint,
        remote_root=str(RUNPOD_DEFAULT_REMOTE_ROOT),
        output_dir=args.output_dir,
        pod_id=pod_id,
        experiment_exit_code=int(exit_code_text),
        experiment_command=args.experiment_command,
    )
    return 0


def cmd_delete_pod(args: argparse.Namespace) -> int:
    pod_id_path = args.runtime_dir / "pod_id"
    if not pod_id_path.is_file():
        LOGGER.info("No pod_id file found — nothing to delete")
        return 0
    pod_id = pod_id_path.read_text(encoding="utf-8").strip()
    if not pod_id:
        LOGGER.info("Empty pod_id — nothing to delete")
        return 0
    delete_pod(api_key=args.api_key, pod_id=pod_id)
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=Path(".scratch/autoresearch-runpod/runtime"))
    parser.add_argument("--api-key", default="", help="RunPod API key (or set RUNPOD_API_KEY env var)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_create = subparsers.add_parser("create-pod")
    p_create.add_argument("--ssh-public-key-path", type=Path, required=True)
    p_create.add_argument("--pod-name", required=True)

    subparsers.add_parser("wait-for-ssh")

    p_sync = subparsers.add_parser("sync-bundle")
    p_sync.add_argument("--ssh-key-path", type=Path, required=True)
    p_sync.add_argument("--staged-bundle-dir", type=Path, required=True)

    p_run = subparsers.add_parser("run-experiment")
    p_run.add_argument("--ssh-key-path", type=Path, required=True)
    p_run.add_argument("--experiment-command", required=True)

    p_collect = subparsers.add_parser("collect-candidate")
    p_collect.add_argument("--ssh-key-path", type=Path, required=True)
    p_collect.add_argument("--experiment-command", required=True)
    p_collect.add_argument("--output-dir", type=Path, required=True)

    subparsers.add_parser("delete-pod")

    import os

    args = parser.parse_args(argv)
    if not args.api_key:
        args.api_key = os.environ.get("RUNPOD_API_KEY", "")
    return args


COMMAND_DISPATCH = {
    "create-pod": cmd_create_pod,
    "wait-for-ssh": cmd_wait_for_ssh,
    "sync-bundle": cmd_sync_bundle,
    "run-experiment": cmd_run_experiment,
    "collect-candidate": cmd_collect_candidate,
    "delete-pod": cmd_delete_pod,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_logging()
    args = parse_args(argv)
    return COMMAND_DISPATCH[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
