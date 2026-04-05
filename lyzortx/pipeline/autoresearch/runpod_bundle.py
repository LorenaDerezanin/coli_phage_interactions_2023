#!/usr/bin/env python3
"""Package the minimal AUTORESEARCH runtime bundle for RunPod experiments."""

from __future__ import annotations

import argparse
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from lyzortx.pipeline.autoresearch import runtime_contract

BUNDLE_FILENAME = "autoresearch_runpod_bundle.tgz"
BUNDLE_MANIFEST_FILENAME = "autoresearch_runpod_bundle_manifest.json"
RUNPOD_ENVIRONMENT_NAME = "runpod-autoresearch"
RUNPOD_REQUIRED_SECRET_NAME = "RUNPOD_API_KEY"
RUNPOD_LOCKED_GPU_TYPE_ID = "NVIDIA A40"
RUNPOD_LOCKED_GPU_DISPLAY_NAME = "A40"
RUNPOD_LOCKED_GPU_VRAM_GB = 48
RUNPOD_LOCKED_HOURLY_COST_USD = 0.35
RUNPOD_LOCKED_IMAGE_NAME = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
RUNPOD_LOCKED_GPU_COUNT = 1
RUNPOD_CONTAINER_DISK_GB = 50
RUNPOD_VOLUME_GB = 20
RUNPOD_SUPPORT_PUBLIC_IP = True
RUNPOD_DEFAULT_EXPERIMENT_TIMEOUT_SECONDS = 2100
RUNPOD_DEFAULT_REMOTE_ROOT = Path("/workspace/autoresearch-runpod")
RUNPOD_DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/autoresearch/train_runs/runpod_candidate")
RUNPOD_DEFAULT_EXPERIMENT_COMMAND = (
    "micromamba run -n phage_env python lyzortx/autoresearch/train.py "
    "--device-type gpu --output-dir lyzortx/generated_outputs/autoresearch/train_runs/runpod_candidate"
)

RUNTIME_BUNDLE_PATHS = (
    Path("environment.yml"),
    Path("pyproject.toml"),
    Path("requirements.txt"),
    Path("lyzortx/__init__.py"),
    Path("lyzortx/log_config.py"),
    Path("lyzortx/autoresearch/README.md"),
    Path("lyzortx/autoresearch/prepare.py"),
    Path("lyzortx/autoresearch/program.md"),
    Path("lyzortx/autoresearch/train.py"),
    Path("lyzortx/pipeline/autoresearch/runtime_contract.py"),
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root containing the AUTORESEARCH runtime files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=runtime_contract.DEFAULT_CACHE_DIR,
        help="Prepared AUTORESEARCH search cache directory from prepare.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".scratch/autoresearch_runpod_bundle"),
        help="Directory where the tarball and manifest should be written.",
    )
    return parser.parse_args(argv)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    return value


def resolve_runtime_bundle_paths(*, repo_root: Path, cache_dir: Path) -> list[Path]:
    resolved_repo_root = repo_root.resolve()
    resolved_cache_dir = cache_dir.resolve()
    bundle_paths: list[Path] = []
    for relative_path in RUNTIME_BUNDLE_PATHS:
        path = (resolved_repo_root / relative_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Required RunPod bundle path does not exist: {relative_path}")
        if not path.is_file():
            raise FileNotFoundError(f"RunPod bundle path must be a file: {relative_path}")
        bundle_paths.append(path)

    required_cache_manifest_paths = (
        resolved_cache_dir / runtime_contract.CACHE_MANIFEST_FILENAME,
        resolved_cache_dir / runtime_contract.SCHEMA_MANIFEST_FILENAME,
        resolved_cache_dir / runtime_contract.PROVENANCE_MANIFEST_FILENAME,
        resolved_cache_dir / "search_pairs" / runtime_contract.TRAIN_PAIR_TABLE_FILENAME,
        resolved_cache_dir / "search_pairs" / runtime_contract.INNER_VAL_PAIR_TABLE_FILENAME,
    )
    for path in required_cache_manifest_paths:
        if not path.exists():
            raise FileNotFoundError(f"Prepared search cache is missing required RunPod bundle input: {path}")

    cache_files = sorted(path for path in resolved_cache_dir.rglob("*") if path.is_file())
    if not cache_files:
        raise FileNotFoundError(f"Prepared search cache has no files to bundle: {cache_dir}")
    bundle_paths.extend(cache_files)
    return bundle_paths


def relative_bundle_paths(*, repo_root: Path, cache_dir: Path) -> list[Path]:
    resolved_repo_root = repo_root.resolve()
    return sorted(
        path.relative_to(resolved_repo_root)
        for path in resolve_runtime_bundle_paths(repo_root=repo_root, cache_dir=cache_dir)
    )


def build_bundle_manifest(*, repo_root: Path, cache_dir: Path, tarball_path: Path) -> dict[str, Any]:
    relative_paths = relative_bundle_paths(repo_root=repo_root, cache_dir=cache_dir)
    resolved_repo_root = repo_root.resolve()
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_contract_id": "autoresearch_runpod_bundle_v1",
        "repo_root": str(resolved_repo_root),
        "cache_dir": str(cache_dir.resolve()),
        "tarball_path": str(tarball_path.resolve()),
        "runtime_bundle_paths": [str(path) for path in relative_paths],
        "path_checksums": {
            str(path): runtime_contract.sha256_file(resolved_repo_root / path) for path in relative_paths
        },
        "runpod_contract": {
            "environment_name": RUNPOD_ENVIRONMENT_NAME,
            "required_secret_name": RUNPOD_REQUIRED_SECRET_NAME,
            "locked_gpu_type_id": RUNPOD_LOCKED_GPU_TYPE_ID,
            "locked_gpu_display_name": RUNPOD_LOCKED_GPU_DISPLAY_NAME,
            "locked_gpu_vram_gb": RUNPOD_LOCKED_GPU_VRAM_GB,
            "locked_hourly_cost_usd": RUNPOD_LOCKED_HOURLY_COST_USD,
            "locked_gpu_count": RUNPOD_LOCKED_GPU_COUNT,
            "locked_image_name": RUNPOD_LOCKED_IMAGE_NAME,
            "container_disk_gb": RUNPOD_CONTAINER_DISK_GB,
            "volume_gb": RUNPOD_VOLUME_GB,
            "support_public_ip": RUNPOD_SUPPORT_PUBLIC_IP,
            "default_experiment_timeout_seconds": RUNPOD_DEFAULT_EXPERIMENT_TIMEOUT_SECONDS,
            "default_remote_root": str(RUNPOD_DEFAULT_REMOTE_ROOT),
            "default_output_dir": str(RUNPOD_DEFAULT_OUTPUT_DIR),
            "default_experiment_command": RUNPOD_DEFAULT_EXPERIMENT_COMMAND,
            "selection_reason": (
                "A40 locks the workflow to a single 48 GB GPU with enough headroom for the frozen cache and "
                "future ablations while staying near the lowest-cost 24 GB community options."
            ),
        },
    }


def write_bundle(*, repo_root: Path, cache_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    resolved_repo_root = repo_root.resolve()
    resolved_output_dir = output_dir.resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    tarball_path = resolved_output_dir / BUNDLE_FILENAME
    manifest_path = resolved_output_dir / BUNDLE_MANIFEST_FILENAME
    paths = resolve_runtime_bundle_paths(repo_root=resolved_repo_root, cache_dir=cache_dir)

    with tarfile.open(tarball_path, "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=path.relative_to(resolved_repo_root))

    manifest = build_bundle_manifest(repo_root=resolved_repo_root, cache_dir=cache_dir, tarball_path=tarball_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    return tarball_path, manifest_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    tarball_path, manifest_path = write_bundle(
        repo_root=args.repo_root, cache_dir=args.cache_dir, output_dir=args.output_dir
    )
    print(f"Created RunPod bundle tarball: {tarball_path}")
    print(f"Created RunPod bundle manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
