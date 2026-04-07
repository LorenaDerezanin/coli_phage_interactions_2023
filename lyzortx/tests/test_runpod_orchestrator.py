"""Tests for the RunPod orchestrator pure data builders."""

from __future__ import annotations

from lyzortx.pipeline.autoresearch import runpod_orchestrator as orch


def test_build_create_pod_payload_field_names() -> None:
    payload = orch.build_create_pod_payload(ssh_public_key="ssh-ed25519 AAAA", pod_name="test-pod-1")

    assert payload["cloudType"] == "COMMUNITY"
    assert payload["gpuTypeId"] == "NVIDIA A40"
    assert payload["gpuCount"] == 1
    assert payload["containerDiskInGb"] == 50
    assert payload["volumeInGb"] == 20
    assert payload["volumeMountPath"] == "/workspace"
    assert payload["imageName"] == orch.RUNPOD_LOCKED_IMAGE_NAME
    assert payload["name"] == "test-pod-1"
    assert payload["ports"] == ["22/tcp"]
    assert payload["supportPublicIp"] is True
    assert payload["env"] == {"SSH_PUBLIC_KEY": "ssh-ed25519 AAAA"}
    assert payload["dockerArgs"] == ""

    # These fields must NOT be present — they cause 500 errors on the RunPod REST v1 API
    assert "gpuTypeIds" not in payload
    assert "computeType" not in payload
    assert "allowedCudaVersions" not in payload


def test_build_create_pod_payload_field_types() -> None:
    """gpuTypeId must be a string, ports must be an array."""
    payload = orch.build_create_pod_payload(ssh_public_key="key", pod_name="p")
    assert isinstance(payload["gpuTypeId"], str)
    assert isinstance(payload["ports"], list)


def test_parse_create_pod_response_extracts_id() -> None:
    assert orch.parse_create_pod_response({"id": "abc123", "status": "ok"}) == "abc123"


def test_parse_create_pod_response_rejects_missing_id() -> None:
    import pytest

    with pytest.raises(ValueError, match="did not return an id"):
        orch.parse_create_pod_response({"error": "something went wrong"})


def test_parse_pod_status_running_with_ssh() -> None:
    body = {"desiredStatus": "RUNNING", "publicIp": "1.2.3.4", "portMappings": {"22": "10022"}}
    endpoint = orch.parse_pod_status(body)
    assert endpoint is not None
    assert endpoint.public_ip == "1.2.3.4"
    assert endpoint.ssh_port == "10022"


def test_parse_pod_status_not_ready() -> None:
    assert orch.parse_pod_status({"desiredStatus": "CREATED", "publicIp": "", "portMappings": {}}) is None
    assert orch.parse_pod_status({"desiredStatus": "RUNNING", "publicIp": "", "portMappings": {}}) is None
    assert orch.parse_pod_status({"desiredStatus": "RUNNING", "publicIp": "1.2.3.4", "portMappings": {}}) is None


def test_build_experiment_metadata_keys() -> None:
    meta = orch.build_experiment_metadata(github_run_id="123", experiment_command="python train.py")
    assert meta["github_run_id"] == "123"
    assert meta["experiment_command"] == "python train.py"
    assert "github_repository" in meta


def test_build_pod_execution_metadata() -> None:
    meta = orch.build_pod_execution_metadata(
        pod_id="pod-1",
        public_ip="1.2.3.4",
        ssh_port="22",
        experiment_exit_code=0,
        experiment_command="python train.py",
    )
    assert meta["pod_id"] == "pod-1"
    assert meta["experiment_exit_code"] == 0
    assert meta["locked_gpu_type_id"] == "NVIDIA A40"


def test_build_remote_experiment_script_contains_command() -> None:
    script = orch.build_remote_experiment_script(
        remote_root="/workspace/test",
        experiment_command="python train.py --gpu",
        experiment_timeout_seconds=600,
    )
    assert "python train.py --gpu" in script
    assert "timeout 600" in script
    assert "micromamba create" in script
    assert "cd /workspace/test" in script
