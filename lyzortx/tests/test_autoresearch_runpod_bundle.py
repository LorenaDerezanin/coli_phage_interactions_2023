from __future__ import annotations

import json
import tarfile
from pathlib import Path

from lyzortx.pipeline.autoresearch import runpod_bundle


def write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def create_minimal_runpod_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo_root = tmp_path / "repo"
    cache_dir = repo_root / "lyzortx" / "generated_outputs" / "autoresearch" / "search_cache_v1"

    write_text(
        repo_root / "environment.yml", "name: phage_env\nchannels:\n  - conda-forge\ndependencies:\n  - python=3.12\n"
    )
    write_text(repo_root / "pyproject.toml", '[build-system]\nrequires = ["setuptools"]\n')
    write_text(repo_root / "requirements.txt", "-e .\nlightgbm==4.6.0\n")
    write_text(repo_root / "lyzortx" / "__init__.py", "")
    write_text(repo_root / "lyzortx" / "log_config.py", "def setup_logging(*_args, **_kwargs):\n    return None\n")
    write_text(repo_root / "lyzortx" / "autoresearch" / "README.md", "# README\n")
    write_text(repo_root / "lyzortx" / "autoresearch" / "prepare.py", "print('prepare')\n")
    write_text(repo_root / "lyzortx" / "autoresearch" / "program.md", "# Program\n")
    write_text(repo_root / "lyzortx" / "autoresearch" / "train.py", "print('train')\n")
    write_text(
        repo_root / "lyzortx" / "pipeline" / "autoresearch" / "runtime_contract.py",
        "CACHE_MANIFEST_FILENAME = 'ar02_search_cache_manifest_v1.json'\n"
        "SCHEMA_MANIFEST_FILENAME = 'ar02_schema_manifest_v1.json'\n"
        "PROVENANCE_MANIFEST_FILENAME = 'ar02_provenance_manifest_v1.json'\n"
        "TRAIN_PAIR_TABLE_FILENAME = 'train_pairs.csv'\n"
        "INNER_VAL_PAIR_TABLE_FILENAME = 'inner_val_pairs.csv'\n"
        "def sha256_file(_path):\n    return 'checksum'\n",
    )
    write_text(cache_dir / "ar02_search_cache_manifest_v1.json", "{}\n")
    write_text(cache_dir / "ar02_schema_manifest_v1.json", "{}\n")
    write_text(cache_dir / "ar02_provenance_manifest_v1.json", "{}\n")
    write_text(cache_dir / "search_pairs" / "train_pairs.csv", "pair_id\nB1__P1\n")
    write_text(cache_dir / "search_pairs" / "inner_val_pairs.csv", "pair_id\nB2__P1\n")
    write_text(cache_dir / "feature_slots" / "host_surface" / "features.csv", "bacteria,host_surface__x\nB1,1\n")
    return repo_root, cache_dir


def test_relative_bundle_paths_include_runtime_support_and_cache(tmp_path: Path) -> None:
    repo_root, cache_dir = create_minimal_runpod_repo(tmp_path)
    write_text(repo_root / "ignore_me.txt", "not bundled\n")

    observed = [str(path) for path in runpod_bundle.relative_bundle_paths(repo_root=repo_root, cache_dir=cache_dir)]

    assert "environment.yml" in observed
    assert "pyproject.toml" in observed
    assert "requirements.txt" in observed
    assert "lyzortx/autoresearch/train.py" in observed
    assert "lyzortx/autoresearch/prepare.py" in observed
    assert "lyzortx/pipeline/autoresearch/runtime_contract.py" in observed
    assert "lyzortx/generated_outputs/autoresearch/search_cache_v1/search_pairs/train_pairs.csv" in observed
    assert "ignore_me.txt" not in observed


def test_write_bundle_creates_manifest_and_tarball(tmp_path: Path) -> None:
    repo_root, cache_dir = create_minimal_runpod_repo(tmp_path)
    output_dir = tmp_path / "bundle-output"

    tarball_path, manifest_path = runpod_bundle.write_bundle(
        repo_root=repo_root, cache_dir=cache_dir, output_dir=output_dir
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["bundle_contract_id"] == "autoresearch_runpod_bundle_v1"
    assert manifest["runpod_contract"]["environment_name"] == runpod_bundle.RUNPOD_ENVIRONMENT_NAME
    assert manifest["runpod_contract"]["required_secret_name"] == runpod_bundle.RUNPOD_REQUIRED_SECRET_NAME
    assert manifest["runpod_contract"]["locked_gpu_type_id"] == runpod_bundle.RUNPOD_LOCKED_GPU_TYPE_ID
    assert manifest["runpod_contract"]["locked_gpu_vram_gb"] == runpod_bundle.RUNPOD_LOCKED_GPU_VRAM_GB
    assert manifest["runpod_contract"]["locked_hourly_cost_usd"] == runpod_bundle.RUNPOD_LOCKED_HOURLY_COST_USD

    with tarfile.open(tarball_path, "r:gz") as archive:
        names = sorted(member.name for member in archive.getmembers() if member.isfile())

    assert "lyzortx/autoresearch/train.py" in names
    assert "lyzortx/autoresearch/prepare.py" in names
    assert "lyzortx/generated_outputs/autoresearch/search_cache_v1/ar02_search_cache_manifest_v1.json" in names
    assert "lyzortx/generated_outputs/autoresearch/search_cache_v1/search_pairs/inner_val_pairs.csv" in names
