"""TL08 generalized inference for arbitrary host and phage genomes."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import pandas as pd

from lyzortx.pipeline.track_g.steps import train_v1_binary_classifier
from lyzortx.pipeline.track_l.steps import run_novel_host_defense_finder
from lyzortx.pipeline.track_l.steps.novel_organism_feature_projection import project_novel_phage

INFER_SCRATCH_ROOT = Path(".scratch/tl08_infer")


def _load_bundle(model_path: Path) -> dict[str, Any]:
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise TypeError(f"Unsupported TL08 bundle payload in {model_path}: {type(bundle)!r}")
    required_keys = {
        "lightgbm_estimator",
        "feature_vectorizer",
        "isotonic_calibrator",
        "feature_space",
        "artifacts",
        "runtime",
    }
    missing = sorted(key for key in required_keys if key not in bundle)
    if missing:
        raise ValueError(f"TL08 bundle loaded from {model_path} is missing keys: {', '.join(missing)}")
    return bundle


def _resolve_artifact_path(model_path: Path, filename: str) -> Path:
    artifact_path = model_path.parent / filename
    if not artifact_path.exists():
        raise FileNotFoundError(f"Bundle artifact not found next to {model_path}: {artifact_path}")
    return artifact_path


def _coerce_paths(paths: Iterable[str | Path]) -> list[Path]:
    resolved = [Path(path) for path in paths]
    if not resolved:
        raise ValueError("phage_fna_paths must contain at least one phage genome.")
    return resolved


def infer(host_genome_path: str | Path, phage_fna_paths: Sequence[str | Path], model_path: str | Path) -> pd.DataFrame:
    """Score one host genome against one or more phage genomes using a saved TL08 bundle."""

    host_genome = Path(host_genome_path)
    bundle_path = Path(model_path)
    phage_paths = _coerce_paths(phage_fna_paths)
    if not host_genome.exists():
        raise FileNotFoundError(f"Host genome FASTA not found: {host_genome}")
    if not bundle_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {bundle_path}")
    for phage_path in phage_paths:
        if not phage_path.exists():
            raise FileNotFoundError(f"Phage genome FASTA not found: {phage_path}")

    bundle = _load_bundle(bundle_path)
    feature_space_payload = dict(bundle["feature_space"])
    feature_space = train_v1_binary_classifier.FeatureSpace(
        categorical_columns=tuple(feature_space_payload.get("categorical_columns", [])),
        numeric_columns=tuple(feature_space_payload.get("numeric_columns", [])),
        track_c_additional_columns=tuple(feature_space_payload.get("host_feature_columns", [])),
        track_d_columns=tuple(feature_space_payload.get("phage_feature_columns", [])),
        track_e_columns=(),
    )
    defense_mask_path = _resolve_artifact_path(bundle_path, str(bundle["artifacts"]["defense_mask_filename"]))
    phage_svd_path = _resolve_artifact_path(bundle_path, str(bundle["artifacts"]["phage_svd_filename"]))
    panel_defense_subtypes_path = _resolve_artifact_path(
        bundle_path,
        str(bundle["artifacts"]["panel_defense_subtypes_filename"]),
    )
    models_dir = bundle_path.parent / str(bundle["runtime"]["defense_finder_models_dirname"])

    inference_id = uuid.uuid4().hex[:12]
    host_output_dir = INFER_SCRATCH_ROOT / f"{host_genome.stem}_{inference_id}"
    run_novel_host_defense_finder.run_novel_host_defense_finder(
        host_genome,
        output_dir=host_output_dir,
        column_mask_path=defense_mask_path,
        panel_defense_subtypes_path=panel_defense_subtypes_path,
        models_dir=models_dir,
        workers=0,
        force_model_update=False,
        force_run=False,
        preserve_raw=False,
    )
    host_feature_path = host_output_dir / "novel_host_defense_features.csv"
    host_row = pd.read_csv(host_feature_path).iloc[0].to_dict()

    feature_rows: list[dict[str, object]] = []
    for phage_path in phage_paths:
        phage_row = project_novel_phage(phage_path, phage_svd_path)
        merged_row: dict[str, object] = {
            "bacteria": str(host_row["bacteria"]),
            "phage": str(phage_row["phage"]),
        }
        for column in feature_space_payload.get("host_feature_columns", []):
            merged_row[column] = host_row[column]
        for column in feature_space_payload.get("phage_feature_columns", []):
            merged_row[column] = phage_row[column]
        feature_rows.append(merged_row)

    vectorizer = bundle["feature_vectorizer"]
    estimator = bundle["lightgbm_estimator"]
    calibrator = bundle["isotonic_calibrator"]
    feature_dicts = [
        train_v1_binary_classifier.build_feature_dict(
            row,
            categorical_columns=feature_space.categorical_columns,
            numeric_columns=feature_space.numeric_columns,
        )
        for row in feature_rows
    ]
    raw_probabilities = train_v1_binary_classifier.predict_probabilities(estimator, vectorizer.transform(feature_dicts))
    calibrated_probabilities = [float(value) for value in calibrator.predict(raw_probabilities)]

    output_rows = [
        {"phage": row["phage"], "p_lysis": probability}
        for row, probability in zip(feature_rows, calibrated_probabilities)
    ]
    ranked = sorted(output_rows, key=lambda row: (-float(row["p_lysis"]), str(row["phage"])))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return pd.DataFrame(ranked, columns=["phage", "p_lysis", "rank"])
