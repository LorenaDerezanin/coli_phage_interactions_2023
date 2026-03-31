"""TL08 generalized inference for arbitrary host and phage genomes."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import pandas as pd

from lyzortx.pipeline.track_g.steps import train_v1_binary_classifier
from lyzortx.pipeline.track_l.steps.deployable_tl04_runtime import (
    TL04_DIRECT_BLOCK_ID,
    Tl04ProfileRuntime,
    build_direct_feature_values,
    build_pairwise_feature_values,
    build_profile_presence,
    extract_antidef_feature_names,
    parse_tl04_runtime_payload,
)
from lyzortx.pipeline.track_l.steps import run_novel_host_defense_finder
from lyzortx.pipeline.track_l.steps.novel_organism_feature_projection import project_novel_phage
from lyzortx.pipeline.track_l.steps.run_pharokka import run_pharokka_on_file, verify_annotations

INFER_SCRATCH_ROOT = Path(".scratch/tl08_infer")


@dataclass(frozen=True)
class InferenceRuntime:
    bundle_path: Path
    bundle: dict[str, Any]
    feature_space_payload: dict[str, Any]
    defense_mask_path: Path
    phage_svd_path: Path
    panel_defense_subtypes_path: Path
    models_dir: Path
    tl04_runtime_payload: dict[str, Any] | None = None
    panel_annotation_cache_dir: Path | None = None


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


def _resolve_runtime(
    model_path: str | Path | None,
    runtime: InferenceRuntime | None,
) -> InferenceRuntime:
    if runtime is not None:
        return runtime
    if model_path is None:
        raise ValueError("model_path is required when runtime is not provided.")
    return load_runtime(model_path)


def load_runtime(model_path: str | Path) -> InferenceRuntime:
    bundle_path = Path(model_path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {bundle_path}")

    bundle = _load_bundle(bundle_path)
    feature_space_payload = dict(bundle["feature_space"])
    defense_mask_path = _resolve_artifact_path(bundle_path, str(bundle["artifacts"]["defense_mask_filename"]))
    phage_svd_path = _resolve_artifact_path(bundle_path, str(bundle["artifacts"]["phage_svd_filename"]))
    panel_defense_subtypes_path = _resolve_artifact_path(
        bundle_path,
        str(bundle["artifacts"]["panel_defense_subtypes_filename"]),
    )
    models_dir = bundle_path.parent / str(bundle["runtime"]["defense_finder_models_dirname"])
    tl04_runtime_payload = dict(bundle.get("deployable_runtime", {}).get(TL04_DIRECT_BLOCK_ID, {}))
    panel_annotation_cache_dir = None
    if tl04_runtime_payload:
        cache_dirname = str(tl04_runtime_payload.get("panel_annotation_cache_dirname", "")).strip()
        if cache_dirname:
            panel_annotation_cache_dir = bundle_path.parent / cache_dirname
    return InferenceRuntime(
        bundle_path=bundle_path,
        bundle=bundle,
        feature_space_payload=feature_space_payload,
        defense_mask_path=defense_mask_path,
        phage_svd_path=phage_svd_path,
        panel_defense_subtypes_path=panel_defense_subtypes_path,
        models_dir=models_dir,
        tl04_runtime_payload=tl04_runtime_payload,
        panel_annotation_cache_dir=panel_annotation_cache_dir,
    )


def _feature_space(feature_space_payload: dict[str, Any]) -> train_v1_binary_classifier.FeatureSpace:
    return train_v1_binary_classifier.FeatureSpace(
        categorical_columns=tuple(feature_space_payload.get("categorical_columns", [])),
        numeric_columns=tuple(feature_space_payload.get("numeric_columns", [])),
        track_c_additional_columns=tuple(feature_space_payload.get("host_feature_columns", [])),
        track_d_columns=tuple(feature_space_payload.get("phage_feature_columns", [])),
        track_e_columns=tuple(feature_space_payload.get("pairwise_feature_columns", [])),
    )


def _resolve_annotation_tsv_path(
    phage_path: Path,
    *,
    runtime: InferenceRuntime,
    annotation_tsv_paths: dict[str, Path] | None,
    pharokka_database_dir: Path | None,
) -> Path:
    if annotation_tsv_paths is not None:
        resolved = annotation_tsv_paths.get(phage_path.stem)
        if resolved is not None:
            return resolved
    if runtime.panel_annotation_cache_dir is not None:
        cached_path = runtime.panel_annotation_cache_dir / f"{phage_path.stem}_cds_final_merged_output.tsv"
        if cached_path.exists():
            return cached_path
    if pharokka_database_dir is None:
        raise FileNotFoundError(
            f"Bundle requires Pharokka anti-defense annotations for {phage_path.stem}, but no cached TSV or "
            "pharokka_database_dir was supplied."
        )
    phage_output_dir = INFER_SCRATCH_ROOT / f"{phage_path.stem}_pharokka"
    run_pharokka_on_file(
        fna_path=phage_path,
        output_dir=phage_output_dir.parent,
        database_dir=pharokka_database_dir,
        threads=1,
        force=False,
    )
    verify_annotations(phage_output_dir, phage_path.stem)
    annotation_tsv_path = phage_output_dir / f"{phage_path.stem}_cds_final_merged_output.tsv"
    if not annotation_tsv_path.exists():
        raise FileNotFoundError(f"Expected Pharokka merged TSV at {annotation_tsv_path}")
    return annotation_tsv_path


def _augment_phage_row_with_tl04_features(
    phage_row: dict[str, object],
    *,
    annotation_tsv_path: Path,
    tl04_profiles: Sequence[Tl04ProfileRuntime],
) -> None:
    antidef_features = extract_antidef_feature_names(annotation_tsv_path)
    profile_presence = build_profile_presence(antidef_features, tl04_profiles)
    phage_row.update(build_direct_feature_values(profile_presence, tl04_profiles))


def project_host_features(
    host_genome_path: str | Path,
    model_path: str | Path | None = None,
    *,
    bacteria_id: str | None = None,
    runtime: InferenceRuntime | None = None,
) -> dict[str, object]:
    """Project one host assembly into the feature space expected by a TL08 bundle."""

    host_genome = Path(host_genome_path)
    if not host_genome.exists():
        raise FileNotFoundError(f"Host genome FASTA not found: {host_genome}")
    runtime = _resolve_runtime(model_path, runtime)

    inference_id = uuid.uuid4().hex[:12]
    host_output_dir = INFER_SCRATCH_ROOT / f"{host_genome.stem}_{inference_id}"
    runner_kwargs = {
        "output_dir": host_output_dir,
        "column_mask_path": runtime.defense_mask_path,
        "panel_defense_subtypes_path": runtime.panel_defense_subtypes_path,
        "models_dir": runtime.models_dir,
        "workers": 0,
        "force_model_update": False,
        "force_run": False,
        "preserve_raw": False,
    }
    if bacteria_id is not None:
        runner_kwargs["bacteria_id"] = bacteria_id
    run_novel_host_defense_finder.run_novel_host_defense_finder(
        host_genome,
        **runner_kwargs,
    )
    host_feature_path = host_output_dir / "novel_host_defense_features.csv"
    return pd.read_csv(host_feature_path).iloc[0].to_dict()


def project_phage_features(
    phage_fna_paths: Sequence[str | Path],
    model_path: str | Path | None = None,
    *,
    runtime: InferenceRuntime | None = None,
    annotation_tsv_paths: dict[str, str | Path] | None = None,
    pharokka_database_dir: str | Path | None = None,
) -> list[dict[str, object]]:
    """Project one or more phage genomes into the feature space expected by a TL08 bundle."""

    phage_paths = _coerce_paths(phage_fna_paths)
    runtime = _resolve_runtime(model_path, runtime)
    resolved_annotation_paths = (
        {name: Path(path) for name, path in annotation_tsv_paths.items()} if annotation_tsv_paths is not None else None
    )
    resolved_pharokka_database_dir = Path(pharokka_database_dir) if pharokka_database_dir is not None else None
    if runtime.tl04_runtime_payload:
        tl04_profiles, _ = parse_tl04_runtime_payload(runtime.tl04_runtime_payload)
    else:
        tl04_profiles = []
    projected_rows: list[dict[str, object]] = []
    for phage_path in phage_paths:
        if not phage_path.exists():
            raise FileNotFoundError(f"Phage genome FASTA not found: {phage_path}")
        projected_row = project_novel_phage(phage_path, runtime.phage_svd_path)
        if runtime.tl04_runtime_payload:
            annotation_tsv_path = _resolve_annotation_tsv_path(
                phage_path,
                runtime=runtime,
                annotation_tsv_paths=resolved_annotation_paths,
                pharokka_database_dir=resolved_pharokka_database_dir,
            )
            _augment_phage_row_with_tl04_features(
                projected_row,
                annotation_tsv_path=annotation_tsv_path,
                tl04_profiles=tl04_profiles,
            )
        projected_rows.append(projected_row)
    return projected_rows


def score_projected_features(
    host_row: dict[str, object],
    phage_rows: Sequence[dict[str, object]],
    model_path: str | Path | None = None,
    *,
    runtime: InferenceRuntime | None = None,
) -> pd.DataFrame:
    """Score one projected host row against projected phage rows using a TL08 bundle."""

    if not phage_rows:
        raise ValueError("phage_rows must contain at least one projected phage feature row.")
    runtime = _resolve_runtime(model_path, runtime)
    feature_space = _feature_space(runtime.feature_space_payload)
    if runtime.tl04_runtime_payload:
        tl04_profiles, tl04_associations = parse_tl04_runtime_payload(runtime.tl04_runtime_payload)
    else:
        tl04_profiles, tl04_associations = [], []

    feature_rows: list[dict[str, object]] = []
    for phage_row in phage_rows:
        merged_row: dict[str, object] = {
            "bacteria": str(host_row["bacteria"]),
            "phage": str(phage_row["phage"]),
        }
        for column in runtime.feature_space_payload.get("host_feature_columns", []):
            merged_row[column] = host_row[column]
        for column in runtime.feature_space_payload.get("phage_feature_columns", []):
            merged_row[column] = phage_row[column]
        if tl04_associations:
            profile_presence = {
                profile.profile_id: int(phage_row.get(profile.direct_column, 0) or 0) for profile in tl04_profiles
            }
            merged_row.update(
                build_pairwise_feature_values(
                    host_row=host_row,
                    profile_presence=profile_presence,
                    associations=tl04_associations,
                )
            )
        feature_rows.append(merged_row)

    vectorizer = runtime.bundle["feature_vectorizer"]
    estimator = runtime.bundle["lightgbm_estimator"]
    calibrator = runtime.bundle["isotonic_calibrator"]
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


def infer(
    host_genome_path: str | Path,
    phage_fna_paths: Sequence[str | Path],
    model_path: str | Path,
    *,
    annotation_tsv_paths: dict[str, str | Path] | None = None,
    pharokka_database_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Score one host genome against one or more phage genomes using a saved TL08 bundle."""

    host_row = project_host_features(host_genome_path, model_path)
    phage_rows = project_phage_features(
        phage_fna_paths,
        model_path,
        annotation_tsv_paths=annotation_tsv_paths,
        pharokka_database_dir=pharokka_database_dir,
    )
    return score_projected_features(host_row, phage_rows, model_path)
