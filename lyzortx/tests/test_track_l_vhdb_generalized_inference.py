from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lyzortx.pipeline.track_l.steps import generalized_inference
from lyzortx.pipeline.track_l.steps import validate_vhdb_generalized_inference as tl09


def test_match_panel_host_name_handles_aliases() -> None:
    panel_lookup = {
        tl09._normalize_host_name("LF82"): "LF82",
        tl09._normalize_host_name("EDL933"): "EDL933",
        tl09._normalize_host_name("55989"): "55989",
    }

    assert tl09.match_panel_host_name("Escherichia coli LF82", panel_lookup) == "LF82"
    assert tl09.match_panel_host_name("Escherichia coli O157:H7 str. EDL933", panel_lookup) == "EDL933"
    assert tl09.match_panel_host_name("Escherichia coli 55989", panel_lookup) == "55989"
    assert tl09.match_panel_host_name("Escherichia coli K-12", panel_lookup) == ""


def test_parse_vhdb_positive_pairs_filters_and_splits_realistic_tsv() -> None:
    vhdb_text = "\n".join(
        [
            "host tax id\thost name\thost lineage\trefseq id\tvirus name",
            (
                "83334\tEscherichia coli O157:H7 str. Sakai\t"
                "Bacteria; Pseudomonadati; Enterobacterales; Escherichia coli\t"
                "NC_001416.1; NC_001802.1\tEnterobacteria phage lambda"
            ),
            (
                "511145\tK-12 substr. MG1655\t"
                "Bacteria; Pseudomonadati; Enterobacterales; Escherichia coli\t"
                "NC_000866.4|NC_001422.1\tEnterobacteria phage P1"
            ),
            (
                "562\tEscherichia coli str. K-12 substr. MG1655\t"
                "Bacteria; Pseudomonadati; Enterobacterales; Escherichia coli\t"
                "NC_002655.2\tlab control phage"
            ),
            "9606\tHomo sapiens\tEukaryota; Metazoa\tNC_045512.2\tSevere acute respiratory syndrome virus",
        ]
    )

    pairs = tl09.parse_vhdb_positive_pairs(vhdb_text)

    assert [(pair.host_tax_id, pair.phage_accession) for pair in pairs] == [
        ("83334", "NC_001416.1"),
        ("83334", "NC_001802.1"),
        ("511145", "NC_000866.4"),
        ("511145", "NC_001422.1"),
    ]
    assert {pair.source_virus_name for pair in pairs} == {"Enterobacteria phage lambda", "Enterobacteria phage P1"}


def test_choose_best_assembly_prefers_latest_complete_reference() -> None:
    records = [
        tl09.AssemblyRecord(
            assembly_accession="GCF_000003.1",
            taxid="1",
            organism_name="Escherichia coli test",
            infraspecific_name="",
            isolate="",
            version_status="latest",
            assembly_level="Contig",
            refseq_category="na",
            ftp_path="ftp://example/3",
        ),
        tl09.AssemblyRecord(
            assembly_accession="GCF_000001.1",
            taxid="1",
            organism_name="Escherichia coli test",
            infraspecific_name="",
            isolate="",
            version_status="latest",
            assembly_level="Complete Genome",
            refseq_category="reference genome",
            ftp_path="ftp://example/1",
        ),
        tl09.AssemblyRecord(
            assembly_accession="GCF_000002.1",
            taxid="1",
            organism_name="Escherichia coli test",
            infraspecific_name="",
            isolate="",
            version_status="suppressed",
            assembly_level="Complete Genome",
            refseq_category="reference genome",
            ftp_path="ftp://example/2",
        ),
    ]

    chosen = tl09.choose_best_assembly(records)

    assert chosen.assembly_accession == "GCF_000001.1"


def test_build_host_candidates_tracks_positive_pairs_separately_from_unique_phages() -> None:
    pairs = [
        tl09.PositivePair(
            host_tax_id="1",
            host_name="Escherichia coli O177",
            phage_accession="NC_001416.1",
            source_virus_name="phage one",
        ),
        tl09.PositivePair(
            host_tax_id="1",
            host_name="Escherichia coli O177",
            phage_accession="NC_001416.1",
            source_virus_name="phage one duplicate evidence",
        ),
        tl09.PositivePair(
            host_tax_id="1",
            host_name="Escherichia coli O177",
            phage_accession="NC_001802.1",
            source_virus_name="phage two",
        ),
    ]
    assemblies_by_taxid = {
        "1": [
            tl09.AssemblyRecord(
                assembly_accession="GCF_000001.1",
                taxid="1",
                organism_name="Escherichia coli O177",
                infraspecific_name="",
                isolate="",
                version_status="latest",
                assembly_level="Complete Genome",
                refseq_category="reference genome",
                ftp_path="ftp://example/1",
            )
        ]
    }

    candidates = tl09.build_host_candidates(pairs, assemblies_by_taxid, panel_lookup={})

    assert len(candidates) == 1
    assert candidates[0].positive_pair_count == 3
    assert candidates[0].unique_phage_count == 2


def test_select_validation_hosts_separates_novel_and_roundtrip_hosts() -> None:
    host_candidates = [
        tl09.HostCandidate(
            host_tax_id="1",
            host_name="Escherichia coli O177",
            positive_pair_count=7,
            unique_phage_count=7,
            panel_match="",
            is_panel_host=False,
            assembly_accession="GCF_A",
            assembly_level="Scaffold",
            assembly_organism_name="Escherichia coli O177",
            assembly_ftp_path="ftp://example/A",
        ),
        tl09.HostCandidate(
            host_tax_id="2",
            host_name="Escherichia coli O78",
            positive_pair_count=5,
            unique_phage_count=5,
            panel_match="",
            is_panel_host=False,
            assembly_accession="GCF_B",
            assembly_level="Complete Genome",
            assembly_organism_name="Escherichia coli O78",
            assembly_ftp_path="ftp://example/B",
        ),
        tl09.HostCandidate(
            host_tax_id="3",
            host_name="Escherichia coli LF82",
            positive_pair_count=24,
            unique_phage_count=24,
            panel_match="LF82",
            is_panel_host=True,
            assembly_accession="GCF_C",
            assembly_level="Complete Genome",
            assembly_organism_name="Escherichia coli LF82",
            assembly_ftp_path="ftp://example/C",
        ),
        tl09.HostCandidate(
            host_tax_id="4",
            host_name="Escherichia coli O157:H7 str. EDL933",
            positive_pair_count=2,
            unique_phage_count=2,
            panel_match="EDL933",
            is_panel_host=True,
            assembly_accession="GCF_D",
            assembly_level="Complete Genome",
            assembly_organism_name="Escherichia coli O157:H7 str. EDL933",
            assembly_ftp_path="ftp://example/D",
        ),
    ]

    novel_hosts, roundtrip_hosts = tl09.select_validation_hosts(
        host_candidates,
        novel_host_count=2,
        min_positive_phages_per_host=5,
    )

    assert [host.host_tax_id for host in novel_hosts] == ["2", "1"]
    assert [host.panel_match for host in roundtrip_hosts] == ["LF82", "EDL933"]


def test_evaluate_positive_only_metrics_builds_rank_and_random_summaries() -> None:
    host_metadata = {
        "T1": tl09.HostCandidate(
            host_tax_id="T1",
            host_name="Host one",
            positive_pair_count=2,
            unique_phage_count=2,
            panel_match="",
            is_panel_host=False,
            assembly_accession="GCF_1",
            assembly_level="Complete Genome",
            assembly_organism_name="Host one",
            assembly_ftp_path="ftp://example/1",
        )
    }
    predictions = pd.DataFrame(
        [
            {"host_tax_id": "T1", "phage": "P1", "p_lysis": 0.9, "rank": 1},
            {"host_tax_id": "T1", "phage": "P2", "p_lysis": 0.7, "rank": 2},
            {"host_tax_id": "T1", "phage": "P3", "p_lysis": 0.2, "rank": 3},
            {"host_tax_id": "T1", "phage": "P4", "p_lysis": 0.1, "rank": 4},
        ]
    )

    positive_df, random_df, host_summary_df, overall = tl09.evaluate_positive_only_metrics(
        prediction_frames=[predictions],
        host_metadata=host_metadata,
        known_phages_by_taxid={"T1": ["P1", "P2"]},
        base_rate=0.30,
        random_seed=42,
    )

    assert set(positive_df["phage"]) == {"P1", "P2"}
    assert len(random_df) == 2
    assert host_summary_df.iloc[0]["positive_median_p_lysis"] == 0.8
    assert host_summary_df.iloc[0]["positive_median_rank_percentile"] == 5.0 / 6.0
    assert overall["positive_median_p_lysis"] == 0.8
    assert overall["host_count_above_median_rank"] == 1.0


def test_score_projected_features_uses_supplied_runtime_without_reloading(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyVectorizer:
        def transform(self, feature_dicts: list[dict[str, object]]) -> list[dict[str, object]]:
            return feature_dicts

    class DummyCalibrator:
        def predict(self, raw_probabilities: list[float]) -> list[float]:
            return raw_probabilities

    runtime = generalized_inference.InferenceRuntime(
        bundle_path=Path("bundle.joblib"),
        bundle={
            "feature_vectorizer": DummyVectorizer(),
            "lightgbm_estimator": object(),
            "isotonic_calibrator": DummyCalibrator(),
        },
        feature_space_payload={
            "categorical_columns": ["bacteria", "phage"],
            "numeric_columns": ["host_signal", "phage_signal"],
            "host_feature_columns": ["host_signal"],
            "phage_feature_columns": ["phage_signal"],
        },
        defense_mask_path=Path("defense-mask.csv"),
        phage_svd_path=Path("phage-svd.joblib"),
        panel_defense_subtypes_path=Path("panel-defense.csv"),
        models_dir=Path("models"),
    )

    def fail_load_runtime(model_path: str | Path) -> generalized_inference.InferenceRuntime:
        raise AssertionError(f"load_runtime should not be called when runtime is supplied: {model_path}")

    monkeypatch.setattr(generalized_inference, "load_runtime", fail_load_runtime)
    monkeypatch.setattr(
        generalized_inference.train_v1_binary_classifier,
        "build_feature_dict",
        lambda row, categorical_columns, numeric_columns: {
            column: row[column] for column in (*categorical_columns, *numeric_columns)
        },
    )
    monkeypatch.setattr(
        generalized_inference.train_v1_binary_classifier,
        "predict_probabilities",
        lambda estimator, matrix: [float(row["host_signal"]) + float(row["phage_signal"]) for row in matrix],
    )

    predictions = generalized_inference.score_projected_features(
        {"bacteria": "HostA", "host_signal": 0.5},
        [
            {"phage": "PhageB", "phage_signal": 0.1},
            {"phage": "PhageA", "phage_signal": 0.3},
        ],
        runtime=runtime,
    )

    assert predictions.to_dict("records") == [
        {"phage": "PhageA", "p_lysis": 0.8, "rank": 1},
        {"phage": "PhageB", "p_lysis": 0.6, "rank": 2},
    ]
