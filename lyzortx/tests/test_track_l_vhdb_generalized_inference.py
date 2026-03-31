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


def test_assess_tl13_gate_requires_extra_block_and_roundtrip_improvement() -> None:
    passed = tl09.assess_tl13_gate(
        {
            "task_id": "TL13",
            "format_version": "tl13_bundle_v1",
            "deployable_feature_blocks": [
                {"block_id": "track_c_defense"},
                {"block_id": "track_d_phage_genomic_kmers"},
                {"block_id": "tl04_antidef_defense_evasion"},
            ],
            "roundtrip_gate": {"improved_metrics": ["max_abs_probability_delta_max"]},
        }
    )
    failed = tl09.assess_tl13_gate(
        {
            "task_id": "TL08",
            "format_version": "tl08_bundle_v2",
            "deployable_feature_blocks": [
                {"block_id": "track_c_defense"},
                {"block_id": "track_d_phage_genomic_kmers"},
            ],
            "roundtrip_gate": {"improved_metrics": []},
        }
    )

    assert passed.passed is True
    assert passed.extra_deployable_block_ids == ("tl04_antidef_defense_evasion",)
    assert passed.improved_roundtrip_metrics == ("max_abs_probability_delta_max",)
    assert failed.passed is False
    assert failed.failure_reasons == (
        "bundle_task_id_was_not_tl13",
        "bundle_did_not_add_any_deployable_feature_block_beyond_defense_and_kmers",
        "bundle_did_not_record_any_improved_roundtrip_metric",
    )


def test_load_saved_roundtrip_contract_returns_named_fields(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    reference_path = bundle_dir / "roundtrip_reference_predictions.csv"
    reference_path.write_text("bacteria\nEDL933\nLF82\n", encoding="utf-8")
    cohort_path = bundle_dir / "roundtrip_host_cohort.csv"
    cohort_path.write_text("panel_match\nEDL933\nLF82\n", encoding="utf-8")
    bundle_path = bundle_dir / "bundle.joblib"
    joblib_payload = {
        "task_id": "TL13",
        "format_version": "tl13_bundle_v1",
        "deployable_feature_blocks": [
            {"block_id": "track_c_defense"},
            {"block_id": "track_d_phage_genomic_kmers"},
            {"block_id": "tl04_antidef_defense_evasion"},
        ],
        "roundtrip_gate": {"improved_metrics": ["max_abs_probability_delta_max"]},
        "artifacts": {
            "roundtrip_reference_predictions_filename": reference_path.name,
            "roundtrip_host_cohort_filename": cohort_path.name,
        },
    }
    tl09.joblib.dump(joblib_payload, bundle_path)

    contract = tl09.load_saved_roundtrip_contract(bundle_path)

    assert isinstance(contract, tl09.SavedRoundtripContract)
    assert contract.gate_assessment.passed is True
    assert contract.roundtrip_reference_path == reference_path
    assert contract.roundtrip_cohort_path == cohort_path
    assert contract.roundtrip_reference_hosts == {"EDL933", "LF82"}
    assert contract.roundtrip_cohort_hosts == {"EDL933", "LF82"}
    assert contract.contract_issues == []


def test_build_validation_cohort_rows_keeps_counts_and_candidate_set_sizes_separate() -> None:
    hosts = [
        tl09.HostCandidate(
            host_tax_id="H1",
            host_name="Host one",
            positive_pair_count=3,
            unique_phage_count=2,
            panel_match="EDL933",
            is_panel_host=True,
            assembly_accession="GCF_1",
            assembly_level="Complete Genome",
            assembly_organism_name="Host one",
            assembly_ftp_path="ftp://example/1",
        ),
        tl09.HostCandidate(
            host_tax_id="H2",
            host_name="Host two",
            positive_pair_count=1,
            unique_phage_count=1,
            panel_match="",
            is_panel_host=False,
            assembly_accession="GCF_2",
            assembly_level="Scaffold",
            assembly_organism_name="Host two",
            assembly_ftp_path="ftp://example/2",
        ),
    ]
    positive_pairs = [
        tl09.PositivePair("H1", "Host one", "P1", "virus-a"),
        tl09.PositivePair("H1", "Host one", "P1", "virus-a-repeat"),
        tl09.PositivePair("H1", "Host one", "P2", "virus-b"),
        tl09.PositivePair("H2", "Host two", "P3", "virus-c"),
    ]

    host_rows, pair_rows, summary = tl09.build_validation_cohort_rows(
        hosts=hosts,
        positive_pairs=positive_pairs,
        known_phages_by_taxid={"H1": ["P1", "P2"], "H2": ["P3"]},
        panel_phage_count=96,
        roundtrip_panel_matches={"EDL933"},
    )

    assert summary == {
        "host_count": 2,
        "positive_pair_count": 4,
        "unique_phage_count": 3,
        "roundtrip_host_count": 1,
    }
    assert [row["candidate_set_size"] for row in host_rows] == [98, 97]
    assert [row["qualifies_for_panel_roundtrip"] for row in host_rows] == [1, 0]
    assert [row["positive_pair_count"] for row in pair_rows if row["host_tax_id"] == "H1"] == [3, 3, 3]
    assert [row["unique_phage_count"] for row in pair_rows if row["host_tax_id"] == "H1"] == [2, 2, 2]


def test_determine_validation_conclusion_distinguishes_failed_and_inconclusive() -> None:
    failed_conclusion, failed_rows = tl09.determine_validation_conclusion(
        gate_passed=False,
        contract_issues=[],
        qualified_roundtrip_host_count=5,
        min_roundtrip_hosts=3,
    )
    inconclusive_conclusion, inconclusive_rows = tl09.determine_validation_conclusion(
        gate_passed=True,
        contract_issues=["saved_roundtrip_reference_predictions_file_was_missing"],
        qualified_roundtrip_host_count=1,
        min_roundtrip_hosts=3,
    )
    validated_conclusion, validated_rows = tl09.determine_validation_conclusion(
        gate_passed=True,
        contract_issues=[],
        qualified_roundtrip_host_count=3,
        min_roundtrip_hosts=3,
        overall_metrics={
            "base_rate": 0.2,
            "positive_median_p_lysis": 0.8,
            "random_median_p_lysis": 0.3,
            "host_median_positive_rank_percentile": 0.7,
        },
    )

    assert failed_conclusion == tl09.CONCLUSION_FAILED
    assert failed_rows[0]["passed"] == 0
    assert inconclusive_conclusion == tl09.CONCLUSION_INCONCLUSIVE
    assert inconclusive_rows[1]["passed"] == 0
    assert validated_conclusion == tl09.CONCLUSION_VALIDATED
    assert all(row["passed"] == 1 for row in validated_rows)


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


def test_main_fails_fast_when_contract_filters_validation_cohort_to_zero_hosts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "outputs"
    panel_hosts_path = tmp_path / "panel_hosts.csv"
    panel_hosts_path.write_text("bacteria\nEDL933\n", encoding="utf-8")
    panel_phage_metadata_path = tmp_path / "panel_phages.csv"
    panel_phage_metadata_path.write_text("phage\nPANEL_1\n", encoding="utf-8")
    bundle_path = tmp_path / "bundle.joblib"
    bundle_path.write_text("placeholder", encoding="utf-8")

    roundtrip_host = tl09.HostCandidate(
        host_tax_id="H1",
        host_name="Host one",
        positive_pair_count=5,
        unique_phage_count=5,
        panel_match="EDL933",
        is_panel_host=True,
        assembly_accession="GCF_1",
        assembly_level="Complete Genome",
        assembly_organism_name="Host one",
        assembly_ftp_path="ftp://example/1",
    )

    monkeypatch.setattr(tl09, "_download_text", lambda url: "placeholder")
    monkeypatch.setattr(tl09, "load_panel_hosts", lambda path: ({"EDL933"}, {"edl933": "EDL933"}))
    monkeypatch.setattr(
        tl09,
        "parse_vhdb_positive_pairs",
        lambda text: [tl09.PositivePair("H1", "Host one", "P1", "virus-one")],
    )
    monkeypatch.setattr(
        tl09,
        "summarize_positive_pairs",
        lambda pairs: {"host_count": 1, "phage_accession_count": 1, "positive_pair_count": 1},
    )
    monkeypatch.setattr(tl09, "parse_assembly_summary", lambda text: {"H1": []})
    monkeypatch.setattr(
        tl09,
        "build_host_candidates",
        lambda positive_pairs, assemblies_by_taxid, panel_lookup: [roundtrip_host],
    )
    monkeypatch.setattr(
        tl09,
        "select_validation_hosts",
        lambda host_candidates, novel_host_count, min_positive_phages_per_host: ([], [roundtrip_host]),
    )
    monkeypatch.setattr(tl09, "require_bundle", lambda path: bundle_path)
    monkeypatch.setattr(
        tl09,
        "load_saved_roundtrip_contract",
        lambda path: tl09.SavedRoundtripContract(
            gate_assessment=tl09.Tl13GateAssessment(
                bundle_task_id="TL13",
                bundle_format_version="tl13_bundle_v1",
                extra_deployable_block_ids=("tl04_antidef_defense_evasion",),
                improved_roundtrip_metrics=("max_abs_probability_delta_max",),
                passed=True,
                failure_reasons=(),
            ),
            roundtrip_reference_path=tmp_path / "saved_roundtrip_reference.csv",
            roundtrip_reference_hosts=set(),
            roundtrip_cohort_path=tmp_path / "saved_roundtrip_cohort.csv",
            roundtrip_cohort_hosts=set(),
            contract_issues=[],
        ),
    )

    with pytest.raises(ValueError, match="Validation cohort selection produced zero hosts"):
        tl09.main(
            [
                "--output-dir",
                str(output_dir),
                "--panel-hosts-path",
                str(panel_hosts_path),
                "--panel-phage-metadata-path",
                str(panel_phage_metadata_path),
                "--bundle-path",
                str(bundle_path),
                "--st02-pair-table-path",
                str(tmp_path / "missing_st02_pair_table.csv"),
            ]
        )
