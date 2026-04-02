import csv
import json
from pathlib import Path

from lyzortx.pipeline.deployment_paired_features import derive_host_surface_features


def test_build_host_surface_schema_uses_continuous_scores_and_drops_legacy_duplicates() -> None:
    schema = derive_host_surface_features.build_host_surface_schema(["KfiA", "cluster_19"])

    assert schema["columns"][:6] == [
        {"name": "bacteria", "dtype": "string"},
        {"name": "host_o_antigen_type", "dtype": "string"},
        {"name": "host_o_antigen_score", "dtype": "float64"},
        {"name": "host_lps_core_type", "dtype": "string"},
        {"name": "host_receptor_btub_score", "dtype": "float64"},
        {"name": "host_receptor_fadL_score", "dtype": "float64"},
    ]
    assert schema["capsule_score_columns"] == [
        "host_capsule_profile_kfia_score",
        "host_capsule_profile_cluster_19_score",
    ]
    assert "host_o_antigen_present" in schema["dropped_legacy_columns"]
    assert "host_surface_lps_core_type" in schema["dropped_legacy_columns"]
    assert "host_k_antigen_type" in schema["dropped_legacy_columns"]


def test_summarize_o_antigen_result_uses_best_total_score_and_retains_score_when_call_unresolved() -> None:
    references = [
        derive_host_surface_features.tl15.OAlleleReference(
            query_id="O1__wzx__a",
            o_type="O1",
            gene_family="wzx",
            allele_key="a",
            sequence="ATGC",
        ),
        derive_host_surface_features.tl15.OAlleleReference(
            query_id="O1__wzy__b",
            o_type="O1",
            gene_family="wzy",
            allele_key="b",
            sequence="ATGC",
        ),
        derive_host_surface_features.tl15.OAlleleReference(
            query_id="O2__wzx__c",
            o_type="O2",
            gene_family="wzx",
            allele_key="c",
            sequence="ATGC",
        ),
    ]
    contract = {
        "O1": {"wzx": ("O1__wzx__a",), "wzy": ("O1__wzy__b",)},
        "O2": {"wzx": ("O2__wzx__c",)},
    }

    called = derive_host_surface_features.summarize_o_antigen_result(
        hits=[
            derive_host_surface_features.tl15.HmmerHit("contig", "O1__wzx__a", 0.0, 320.0, ""),
            derive_host_surface_features.tl15.HmmerHit("contig", "O1__wzy__b", 0.0, 180.0, ""),
            derive_host_surface_features.tl15.HmmerHit("contig", "O2__wzx__c", 0.0, 500.0, ""),
        ],
        references=references,
        o_type_contract=contract,
    )
    unresolved = derive_host_surface_features.summarize_o_antigen_result(
        hits=[derive_host_surface_features.tl15.HmmerHit("contig", "O2__wzx__c", 0.0, 500.0, "")],
        references=references,
        o_type_contract=contract,
    )

    assert called["o_type"] == "O1"
    assert called["continuous_score"] == 500.0
    assert unresolved["o_type"] == ""
    assert unresolved["continuous_score"] == 500.0


def test_summarize_receptor_scores_uses_best_score_per_receptor_and_zero_fills_missing() -> None:
    scores = derive_host_surface_features.summarize_receptor_scores(
        [
            derive_host_surface_features.tl15.HmmerHit("gene_1", "sp|P06129|BTUB_ECOLI", 1e-20, 220.0, ""),
            derive_host_surface_features.tl15.HmmerHit("gene_2", "sp|P06129|BTUB_ECOLI", 1e-30, 240.0, ""),
            derive_host_surface_features.tl15.HmmerHit("gene_3", "sp|P21420|PQQU_ECOLI", 1e-12, 150.0, ""),
        ]
    )

    assert scores["BTUB"] == 240.0
    assert scores["YNCD"] == 150.0
    assert scores["FADL"] == 0.0


def test_build_host_surface_feature_row_emits_zero_filled_continuous_schema() -> None:
    schema = derive_host_surface_features.build_host_surface_schema(["KfiA", "KpsC_2"])

    row = derive_host_surface_features.build_host_surface_feature_row(
        bacteria="LF82",
        schema=schema,
        o_antigen_type="O83",
        o_antigen_score=812.4,
        lps_core_type="R1",
        receptor_scores={"BTUB": 120.5},
        capsule_profile_scores={"KfiA": 88.1},
    )

    assert row["host_o_antigen_type"] == "O83"
    assert row["host_lps_core_type"] == "R1"
    assert row["host_receptor_btub_score"] == 120.5
    assert row["host_receptor_fadL_score"] == 0.0
    assert row["host_capsule_profile_kfia_score"] == 88.1
    assert row["host_capsule_profile_kpsc_2_score"] == 0.0


def test_run_validation_subset_writes_feature_csv_and_report(monkeypatch, tmp_path: Path) -> None:
    validation_dir = tmp_path / "fastas"
    validation_dir.mkdir()
    for host in derive_host_surface_features.VALIDATION_HOSTS:
        (validation_dir / f"{host}.fasta").write_text(">contig\nATGC\n", encoding="utf-8")

    runtime_inputs = derive_host_surface_features.SurfaceRuntimeInputs(
        references=(),
        o_type_contract={},
        o_antigen_query_path=tmp_path / "queries.fna",
        lps_lookup={"O83": {"proxy_type": "R1"}},
        capsule_hmm_bundle_path=tmp_path / "capsule.hmm",
        capsule_profile_names=("KfiA",),
        omp_reference_path=tmp_path / "omp.faa",
    )

    monkeypatch.setattr(
        derive_host_surface_features,
        "prepare_host_surface_runtime_inputs",
        lambda **_: runtime_inputs,
    )
    monkeypatch.setattr(
        derive_host_surface_features.tl18_runtime,
        "build_tl15_panel_training_rows",
        lambda **_: [
            {
                "bacteria": "55989",
                "host_o_antigen_type": "O83",
                "host_lps_core_type": "R1",
                "host_receptor_btub_present": 1,
            },
            {
                "bacteria": "EDL933",
                "host_o_antigen_type": "O157",
                "host_lps_core_type": "R3",
                "host_receptor_btub_present": 0,
            },
            {
                "bacteria": "LF82",
                "host_o_antigen_type": "O83",
                "host_lps_core_type": "R1",
                "host_receptor_btub_present": 1,
            },
        ],
    )

    rows_by_host = {
        "55989": {
            "bacteria": "55989",
            "host_o_antigen_type": "O83",
            "host_o_antigen_score": 500.0,
            "host_lps_core_type": "R1",
            "host_receptor_btub_score": 200.0,
            "host_capsule_profile_kfia_score": 10.0,
        },
        "EDL933": {
            "bacteria": "EDL933",
            "host_o_antigen_type": "",
            "host_o_antigen_score": 250.0,
            "host_lps_core_type": "",
            "host_receptor_btub_score": 0.0,
            "host_capsule_profile_kfia_score": 0.0,
        },
        "LF82": {
            "bacteria": "LF82",
            "host_o_antigen_type": "O83",
            "host_o_antigen_score": 700.0,
            "host_lps_core_type": "R1",
            "host_receptor_btub_score": 50.0,
            "host_capsule_profile_kfia_score": 25.0,
        },
    }

    def fake_derive_host_surface_features(
        assembly_path: Path,
        *,
        bacteria_id: str | None = None,
        output_dir: Path,
        picard_metadata_path: Path,
        o_type_output_path: Path,
        o_type_allele_path: Path,
        o_antigen_override_path: Path,
        abc_capsule_profile_dir: Path,
        omp_reference_path: Path,
        runtime_inputs,
    ) -> dict[str, object]:
        bacteria = bacteria_id or assembly_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        schema = derive_host_surface_features.build_host_surface_schema(["KfiA"])
        return {
            "schema": schema,
            "feature_row": rows_by_host[bacteria],
            "manifest": {},
        }

    monkeypatch.setattr(
        derive_host_surface_features,
        "derive_host_surface_features",
        fake_derive_host_surface_features,
    )

    summary = derive_host_surface_features.run_validation_subset(
        validation_fastas_dir=validation_dir,
        output_dir=tmp_path / "output",
        picard_metadata_path=tmp_path / "picard.csv",
        o_type_output_path=tmp_path / "otype.tsv",
        o_type_allele_path=tmp_path / "alleles.tsv",
        o_antigen_override_path=tmp_path / "override.tsv",
        abc_capsule_profile_dir=tmp_path / "capsules",
        omp_reference_path=tmp_path / "omp.faa",
    )

    assert summary["average_receptor_binary_mismatches_per_host"] == 0.0
    assert summary["o_antigen_type_exact_match_count"] == 2
    counts_path = tmp_path / "output" / "validation_host_surface_features.csv"
    with counts_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert [row["bacteria"] for row in rows] == ["55989", "EDL933", "LF82"]

    report = json.loads((tmp_path / "output" / "validation_report.json").read_text(encoding="utf-8"))
    host_report_by_name = {entry["bacteria"]: entry for entry in report["host_reports"]}
    assert host_report_by_name["55989"]["o_antigen_type_match"] is True
    assert host_report_by_name["EDL933"]["o_antigen_type_match"] is False
    assert host_report_by_name["LF82"]["nonzero_capsule_profile_count"] == 1
