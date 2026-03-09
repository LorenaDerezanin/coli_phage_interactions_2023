from pathlib import Path

from lyzortx.pipeline.track_a.steps.build_mechanistic_proxy_features import (
    build_host_proxy_rows,
    build_manifest,
    build_phage_proxy_rows,
)


def test_build_host_proxy_rows_confidence_and_ratio() -> None:
    rows = [
        {
            "bacteria": "B1",
            "LPS_type": "LPS-A",
            "O-type": "O1",
            "H-type": "H7",
            "n_defense_systems": "8",
            "n_infections": "4",
        },
        {"bacteria": "B2", "n_defense_systems": "", "n_infections": ""},
    ]

    out = build_host_proxy_rows(rows)

    assert len(out) == 2
    assert out[0]["bacteria"] == "B1"
    assert out[0]["host_receptor_confidence"] == 0.85
    assert out[0]["host_defense_per_infection_proxy"] == 2.0
    assert out[1]["host_receptor_proxy_available"] == 0
    assert out[1]["host_defense_confidence"] == 0.25


def test_build_phage_proxy_rows_keyword_proxy() -> None:
    rows = [
        {"phage": "P1", "Family": "Podoviridae", "Genome_size": "42000"},
        {"phage": "P2", "Family": "", "Genome_size": ""},
    ]

    out = build_phage_proxy_rows(rows)

    assert out[0]["phage_depolymerase_proxy"] == 1
    assert out[0]["phage_domain_complexity_proxy"] == 1
    assert out[1]["phage_domain_complexity_proxy"] == ""


def test_build_manifest_has_missingness_and_feature_definitions(tmp_path: Path) -> None:
    host_in = tmp_path / "host.csv"
    phage_in = tmp_path / "phage.csv"
    host_in.write_text("h", encoding="utf-8")
    phage_in.write_text("p", encoding="utf-8")

    host_rows = [
        {
            "bacteria": "B1",
            "host_receptor_confidence": 0.5,
            "host_defense_confidence": 0.5,
            "host_defense_per_infection_proxy": "",
        }
    ]
    phage_rows = [
        {
            "phage": "P1",
            "phage_rbp_confidence": 0.5,
            "phage_depolymerase_confidence": 0.5,
        }
    ]

    manifest = build_manifest(
        version="v1",
        host_rows=host_rows,
        phage_rows=phage_rows,
        host_input_path=host_in,
        phage_input_path=phage_in,
        host_output_path=tmp_path / "h_out.csv",
        phage_output_path=tmp_path / "p_out.csv",
    )

    assert "missingness" in manifest
    assert manifest["missingness"]["host"]["host_defense_per_infection_proxy"]["missing_count"] == 1
    assert manifest["schema"]["confidence_fields"]["host"] == ["host_receptor_confidence", "host_defense_confidence"]
    assert manifest["schema"]["confidence_fields"]["phage"] == ["phage_rbp_confidence", "phage_depolymerase_confidence"]
    assert "feature_definitions" in manifest["schema"]
    assert "host_receptor_surface_proxy_score" in manifest["schema"]["feature_definitions"]["host"]
    assert manifest["confidence_summary"]["host"]["host_receptor_confidence"]["mean"] == 0.5
