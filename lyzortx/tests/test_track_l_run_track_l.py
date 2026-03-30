from lyzortx.pipeline.track_l import run_track_l


def _stub_all_steps(monkeypatch, calls):
    """Replace every step's module main() with a recorder."""
    monkeypatch.setattr(run_track_l.run_pharokka, "main", lambda argv: calls.append("annotate"))
    monkeypatch.setattr(run_track_l.parse_annotations, "main", lambda argv: calls.append("parse"))
    monkeypatch.setattr(run_track_l.run_enrichment_analysis, "main", lambda argv: calls.append("enrich"))
    monkeypatch.setattr(
        run_track_l.build_mechanistic_rbp_receptor_features, "main", lambda argv: calls.append("rbp-features")
    )
    monkeypatch.setattr(
        run_track_l.build_mechanistic_defense_evasion_features, "main", lambda argv: calls.append("defense-features")
    )
    monkeypatch.setattr(
        run_track_l.retrain_mechanistic_v1_model, "main", lambda argv: calls.append("retrain-mechanistic-v1")
    )
    monkeypatch.setattr(
        run_track_l.build_generalized_inference_bundle,
        "main",
        lambda argv: calls.append("generalized-inference-bundle"),
    )
    monkeypatch.setattr(
        run_track_l.validate_vhdb_generalized_inference,
        "main",
        lambda argv: calls.append("validate-vhdb-generalized-inference"),
    )


def test_single_step_dispatch(monkeypatch) -> None:
    calls: list[str] = []
    _stub_all_steps(monkeypatch, calls)
    run_track_l.main(["--step", "retrain-mechanistic-v1"])
    assert calls == ["retrain-mechanistic-v1"]


def test_features_group_dispatch(monkeypatch) -> None:
    calls: list[str] = []
    _stub_all_steps(monkeypatch, calls)
    run_track_l.main(["--step", "features"])
    assert calls == ["parse", "enrich", "rbp-features", "defense-features"]


def test_inference_group_dispatch(monkeypatch) -> None:
    calls: list[str] = []
    _stub_all_steps(monkeypatch, calls)
    run_track_l.main(["--step", "inference"])
    assert calls == ["generalized-inference-bundle", "validate-vhdb-generalized-inference"]


def test_all_runs_every_step_in_order(monkeypatch) -> None:
    calls: list[str] = []
    _stub_all_steps(monkeypatch, calls)
    # annotate requires --database-dir
    run_track_l.main(["--step", "all", "--database-dir", "/fake/db"])
    assert calls == [
        "annotate",
        "parse",
        "enrich",
        "rbp-features",
        "defense-features",
        "retrain-mechanistic-v1",
        "generalized-inference-bundle",
        "validate-vhdb-generalized-inference",
    ]
