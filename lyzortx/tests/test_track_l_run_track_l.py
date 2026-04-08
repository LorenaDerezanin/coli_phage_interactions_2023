from lyzortx.pipeline.track_l import run_track_l


def _stub_all_steps(monkeypatch, calls):
    """Replace every step's module main() with a recorder."""
    monkeypatch.setattr(run_track_l.run_pharokka, "main", lambda argv: calls.append("annotate"))
    monkeypatch.setattr(
        run_track_l, "cache_key_tsvs", lambda annotations_dir, cached_dir: calls.append("cache-key-tsvs")
    )
    monkeypatch.setattr(run_track_l.parse_annotations, "main", lambda argv: calls.append("parse"))
    monkeypatch.setattr(run_track_l.run_enrichment_analysis, "main", lambda argv: calls.append("enrich"))
    monkeypatch.setattr(
        run_track_l.build_tl17_phage_compatibility_preprocessor,
        "main",
        lambda argv: calls.append("tl17-phage-compatibility-preprocessor"),
    )
    monkeypatch.setattr(
        run_track_l.build_generalized_inference_bundle,
        "main",
        lambda argv: calls.append("generalized-inference-bundle"),
    )
    monkeypatch.setattr(
        run_track_l.build_tl13_generalized_inference_bundle,
        "main",
        lambda argv: calls.append("deployable-generalized-inference-bundle"),
    )
    monkeypatch.setattr(
        run_track_l.build_tl18_generalized_inference_bundle,
        "main",
        lambda argv: calls.append("richer-deployable-generalized-inference-bundle"),
    )


def test_single_step_dispatch(monkeypatch) -> None:
    calls: list[str] = []
    _stub_all_steps(monkeypatch, calls)
    run_track_l.main(["--step", "tl17-phage-compatibility-preprocessor"])
    assert calls == ["tl17-phage-compatibility-preprocessor"]


def test_features_group_dispatch(monkeypatch) -> None:
    calls: list[str] = []
    _stub_all_steps(monkeypatch, calls)
    run_track_l.main(["--step", "features"])
    assert calls == ["parse", "enrich"]


def test_inference_group_dispatch(monkeypatch) -> None:
    calls: list[str] = []
    _stub_all_steps(monkeypatch, calls)
    run_track_l.main(["--step", "inference"])
    assert calls == [
        "generalized-inference-bundle",
        "deployable-generalized-inference-bundle",
        "richer-deployable-generalized-inference-bundle",
    ]


def test_all_runs_every_step_in_order(monkeypatch) -> None:
    calls: list[str] = []
    _stub_all_steps(monkeypatch, calls)
    # annotate requires --database-dir
    run_track_l.main(["--step", "all", "--database-dir", "/fake/db"])
    assert calls == [
        "annotate",
        "cache-key-tsvs",
        "parse",
        "enrich",
        "tl17-phage-compatibility-preprocessor",
        "generalized-inference-bundle",
        "deployable-generalized-inference-bundle",
        "richer-deployable-generalized-inference-bundle",
    ]
