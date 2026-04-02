from lyzortx.pipeline.track_j import run_track_j


def test_run_track_j_dispatches_release_sequence_in_dependency_order(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(run_track_j.st01_label_policy, "main", lambda argv: calls.append("st01"))
    monkeypatch.setattr(run_track_j.st01b_confidence_tiers, "main", lambda argv: calls.append("st01b"))
    monkeypatch.setattr(run_track_j.st02_build_pair_table, "main", lambda argv: calls.append("st02"))
    monkeypatch.setattr(run_track_j.st03_build_splits, "main", lambda argv: calls.append("st03"))
    monkeypatch.setattr(
        run_track_j.build_receptor_surface_feature_block,
        "main",
        lambda argv: calls.append("track-c-receptor-surface"),
    )
    monkeypatch.setattr(
        run_track_j.build_omp_receptor_variant_feature_block,
        "main",
        lambda argv: calls.append("track-c-omp-variants"),
    )
    monkeypatch.setattr(
        run_track_j.build_extended_host_surface_feature_block,
        "main",
        lambda argv: calls.append("track-c-extended-surface"),
    )
    monkeypatch.setattr(
        run_track_j.build_v1_host_feature_pair_table,
        "main",
        lambda argv: calls.append("track-c-v1-pair-table"),
    )
    monkeypatch.setattr(run_track_j.run_track_d, "main", lambda argv: calls.append("track-d"))
    monkeypatch.setattr(run_track_j.run_track_e, "main", lambda argv: calls.append("track-e"))
    monkeypatch.setattr(
        run_track_j.run_track_g.train_v1_binary_classifier,
        "main",
        lambda argv: calls.append("track-g-train-v1-binary"),
    )
    monkeypatch.setattr(
        run_track_j.run_track_g.calibrate_gbm_outputs,
        "main",
        lambda argv: calls.append("track-g-calibrate-gbm"),
    )
    monkeypatch.setattr(
        run_track_j.run_track_g.run_feature_block_ablation_suite,
        "main",
        lambda argv: calls.append("track-g-feature-block-ablation"),
    )
    monkeypatch.setattr(
        run_track_j.run_track_g.compute_shap_explanations,
        "main",
        lambda argv: calls.append("track-g-compute-shap"),
    )

    run_track_j.main([])

    assert calls == [
        "st01",
        "st01b",
        "st02",
        "st03",
        "track-c-receptor-surface",
        "track-c-omp-variants",
        "track-c-extended-surface",
        "track-c-v1-pair-table",
        "track-d",
        "track-e",
        "track-g-train-v1-binary",
        "track-g-calibrate-gbm",
        "track-g-feature-block-ablation",
        "track-g-compute-shap",
    ]


def test_run_track_j_uses_dynamic_runner_boundaries_and_logs_steps(monkeypatch, caplog) -> None:
    foundation_calls: list[str] = []
    feature_calls: list[str] = []
    modeling_calls: list[str] = []

    monkeypatch.setattr(
        run_track_j,
        "foundation_runners",
        lambda: (
            ("foundation-a", lambda: foundation_calls.append("foundation-a")),
            ("foundation-b", lambda: foundation_calls.append("foundation-b")),
        ),
    )
    monkeypatch.setattr(
        run_track_j,
        "feature_block_runners",
        lambda: (("feature-a", lambda: feature_calls.append("feature-a")),),
    )
    monkeypatch.setattr(run_track_j.run_track_d, "main", lambda argv: modeling_calls.append("track-d"))
    monkeypatch.setattr(run_track_j.run_track_e, "main", lambda argv: modeling_calls.append("track-e"))
    monkeypatch.setattr(
        run_track_j.run_track_g.train_v1_binary_classifier,
        "main",
        lambda argv: modeling_calls.append("track-g-train-v1-binary"),
    )
    monkeypatch.setattr(
        run_track_j.run_track_g.calibrate_gbm_outputs,
        "main",
        lambda argv: modeling_calls.append("track-g-calibrate-gbm"),
    )
    monkeypatch.setattr(
        run_track_j.run_track_g.run_feature_block_ablation_suite,
        "main",
        lambda argv: modeling_calls.append("track-g-feature-block-ablation"),
    )
    monkeypatch.setattr(
        run_track_j.run_track_g.compute_shap_explanations,
        "main",
        lambda argv: modeling_calls.append("track-g-compute-shap"),
    )

    foundation_runners = tuple(run_track_j._runners_for_step("foundation"))
    feature_runners = tuple(run_track_j._runners_for_step("feature-blocks"))
    modeling_runners = tuple(run_track_j._runners_for_step("modeling"))

    assert [name for name, _ in foundation_runners] == ["foundation-a", "foundation-b"]
    assert [name for name, _ in feature_runners] == ["feature-a"]
    assert [name for name, _ in modeling_runners] == [
        "track-d",
        "track-e",
        "track-g-train-v1-binary",
        "track-g-calibrate-gbm",
        "track-g-feature-block-ablation",
        "track-g-compute-shap",
    ]

    with caplog.at_level("INFO", logger="lyzortx.pipeline.track_j.run_track_j"):
        run_track_j.main(["--step", "all"])

    assert foundation_calls == ["foundation-a", "foundation-b"]
    assert feature_calls == ["feature-a"]
    assert modeling_calls == [
        "track-d",
        "track-e",
        "track-g-train-v1-binary",
        "track-g-calibrate-gbm",
        "track-g-feature-block-ablation",
        "track-g-compute-shap",
    ]
    step_messages = [r.message for r in caplog.records if r.message.startswith("[track-j]")]
    assert step_messages == [
        "[track-j] foundation-a",
        "[track-j] foundation-b",
        "[track-j] feature-a",
        "[track-j] track-d",
        "[track-j] track-e",
        "[track-j] track-g-train-v1-binary",
        "[track-j] track-g-calibrate-gbm",
        "[track-j] track-g-feature-block-ablation",
        "[track-j] track-g-compute-shap",
    ]


def test_run_track_j_can_limit_to_feature_blocks(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(run_track_j.st01_label_policy, "main", lambda argv: calls.append("st01"))
    monkeypatch.setattr(run_track_j.st01b_confidence_tiers, "main", lambda argv: calls.append("st01b"))
    monkeypatch.setattr(run_track_j.st02_build_pair_table, "main", lambda argv: calls.append("st02"))
    monkeypatch.setattr(run_track_j.st03_build_splits, "main", lambda argv: calls.append("st03"))
    monkeypatch.setattr(
        run_track_j.build_receptor_surface_feature_block,
        "main",
        lambda argv: calls.append("track-c-receptor-surface"),
    )
    monkeypatch.setattr(
        run_track_j.build_omp_receptor_variant_feature_block,
        "main",
        lambda argv: calls.append("track-c-omp-variants"),
    )
    monkeypatch.setattr(
        run_track_j.build_extended_host_surface_feature_block,
        "main",
        lambda argv: calls.append("track-c-extended-surface"),
    )
    monkeypatch.setattr(
        run_track_j.build_v1_host_feature_pair_table,
        "main",
        lambda argv: calls.append("track-c-v1-pair-table"),
    )
    monkeypatch.setattr(run_track_j.run_track_d, "main", lambda argv: calls.append("track-d"))
    monkeypatch.setattr(run_track_j.run_track_e, "main", lambda argv: calls.append("track-e"))
    monkeypatch.setattr(
        run_track_j.run_track_g.train_v1_binary_classifier,
        "main",
        lambda argv: calls.append("track-g-train-v1-binary"),
    )
    monkeypatch.setattr(
        run_track_j.run_track_g.calibrate_gbm_outputs,
        "main",
        lambda argv: calls.append("track-g-calibrate-gbm"),
    )
    monkeypatch.setattr(
        run_track_j.run_track_g.run_feature_block_ablation_suite,
        "main",
        lambda argv: calls.append("track-g-feature-block-ablation"),
    )
    monkeypatch.setattr(
        run_track_j.run_track_g.compute_shap_explanations,
        "main",
        lambda argv: calls.append("track-g-compute-shap"),
    )

    run_track_j.main(["--step", "feature-blocks"])

    assert calls == [
        "track-c-receptor-surface",
        "track-c-omp-variants",
        "track-c-extended-surface",
        "track-c-v1-pair-table",
    ]
