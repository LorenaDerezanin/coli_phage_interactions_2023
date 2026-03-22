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
    monkeypatch.setattr(run_track_j.run_track_g, "main", lambda argv: calls.append("track-g"))
    monkeypatch.setattr(run_track_j.run_track_h, "main", lambda argv: calls.append("track-h"))

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
        "track-g",
        "track-h",
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
    monkeypatch.setattr(run_track_j.run_track_g, "main", lambda argv: calls.append("track-g"))
    monkeypatch.setattr(run_track_j.run_track_h, "main", lambda argv: calls.append("track-h"))

    run_track_j.main(["--step", "feature-blocks"])

    assert calls == [
        "track-c-receptor-surface",
        "track-c-omp-variants",
        "track-c-extended-surface",
        "track-c-v1-pair-table",
    ]
