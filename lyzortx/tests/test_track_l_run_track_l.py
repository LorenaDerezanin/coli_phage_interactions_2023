from lyzortx.pipeline.track_l import run_track_l


def test_run_track_l_dispatches_tl05_step(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(run_track_l.retrain_mechanistic_v1_model, "main", lambda argv: calls.append("tl05"))

    run_track_l.main(["--step", "retrain-mechanistic-v1"])

    assert calls == ["tl05"]
