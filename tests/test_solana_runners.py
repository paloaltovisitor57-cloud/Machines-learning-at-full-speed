import numpy as np
from sklearn.metrics import roc_auc_score

from nardis_neural.solana.runners import RunnerDetector, runner_labels


def test_censored_rows_that_already_hit_are_known_positives() -> None:
    peak = np.array([12.0, 3.0, 15.0, 1.5])
    censored = np.array([False, False, True, True])
    y, known = runner_labels(peak, censored, 10.0)
    assert y.tolist() == [1, 0, 1, 0] and known.tolist() == [True, True, True, False]


def _data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runners need two things at once (fast buying and low concentration); noise elsewhere."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5))
    both = (x[:, 0] > 0.8) & (x[:, 1] < -0.3)
    peak = np.where(both, np.exp(rng.normal(2.6, 0.8, n)), np.exp(rng.normal(0.0, 0.5, n)))
    censored = rng.random(n) < 0.2
    return x, peak, censored


def test_detector_learns_an_interaction_and_round_trips(tmp_path) -> None:  # type: ignore[no-untyped-def]
    x, peak, cens = _data(6000, 0)
    det = RunnerDetector([f"f{i}" for i in range(5)], min_positives=15)
    rep = det.fit(x, peak, cens)
    assert rep["2x"]["trained"] and rep["10x"]["trained"] and not rep["1000x"]["trained"]
    xt, pt, _ = _data(3000, 1)
    p = det.predict(xt)
    assert roc_auc_score(pt >= 10, p[10.0]) > 0.9
    assert np.all(p[10.0] <= p[5.0] + 1e-12) and np.all(p[5.0] <= p[2.0] + 1e-12)
    det.save(tmp_path)
    again = RunnerDetector.load(tmp_path)
    np.testing.assert_allclose(again.predict(xt)[10.0], p[10.0])


def test_runner_research_and_brain_integration(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from nardis_neural.solana import LaunchSimSpec, SolanaBrain, SolanaConfig, simulate_launches
    from nardis_neural.solana.moonshot import MoonshotSpec
    from nardis_neural.solana.simulator import MARKET_PRESETS
    from tests.conftest import make_tiny_config

    hist, _ = simulate_launches(
        LaunchSimSpec(
            n_tokens=40, seed=6, n_retail=200, duration_seconds=2 * 3600,
            archetype_weights=dict(MARKET_PRESETS["degen"]),
        )
    )  # fmt: skip
    base = make_tiny_config()
    base.training.epochs = 1
    base.ensemble.size = 1
    base.ensemble.mc_dropout_samples = 0
    SolanaBrain.bootstrap(
        tmp_path / "ws", hist, SolanaConfig(sample_interval_seconds=20.0), base, device="cpu"
    )
    brain = SolanaBrain(tmp_path / "ws", device="cpu")
    report = brain.fit_runners(MoonshotSpec(horizon_seconds=1800), test_fraction=0.3)
    assert set(report["targets"]) == {"2x", "5x", "10x", "100x", "1000x"}
    assert "tail" in report["targets"]["2x"] and report["tokens"]["test"] > 0
    assert (tmp_path / "ws" / "runners" / "REPORT.md").exists()
    again = SolanaBrain(tmp_path / "ws", device="cpu")
    assert again.runners is not None and again.runners.report["research"]["tokens"] == report["tokens"]
