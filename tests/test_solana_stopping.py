import numpy as np

from nardis_neural.solana.stopping import StoppingModel, StoppingPath, _thin, state_matrix


def _paths(n: int, seed: int) -> list[StoppingPath]:
    """Pumps that dump: value rises to a peak whose timing is signalled by a feature, then
    collapses.  Holding to the end always loses; selling once the signal flips wins."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        k = 20
        top = int(rng.integers(4, 16))
        up = np.exp(np.linspace(0, rng.uniform(0.5, 1.5), top + 1))
        down = up[-1] * np.exp(-np.linspace(0, 3.0, k - top))[1:]
        marks = np.r_[up, down]
        signal = (np.arange(k) >= top).astype(np.float32)
        feats = np.stack([signal, rng.normal(size=k).astype(np.float32)], axis=1)
        out.append(StoppingPath(f"m{i}", 0.0, np.arange(k, dtype=np.float64) * 30 + 1, marks, feats))
    return out


def test_state_matrix_tracks_peak_and_drawdown() -> None:
    p = StoppingPath(
        "m", 0.0, np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0]), np.zeros((3, 1), np.float32)
    )
    x = state_matrix(p)
    assert x.shape == (3, 5)
    assert np.allclose(x[:, 3], np.log([1.0, 2.0, 2.0]))
    assert np.allclose(x[:, 4], [0.0, 0.0, -np.log(2.0)])


def test_stopping_beats_holding_out_of_sample(tmp_path) -> None:  # type: ignore[no-untyped-def]
    model = StoppingModel("log", iterations=3, seed=0)
    rep = model.fit(_paths(200, 0), max_iter=80)
    assert rep["paths"] == 200 and len(rep["iterations"]) == 3
    test = _paths(100, 1)
    stopped = np.array([p.marks[model.stop_index(p)] for p in test])
    held = np.array([p.marks[-1] for p in test])
    assert np.log(stopped).mean() > np.log(held).mean() + 1.0
    assert (stopped > 1.0).mean() > 0.9
    model.save(tmp_path)
    again = StoppingModel.load(tmp_path)
    assert [again.stop_index(p) for p in test] == [model.stop_index(p) for p in test]


def test_unfitted_model_holds_and_thin_spacing() -> None:
    p = _paths(1, 0)[0]
    assert StoppingModel().stop_index(p) == len(p.marks) - 1
    assert _thin(np.array([0.0, 10.0, 31.0, 40.0, 70.0]), 30.0).tolist() == [0, 2, 4]


def test_stopping_research_and_brain_integration(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from nardis_neural.solana import LaunchSimSpec, SolanaBrain, SolanaConfig, simulate_launches
    from nardis_neural.solana.moonshot import MoonshotSpec
    from nardis_neural.solana.simulator import MARKET_PRESETS
    from tests.conftest import make_tiny_config

    hist, arch = simulate_launches(
        LaunchSimSpec(
            n_tokens=22, seed=6, n_retail=200, duration_seconds=2 * 3600,
            archetype_weights=dict(MARKET_PRESETS["degen"]),
        )
    )  # fmt: skip
    cfg = SolanaConfig(sample_interval_seconds=20.0, graph_top_k=6)
    base = make_tiny_config()
    base.training.epochs = 1
    base.ensemble.size = 1
    base.ensemble.mc_dropout_samples = 0
    SolanaBrain.bootstrap(tmp_path / "ws", hist, cfg, base, device="cpu")
    brain = SolanaBrain(tmp_path / "ws", device="cpu")
    report = brain.fit_stopping(MoonshotSpec(horizon_seconds=3600), test_fraction=0.3, archetypes=arch)
    assert {"optimal_stopping_log", "hold_to_horizon", "ladder", "hindsight_best"} <= set(report["exits"])
    best = report["exits"]["hindsight_best"]["mean_log_multiple"]
    assert report["exits"]["optimal_stopping_log"]["mean_log_multiple"] <= best + 1e-9
    reloaded = SolanaBrain(tmp_path / "ws", device="cpu")
    assert reloaded.stopping is not None and reloaded.stopping.trees is not None
    mint = next(iter(reloaded.market.tokens))
    advice = reloaded.hold_advice(mint, reloaded.market.token(mint).launch.t + 30.0)
    assert {"liquidation_multiple", "sell_now_utility", "continuation_utility", "advantage"} <= set(advice)
    assert np.isfinite(advice["advantage"])
