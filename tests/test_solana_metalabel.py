import numpy as np
from sklearn.metrics import roc_auc_score

from nardis_neural.solana.metalabel import MetaLearner, TradeOutcome, TradeProposal


def _stream(n: int, seed: int, t0: float = 0.0) -> list[tuple[TradeProposal, TradeOutcome, float]]:
    """Proposals whose hidden quality drives both the win rate and the tail; one noise feature."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        q = float(rng.normal())
        p_win = 1.0 / (1.0 + np.exp(-(2.0 * q - 0.5)))
        win = rng.random() < p_win
        mult = float(np.exp(rng.normal(0.6 + 0.5 * q, 0.5))) if win else float(rng.uniform(0.2, 0.95))
        peak = mult * (12.0 if win and q > 1.2 and rng.random() < 0.5 else 1.0)
        prop = TradeProposal(f"t{seed}-{i}", f"m{i}", t0 + i, {"quality": q, "noise": float(rng.normal())})
        out.append((prop, TradeOutcome(prop.trade_id, t0 + i + 60, mult, peak), q))
    return out


def test_cold_start_answers_from_base_rates() -> None:
    m = MetaLearner()
    a = m.advise(TradeProposal("a", "m", 0.0, {"x": 1.0}))
    assert a.source == "prior" and a.size_multiplier == 1.0 and not a.veto and a.evidence == 0
    assert a.p_100x <= a.p_10x <= a.p_win


def test_learns_which_proposals_win_and_vetoes_losing_patterns(tmp_path) -> None:  # type: ignore[no-untyped-def]
    m = MetaLearner(min_trades=100, refit_every=50)
    for prop, outcome, _ in _stream(800, 0):
        m.advise(prop)
        m.settle(outcome)
    assert m.report["levels"]["1.0"]["deployed"]
    test = _stream(400, 1, t0=10_000.0)
    advice = [m.advise(p, record=False) for p, _, _ in test]
    wins = np.array([o.multiple > 1 for _, o, _ in test])
    q = np.array([qq for _, _, qq in test])
    assert roc_auc_score(wins, [a.p_win for a in advice]) > 0.75
    size = np.array([a.size_multiplier for a in advice])
    assert size[q > 1].mean() > 1.2
    assert size[q < -1].mean() < 0.8
    veto = np.array([a.veto for a in advice])
    realised = np.array([o.multiple for _, o, _ in test])
    assert veto.sum() > 0 and realised[veto].mean() < 1.0 < realised[~veto].mean()
    m.save(tmp_path)
    again = MetaLearner.load(tmp_path)
    b = again.advise(test[0][0], record=False)
    assert abs(b.p_win - advice[0].p_win) < 1e-9 and b.source == "learned"


def test_noise_features_do_not_get_deployed() -> None:
    rng = np.random.default_rng(3)
    m = MetaLearner(min_trades=100, refit_every=100)
    for i in range(600):
        p = TradeProposal(str(i), "m", float(i), {"a": float(rng.normal()), "b": float(rng.normal())})
        m.advise(p)
        m.settle(TradeOutcome(str(i), i + 1.0, float(rng.choice([0.5, 1.8]))))
    assert not m.report["levels"]["1.0"]["deployed"]
    a = m.advise(TradeProposal("x", "m", 1e6, {"a": 3.0, "b": -3.0}), record=False)
    assert a.source in ("prior", "learned") and not a.veto


def test_brain_advises_and_learns_from_trades(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from nardis_neural.solana import LaunchSimSpec, SolanaBrain, SolanaConfig, simulate_launches
    from tests.conftest import make_tiny_config

    hist, _ = simulate_launches(LaunchSimSpec(n_tokens=8, seed=2, n_retail=80, duration_seconds=1800))
    base = make_tiny_config()
    base.training.epochs = 1
    base.ensemble.size = 1
    base.ensemble.mc_dropout_samples = 0
    cfg = SolanaConfig(sample_interval_seconds=30.0)
    SolanaBrain.bootstrap(tmp_path / "ws", hist, cfg, base, device="cpu")
    brain = SolanaBrain(tmp_path / "ws", device="cpu")
    mint = next(iter(brain.market.tokens))
    advice = brain.advise_trade(TradeProposal("n1", mint, brain.market.now, {"nardis_score": 0.7}))
    assert advice.source == "prior" and 0 <= advice.p_win <= 1
    assert any(k.startswith("addon_") for k in brain.meta.pending["n1"][1])
    assert brain.settle_trade(TradeOutcome("n1", brain.market.now + 60, 1.4))
    brain.save()
    again = SolanaBrain(tmp_path / "ws", device="cpu")
    assert len(again.meta.multiple) == 1 and not again.meta.pending
