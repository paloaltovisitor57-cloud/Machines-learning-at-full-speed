import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from nardis_neural.solana.edge.trees import TreeEnsemble
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


def test_a_settled_trade_id_is_never_learned_twice() -> None:
    m = MetaLearner()
    m.advise(TradeProposal("a", "m", 0.0, {"x": 1.0}))
    assert m.settle(TradeOutcome("a", 60.0, 2.0))
    m.advise(TradeProposal("a", "m", 100.0, {"x": 1.0}))
    assert "a" not in m.pending, "an already settled id is not recorded again"
    assert not m.settle(TradeOutcome("a", 160.0, 2.0))
    m.pending["a"] = (100.0, {"x": 1.0})  # even when it reaches the pending book some other way
    assert not m.settle(TradeOutcome("a", 160.0, 2.0))
    assert m.multiple == [2.0] and m.trade_ids == ["a"]


def test_pending_proposals_are_bounded_by_count_whatever_the_clock(tmp_path) -> None:  # type: ignore[no-untyped-def]
    m = MetaLearner(max_pending=3)
    t0 = 1.75e12  # milliseconds: no clock unit or far-future t may expire the others
    m.advise(TradeProposal("a", "m", t0))
    m.advise(TradeProposal("b", "m", t0 + 11 * 60_000))
    m.advise(TradeProposal("far", "m", 1e18))
    m.advise(TradeProposal("c", "m", 5.0))
    assert list(m.pending) == ["b", "far", "c"], "only the oldest recorded one is dropped"
    assert not m.settle(TradeOutcome("a", 0.0, 3.0))
    assert m.settle(TradeOutcome("b", 0.0, 3.0))
    m.advise(TradeProposal("far", "m", 1e18))  # re-advised: becomes the newest
    m.advise(TradeProposal("d", "m", 6.0))
    m.advise(TradeProposal("e", "m", 7.0))
    assert list(m.pending) == ["far", "d", "e"]
    m.save(tmp_path)
    back = MetaLearner.load(tmp_path)
    assert back.max_pending == 3 and list(back.pending) == ["far", "d", "e"]
    assert back.settle(TradeOutcome("d", 0.0, 1.5))
    meta = json.loads((tmp_path / "meta.json").read_text())
    del meta["config"]["max_pending"]
    meta["config"]["pending_ttl_seconds"] = 3600.0  # an interim file: the key is ignored
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    assert MetaLearner.load(tmp_path).max_pending == 20_000


def test_cold_start_chase_stays_at_break_even_until_targets_are_proven() -> None:
    from nardis_neural.solana.chase import CHASE_TARGETS, break_even

    a = MetaLearner().advise(TradeProposal("a", "m", 0.0), record=False)
    assert a.source == "prior" and a.p_win == 0.5
    assert all(a.chase[f"edge_{k:g}x"] <= 1.0 + 1e-9 for k in CHASE_TARGETS), a.chase
    assert a.chase["tail_ev"] == pytest.approx(0.7) and a.chase["chase_target"] == 0.0, "nothing banked"
    assert a.p_10x <= break_even(10.0) + 1e-12 and a.p_100x <= break_even(100.0) + 1e-12
    m = MetaLearner()
    for i in range(20):  # 20 real 12x peaks: 2x, 5x and 10x are proven, 100x is not
        m.advise(TradeProposal(str(i), "m", float(i)))
        m.settle(TradeOutcome(str(i), i + 60.0, 1.5, peak_multiple=12.0))
    b = m.advise(TradeProposal("b", "m", 100.0), record=False)
    assert b.p_10x > 0.9 and b.chase["chase_target"] == 10.0
    assert b.chase["edge_100x"] <= 1.0 + 1e-9
    # only the proven 2x / 5x / 10x rungs are banked: the ladder is worth at most 10x
    p2, p5, p10 = (b.chase[f"p_{k}x"] for k in ("2", "5", "10"))
    ladder = (1 - p2) * 0.7 + (p2 - p5) * 2 + (p5 - p10) * 5 + p10 * 10
    assert b.chase["tail_ev"] == pytest.approx(ladder) and b.chase["tail_ev"] <= 10.0


def _tree() -> TreeEnsemble:
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 1))
    reg = HistGradientBoostingRegressor(max_iter=5).fit(x, x[:, 0])
    return TreeEnsemble.export(reg)


def test_meta_save_is_crash_safe_and_reads_the_old_layout(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from pathlib import Path

    m = MetaLearner()
    m.feature_names = ["x"]
    m.value_model = _tree()
    m.models = {1.0: _tree()}
    m.save(tmp_path)
    assert sorted(f.name for f in tmp_path.glob("*.npz")) == ["level_1.g1.npz", "value.g1.npz"]
    m.value_model = None

    def crash(self: Path, target: Path) -> Path:
        raise OSError("disk gone")

    monkeypatch.setattr(Path, "replace", crash)
    with pytest.raises(OSError):
        m.save(tmp_path)  # new trees written, meta.json not yet replaced
    monkeypatch.undo()
    back = MetaLearner.load(tmp_path)
    assert back.value_model is not None and set(back.models) == {1.0}, "the last complete save"
    # the layout written before generations: untagged files, no "generation" / "value" keys
    meta = json.loads((tmp_path / "meta.json").read_text())
    del meta["generation"], meta["value"]
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "level_1.g1.npz").rename(tmp_path / "level_1.npz")
    (tmp_path / "value.g1.npz").rename(tmp_path / "value.npz")
    old = MetaLearner.load(tmp_path)
    assert old.value_model is not None and set(old.models) == {1.0}
    old.save(tmp_path)
    assert sorted(f.name for f in tmp_path.glob("*.npz")) == ["level_1.g1.npz", "value.g1.npz"]
