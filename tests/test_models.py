from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from nardis_neural.config import NeuralConfig, RecurrentConfig, TCNConfig, TransformerConfig
from nardis_neural.data.datasets import Batch, MarketDataset, SequenceBatch
from nardis_neural.data.loaders import ArrayStore
from nardis_neural.data.normalization import FeatureNormalizer
from nardis_neural.models.experts import SequenceExpert
from nardis_neural.models.gating import GatingNetwork
from nardis_neural.models.graph import GraphEncoder
from nardis_neural.models.main import NardisNeuralNetwork
from nardis_neural.models.recurrent import RecurrentCore
from nardis_neural.models.tabular import ResidualMLP
from nardis_neural.models.tcn import TCNCore
from nardis_neural.models.transformer import TemporalTransformerCore, build_attention_mask
from nardis_neural.synthetic import SyntheticSpec, generate_synthetic
from tests.conftest import make_tiny_config

D = 16


def _mask(b: int, t: int, lengths: list[int]) -> torch.Tensor:
    m = torch.zeros(b, t, dtype=torch.bool)
    for i, n in enumerate(lengths):
        if n:
            m[i, -n:] = True
    return m


# ---------------------------------------------------------------- transformer
def test_transformer_shapes_and_masking() -> None:
    torch.manual_seed(0)
    core = TemporalTransformerCore(D, TransformerConfig(layers=2, heads=4), 0.0).eval()
    x = torch.randn(3, 10, D)
    mask = _mask(3, 10, [10, 4, 0])
    out = core(x, mask)
    assert out.shape == (3, 10, D)
    assert torch.isfinite(out).all()
    assert (out[~mask] == 0).all()
    # padded positions must not influence observed ones
    x2 = x.clone()
    x2[1, :6] = 1e3
    assert torch.allclose(core(x2, mask)[1, 6:], out[1, 6:], atol=1e-5)


def test_transformer_is_causal() -> None:
    torch.manual_seed(0)
    core = TemporalTransformerCore(D, TransformerConfig(layers=2, heads=2, causal=True), 0.0).eval()
    x = torch.randn(2, 12, D)
    mask = torch.ones(2, 12, dtype=torch.bool)
    base = core(x, mask)
    x2 = x.clone()
    x2[:, 8:] += 5.0  # change the future
    out = core(x2, mask)
    assert torch.allclose(out[:, :8], base[:, :8], atol=1e-5)
    assert not torch.allclose(out[:, 8:], base[:, 8:])
    am = build_attention_mask(mask, causal=True)[0, 0]
    assert not am.triu(1).any()


def test_transformer_variable_lengths() -> None:
    core = TemporalTransformerCore(D, TransformerConfig(layers=1, heads=2), 0.0).eval()
    for t in (1, 5, 33):
        assert core(torch.randn(2, t, D), torch.ones(2, t, dtype=torch.bool)).shape == (2, t, D)


# ---------------------------------------------------------------- recurrent
@pytest.mark.parametrize("cell", ["gru", "lstm"])
def test_recurrent_ignores_padding_and_gaps(cell: str) -> None:
    torch.manual_seed(0)
    core = RecurrentCore(D, RecurrentConfig(cell=cell, hidden_dim=12, layers=2), 0.0).eval()
    x = torch.randn(3, 9, D)
    mask = _mask(3, 9, [9, 5, 0])
    mask[0, 3] = False  # interior gap
    out = core(x, mask)
    assert out.shape == (3, 9, D) and torch.isfinite(out).all()
    x2 = x.clone()
    x2[~mask] = 100.0
    assert torch.allclose(core(x2, mask), out, atol=1e-5)
    # recurrent state is causal as well
    x3 = x.clone()
    x3[0, 7:] = -3.0
    assert torch.allclose(core(x3, mask)[0, :7], out[0, :7], atol=1e-5)


# ---------------------------------------------------------------- TCN
def test_tcn_causality_and_receptive_field() -> None:
    torch.manual_seed(0)
    core = TCNCore(D, TCNConfig(channels=[8, 8, 8], kernel_size=3), 0.0).eval()
    assert core.receptive_field == 1 + 2 * 2 * (1 + 2 + 4)
    x = torch.randn(2, 40, D)
    mask = torch.ones(2, 40, dtype=torch.bool)
    base = core(x, mask)
    x2 = x.clone()
    x2[:, 30:] += 3.0
    out = core(x2, mask)
    assert torch.allclose(out[:, :30], base[:, :30], atol=1e-5), "TCN leaked future information"
    # a step far outside the receptive field cannot affect the last output
    x3 = x.clone()
    x3[:, 0] += 50.0
    assert torch.allclose(core(x3, mask)[:, -1], base[:, -1], atol=1e-5)


def test_tcn_masked_input_zeroed() -> None:
    core = TCNCore(D, TCNConfig(channels=[8], kernel_size=2), 0.0).eval()
    x = torch.randn(1, 6, D)
    mask = _mask(1, 6, [3])
    x2 = x.clone()
    x2[0, :3] = 77.0
    assert torch.allclose(core(x, mask), core(x2, mask))


# ---------------------------------------------------------------- tabular / graph
def test_residual_mlp() -> None:
    cfg = make_tiny_config().model.tabular
    mlp = ResidualMLP(10, 7, cfg, 0.1)
    out = mlp(torch.randn(5, 10))
    assert out.shape == (5, 7)
    out.sum().backward()
    assert all(p.grad is not None for p in mlp.parameters())


@pytest.mark.parametrize("kind", ["sage", "gat"])
def test_graph_encoder(kind: str) -> None:
    cfg = make_tiny_config(graph=True).model.graph.model_copy(update={"kind": kind})
    enc = GraphEncoder(4, D, 3, cfg, 0.0)
    nf = torch.randn(7, 4)
    ei = torch.tensor([[1, 2, 3, 5, 6], [0, 0, 1, 4, 4]])
    et = torch.tensor([0, 0, 1, 0, 2])
    node_batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    target = torch.tensor([0, 4, -1])
    out = enc(nf, ei, et, node_batch, target, 3)
    assert out.shape == (3, D) and torch.isfinite(out).all()
    out[:2].sum().backward()
    assert enc.inp.weight.grad is not None and enc.inp.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------- gating
def test_gating_weights_sum_to_one_and_respect_availability() -> None:
    cfg = make_tiny_config().model.gating
    gate = GatingNetwork(4, D, cfg, 0.0).eval()
    latents = torch.randn(6, 4, D)
    avail = torch.ones(6, 4, dtype=torch.bool)
    avail[0, 1] = False
    avail[1, :3] = False
    out = gate(latents, avail)
    assert torch.allclose(out.weights.sum(-1), torch.ones(6), atol=1e-6)
    assert out.weights[0, 1] == 0 and torch.allclose(out.weights[1, 3], torch.tensor(1.0))
    assert set(out.aux_losses) == {"gate_load_balance", "gate_entropy", "gate_z_loss"}
    # no expert available at all → falls back to uniform instead of NaN
    none = gate(latents[:1], torch.zeros(1, 4, dtype=torch.bool))
    assert torch.isfinite(none.weights).all()


def test_load_balance_penalises_collapse() -> None:
    cfg = make_tiny_config().model.gating.model_copy(update={"load_balance_weight": 1.0, "noise_std": 0.0})
    gate = GatingNetwork(3, D, cfg, 0.0).train()
    last = gate.net[-1]
    assert isinstance(last, torch.nn.Linear)
    with torch.no_grad():
        last.bias.copy_(torch.tensor([10.0, 0.0, 0.0]))  # collapse onto expert 0
    collapsed = gate(torch.randn(8, 3, D), torch.ones(8, 3, dtype=torch.bool)).aux_losses["gate_load_balance"]
    with torch.no_grad():
        last.bias.zero_()
    uniform = gate(torch.randn(8, 3, D), torch.ones(8, 3, dtype=torch.bool)).aux_losses["gate_load_balance"]
    assert collapsed > 1.5 and uniform < 0.05


# ---------------------------------------------------------------- full model
@pytest.fixture
def model_and_batch(norm_batch: Batch, tiny_config: NeuralConfig) -> tuple[NardisNeuralNetwork, Batch]:
    torch.manual_seed(0)
    return NardisNeuralNetwork(tiny_config), norm_batch


def test_full_forward_shapes(
    model_and_batch: tuple[NardisNeuralNetwork, Batch], tiny_config: NeuralConfig
) -> None:
    model, batch = model_and_batch
    model.eval()
    out = model(batch)
    b, h = batch.size, len(tiny_config.targets.horizons)
    for t in ("return", "max_upside", "max_drawdown", "volatility"):
        assert out.means[t].shape == (b, h) and out.logvars[t].shape == (b, h)
    assert out.logits["upside"].shape == (b, h)
    assert out.quantiles is not None and out.quantiles.shape == (b, h, 3)
    assert (out.quantiles.diff(dim=-1) > 0).all(), "quantiles must be monotone"
    for t in ("max_upside", "max_drawdown", "volatility"):
        assert (out.means[t] >= 0).all()
    assert out.embedding.shape == (b, tiny_config.model.latent_dim)
    assert out.expert_weights.shape == (b, 4)
    assert torch.allclose(out.expert_weights.sum(-1), torch.ones(b), atol=1e-5)
    assert set(out.timescale_weights) == {"transformer", "recurrent", "tcn"}
    for tensor in [*out.means.values(), *out.logvars.values(), *out.logits.values(), out.embedding]:
        assert torch.isfinite(tensor).all()


def test_unnormalized_batch_rejected(tiny_config: NeuralConfig, base_store: ArrayStore) -> None:
    model = NardisNeuralNetwork(tiny_config)
    with pytest.raises(ValueError, match="normalised"):
        model(MarketDataset(base_store, tiny_config)[np.arange(2)])


def test_backward_reaches_every_expert(model_and_batch: tuple[NardisNeuralNetwork, Batch]) -> None:
    model, batch = model_and_batch
    model.train()
    out = model(batch)
    loss = sum(v.pow(2).mean() for v in out.means.values()) + sum(v.mean() for v in out.logits.values())
    loss = loss + sum(out.aux_losses.values())
    loss.backward()
    for name, expert in model.experts.items():
        grads = [p.grad.abs().sum() for p in expert.parameters() if p.grad is not None]
        assert grads and sum(grads) > 0, f"no gradient reached expert {name}"
    for name, p in model.gate.named_parameters():
        assert p.grad is not None, name


def test_disabled_experts_get_zero_weight(model_and_batch: tuple[NardisNeuralNetwork, Batch]) -> None:
    model, batch = model_and_batch
    model.eval()
    model.set_disabled_experts({"transformer", "tcn"})
    out = model(batch)
    idx = [model.expert_names.index(n) for n in ("transformer", "tcn")]
    assert (out.expert_weights[:, idx] == 0).all()
    assert torch.allclose(out.expert_weights.sum(-1), torch.ones(batch.size), atol=1e-5)
    with pytest.raises(ValueError):
        model.set_disabled_experts({"nonexistent"})


def test_config_disabled_experts_not_built(tiny_config: NeuralConfig, norm_batch: Batch) -> None:
    tiny_config.model.experts = ["tabular", "tcn"]
    model = NardisNeuralNetwork(tiny_config).eval()
    assert model.expert_names == ("tcn", "tabular")
    assert model(norm_batch).expert_weights.shape[1] == 2


def test_gating_is_dynamic_per_observation(model_and_batch: tuple[NardisNeuralNetwork, Batch]) -> None:
    model, batch = model_and_batch
    last = model.gate.net[-1]
    assert isinstance(last, torch.nn.Linear)
    torch.nn.init.normal_(last.weight, std=0.5)
    model.eval()
    w = model(batch).expert_weights
    assert w.std(dim=0).max() > 1e-3, "gate weights must vary across observations"


def test_missing_sequences_mark_experts_unavailable(
    model_and_batch: tuple[NardisNeuralNetwork, Batch],
) -> None:
    model, batch = model_and_batch
    model.eval()
    empty = {
        k: SequenceBatch(torch.zeros_like(s.values), torch.zeros_like(s.mask), s.time_deltas)
        for k, s in batch.sequences.items()
    }
    out = model(replace(batch, sequences=empty))
    seq_idx = [model.expert_names.index(n) for n in ("transformer", "recurrent", "tcn")]
    assert (out.expert_weights[:, seq_idx] == 0).all()
    assert torch.allclose(out.expert_weights[:, model.expert_names.index("tabular")], torch.ones(batch.size))
    # a batch without some timescales entirely still works
    partial = replace(batch, sequences={"fast": batch.sequences["fast"]})
    assert torch.isfinite(model(partial).means["return"]).all()


def test_multitimescale_independent_encoders(tiny_config: NeuralConfig, norm_batch: Batch) -> None:
    tiny_config.model.share_encoder_across_timescales = False
    model = NardisNeuralNetwork(tiny_config).eval()
    exp = model.experts["transformer"]
    assert isinstance(exp, SequenceExpert) and len(exp.cores) == 3
    out = model(norm_batch)
    tw = out.timescale_weights["transformer"]
    assert tw.shape == (norm_batch.size, 3)
    assert torch.allclose(tw.sum(-1)[norm_batch.sequences["fast"].available], torch.ones(1), atol=1e-5)


def test_graph_pathway_end_to_end() -> None:
    cfg = make_tiny_config(graph=True)
    arrays = generate_synthetic(cfg, SyntheticSpec(n_observations=80, seed=4, graph=True))
    store = ArrayStore(arrays)
    norm = FeatureNormalizer.fit(store, np.arange(60), cfg)
    batch = norm.transform_batch(MarketDataset(store, cfg)[np.arange(40)])
    model = NardisNeuralNetwork(cfg)
    assert "graph" in model.expert_names
    out = model(batch)
    gi = model.expert_names.index("graph")
    assert batch.graph is not None
    no_graph = ~batch.graph.available
    assert no_graph.any() and (out.expert_weights[no_graph, gi] == 0).all()
    out.means["return"].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.experts["graph"].parameters())


def test_model_without_graph_ignores_graph_inputs() -> None:
    cfg = make_tiny_config(graph=False)
    graph_cfg = make_tiny_config(graph=True)
    arrays = generate_synthetic(graph_cfg, SyntheticSpec(n_observations=30, seed=4, graph=True))
    store = ArrayStore(arrays)
    batch = MarketDataset(store, cfg)[np.arange(10)]
    assert batch.graph is None


def test_batched_timescales_match_per_timescale_encoding(
    norm_batch: Batch, tiny_config: NeuralConfig
) -> None:
    """The shared core encodes all timescales in one call; this must equal separate calls."""
    model = NardisNeuralNetwork(tiny_config).eval()
    for name, expert in model.sequence_experts().items():
        tokens, avail = expert.encode_timescales(norm_batch)
        for i, ts in enumerate(expert.timescales):
            seq = norm_batch.sequences[ts]
            ref = expert.summarize(expert.encode_timesteps(ts, seq), seq.mask)
            a = avail[:, i]
            assert torch.allclose(tokens[a, i], ref[a], atol=1e-5), f"{name}/{ts}"
