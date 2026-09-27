"""NeuralEngine — the tiny public inference API.

    engine = NeuralEngine.load("models/champion")
    prediction = engine.predict(observation)

The engine bundles the ensemble, normaliser, calibrators, OOD detector and optional
regime model of one immutable model version.  It never emits trading decisions.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from nardis_neural.config import CLASSIFICATION_TASKS, REGRESSION_TASKS, NeuralConfig
from nardis_neural.data.datasets import Batch, MarketDataset, arrays_to_batch, make_loader
from nardis_neural.data.normalization import FeatureNormalizer
from nardis_neural.data.sequences import observations_to_arrays
from nardis_neural.inference.calibration import CalibrationSet
from nardis_neural.inference.ood import OODDetector
from nardis_neural.inference.uncertainty import (
    aggregate_classification,
    aggregate_regression,
    confidence_score,
    member_disagreement,
)
from nardis_neural.lifecycle.checkpoints import (
    CheckpointContents,
    ModelMetadata,
    load_checkpoint,
    save_checkpoint,
)
from nardis_neural.models.ensemble import DeepEnsemble
from nardis_neural.regimes.clustering import RegimeClusterer
from nardis_neural.schemas import NeuralObservation, NeuralPrediction
from nardis_neural.training.trainer import resolve_device

Array = npt.NDArray[Any]


class NeuralEngine:
    """One immutable model version: ensemble, normaliser, calibration, OOD detector and regimes.

    Produces probabilistic :class:`NeuralPrediction` objects only, never trading decisions.
    """

    def __init__(
        self,
        config: NeuralConfig,
        ensemble: DeepEnsemble,
        normalizer: FeatureNormalizer,
        calibration: CalibrationSet,
        metadata: ModelMetadata,
        ood: OODDetector | None = None,
        regimes: RegimeClusterer | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device) if device is not None else resolve_device(config.training.device)
        self.ensemble = ensemble.to(self.device).eval()
        self.normalizer = normalizer
        self.calibration = calibration
        self.metadata = metadata
        self.ood = ood
        self.regimes = regimes

    # ------------------------------------------------------------------ persistence
    @property
    def version(self) -> str:
        """Model version from the metadata."""
        return self.metadata.version

    @property
    def expert_names(self) -> tuple[str, ...]:
        """Names of the enabled experts."""
        return self.ensemble.member(0).expert_names

    def contents(self) -> CheckpointContents:
        """The engine's components as :class:`CheckpointContents`."""
        return CheckpointContents(
            self.config,
            self.ensemble,
            self.normalizer,
            self.calibration,
            self.ood,
            self.regimes,
            self.metadata,
        )

    def save(self, path: str | Path, overwrite: bool = False) -> Path:
        """Write a checkpoint directory to ``path`` and return it."""
        return save_checkpoint(path, self.contents(), overwrite=overwrite)

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str | None = None) -> NeuralEngine:
        """Load a model directory, or a model registry root (loads its champion)."""
        p = Path(path)
        if not (p / "manifest.json").exists() and (p / "registry.json").exists():
            from nardis_neural.lifecycle.champion import ModelRegistry

            p = ModelRegistry(p).champion_path()
        dev = torch.device(device) if device is not None else resolve_device("auto")
        c = load_checkpoint(p, dev)
        return cls(c.config, c.ensemble, c.normalizer, c.calibration, c.metadata, c.ood, c.regimes, dev)

    def clone(self, version: str, origin: str = "adapt") -> NeuralEngine:
        """Deep, fully isolated copy (weights, normaliser, calibration, metadata)."""
        md = self.metadata.model_copy(deep=True)
        md.parent_version = self.metadata.version
        md.version = version
        md.origin = origin
        return NeuralEngine(
            self.config.model_copy(deep=True),
            copy.deepcopy(self.ensemble),
            copy.deepcopy(self.normalizer),
            copy.deepcopy(self.calibration),
            md,
            copy.deepcopy(self.ood),
            copy.deepcopy(self.regimes),
            self.device,
        )

    # ------------------------------------------------------------------ core inference
    @torch.no_grad()
    def forward_arrays(self, batch: Batch, mc_samples: int | None = None) -> dict[str, Array]:
        """Full probabilistic forward pass on a raw (un-normalised) batch → NumPy arrays."""
        ec = self.config.ensemble
        mc = ec.mc_dropout_samples if mc_samples is None else mc_samples
        b = self.normalizer.transform_batch(batch.to(self.device))
        stacked = self.ensemble.forward_samples(b, mc_samples=mc, mc_seed=ec.mc_seed)
        out: dict[str, Array] = {}
        eps_parts, alea_parts, tot_parts = [], [], []
        for task in REGRESSION_TASKS:
            agg = aggregate_regression(stacked.means[task], stacked.logvars[task])
            mean, alea = self.normalizer.denormalize(task, agg.mean, agg.aleatoric_var)
            _, epi = self.normalizer.denormalize(task, agg.mean, agg.epistemic_var)
            assert alea is not None and epi is not None
            out[f"{task}.mean"] = mean.cpu().numpy()
            out[f"{task}.aleatoric_var"] = alea.cpu().numpy()
            out[f"{task}.epistemic_var"] = epi.cpu().numpy()
            out[f"{task}.std"] = torch.sqrt(alea + epi).cpu().numpy()
            if task == "return":
                # scalar summaries in normalised units (fraction of typical move per horizon)
                eps_parts.append(torch.sqrt(agg.epistemic_var).mean(dim=-1))
                alea_parts.append(torch.sqrt(agg.aleatoric_var).mean(dim=-1))
                tot_parts.append(torch.sqrt(agg.total_var).mean(dim=-1))
        if stacked.quantiles is not None:
            q = stacked.quantiles.mean(dim=0)
            scale = self.normalizer.target_scale_tensor("return", q[..., 0]).unsqueeze(-1)
            out["return.quantiles"] = (q * scale).cpu().numpy()
        cls_epi = []
        for task in CLASSIFICATION_TASKS:
            agg_c = aggregate_classification(stacked.logits[task])
            raw = agg_c.prob.cpu().numpy().astype(np.float64)
            out[f"{task}.prob_raw"] = raw
            out[f"{task}.prob"] = self.calibration.apply(task, raw)
            out[f"{task}.epistemic"] = agg_c.mutual_information.cpu().numpy()
            out[f"{task}.aleatoric"] = agg_c.aleatoric_entropy.cpu().numpy()
            cls_epi.append(agg_c.mutual_information.mean(dim=-1))
        epistemic = eps_parts[0]
        out["epistemic"] = epistemic.cpu().numpy()
        out["aleatoric"] = alea_parts[0].cpu().numpy()
        out["total_uncertainty"] = tot_parts[0].cpu().numpy()
        out["classification_epistemic"] = torch.stack(cls_epi).mean(dim=0).cpu().numpy()
        out["disagreement"] = member_disagreement(stacked.means["return"], stacked.member_index).cpu().numpy()
        emb = stacked.embeddings[min(ec.embedding_member, stacked.embeddings.shape[0] - 1)]
        out["embedding"] = emb.cpu().numpy()
        out["expert_weights"] = stacked.expert_weights.mean(dim=0).cpu().numpy()
        input_rms = torch.sqrt(
            (self.normalizer.current_zscore(batch.current.to(self.device)) ** 2).mean(dim=-1)
        )
        out["input_rms"] = input_rms.cpu().numpy()
        if self.ood is not None:
            comp = self.ood.score(out["embedding"], out["input_rms"], out["epistemic"])
            for k, v in comp.items():
                out["ood_score" if k == "score" else f"ood.{k}"] = v
        else:
            out["ood_score"] = np.zeros(b.size)
        ref = float(self.metadata.reference.get("epistemic_median", 0.0)) or float(
            np.median(out["epistemic"]) + 1e-8
        )
        conf, c_epi, c_ood = confidence_score(
            epistemic,
            ref,
            torch.as_tensor(out["ood_score"], device=epistemic.device),
            self.config.ood.confidence_penalty,
        )
        out["confidence"] = conf.cpu().numpy()
        out["confidence.epistemic"] = c_epi.cpu().numpy()
        out["confidence.ood"] = c_ood.cpu().numpy()
        out["regime"] = (
            self.regimes.predict(out["embedding"])
            if self.regimes is not None
            else np.full(b.size, -1, np.int64)
        )
        out["observation_id"] = np.asarray(batch.observation_ids)
        out["timestamp"] = batch.timestamps
        return out

    def predict_arrays(self, arrays: dict[str, Array], mc_samples: int | None = None) -> dict[str, Array]:
        """Predict canonical row arrays; returns the :meth:`forward_arrays` dict."""
        return self.forward_arrays(arrays_to_batch(arrays, self.config), mc_samples)

    def predict_dataset(
        self, dataset: MarketDataset, batch_size: int | None = None, mc_samples: int | None = None
    ) -> dict[str, Array]:
        """Predict every row of a dataset in its stored order."""
        bs = batch_size or self.config.training.eval_batch_size
        loader = make_loader(dataset, bs, shuffle=False)
        parts = [self.forward_arrays(batch, mc_samples) for batch in loader]
        if not parts:
            raise ValueError("dataset is empty")
        return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}

    # ------------------------------------------------------------------ public API
    def predict_batch(self, observations: Sequence[NeuralObservation]) -> list[NeuralPrediction]:
        """Predict several observations in one pass (empty input → empty list)."""
        if not observations:
            return []
        arrays = observations_to_arrays(observations, self.config)
        out = self.predict_arrays(arrays)
        return self.to_predictions(out)

    def predict(self, observation: NeuralObservation) -> NeuralPrediction:
        """Predict a single observation."""
        return self.predict_batch([observation])[0]

    def to_predictions(self, out: dict[str, Array]) -> list[NeuralPrediction]:
        """Convert a :meth:`forward_arrays` result to one :class:`NeuralPrediction` per row."""
        horizons = self.config.horizon_names
        qs = [f"q{round(q * 100):02d}" for q in self.config.targets.quantiles]
        names = list(self.expert_names)
        preds = []
        for i in range(len(out["observation_id"])):

            def hz(key: str, i: int = i) -> dict[str, float]:
                return {h: float(out[key][i, j]) for j, h in enumerate(horizons)}

            quant = (
                {
                    h: {q: float(out["return.quantiles"][i, j, k]) for k, q in enumerate(qs)}
                    for j, h in enumerate(horizons)
                }
                if "return.quantiles" in out
                else {}
            )
            regime = int(out["regime"][i])
            preds.append(
                NeuralPrediction(
                    observation_id=str(out["observation_id"][i]),
                    model_version=self.version,
                    timestamp=float(out["timestamp"][i]),
                    horizons=horizons,
                    expected_returns=hz("return.mean"),
                    return_std=hz("return.std"),
                    return_quantiles=quant,
                    upside_probabilities=hz("upside.prob"),
                    downside_probabilities=hz("downside.prob"),
                    maximum_upside=hz("max_upside.mean"),
                    maximum_drawdown=hz("max_drawdown.mean"),
                    predicted_volatility=hz("volatility.mean"),
                    epistemic_uncertainty=float(out["epistemic"][i]),
                    aleatoric_uncertainty=float(out["aleatoric"][i]),
                    total_uncertainty=float(out["total_uncertainty"][i]),
                    confidence=float(out["confidence"][i]),
                    confidence_components={
                        "epistemic": float(out["confidence.epistemic"][i]),
                        "ood": float(out["confidence.ood"][i]),
                    },
                    ood_score=float(out["ood_score"][i]),
                    member_disagreement=float(out["disagreement"][i]),
                    market_embedding=[float(v) for v in out["embedding"][i]],
                    expert_weights={n: float(out["expert_weights"][i, k]) for k, n in enumerate(names)},
                    regime_cluster=None if regime < 0 else regime,
                )
            )
        return preds

    def describe(self) -> dict[str, Any]:
        """Summary of version lineage, architecture, calibration and reference statistics."""
        return {
            "version": self.version,
            "parent_version": self.metadata.parent_version,
            "origin": self.metadata.origin,
            "created_at": self.metadata.created_at_iso,
            "git_commit": self.metadata.git_commit,
            "data_fingerprint": self.metadata.data_fingerprint,
            "ensemble_size": self.ensemble.size,
            "experts": list(self.expert_names),
            "horizons": self.config.horizon_names,
            "parameters_per_member": self.ensemble.member(0).parameter_count(),
            "device": str(self.device),
            "calibration": {k: [c.method for c in v] for k, v in self.calibration.calibrators.items()},
            "has_ood": self.ood is not None,
            "has_regimes": self.regimes is not None,
            "reference": self.metadata.reference,
            "training_stats": self.metadata.training_stats,
        }
