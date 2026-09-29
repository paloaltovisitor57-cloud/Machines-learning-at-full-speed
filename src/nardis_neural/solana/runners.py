"""Runner identification: recognise, early, the tokens that go on to 2x, 5x, 10x, 100x, 1000x.

The tail model (:mod:`nardis_neural.solana.moonshot.tail`) describes the whole payoff
distribution with a parametric mixture.  The runner detector is its complement: one
gradient-boosted classifier per chase target, answering directly "will this token reach k?"
from the same causal features.  Trees pick up feature *combinations* (e.g. fast organic
buying *and* low cluster concentration *and* a rising branching ratio) that a smooth
mixture may miss.

Labels make the most of rare runners:

* a row whose outcome is resolved is labelled normally;
* a row still censored at the data end, but whose peak **already** reached ``k``, is a known
  positive (it cannot un-reach it), so it is kept;
* a censored row below ``k`` is unknown and left out.

A target is only trained when it has at least ``min_positives`` known hits, and evaluation is
always on later-launched tokens than training (see :func:`run_runner_research`), with
permutation importance showing *which signals identify runners*.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from nardis_neural.solana.chase import CHASE_TARGETS
from nardis_neural.solana.edge.trees import TreeEnsemble

F64 = npt.NDArray[np.float64]
_PARAMS: dict[str, Any] = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "max_iter": 300,
    "l2_regularization": 1.0,
    "min_samples_leaf": 40,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 25,
}


def runner_labels(
    peak: F64, censored: npt.NDArray[np.bool_], target: float
) -> tuple[F64, npt.NDArray[np.bool_]]:
    """``(label, known)`` for reaching ``target``: resolved rows, plus censored rows that already hit it."""
    hit = peak >= target
    known = ~censored | hit
    return hit.astype(np.float64), known


def _sigmoid(z: F64) -> F64:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class RunnerDetector:
    """One boosted classifier per chase target, exported to plain arrays."""

    def __init__(self, feature_names: list[str], min_positives: int = 15, seed: int = 0) -> None:
        self.feature_names = list(feature_names)
        self.min_positives, self.seed = min_positives, seed
        self.models: dict[float, TreeEnsemble] = {}
        self.report: dict[str, Any] = {}

    def fit(self, x: npt.NDArray[Any], peak: F64, censored: npt.NDArray[np.bool_]) -> dict[str, Any]:
        """Fit every target with enough known hits; returns per-target counts."""
        from sklearn.ensemble import HistGradientBoostingClassifier

        x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=np.nan)
        rep: dict[str, Any] = {}
        self.models = {}
        for k in CHASE_TARGETS:
            y, known = runner_labels(peak, censored, k)
            pos = int(y[known].sum())
            rep[f"{k:g}x"] = {"known_rows": int(known.sum()), "hits": pos, "trained": False}
            if pos < self.min_positives or int(known.sum()) - pos < self.min_positives:
                continue
            clf = HistGradientBoostingClassifier(random_state=self.seed, **_PARAMS)
            clf.fit(x[known], y[known])
            self.models[k] = TreeEnsemble.export(clf)
            rep[f"{k:g}x"]["trained"] = True
        self.report = rep
        return rep

    def predict(self, x: npt.NDArray[Any]) -> dict[float, F64]:
        """P(reach k) per trained target, made monotone in k."""
        x = np.asarray(x, dtype=np.float64)
        out: dict[float, F64] = {}
        prev: F64 | None = None
        for k in CHASE_TARGETS:
            if k not in self.models:
                continue
            p = _sigmoid(self.models[k].predict(x))
            if prev is not None:
                p = np.minimum(p, prev)
            out[k] = p
            prev = p
        return out

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("runner_*.npz"):
            f.unlink()
        for k, m in self.models.items():
            m.save(d / f"runner_{k:g}x.npz")
        meta = {
            "feature_names": self.feature_names,
            "min_positives": self.min_positives,
            "seed": self.seed,
            "targets": list(self.models),
            "report": self.report,
        }
        (d / "runners.json").write_text(json.dumps(meta, indent=2, default=float))

    @classmethod
    def load(cls, directory: str | Path) -> RunnerDetector:
        d = Path(directory)
        meta = json.loads((d / "runners.json").read_text())
        det = cls(list(meta["feature_names"]), int(meta["min_positives"]), int(meta["seed"]))
        det.models = {float(k): TreeEnsemble.load(d / f"runner_{float(k):g}x.npz") for k in meta["targets"]}
        det.report = dict(meta.get("report", {}))
        return det


def _auc(y: F64, s: F64) -> float | None:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def run_runner_research(
    store: Any,
    cfg: Any,
    spec: Any = None,
    test_fraction: float = 0.35,
    seed: int = 0,
    log: Any = None,
    importance_repeats: int = 5,
) -> tuple[dict[str, Any], RunnerDetector]:
    """Train on earlier tokens (labels truncated at the cutoff), score once on later tokens.

    Compares the detector with the tail model and their average on the same rows (AUC per
    target, hit rates of the top 5 / 10 % of tokens, one entry per token) and reports the
    permutation importance of the features for identifying 2x and 10x runners.  Returns the
    report and a detector refitted on every token.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance

    from nardis_neural.solana.moonshot import MoonshotSpec
    from nardis_neural.solana.moonshot.research import build_moonshot_dataset, first_signal, moonshot_features
    from nardis_neural.solana.moonshot.tail import TailModel

    say = log or (lambda _m: None)
    spec = spec or MoonshotSpec()
    ncfg = cfg.neural_config()
    mds = build_moonshot_dataset(store, cfg, spec, ncfg)
    x, names = moonshot_features(mds, None, ncfg.horizon_names)
    x = np.asarray(x, dtype=np.float64)
    ts, mints = mds.timestamps, mds.mints
    launch = {str(m): mds.market.token(str(m)).launch.t for m in np.unique(mints)}
    tokens = sorted(launch, key=lambda m: launch[m])
    n_test = max(1, round(len(tokens) * test_fraction))
    if len(tokens) - n_test < 20:
        raise ValueError("not enough tokens for a train / test split")
    test_tokens = set(tokens[-n_test:])
    is_test = np.array([str(m) in test_tokens for m in mints])
    cutoff = float(ts[is_test].min())
    tl = mds.labels(cutoff)
    train = np.flatnonzero(~is_test & tl.valid & (ts + spec.latency_seconds < cutoff))
    say(f"runners: {len(tokens)} tokens; train {len(train)} rows; test {n_test} tokens")

    det = RunnerDetector(list(names), seed=seed)
    fit_rep = det.fit(x[train], tl.peak[train], tl.censored[train])
    tail = TailModel(x.shape[1], spec, members=5, feature_names=list(names), seed=seed)
    tail.fit(x[train], tl.peak[train], tl.censored[train], ts[train], mints[train])

    lab = mds.labels()
    test = np.flatnonzero(is_test & lab.valid)
    first = test[first_signal(ts[test], mints[test], np.ones(len(test), dtype=bool))]
    p_det = det.predict(x[first])
    p_tail_all = tail.predict(x[first]).survival
    levels = list(spec.levels)
    report: dict[str, Any] = {
        "tokens": {"train": len(tokens) - n_test, "test": n_test, "test_entries": len(first)},
        "horizon_seconds": spec.horizon_seconds,
        "fit": fit_rep,
        "targets": {},
    }
    for k in CHASE_TARGETS:
        y, known = runner_labels(lab.peak[first], lab.censored[first], k)
        rows = np.flatnonzero(known)
        p_tail = p_tail_all[:, levels.index(k)]
        entry: dict[str, Any] = {"known_entries": len(rows), "hits": int(y[rows].sum())}
        scores = {"tail": p_tail}
        if k in p_det:
            scores["detector"] = p_det[k]
            scores["blend"] = 0.5 * (p_det[k] + p_tail)
        for name, s in scores.items():
            e: dict[str, Any] = {"auc": _auc(y[rows], s[rows])}
            order = rows[np.argsort(-s[rows])]
            for q in (0.05, 0.10):
                top = order[: max(1, round(len(order) * q))]
                e[f"top{int(q * 100)}_hit_rate"] = float(y[top].mean()) if len(top) else None
                e[f"top{int(q * 100)}_hits"] = int(y[top].sum())
            entry[name] = e
        entry["base_hit_rate"] = float(y[rows].mean()) if len(rows) else None
        report["targets"][f"{k:g}x"] = entry

    # which signals identify runners: permutation importance on the test entries
    importance: dict[str, list[tuple[str, float]]] = {}
    for k in (2.0, 10.0):
        if k not in det.models:
            continue
        y_tr, kn_tr = runner_labels(tl.peak[train], tl.censored[train], k)
        clf = HistGradientBoostingClassifier(random_state=seed, **_PARAMS).fit(x[train][kn_tr], y_tr[kn_tr])
        y_te, kn_te = runner_labels(lab.peak[first], lab.censored[first], k)
        if not 0 < y_te[kn_te].sum() < kn_te.sum():
            continue
        imp = permutation_importance(
            clf,
            x[first][kn_te],
            y_te[kn_te],
            scoring="roc_auc",
            n_repeats=importance_repeats,
            random_state=seed,
        )
        order_i = np.argsort(-imp.importances_mean)[:12]
        importance[f"{k:g}x"] = [(names[i], float(imp.importances_mean[i])) for i in order_i]
    report["importance"] = importance

    # production detector: every token, labels as known at the data end
    rows_all = np.flatnonzero(lab.valid)
    final = RunnerDetector(list(names), seed=seed)
    final.fit(x[rows_all], lab.peak[rows_all], lab.censored[rows_all])
    return report, final


def runner_markdown(report: dict[str, Any]) -> str:
    """Readable summary of :func:`run_runner_research`."""
    t = report["tokens"]
    lines = [
        "# Runner identification (out-of-sample tokens)",
        "",
        f"Train tokens {t['train']}, test tokens {t['test']} ({t['test_entries']} entries); "
        f"outcome horizon {report['horizon_seconds'] / 60:.0f} min.",
        "",
        "| target | known / hits | base rate | model | AUC | top 5 % hit rate | top 10 % hit rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for tag, e in report["targets"].items():
        for name in ("tail", "detector", "blend"):
            if name not in e:
                continue
            m = e[name]
            auc = f"{m['auc']:.3f}" if m["auc"] is not None else "–"
            t5 = f"{m['top5_hit_rate']:.1%} ({m['top5_hits']})" if m["top5_hit_rate"] is not None else "–"
            t10 = f"{m['top10_hit_rate']:.1%} ({m['top10_hits']})" if m["top10_hit_rate"] is not None else "–"
            base = f"{e['base_hit_rate']:.1%}" if e["base_hit_rate"] is not None else "–"
            lines.append(
                f"| {tag} | {e['known_entries']} / {e['hits']} | {base} | {name} | {auc} | {t5} | {t10} |"
            )
    for tag, feats in report.get("importance", {}).items():
        lines += ["", f"What identifies {tag} runners (permutation importance, AUC drop):", ""]
        lines += [f"* `{n}`: {v:+.4f}" for n, v in feats if v > 0]
    return "\n".join(lines) + "\n"
