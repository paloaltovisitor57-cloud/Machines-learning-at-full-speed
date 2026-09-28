"""Criticality of trade flow: an online self-exciting (Hawkes) point-process estimator.

A runner is a cascade: buys trigger more buys.  The exponential Hawkes process models
exactly that,

    λ(t) = μ + Σ_{t_j < t} n · β · exp(−β (t − t_j))

where ``μ`` is the exogenous rate (news, insiders, bots acting on their own schedule),
``β`` the decay rate of excitation (1/β is the herding timescale) and ``n`` the
**branching ratio**: the expected number of events each event directly triggers.

* ``n ≪ 1`` — activity is driven from outside and dies out when the drivers stop;
* ``n → 1`` — the process is at a phase transition: cascades become arbitrarily large
  (the epidemiologists' R₀ = 1, the physicists' criticality; Filimonov & Sornette's
  "endogeneity" of financial markets);
* the share of events attributed to excitation separates **organic herding** from
  **scripted** flow (wash bots and staged insiders trade on a clock, not on each other).

The estimator maximises the exact likelihood with expectation–maximisation for each decay
rate on a small log-spaced grid and keeps the best.  For a fixed ``β`` each EM step is
O(N); the O(N²) kernel sums are computed once per ``β`` on at most ``max_events`` recent
events, so a fit costs a few milliseconds.

Near criticality with slow decay the estimate is biased *downwards* in short windows
(measured: true n = 0.9 at β = 0.1 reads ≈ 0.4–0.65 on 200–600 events), a known property
of finite-window Hawkes estimation; the ordering of tokens by n is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
BETAS = np.geomspace(0.02, 5.0, 8)
"""Decay rates tried (per second): herding timescales from 0.2 s to 50 s."""
TRENDS = np.array([-0.03, -0.015, -0.008, -0.004, -0.002, 0.0, 0.003, 0.008])
"""Background-rate trends tried by the detrended fit (per second): a launch rush that fades
over ~50 s … steady … demand building over minutes."""


@dataclass(frozen=True)
class HawkesFit:
    """Maximum-likelihood exponential Hawkes parameters for one event stream."""

    mu: float
    """Exogenous (background) rate, events per second."""
    branching: float
    """Branching ratio n: expected events directly triggered by each event."""
    beta: float
    """Excitation decay rate (per second); 1/β is the herding timescale."""
    endogenous_share: float
    """Share of events attributed to excitation by other events (vs background)."""
    loglik: float
    n_events: int
    trend: float = 0.0
    """Exponential trend γ of the background rate, μ(t) = μ · exp(γ (t − t_end)), per second."""
    reflex: float = 0.0
    """Branching carried by the fixed fast kernel (two-kernel fits only): sub-second reflexes such
    as bots and bundles reacting in the same slot.  ``branching`` is the total; ``branching −
    reflex`` is the slower herding part."""


EMPTY = HawkesFit(0.0, 0.0, 1.0, 0.0, 0.0, 0)


def excitation_sums(times: F64, betas: F64) -> F64:
    """``A[k, i] = Σ_{j: t_j < t_i} exp(−β_k (t_i − t_j))`` for sorted ``times``, exactly, in O(N) per β.

    Prefix sums of ``exp(β t)`` are taken in blocks short enough that the exponentials
    cannot overflow, with the running sum carried across block boundaries.  Events at the
    same timestamp do not excite each other.
    """
    x = np.asarray(times, dtype=np.float64)
    x = x - x[0] if len(x) else x
    out = np.zeros((len(betas), len(x)))
    for k, beta in enumerate(np.asarray(betas, dtype=np.float64)):
        e = beta * x
        start, carry = 0, 0.0  # carry: Σ over all earlier events, measured at the block's start
        while start < len(x):
            stop = max(start + int(np.searchsorted(e[start:], e[start] + 600.0, side="left")), start + 1)
            seg = e[start:stop] - e[start]
            prefix = np.concatenate([[0.0], np.cumsum(np.exp(seg))])
            # an event is excited only by events at strictly earlier timestamps
            first_of_tie = np.searchsorted(seg, seg, side="left")
            out[k, start:stop] = np.exp(-seg) * (prefix[first_of_tie] + carry)
            if stop < len(x):
                carry = float(np.exp(-(e[stop] - e[start])) * (prefix[-1] + carry))
            start = stop
    return out


def fit_hawkes(
    times: F64,
    t_start: float,
    t_end: float,
    betas: F64 = BETAS,
    trends: F64 | tuple[float, ...] = (0.0,),
    fast_beta: float | None = None,
    iterations: int = 60,
    max_events: int = 300,
    min_events: int = 8,
    max_branching: float = 1.5,
) -> HawkesFit:
    """Fit an exponential Hawkes process to event ``times`` observed on ``[t_start, t_end]``.

    Only the most recent ``max_events`` events are used.  Streams with fewer than
    ``min_events`` events return :data:`EMPTY`.  The branching ratio may exceed 1 in a
    finite window (a supercritical burst) and is capped at ``max_branching``.

    ``trends`` lets the background rate rise or decay exponentially (``γ`` per second) so
    that a fading launch rush is not mistaken for self-excitation; the best (β, γ) pair by
    likelihood wins.

    ``fast_beta`` adds a second, fixed fast kernel ``n_r · β_f · exp(−β_f Δ)`` alongside the
    fitted one: order flow has a sub-second reflex layer (bots, same-slot bundles) and a
    slower human herding layer, and one exponential can only describe one of them.
    """
    t = np.sort(np.asarray(times, dtype=np.float64))
    t = t[(t >= t_start) & (t <= t_end)]
    # the likelihood covers the most recent ``max_events``; up to as many earlier events are
    # kept as *history* so that their excitation is not misread as background (edge effect)
    hist = t[max(0, len(t) - 2 * max_events) : max(0, len(t) - max_events)]
    t = t[-max_events:]
    n_ev = len(t)
    if n_ev < min_events:
        return EMPTY
    w0 = float(t[0]) if len(hist) else t_start
    horizon = t_end - w0
    if horizon <= 0:
        return EMPTY
    src = np.concatenate([hist, t])
    b = np.asarray(betas, dtype=np.float64)
    a = excitation_sums(src, b)[:, len(hist) :]  # (B, N) excitation reaching each target event
    comp = (1.0 - np.exp(-b[:, None] * (t_end - t)[None, :])).sum(axis=1)  # kernel mass in window
    if len(hist):
        comp += (
            np.exp(-b[:, None] * (w0 - hist)[None, :]) - np.exp(-b[:, None] * (t_end - hist)[None, :])
        ).sum(axis=1)
    g_rates = np.asarray(trends, dtype=np.float64)
    nb, ng = len(b), len(g_rates)
    # every (β, γ) pair at once: rows are β-major
    ba = np.repeat(b[:, None] * a, ng, axis=0)  # (B·G, N) slow-kernel excitation
    comp_all = np.repeat(comp, ng)
    if fast_beta is not None:
        fb = np.array([fast_beta])
        fa = fast_beta * excitation_sums(src, fb)[0, len(hist) :]  # (N,) fast-kernel excitation
        fcomp = float((1.0 - np.exp(-fast_beta * (t_end - t))).sum())
        if len(hist):
            fcomp += float((np.exp(-fast_beta * (w0 - hist)) - np.exp(-fast_beta * (t_end - hist))).sum())
    else:
        fa, fcomp = np.zeros(n_ev), 1.0
    shape = np.tile(np.exp(g_rates[:, None] * (t - t_end)[None, :]), (nb, 1))  # background shape
    gint = np.where(
        np.abs(g_rates) > 1e-12,
        (1.0 - np.exp(g_rates * (w0 - t_end))) / np.where(np.abs(g_rates) > 1e-12, g_rates, 1.0),
        horizon,
    )
    gint_all = np.tile(gint, nb)  # ∫ of the background shape over the window
    mu = 0.5 * n_ev / gint_all
    n = np.full(nb * ng, 0.4 if fast_beta is not None else 0.5)
    r = np.full(nb * ng, 0.1 if fast_beta is not None else 0.0)
    for _ in range(iterations):  # EM for every (β, γ) at once
        lam = mu[:, None] * shape + n[:, None] * ba + r[:, None] * fa[None, :]
        bg = mu[:, None] * shape / lam
        slow = n[:, None] * ba / lam
        mu = bg.sum(axis=1) / gint_all
        n = np.minimum(slow.sum(axis=1) / np.maximum(comp_all, 1e-12), max_branching)
        if fast_beta is not None:
            r = np.minimum((1.0 - bg - slow).sum(axis=1) / max(fcomp, 1e-12), max_branching)
    lam = mu[:, None] * shape + n[:, None] * ba + r[:, None] * fa[None, :]
    ll = np.log(np.maximum(lam, 1e-300)).sum(axis=1) - mu * gint_all - n * comp_all - r * fcomp
    k = int(np.argmax(ll))
    share = float(((n[k] * ba[k] + r[k] * fa) / lam[k]).mean())
    best = HawkesFit(
        float(mu[k]),
        float(n[k] + r[k]),
        float(b[k // ng]),
        share,
        float(ll[k]),
        n_ev,
        float(g_rates[k % ng]),
        float(r[k]),
    )
    return best


def simulate_hawkes(
    mu: float, branching: float, beta: float, horizon: float, rng: np.random.Generator
) -> F64:
    """Exact simulation of an exponential Hawkes process on ``[0, horizon]`` (Ogata thinning)."""
    out: list[float] = []
    t, excite = 0.0, 0.0  # excite = Σ n β exp(−β (t − t_j))
    while True:
        bound = mu + excite
        w = rng.exponential(1.0 / bound)
        t += w
        if t > horizon:
            break
        excite *= np.exp(-beta * w)
        if rng.random() * bound <= mu + excite:
            out.append(t)
            excite += branching * beta
    return np.asarray(out, dtype=np.float64)
