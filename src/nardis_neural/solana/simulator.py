"""Agent-based simulator of Solana memecoin launches — for testing and demos only.

It is *not* a profitability model.  It produces raw on-chain-like event streams whose
structure the Solana stack must be able to discover:

Wallet population
    retail (CEX-funded long ago), smart money (joins healthy launches early and takes
    profit), snipers (buy in the first slots with high priority fees / Jito tips and flip),
    wash bots, honest devs, and rug crews (dev + bundle wallets funded minutes before
    launch from a shared funder).

Launch archetypes
    ``organic``   genuine but fading demand;
    ``graduate``  strong demand that completes the pump.fun curve and migrates to an AMM;
    ``rug``       bundled launch, hype, then dev + bundle dump (or an LP pull for
                  AMM-launched rugs), frequently with live mint authority;
    ``dud``       barely traded, slow bleed;
    ``wash``      bots churning volume with little real participation.

Pricing uses exact pump.fun / constant-product maths, so price impact and graduation
behave like the real venues.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nardis_neural.solana.amm import (
    AMM_FEE,
    PUMP_GRADUATION_SOL,
    Pool,
    pump_curve,
)
from nardis_neural.solana.events import (
    SLOT_SECONDS,
    Event,
    LiquidityChange,
    Migration,
    Swap,
    TokenLaunch,
    Transfer,
    Venue,
)
from nardis_neural.solana.market import EventStore

ARCHETYPES: tuple[str, ...] = ("organic", "graduate", "rug", "dud", "wash", "runner", "trap")
MARKET_PRESETS: dict[str, dict[str, float]] = {
    "default": {"organic": 0.3, "graduate": 0.15, "rug": 0.3, "dud": 0.15, "wash": 0.1, "runner": 0.0},
    # most launches go nowhere, a few percent become multi-hour runners
    "degen": {"organic": 0.12, "graduate": 0.06, "rug": 0.3, "dud": 0.42, "wash": 0.04, "runner": 0.06},
    # degen plus traps: launches staged to look like early runners (insiders funded through a
    # relay wallet, wash volume, runner-like demand) that dump after the entry window
    "adversarial": {
        "organic": 0.12,
        "graduate": 0.06,
        "rug": 0.2,
        "dud": 0.4,
        "wash": 0.04,
        "runner": 0.06,
        "trap": 0.12,
    },
}


@dataclass
class LaunchSimSpec:
    """Simulated market settings: token count, duration, seed, wallet population and archetype mix."""

    n_tokens: int = 40
    duration_seconds: float = 3 * 3600.0
    seed: int = 0
    start_time: float = 1_750_000_000.0
    n_retail: int = 800
    n_smart: int = 30
    n_snipers: int = 15
    n_bots: int = 8
    mint_prefix: str = "Mint"
    """Distinct prefixes let several simulated eras share one wallet population."""
    stealth_rug_fraction: float = 0.4
    """Rugs run from aged, hub-funded wallets that buy over the first minute with authorities
    revoked — no obvious bundle / fresh-wallet / mint-authority footprint."""
    decoy_fraction: float = 0.25
    """Honest launches where the dev's co-funded friends buy in the launch slots (bundle-like
    footprint without a rug)."""
    archetype_weights: dict[str, float] = field(default_factory=lambda: dict(MARKET_PRESETS["default"]))
    """``runner`` launches (rare viral tokens that compound for hours, 100–1000x+) are off by
    default; the ``degen`` preset turns them on."""


@dataclass
class _Token:
    mint: str
    archetype: str
    launch_t: float
    creator: str
    pool: Pool
    venue: Venue
    crew: list[str]
    friends: list[str]
    stealth: bool
    lifetime: float
    dump_at: float
    holdings: dict[str, float] = field(default_factory=dict)
    entry_price: dict[str, float] = field(default_factory=dict)
    migrated: bool = False
    dead: bool = False
    clock: float = 0.0

    def stamp(self, t: float) -> float:
        """Monotone per-token timestamps so event order always equals pricing order."""
        self.clock = max(t, self.clock + 1e-3)
        return self.clock


class LaunchSimulator:
    """Agent-based generator of launch event streams with ground-truth archetypes (tests and demos only)."""

    def __init__(self, spec: LaunchSimSpec | None = None) -> None:
        self.spec = spec or LaunchSimSpec()
        self.rng = np.random.default_rng(self.spec.seed)
        s = self.spec
        self.retail = [f"retail_{i}" for i in range(s.n_retail)]
        self.smart = [f"smart_{i}" for i in range(s.n_smart)]
        self.snipers = [f"sniper_{i}" for i in range(s.n_snipers)]
        self.bots = [f"bot_{i}" for i in range(s.n_bots)]
        self.events: list[Event] = []
        self.archetypes: dict[str, str] = {}

    # ------------------------------------------------------------------ helpers
    def _fund_population(self) -> None:
        s = self.spec
        for w in self.retail + self.smart + self.snipers + self.bots:
            t = s.start_time - float(self.rng.uniform(3, 60)) * 86400
            self.events.append(
                Transfer(t=t, source="cex_hot_wallet", dest=w, sol_amount=float(self.rng.uniform(1, 50)))
            )

    def _swap(
        self, tok: _Token, t: float, wallet: str, is_buy: bool, amount: float, fee: float, tip: float = 0.0
    ) -> None:
        if tok.dead:
            return
        if is_buy:
            if amount <= 0:
                return
            sol_in = amount
            tokens, tok.pool = tok.pool.buy(sol_in)
            if tokens <= 0:
                return
            prev = tok.holdings.get(wallet, 0.0)
            tok.holdings[wallet] = prev + tokens
            tok.entry_price.setdefault(wallet, tok.pool.price)
            sol_amt, tok_amt = sol_in, tokens
        else:
            held = tok.holdings.get(wallet, 0.0)
            tok_amt = min(amount, held)
            if tok_amt <= 1.0:
                return
            sol_amt, tok.pool = tok.pool.sell(tok_amt)
            tok.holdings[wallet] = held - tok_amt
        t = tok.stamp(t)
        self.events.append(
            Swap(
                mint=tok.mint,
                t=t,
                wallet=wallet,
                is_buy=is_buy,
                sol_amount=float(sol_amt),
                token_amount=float(tok_amt),
                sol_reserve=tok.pool.sol,
                token_reserve=tok.pool.tokens,
                priority_fee=fee,
                jito_tip=tip,
                slot=int(t / SLOT_SECONDS),
            )
        )
        if tok.venue == "pump_fun" and not tok.migrated and tok.pool.real_sol >= PUMP_GRADUATION_SOL:
            tok.migrated = True
            tok.venue = "pumpswap"
            sol = tok.pool.real_sol * 0.94
            tok.pool = Pool(sol, sol / tok.pool.price, AMM_FEE)
            tm = tok.stamp(t + 0.4)
            self.events.append(
                Migration(
                    mint=tok.mint,
                    t=tm,
                    venue="pumpswap",
                    sol_reserve=tok.pool.sol,
                    token_reserve=tok.pool.tokens,
                    slot=int(tm / SLOT_SECONDS),
                )
            )

    def _fee(self, kind: str) -> tuple[float, float]:
        if kind in ("sniper", "bot", "crew"):
            fee = float(self.rng.lognormal(np.log(5e-4), 0.5))
            tip = float(self.rng.lognormal(np.log(1e-3), 0.5)) if kind != "bot" else 0.0
            return fee, tip
        return float(self.rng.lognormal(np.log(1e-5), 1.0)), 0.0

    def _seller(self, tok: _Token, exclude: set[str]) -> str | None:
        cands = [w for w, b in tok.holdings.items() if b > 1.0 and w not in exclude]
        if not cands:
            return None
        bal = np.array([tok.holdings[w] for w in cands])
        return cands[int(self.rng.choice(len(cands), p=bal / bal.sum()))]

    # ------------------------------------------------------------------ token lifecycle
    def _launch(self, k: int) -> _Token:
        s = self.spec
        names = list(s.archetype_weights)
        w = np.array([s.archetype_weights[n] for n in names])
        arch = names[int(self.rng.choice(len(names), p=w / w.sum()))]
        t0 = s.start_time + float(self.rng.uniform(0, 0.75 * s.duration_seconds))
        mint = f"{s.mint_prefix}{k:04d}{arch[:3].upper()}"
        creator = f"dev_{s.mint_prefix}_{k}"
        crew: list[str] = []
        friends: list[str] = []
        stealth = arch == "rug" and self.rng.random() < s.stealth_rug_fraction
        if arch == "rug":
            n_crew = int(self.rng.integers(2, 5)) if stealth else int(self.rng.integers(4, 9))
            crew = [f"bundle_{s.mint_prefix}_{k}_{j}" for j in range(n_crew)]
            for i, wlt in enumerate([creator, *crew]):
                if stealth:  # aged wallets funded through an exchange hot wallet
                    ft, src = t0 - float(self.rng.uniform(3, 40)) * 86400, "cex_hot_wallet"
                else:  # rug crews reuse a handful of funders, minutes before launch
                    ft, src = t0 - float(self.rng.uniform(60, 900)) - i, f"rugfunder_{k % 4}"
                self.events.append(
                    Transfer(t=ft, source=src, dest=wlt, sol_amount=float(self.rng.uniform(2, 6)))
                )
        elif arch == "trap":
            stealth = True  # insiders trickle in over the first minute instead of bundling
            crew = [f"staged_{s.mint_prefix}_{k}_{j}" for j in range(int(self.rng.integers(4, 8)))]
            relay = f"relay_{s.mint_prefix}_{k}"
            self.events.append(
                Transfer(
                    t=t0 - float(self.rng.uniform(6, 24)) * 3600,
                    source=f"trapfunder_{k % 3}",
                    dest=relay,
                    sol_amount=float(self.rng.uniform(15, 40)),
                )
            )
            for i, wlt in enumerate(crew):  # two hops away from the funder, hours before launch
                self.events.append(
                    Transfer(
                        t=t0 - float(self.rng.uniform(1, 6)) * 3600 - i,
                        source=relay,
                        dest=wlt,
                        sol_amount=float(self.rng.uniform(2, 5)),
                    )
                )
            self.events.append(  # a clean-looking creator: aged, exchange-funded
                Transfer(
                    t=t0 - float(self.rng.uniform(5, 90)) * 86400,
                    source="cex_hot_wallet",
                    dest=creator,
                    sol_amount=float(self.rng.uniform(2, 20)),
                )
            )
        else:
            if self.rng.random() < s.decoy_fraction:
                friends = [f"friend_{s.mint_prefix}_{k}_{j}" for j in range(int(self.rng.integers(2, 5)))]
                for i, wlt in enumerate(friends):
                    self.events.append(
                        Transfer(
                            t=t0 - float(self.rng.uniform(60, 900)) - i,
                            source=creator,
                            dest=wlt,
                            sol_amount=float(self.rng.uniform(1, 3)),
                        )
                    )
            self.events.append(
                Transfer(
                    t=t0 - float(self.rng.uniform(5, 90)) * 86400,
                    source="cex_hot_wallet",
                    dest=creator,
                    sol_amount=float(self.rng.uniform(2, 20)),
                )
            )
        amm_launch = arch == "rug" and self.rng.random() < 0.3
        if amm_launch:
            pool = Pool(float(self.rng.uniform(20, 60)), 0.0, AMM_FEE)
            pool = Pool(pool.sol, pool.sol / 3e-8, AMM_FEE)
            venue: Venue = "raydium"
        else:
            pool = pump_curve()
            venue = "pump_fun"
        lifetime = {
            "organic": (1200, 3600),
            "graduate": (1800, 4200),
            "rug": (300, 1500),
            "dud": (300, 900),
            "wash": (900, 2400),
            "runner": (3 * 3600, 5 * 3600),
            "trap": (1800, 3600),
        }[arch]
        tok = _Token(
            mint=mint,
            archetype=arch,
            launch_t=t0,
            creator=creator,
            pool=pool,
            venue=venue,
            crew=crew,
            friends=friends,
            stealth=stealth,
            lifetime=float(self.rng.uniform(*lifetime)),
            dump_at=t0 + float(self.rng.uniform(60, 420)),
            clock=t0,
        )
        if arch == "trap":  # the dump comes after the moonshot entry window has opened
            tok.dump_at = t0 + float(self.rng.uniform(300, 1800))
        mint_live = (arch == "rug" and not stealth and self.rng.random() < 0.5) or self.rng.random() < 0.03
        self.events.append(
            TokenLaunch(
                mint=mint,
                t=t0,
                creator=creator,
                venue=venue,
                sol_reserve=pool.sol,
                token_reserve=pool.tokens,
                mint_authority_revoked=not mint_live,
                freeze_authority_revoked=not (arch == "rug" and mint_live),
                lp_burned_fraction=0.0 if amm_launch else 1.0,
                slot=int(t0 / SLOT_SECONDS),
            )
        )
        if amm_launch:
            self.events.append(
                LiquidityChange(
                    mint=mint,
                    t=t0,
                    wallet=creator,
                    sol_delta=pool.sol,
                    token_delta=pool.tokens,
                    sol_reserve=pool.sol,
                    token_reserve=pool.tokens,
                    slot=int(t0 / SLOT_SECONDS),
                )
            )
        self.archetypes[mint] = arch
        return tok

    def _run_token(self, tok: _Token) -> None:
        rng = self.rng
        t0 = tok.launch_t
        arch = tok.archetype
        dev_size = float(rng.uniform(0.5, 3.0) if arch == "rug" else rng.uniform(0.1, 1.0))
        self._swap(tok, t0 + 0.01, tok.creator, True, dev_size, *self._fee("crew"))
        # loud rugs bundle in the launch slots; stealth crews trickle in over the first minute
        stealth_entries = {w: t0 + float(rng.uniform(3, 60)) for w in tok.crew} if tok.stealth else {}
        for j, w in enumerate([] if tok.stealth else tok.crew):
            self._swap(tok, t0 + 0.05 + 0.4 * (j % 2), w, True, float(rng.uniform(2, 5)), *self._fee("crew"))
        for w in tok.friends:  # decoys: co-funded friends of an honest dev buying at launch
            self._swap(
                tok,
                t0 + 0.05 + float(rng.uniform(0, 0.8)),
                w,
                True,
                float(rng.uniform(0.5, 2)),
                *self._fee("crew"),
            )
        for w in rng.choice(self.snipers, size=int(rng.integers(2, 6)), replace=False):
            self._swap(
                tok,
                t0 + float(rng.uniform(0.1, 1.0)),
                str(w),
                True,
                float(rng.uniform(0.3, 1.5)),
                *self._fee("sniper"),
            )
        sniper_exit = {w: t0 + float(rng.uniform(8, 90)) for w in tok.holdings if w.startswith("sniper")}
        smart_in = (
            rng.random()
            < {
                "organic": 0.7,
                "graduate": 0.95,
                "rug": 0.1,
                "dud": 0.15,
                "wash": 0.05,
                "runner": 0.95,
                "trap": 0.1,
            }[arch]
        )
        # runner virality is heavy-tailed: most stall at tens of x, a few go four figures
        # (drawn only for runners so default simulations keep their random stream)
        viral = float(np.exp(rng.uniform(np.log(0.15), np.log(1.2)))) if arch == "runner" else 0.0
        dumped = False
        pulled = False
        end = t0 + tok.lifetime
        t = t0 + 1.0
        while t < end and not tok.dead:
            age = t - t0
            if arch == "organic":
                lam_b, lam_s = 0.9 * np.exp(-age / 900) + 0.08, 0.35 * np.exp(-age / 1500) + 0.1
            elif arch == "graduate":
                lam_b, lam_s = (2.2 if not tok.migrated else 0.6) * np.exp(-age / 3000) + 0.1, 0.5
            elif arch == "rug":
                lam_b = (1.4 if t < tok.dump_at else 0.05) * np.exp(-age / 600)
                lam_s = 0.2 if t < tok.dump_at else 1.2 * np.exp(-(t - tok.dump_at) / 60)
            elif arch == "dud":
                lam_b, lam_s = 0.12 * np.exp(-age / 400), 0.08
            elif arch == "trap":  # runner-like demand until the insiders dump
                lam_b = 2.2 if t < tok.dump_at else 0.05
                lam_s = 0.3 if t < tok.dump_at else 1.2 * np.exp(-(t - tok.dump_at) / 60)
            elif arch == "runner":  # viral: demand compounds for hours after graduation
                if not tok.migrated:
                    lam_b, lam_s = 2.5, 0.35
                else:
                    lam_b = 0.8 + 0.6 * viral * np.log1p(age / 600)
                    lam_s = 0.3 + 0.1 * np.log1p(age / 600)
            else:
                lam_b, lam_s = 0.25 * np.exp(-age / 600) + 0.02, 0.1
            for _ in range(int(rng.poisson(lam_b))):
                ts = t + float(rng.uniform(0, 1))
                if smart_in and rng.random() < (0.35 if age < 300 else 0.05):
                    wlt, kind = str(rng.choice(self.smart)), "smart"
                elif rng.random() < 0.08:
                    wlt, kind = str(rng.choice(self.bots)), "bot"
                else:
                    wlt, kind = str(rng.choice(self.retail)), "retail"
                size = float(rng.lognormal(np.log(0.6 if kind == "smart" else 0.25), 0.8))
                if kind == "retail" and age < 120 and rng.random() < 0.06:
                    size = float(rng.uniform(2, 6))  # early whales concentrate honest launches too
                if arch == "runner":
                    size *= min(
                        1.0 + viral * age / 900.0, 25.0 * viral + 1.0
                    )  # FOMO: tickets grow with the market cap
                self._swap(tok, ts, wlt, True, size, *self._fee(kind))
            for _ in range(int(rng.poisson(lam_s))):
                seller = self._seller(tok, exclude={tok.creator, *tok.crew})
                if seller is not None:
                    # runner holders take partial profits and keep a moon bag
                    frac = float(rng.uniform(0.1, 0.5) if arch == "runner" else rng.uniform(0.3, 1.0))
                    self._swap(
                        tok,
                        t + float(rng.uniform(0, 1)),
                        seller,
                        False,
                        tok.holdings[seller] * frac,
                        *self._fee("retail"),
                    )
            for w, at in list(stealth_entries.items()):
                if t >= at:
                    self._swap(tok, t + 0.5, w, True, float(rng.uniform(0.8, 2.5)), *self._fee("retail"))
                    del stealth_entries[w]
            for w, ex in list(sniper_exit.items()):
                if t >= ex:
                    self._swap(tok, t + 0.2, w, False, tok.holdings.get(w, 0.0), *self._fee("sniper"))
                    del sniper_exit[w]
            for w in [w for w in tok.holdings if w.startswith("smart") and tok.holdings[w] > 1]:
                gain = tok.pool.price / tok.entry_price[w] - 1
                if gain > 0.4 or gain < -0.25 or (arch == "rug" and t > tok.dump_at - 20):
                    self._swap(tok, t + 0.3, w, False, tok.holdings[w], *self._fee("smart"))
            if (arch == "wash" and rng.random() < 0.8) or (
                arch == "trap" and t < tok.dump_at and rng.random() < 0.6
            ):
                bot = str(rng.choice(self.bots[:4]))
                size = float(rng.uniform(0.5, 2.0))
                self._swap(tok, t + 0.1, bot, True, size, *self._fee("bot"))
                self._swap(tok, t + 0.5, bot, False, tok.holdings.get(bot, 0.0), *self._fee("bot"))
            if arch in ("rug", "trap") and not dumped and t >= tok.dump_at:
                dumped = True
                for j, w in enumerate([tok.creator, *tok.crew]):
                    self._swap(tok, t + 0.1 * j, w, False, tok.holdings.get(w, 0.0), *self._fee("crew"))
                if tok.venue == "raydium" and not pulled:
                    pulled = True
                    sol_out, tok_out = tok.pool.sol * 0.97, tok.pool.tokens * 0.97
                    tok.pool = Pool(tok.pool.sol - sol_out, tok.pool.tokens - tok_out, AMM_FEE)
                    tp = tok.stamp(t + 2.0)
                    self.events.append(
                        LiquidityChange(
                            mint=tok.mint,
                            t=tp,
                            wallet=tok.creator,
                            sol_delta=-sol_out,
                            token_delta=-tok_out,
                            sol_reserve=tok.pool.sol,
                            token_reserve=tok.pool.tokens,
                            slot=int(tp / SLOT_SECONDS),
                        )
                    )
            t += 1.0

    def run(self) -> tuple[EventStore, dict[str, str]]:
        """Simulate every launch; returns the event store and each mint's archetype.

        Call once per instance: generated events accumulate on the simulator.
        """
        self._fund_population()
        tokens = [self._launch(k) for k in range(self.spec.n_tokens)]
        for tok in tokens:
            self._run_token(tok)
        store = EventStore()
        store.extend(self.events)
        return store, dict(self.archetypes)


def simulate_launches(spec: LaunchSimSpec | None = None) -> tuple[EventStore, dict[str, str]]:
    """Returns the event history and the ground-truth archetype of every mint."""
    return LaunchSimulator(spec).run()
