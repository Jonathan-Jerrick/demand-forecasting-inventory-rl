"""Inventory digital twin -- an M5-grounded multi-echelon replenishment simulator.

Why this exists
---------------
The earlier version of this project trained an RL agent on a synthetic Gaussian
demand process whose only link to the data was that its *mean* was scaled to the
M5 average. The "forecast" handed to the agent was fabricated inside the env. As a
result the agent never saw real demand or a real forecast, and any reported result
was an artefact of the toy generator.

This simulator fixes that. It is a digital twin in the literal sense: a virtual
replica of a small distribution network whose demand is the *actual* historical
M5 series being replayed, and whose state exposes the *actual* probabilistic
forecast produced by the forecast layer (notebook 02 / 02b). The agent therefore
learns to act on the same information a real planner would have: on-hand stock,
in-transit pipeline, and a quantile forecast of demand over the lead time.

Regime rework (T4)
------------------
A single-DC, lost-sales problem with linear holding/stockout cost and daily
ordering is solved near-optimally by a base-stock / newsvendor level, so an RL
agent has no room to win. To create genuine, non-myopic, cross-DC structure the
twin adds four levers (all defaulted ON, set to zero to recover the old regime):

1. **Correlated demand shocks.** Real base demand is multiplied by a stochastic
   log-normal shock that shares a common component across DCs (``shock_corr``).
   Aggregate variance can now be reduced by **risk-pooling / transshipment** --
   something a per-DC newsvendor cannot do.
2. **Stochastic, occasionally-long, correlated lead times.** Orders arrive after a
   random lead time whose noise is partly shared across DCs, so the protection
   window itself is uncertain.
3. **Fixed ordering cost + minimum / batch order quantity.** A non-trivial setup
   cost and an order multiple make "order to a level every day" suboptimal; the
   optimal policy becomes an (s, S) / non-myopic one.
4. **Non-linear (convex) holding.** Inventory above a soft capacity threshold is
   charged at a higher rate (overflow / spoilage), punishing over-ordering
   super-linearly.
5. **Emergency lateral transshipment (risk-pooling).** After demand is realised, a
   DC that would stock out pulls from sister DCs' leftover stock at a small
   transshipment cost instead of losing the sale. This makes the multi-DC network
   genuinely valuable: the cost-minimising strategy is to hold *less* safety stock
   per DC and let the pool cover shortfalls — something a per-DC heuristic cannot do,
   and the source of the learned agent's edge (see notebook 05's ablations).

The stochastic shocks and lead times are **pre-drawn at ``reset``**, so an episode
is reproducible from its seed and a *clairvoyant* oracle (``oracle_policy``) can
peek at realised demand to give a lower-bound cost -- the gap to that bound is the
room a learned policy can capture.

Gym-style API (reset / step). Pure numpy -- no GPU, no gym dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CostParams:
    holding: float = 0.20        # $ per unit per day held at end of day
    stockout: float = 2.50       # $ per unit of unmet demand (lost-sales penalty)
    order_fixed: float = 20.0    # $ per order placed (per node), if qty > 0
    order_var: float = 0.10      # $ per unit ordered
    transship: float = 0.10      # $ per unit moved between nodes (cheap vs a lost sale)
    holding_overflow: float = 1.20   # extra $/unit/day above the soft-capacity threshold


@dataclass
class TwinConfig:
    nodes: tuple = ("CA", "TX")
    capacity: float = 4000.0
    init_inventory: float = 1500.0
    lead_time_mean: int = 4
    lead_time_std: float = 0.6       # mild lead-time noise (kept low so the oracle is feasible)
    lead_time_corr: float = 0.5      # cross-node correlation of lead-time noise
    episode_len: int = 90
    review_window: int = 5           # days of forecast the state summarises (lead + 1)
    lost_sales: bool = True          # if False, unmet demand backorders
    allow_transship: bool = True
    service_target: dict = field(default_factory=lambda: {"CA": 0.95, "TX": 0.95})
    # --- T4 stochastic-regime levers (set to 0 to recover the deterministic twin) ---
    demand_shock_std: float = 0.35   # log-normal demand shock magnitude (0 = off)
    shock_corr: float = 0.25         # shared shock fraction (low -> risk-pooling pays)
    order_batch_frac: float = 0.5    # orders rounded up to this multiple of mean daily demand
    order_min_frac: float = 0.0      # minimum non-zero order, as a fraction of mean daily demand
    overflow_frac: float = 0.55      # soft holding threshold as a fraction of capacity
    costs: CostParams = field(default_factory=CostParams)


class InventoryDigitalTwin:
    """Multi-echelon inventory simulator driven by real M5 demand + real forecasts.

    Parameters
    ----------
    demand : dict[str, np.ndarray]
        node -> 1-D array of real daily demand (length >= total horizon).
    forecast_q : dict[str, np.ndarray] | None
        node -> array of shape (T, n_quantiles) giving the forecast distribution
        per day, aligned index-for-index with `demand`. If None the twin falls
        back to a trailing-mean forecast so it can run standalone.
    quantile_levels : list[float]
        The quantile levels of `forecast_q`'s columns (e.g. [0.5, 0.9]). The state
        uses the median and the highest provided quantile as a safety signal.
    config : TwinConfig
    """

    def __init__(self, demand, forecast_q=None, quantile_levels=(0.5, 0.9),
                 config: TwinConfig | None = None, seed: int = 0):
        self.cfg = config or TwinConfig()
        self.nodes = list(self.cfg.nodes)
        self.demand = {n: np.asarray(demand[n], dtype=float) for n in self.nodes}
        self.T = min(len(self.demand[n]) for n in self.nodes)
        self.forecast_q = forecast_q
        self.q_levels = list(quantile_levels)
        self._q_med = int(np.argmin(np.abs(np.array(self.q_levels) - 0.5)))
        self._q_hi = int(np.argmax(self.q_levels))
        # demand scale used to normalise observations + batch sizes (per node)
        self.scale = {n: max(self.demand[n].mean(), 1.0) for n in self.nodes}
        self.rng = np.random.default_rng(seed)
        # per-node observation features:
        #   inv, pipeline, last_demand, fc_median_window, fc_hi_window, service_target
        self.node_obs_dim = 6
        self.obs_dim = self.node_obs_dim * len(self.nodes) + 2  # + dow sin/cos
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self, start: int | None = None, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        max_start = max(self.T - self.cfg.episode_len - 1, 1)
        self.start = self.rng.integers(0, max_start) if start is None else start
        self.t = 0
        self.inv = {n: float(self.cfg.init_inventory) for n in self.nodes}
        # pipeline holds [remaining_lead_time, qty]; per-node list
        self.pipe = {n: [] for n in self.nodes}
        self._predraw()
        return self._obs()

    def _predraw(self):
        """Pre-draw the episode's stochastic shocks and lead times so the episode
        is reproducible and a clairvoyant oracle can peek at realised demand."""
        cfg = self.cfg
        H = cfg.episode_len + cfg.lead_time_mean + cfg.review_window + 5  # buffer
        # correlated multiplicative demand shock (log-normal, mean ~1)
        s = cfg.demand_shock_std
        if s > 0:
            rho = float(np.clip(cfg.shock_corr, 0.0, 1.0))
            common = self.rng.normal(0, s * np.sqrt(rho), size=H)
            self.shock = {}
            for n in self.nodes:
                idio = self.rng.normal(0, s * np.sqrt(1 - rho), size=H)
                self.shock[n] = np.exp(common + idio - 0.5 * s ** 2)  # E[shock]~1
        else:
            self.shock = {n: np.ones(H) for n in self.nodes}
        # correlated lead-time draws, indexed by the day an order is placed
        lt_common = self.rng.normal(0, 1, size=H)
        self.lt_seq = {}
        for n in self.nodes:
            lt_idio = self.rng.normal(0, 1, size=H)
            z = (np.sqrt(cfg.lead_time_corr) * lt_common
                 + np.sqrt(1 - cfg.lead_time_corr) * lt_idio)
            lt = np.round(cfg.lead_time_mean + cfg.lead_time_std * z)
            self.lt_seq[n] = np.clip(lt, 1, None).astype(int)

    def realized_demand(self, node, day):
        """Realised demand at episode-day offset (real base * pre-drawn shock)."""
        idx = self.start + day
        idx = min(idx, len(self.demand[node]) - 1)
        return float(self.demand[node][idx] * self.shock[node][day])

    # ------------------------------------------------------------------ #
    def _forecast_window(self, node, day):
        """Median and high-quantile forecast summed over the review window."""
        w = self.cfg.review_window
        if self.forecast_q is not None:
            fc = self.forecast_q[node]
            lo = min(day, len(fc) - 1)
            hi = min(day + w, len(fc))
            seg = fc[lo:hi]
            med = seg[:, self._q_med].sum()
            up = seg[:, self._q_hi].sum()
            return med, up
        # fallback: trailing 28-day mean * window, with a +30% safety band
        lo = max(self.start + day - 28, 0)
        recent = self.demand[node][lo:self.start + day]
        mu = recent.mean() if len(recent) else self.scale[node]
        return mu * w, mu * w * 1.3

    def _obs(self):
        day = self.t
        idx = self.start + day
        feats = []
        for n in self.nodes:
            pipe_qty = sum(q for _, q in self.pipe[n])
            last_d = self.realized_demand(n, day - 1) if day > 0 else self.demand[n][max(idx - 1, 0)]
            med, up = self._forecast_window(n, idx)
            feats += [
                self.inv[n] / self.cfg.capacity,
                pipe_qty / self.cfg.capacity,
                last_d / self.scale[n],
                med / (self.scale[n] * self.cfg.review_window),
                up / (self.scale[n] * self.cfg.review_window),
                self.cfg.service_target.get(n, 0.95),
            ]
        dow = idx % 7
        feats += [np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)]
        return np.asarray(feats, dtype=np.float32)

    # ------------------------------------------------------------------ #
    def _batch(self, node, q):
        """Apply minimum and batch-multiple constraints to a raw order quantity."""
        cfg = self.cfg
        if q <= 0:
            return 0.0
        q = max(q, cfg.order_min_frac * self.scale[node])
        mult = cfg.order_batch_frac * self.scale[node]
        if mult > 0:
            q = np.ceil(q / mult) * mult
        return float(q)

    def _apply_transship(self, transship):
        """Redistribute on-hand stock between DCs and return units moved.

        `transship` may be:
          * a scalar  -> legacy 2-node flow (positive = nodes[0]->nodes[1]); or
          * an array/list of length n_nodes -> desired NET flow per node (donors
            negative, receivers positive). Stock is conserved: donors give up to
            their on-hand, receivers take proportionally, and the cost is charged on
            the units actually moved. This is what lets the agent risk-pool across DCs.
        """
        cfg, c = self.cfg, self.cfg.costs
        if not (cfg.allow_transship and len(self.nodes) >= 2):
            return 0.0
        if np.isscalar(transship):
            if abs(transship) < 1e-9:
                return 0.0
            a, b = self.nodes[0], self.nodes[1]
            if transship > 0:
                qty = min(transship, max(self.inv[a], 0))
                self.inv[a] -= qty; self.inv[b] = min(self.inv[b] + qty, cfg.capacity)
            else:
                qty = min(-transship, max(self.inv[b], 0))
                self.inv[b] -= qty; self.inv[a] = min(self.inv[a] + qty, cfg.capacity)
            return qty
        net = np.asarray(transship, dtype=float)
        out = np.maximum(-net, 0.0)
        out = np.array([min(out[i], max(self.inv[n], 0)) for i, n in enumerate(self.nodes)])
        inflow = np.maximum(net, 0.0)
        moved = min(out.sum(), inflow.sum())
        if moved < 1e-9:
            return 0.0
        give = out * (moved / out.sum()) if out.sum() > 0 else out * 0
        take = inflow * (moved / inflow.sum()) if inflow.sum() > 0 else inflow * 0
        for i, n in enumerate(self.nodes):
            self.inv[n] = min(max(self.inv[n] - give[i] + take[i], 0.0), cfg.capacity)
        return float(moved)

    def step(self, orders, transship=0.0):
        """Advance one day.

        orders : dict[node]->qty  OR  array aligned to self.nodes.
        transship : scalar (legacy 2-node) OR array of length n_nodes giving the
                    desired net stock redistribution across DCs. Ignored if disabled.
        """
        cfg, c = self.cfg, self.cfg.costs
        if not isinstance(orders, dict):
            orders = {n: float(orders[i]) for i, n in enumerate(self.nodes)}

        day_holding = day_stockout = day_order = day_trans = 0.0
        info_demand, info_so, info_inv, info_ful = {}, {}, {}, {}

        # 1. place orders -> schedule arrival after the pre-drawn lead time for today
        for n in self.nodes:
            q = self._batch(n, max(float(orders.get(n, 0.0)), 0.0))
            if q > 0:
                lt = int(self.lt_seq[n][self.t])
                self.pipe[n].append([lt, q])
                day_order += c.order_fixed + c.order_var * q

        # 2. receive arrivals due today; decrement remaining lead times
        for n in self.nodes:
            arrived = 0.0
            still = []
            for lt, q in self.pipe[n]:
                lt -= 1
                if lt <= 0:
                    arrived += q
                else:
                    still.append([lt, q])
            self.pipe[n] = still
            self.inv[n] = min(self.inv[n] + arrived, cfg.capacity)

        # 3. optional transshipment / risk-pooling between DCs (scalar or N-node vector)
        moved = self._apply_transship(transship)
        day_trans += c.transship * moved

        # 4. realise REAL (shock-scaled) demand for this day and fulfil from own stock
        unmet = {}
        for n in self.nodes:
            d = self.realized_demand(n, self.t)
            ful = min(d, self.inv[n])
            self.inv[n] -= ful
            unmet[n] = d - ful
            info_demand[n], info_ful[n] = d, ful

        # 4b. EMERGENCY lateral transshipment (risk-pooling): cover a DC's unmet demand
        # from sister DCs' leftover stock at transshipment cost — far cheaper than a
        # lost sale, and the core reason a multi-DC network beats independent DCs. This
        # is where pooling actually pays: a per-DC heuristic cannot use it.
        if cfg.allow_transship and len(self.nodes) >= 2:
            for n in self.nodes:
                if unmet[n] <= 1e-9:
                    continue
                for m in self.nodes:
                    if m == n or self.inv[m] <= 1e-9:
                        continue
                    move = min(unmet[n], self.inv[m])
                    self.inv[m] -= move; unmet[n] -= move
                    day_trans += c.transship * move
                    if unmet[n] <= 1e-9:
                        break

        # 4c. remaining unmet demand is lost (or backordered)
        for n in self.nodes:
            so = unmet[n]
            if not cfg.lost_sales:
                self.inv[n] -= so
            day_stockout += so * c.stockout
            info_so[n] = so

        # 5. end-of-day holding cost (convex: overflow above the soft threshold)
        thresh = cfg.overflow_frac * cfg.capacity
        for n in self.nodes:
            inv = max(self.inv[n], 0)
            day_holding += inv * c.holding + max(inv - thresh, 0) * c.holding_overflow
            info_inv[n] = self.inv[n]

        cost = day_holding + day_stockout + day_order + day_trans
        self.t += 1
        done = self.t >= cfg.episode_len
        info = {
            "cost": cost,
            "holding": day_holding, "stockout": day_stockout,
            "ordering": day_order, "transship": day_trans,
            "demand": info_demand, "unmet": info_so,
            "fulfilled": info_ful, "inventory": info_inv,
        }
        return self._obs(), -cost, done, info

    # ------------------------------------------------------------------ #
    def run_policy(self, policy_fn, n_episodes=30, seed_base=1000):
        """Evaluate a policy_fn(obs, twin)->(orders, transship) over fixed seeds."""
        rows = []
        for ep in range(n_episodes):
            obs = self.reset(seed=seed_base + ep)
            tot = dict(cost=0.0, holding=0.0, stockout=0.0, ordering=0.0,
                       transship=0.0, demand=0.0, fulfilled=0.0, inv=0.0)
            for _ in range(self.cfg.episode_len):
                action = policy_fn(obs, self)
                orders, trans = action if isinstance(action, tuple) else (action, 0.0)
                obs, r, done, info = self.step(orders, trans)
                tot["cost"] += info["cost"]
                for k in ("holding", "stockout", "ordering", "transship"):
                    tot[k] += info[k]
                tot["demand"] += sum(info["demand"].values())
                tot["fulfilled"] += sum(info["fulfilled"].values())
                tot["inv"] += sum(max(v, 0) for v in info["inventory"].values())
                if done:
                    break
            rows.append({
                "cost": tot["cost"],
                "holding": tot["holding"], "stockout": tot["stockout"],
                "ordering": tot["ordering"], "transship": tot["transship"],
                "fill_rate": tot["fulfilled"] / max(tot["demand"], 1),
                "avg_inventory": tot["inv"] / (self.cfg.episode_len * len(self.nodes)),
            })
        agg = {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}
        agg["cost_std"] = float(np.std([row["cost"] for row in rows]))
        agg["fill_std"] = float(np.std([row["fill_rate"] for row in rows]))
        return agg
