"""Classical replenishment policies + forecast-to-order translation.

Every policy is a callable ``policy(obs, twin) -> (orders_dict, transship)`` so it
plugs straight into ``InventoryDigitalTwin.run_policy`` and into the RL benchmark
table. Policies read the *real* forecast window from the twin (the same signal the
RL agent sees), so the comparison is apples-to-apples.

The module also exposes the translation that turns a probabilistic forecast into a
concrete order quantity -- this is what makes a model score mean something in
operational terms: a quantile of lead-time demand becomes an order-up-to level,
and the gap to current inventory position becomes "order N units today".
"""

from __future__ import annotations

import numpy as np


def _norm_ppf(p):
    """Inverse standard-normal CDF (Acklam's rational approximation).

    Avoids a scipy dependency for the single newsvendor z-score we need.
    Accurate to ~1e-9 over the open interval (0, 1).
    """
    p = float(np.clip(p, 1e-9, 1 - 1e-9))
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def inventory_position(twin, node):
    """On-hand + in-transit -- the quantity a planner reasons about."""
    pipe = sum(q for _, q in twin.pipe[node])
    return twin.inv[node] + pipe


def order_up_to_from_quantile(forecast_hi, position, target_S=None):
    """Bring inventory position up to an order-up-to level.

    forecast_hi : units of demand expected over the protection window at the
                  chosen service quantile (e.g. the p90 forecast). This is the
                  order-up-to level S unless `target_S` is given explicitly.
    Returns the non-negative order quantity.
    """
    S = target_S if target_S is not None else forecast_hi
    return max(S - position, 0.0)


# --------------------------------------------------------------------------- #
# Baseline policies
# --------------------------------------------------------------------------- #
def random_policy(obs, twin):
    """Order a random multiple of mean daily demand (a naive, demand-scaled baseline)."""
    rng = twin.rng
    return {n: rng.choice([0.0, 0.5, 1.0, 1.5, 2.0]) * twin.scale[n]
            for n in twin.nodes}, 0.0


def fixed_order_policy(obs, twin):
    """Replenish roughly what is sold each day: order ~ mean daily demand per node."""
    return {n: twin.scale[n] for n in twin.nodes}, 0.0


def min_max_policy(obs, twin):
    """(s, S) expressed in demand-days: reorder below lead-time cover, top up to a
    lead+review cover plus a flat safety margin."""
    lead = twin.cfg.lead_time_mean
    review = twin.cfg.review_window
    orders = {}
    for n in twin.nodes:
        mu = twin.scale[n]
        s = (lead + 1) * mu
        S = (lead + review + 2) * mu
        pos = inventory_position(twin, n)
        orders[n] = max(S - pos, 0.0) if pos < s else 0.0
    return orders, 0.0


def newsvendor_policy(obs, twin):
    """Critical-ratio newsvendor on lead-time demand using the forecast median.

    Service-level z comes from the cost critical ratio cr = Cu/(Cu+Co), where
    Cu is the per-unit stockout cost and Co the per-unit holding cost over the
    protection window. Demand sigma is approximated from the median/high-quantile
    spread the forecaster provides.
    """
    c = twin.cfg.costs
    cu, co = c.stockout, c.holding * (twin.cfg.lead_time_mean + 1)
    cr = cu / (cu + co)
    z = _norm_ppf(np.clip(cr, 0.5, 0.999))
    orders = {}
    for n in twin.nodes:
        med, hi = twin._forecast_window(n, twin.start + twin.t)
        sigma = max((hi - med), 1.0)  # high-quantile gap as a spread proxy
        S = med + z * sigma
        orders[n] = order_up_to_from_quantile(None, inventory_position(twin, n), target_S=S)
    return orders, 0.0


def base_stock_forecast_policy(obs, twin):
    """Order-up-to the high-quantile forecast of protection-window demand.

    This is the policy that most directly *uses* the probabilistic forecast: the
    order-up-to level is the p90 (or whatever top quantile the forecaster emits)
    of demand over the review window, so safety stock scales with predicted
    uncertainty rather than a flat rule.
    """
    orders = {}
    for n in twin.nodes:
        _, hi = twin._forecast_window(n, twin.start + twin.t)
        orders[n] = order_up_to_from_quantile(hi, inventory_position(twin, n))
    return orders, 0.0


def oracle_policy(obs, twin):
    """Clairvoyant lower bound: see the episode's realised (shock-scaled) demand and
    order *exactly* enough to cover the protection window, with same-day transshipment
    to rebalance the two DCs.

    This is not achievable in practice (it peeks at the future), but the gap between
    the best deployable heuristic and this oracle quantifies how much room a learned
    policy could ever capture -- the headline T4 diagnostic.
    """
    # cover realised demand over a window that safely spans the (low-variance) lead
    # time, so the clairvoyant never stocks out; it simply avoids the shock-safety
    # buffer a forecast-based policy must carry.
    protect = twin.cfg.lead_time_mean + 1
    orders = {}
    for n in twin.nodes:
        future = sum(twin.realized_demand(n, twin.t + k) for k in range(protect))
        orders[n] = max(future - inventory_position(twin, n), 0.0)
    # rebalance today's on-hand toward today's realised demand (N-node risk-pooling):
    # the net flow per DC = its shortfall (positive) / surplus (negative) vs realised demand.
    trans = 0.0
    if twin.cfg.allow_transship and len(twin.nodes) >= 2:
        trans = np.array([twin.realized_demand(n, twin.t) - twin.inv[n] for n in twin.nodes])
    return orders, trans


CLASSICAL_POLICIES = {
    "Random": random_policy,
    "Fixed-Order": fixed_order_policy,
    "Min/Max": min_max_policy,
    "Newsvendor": newsvendor_policy,
    "Base-Stock(forecast)": base_stock_forecast_policy,
}
