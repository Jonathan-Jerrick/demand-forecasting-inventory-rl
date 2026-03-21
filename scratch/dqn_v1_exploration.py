"""
Initial DQN attempt — from scratch, single DC, linear costs.

Abandoned in March 2026. Notes on why it didn't work:

The agent learned to under-order. Total cost ended up ~27% WORSE than Newsvendor,
with fill rate around 0.90. The holding cost is smooth and daily; the stockout penalty
hits infrequently but large. With a 6-level discrete action space and ~400 episodes,
the DQN never saw enough stockout events to weigh them properly — it converged to
"order a bit less than Newsvendor" because that reduced the holding terms it saw every
step.

Root problem: the single-DC, lost-sales, linear-cost, daily-ordering environment is
near-optimally solved by the Newsvendor formula analytically. There's no structure for
RL to find that a closed-form solution doesn't already exploit. Running RL here is like
fitting a neural net to y = 2x — it works but you've learned nothing.

What I did instead: reworked the simulator (src/simulator.py) to add correlated demand
shocks across DCs, stochastic lead times, fixed ordering cost, and lateral transshipment.
Those features create a genuinely non-myopic, cross-DC problem that Newsvendor (which
is per-DC and myopic) can't solve optimally. Then switched to residual PPO where the
action is a correction on top of the Newsvendor level — so RL starts at the baseline
and only needs to learn when to deviate from it.
"""

import numpy as np

# --- environment (the old, too-simple version) ---

class SimpleDCEnv:
    """Single DC, lost-sales, linear holding/stockout, discrete action."""
    def __init__(self, demand_mean=100, lead_time=3, h=0.2, p=2.5, episode_len=90):
        self.mu = demand_mean
        self.L = lead_time
        self.h = h        # holding cost per unit per day
        self.p = p        # stockout penalty per unit
        self.T = episode_len

    def reset(self):
        self.inv = self.mu * (self.L + 1)   # start with ~lead-time cover
        self.pipe = []
        self.t = 0
        return self._obs()

    def step(self, action_idx):
        # action: 0=0, 1=0.5*mu, 2=1*mu, 3=1.5*mu, 4=2*mu, 5=2.5*mu
        order = action_idx * 0.5 * self.mu
        self.pipe.append((self.t + self.L, order))

        # receive arrivals
        arrived = sum(q for t, q in self.pipe if t <= self.t)
        self.pipe = [(t, q) for t, q in self.pipe if t > self.t]
        self.inv += arrived

        # realise demand
        d = max(np.random.poisson(self.mu), 0)
        fulfilled = min(self.inv, d)
        unmet = d - fulfilled
        self.inv = max(self.inv - d, 0)

        cost = self.h * self.inv + self.p * unmet
        self.t += 1
        done = self.t >= self.T
        return self._obs(), -cost, done, {"unmet": unmet, "inv": self.inv}

    def _obs(self):
        pipeline = sum(q for _, q in self.pipe)
        return np.array([self.inv / self.mu, pipeline / self.mu, self.t / self.T])


# --- DQN (minimal, the version that failed) ---
# Full training loop was in notebooks/05_rl_inventory_agent.ipynb (v1).
# WRMSSE against Newsvendor: +27% cost, fill 0.90. Not competitive.
# See src/simulator.py for the reworked environment that actually gives RL room.
