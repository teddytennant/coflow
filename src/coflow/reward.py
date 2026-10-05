"""Trajectory variation and the step-budget reward.

vhat(x, alpha) = (1 / (alpha - 1)) * sum_{k=1}^{alpha-1}
    ||v(x_{t_{k+1}}, t_{k+1}) - v(x_{t_k}, t_k)||_1
    / max(||v(x_{t_k}, t_k)||_1, eps)

v_k is the velocity at state x_k and time t_k = k / alpha. The sum starts
at k = 1, so the initial velocity v_0 is not included. The default norm is
L1 and the denominator is max(||v_k||, eps), matching the paper. `ord` only
changes the norm; the max-eps denominator stays.

r(x, alpha) = max(0, (1 - vhat(x, alpha)) - mu * alpha), with mu = 1 / max(A).
eps = 1e-8.
"""

from __future__ import annotations

import numpy as np

VARIATION_EPS = 1e-8


def trajectory_variation(velocities, eps=VARIATION_EPS, ord=1):
    """Mean relative velocity change along a recorded trajectory.

    velocities has shape (alpha + 1, dim), indexed by k = 0, ..., alpha.
    The default path is the paper's L1 ratio with denominator max(||v||_1, eps).
    """
    vs = np.asarray(velocities, dtype=np.float64)
    if vs.ndim == 1:
        vs = vs.reshape(-1, 1)
    alpha = vs.shape[0] - 1
    if alpha < 2:
        return 0.0
    diff = np.linalg.norm(vs[2:] - vs[1:-1], axis=-1, ord=ord)
    base = np.maximum(np.linalg.norm(vs[1:-1], axis=-1, ord=ord), float(eps))
    return float(np.mean(diff / base))


def cost_weight(arms):
    """mu = 1 / max(A)."""
    return 1.0 / float(max(arms))


def step_reward(variation, alpha, mu):
    """r = max(0, (1 - vhat) - mu * alpha)."""
    raw = (1.0 - float(variation)) - float(mu) * float(alpha)
    return float(max(0.0, raw))
