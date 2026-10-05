"""Uniform forward Euler on the unit interval.

Budget alpha means alpha steps. With t_k = k / alpha,

    x_{k+1} = x_k + (1 / alpha) * v(x_k, t_k),  k = 0, ..., alpha - 1.

The paper's discretization theorem says that if the velocity is Lipschitz
and the second derivative of the flow is bounded, the global error of this
scheme is O(1/K). For the linear field dx/dt = -x the exact endpoint is
x0 * exp(-1), and K * error(K) stays within a small constant factor as K
grows, which is the discrete signature of that O(1/K) bound.
"""

from __future__ import annotations

import numpy as np


def rollout(velocity, x0, n_steps):
    """Integrate and record states, velocities, and times.

    Returns xs, vs, ts with length n_steps + 1. vs[k] = v(xs[k], ts[k]),
    including a terminal evaluation at t = 1 that is not used to advance
    the state. That terminal sample is required by the variation sum, which
    runs over velocities v_1, ..., v_alpha.
    """
    n_steps = int(n_steps)
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    x = np.asarray(x0, dtype=np.float64).copy()
    xs = np.empty((n_steps + 1,) + x.shape, dtype=np.float64)
    vs = np.empty_like(xs)
    ts = np.empty(n_steps + 1, dtype=np.float64)
    xs[0] = x
    step = 1.0 / n_steps
    for k in range(n_steps):
        t = k / n_steps
        ts[k] = t
        v = np.asarray(velocity(x, t), dtype=np.float64)
        if v.shape != x.shape:
            raise ValueError(
                f"velocity shape {v.shape} does not match state shape {x.shape}"
            )
        vs[k] = v
        x = x + step * v
        xs[k + 1] = x
    ts[n_steps] = 1.0
    vs[n_steps] = np.asarray(velocity(x, 1.0), dtype=np.float64)
    return xs, vs, ts


def forward_euler(velocity, x0, n_steps):
    """Return x_alpha after n_steps uniform forward Euler steps on [0, 1]."""
    xs, _, _ = rollout(velocity, x0, n_steps)
    return xs[-1]
