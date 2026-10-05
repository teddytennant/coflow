"""NeuralUCB over a discrete step-budget set.

Schedule, matching the experiment text rather than the pseudocode's single
epsilon-greedy block:

1. Cold start: pull each arm once, in the given order.
2. A short epsilon-greedy phase (default: one extra pass over the arms).
3. NeuralUCB: alpha = argmax_a [ f(phi, a) + beta_t * sqrt(g^T V^{-1} g) ],
   with g = grad_theta f at (phi, a).

V_0 = lambda * I with lambda = 1. After the chosen arm, V <- V + g g^T.
The inverse is maintained by Sherman-Morrison, which the paper cites as the
rank-one update.

Every P observations the replay buffer is fit by full-batch MSE with Adam
at learning rate 5e-3 for `epochs` epochs (paper default 20). Adam moments
are reset at each fit; parameters are not. The buffer keeps the full history
by default. The paper text says the buffer size is P, while the algorithm
minimizes over all observed data.

beta_t = beta0 * sqrt(log(t + 1)), beta0 = 0.5 by default. The paper cites
Zhou et al. and does not give a numeric schedule. log is natural log.
t is the 1-indexed decision step.
"""

from __future__ import annotations

import math

import numpy as np
import jax
import jax.numpy as jnp

from coflow.model import (
    adam_epoch,
    init_params,
    model_input,
    parameter_count,
    predict_and_grads,
    predict_batch,
)

DEFAULT_ARMS = (10, 15, 20, 25, 30, 35, 40, 45, 50)


def _normalize_arms(arms):
    if arms is None:
        arms = DEFAULT_ARMS
    out = []
    for raw in arms:
        arm = int(raw)
        if arm < 2:
            raise ValueError("each arm must be an integer >= 2")
        if arm not in out:
            out.append(arm)
    if not out:
        raise ValueError("arms must be non-empty")
    return tuple(out)


def _bucket(n):
    for size in (16, 32, 64, 128, 256, 512, 1024):
        if n <= size:
            return size
    return int(2 ** math.ceil(math.log2(n)))


class NeuralUCB:
    def __init__(
        self,
        arms=None,
        context_dim=40,
        hidden=64,
        lambda_reg=1.0,
        lr=5e-3,
        epochs=20,
        update_interval=10,
        epsilon=0.1,
        epsilon_steps=None,
        beta0=0.5,
        seed=0,
        buffer_size=None,
    ):
        if lambda_reg <= 0:
            raise ValueError("lambda_reg must be positive")
        if update_interval < 1:
            raise ValueError("update_interval must be >= 1")
        self.arms = _normalize_arms(arms)
        self.max_arm = max(self.arms)
        self.context_dim = int(context_dim)
        self.hidden = int(hidden)
        self.lambda_reg = float(lambda_reg)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.update_interval = int(update_interval)
        self.epsilon = float(epsilon)
        self.epsilon_steps = (
            len(self.arms) if epsilon_steps is None else int(epsilon_steps)
        )
        self.beta0 = float(beta0)
        self.buffer_size = None if buffer_size is None else int(buffer_size)
        self.seed = int(seed)
        self.params = init_params(self.seed, self.context_dim + 1, self.hidden)
        self.rng = np.random.default_rng(self.seed)
        n = parameter_count(self.context_dim, self.hidden)
        self.V = np.eye(n, dtype=np.float64) * self.lambda_reg
        self.V_inv = np.eye(n, dtype=np.float64) / self.lambda_reg
        self.t = 0
        self.buffer_phi: list[np.ndarray] = []
        self.buffer_alpha: list[int] = []
        self.buffer_reward: list[float] = []
        self.history: list[dict] = []
        self.n_fits = 0
        self.last_loss = None
        self._pending = None

    def beta(self, t):
        """beta_t = beta0 * sqrt(log(t + 1)). t is 1-indexed."""
        return self.beta0 * math.sqrt(math.log(int(t) + 1))

    def _as_phi(self, phi):
        arr = np.asarray(phi, dtype=np.float32).reshape(-1)
        if arr.shape != (self.context_dim,):
            raise ValueError(
                f"phi has shape {arr.shape}, expected ({self.context_dim},)"
            )
        return arr

    def _inputs(self, phi):
        phi = self._as_phi(phi)
        rows = [model_input(phi, arm, self.max_arm) for arm in self.arms]
        return np.stack(rows, axis=0)

    def predict(self, phi, alpha):
        x = model_input(self._as_phi(phi), alpha, self.max_arm)
        value = predict_batch(self.params, jnp.asarray(x)[None, :])
        return float(value[0])

    def predict_arms(self, phi):
        xs = jnp.asarray(self._inputs(phi))
        values = predict_batch(self.params, xs)
        return np.asarray(values, dtype=np.float64)

    def grad_feature(self, phi, alpha):
        """Flattened g = grad_theta f(phi, alpha) at the current parameters."""
        x = model_input(self._as_phi(phi), alpha, self.max_arm)
        _, grads = predict_and_grads(self.params, jnp.asarray(x)[None, :])
        return np.asarray(grads[0], dtype=np.float64)

    def arm_stats(self, phi, t=None):
        """Predictions, gradient features, and UCB scores for every arm.

        t defaults to the upcoming decision index (self.t + 1).
        """
        if t is None:
            t = self.t + 1
        xs = jnp.asarray(self._inputs(phi))
        preds, grads = predict_and_grads(self.params, xs)
        preds = np.asarray(preds, dtype=np.float64)
        grads = np.asarray(grads, dtype=np.float64)
        # g^T V^{-1} g for each arm.
        quad = np.einsum("ij,jk,ik->i", grads, self.V_inv, grads)
        sigma = np.sqrt(np.maximum(quad, 0.0))
        bonus = self.beta(t) * sigma
        return {
            "pred": preds,
            "grad": grads,
            "sigma": sigma,
            "ucb": preds + bonus,
            "t": int(t),
        }

    def greedy_action(self, phi):
        """argmax_a f(phi, a). Ties keep the earliest arm in self.arms."""
        scores = self.predict_arms(phi)
        return self.arms[int(np.argmax(scores))]

    def ucb_action(self, phi, t=None):
        stats = self.arm_stats(phi, t=t)
        return self.arms[int(np.argmax(stats["ucb"]))], stats

    def select(self, phi):
        """Choose an arm. Pair with observe. Does not increment t."""
        phi = self._as_phi(phi)
        step = self.t + 1
        n = len(self.arms)
        stats = None
        if step <= n:
            alpha = self.arms[step - 1]
            mode = "cold"
        elif step <= n + self.epsilon_steps:
            explore = self.epsilon > 0.0 and float(self.rng.random()) < self.epsilon
            if explore:
                alpha = self.arms[int(self.rng.integers(0, n))]
                mode = "epsilon"
            else:
                alpha = self.greedy_action(phi)
                mode = "greedy"
        else:
            alpha, stats = self.ucb_action(phi, t=step)
            mode = "ucb"
        if stats is None:
            g = self.grad_feature(phi, alpha)
        else:
            g = stats["grad"][self.arms.index(alpha)]
        self._pending = (phi, int(alpha), g, mode, step)
        return int(alpha)

    def _update_design(self, g):
        g = np.asarray(g, dtype=np.float64).reshape(-1)
        self.V = self.V + np.outer(g, g)
        solved = self.V_inv @ g
        denom = 1.0 + float(g @ solved)
        self.V_inv = self.V_inv - np.outer(solved, solved) / denom

    def _trim_buffer(self):
        if self.buffer_size is None:
            return
        extra = len(self.buffer_reward) - self.buffer_size
        if extra > 0:
            del self.buffer_phi[:extra]
            del self.buffer_alpha[:extra]
            del self.buffer_reward[:extra]

    def observe(self, phi, alpha, reward):
        """Store the reward, update V, and fit every P steps."""
        phi = self._as_phi(phi)
        alpha = int(alpha)
        reward = float(reward)
        pending = self._pending
        if (
            pending is not None
            and pending[1] == alpha
            and np.array_equal(pending[0], phi)
        ):
            g = pending[2]
            mode = pending[3]
        else:
            g = self.grad_feature(phi, alpha)
            mode = "observe"
        self._pending = None
        self._update_design(g)
        self.buffer_phi.append(phi.copy())
        self.buffer_alpha.append(alpha)
        self.buffer_reward.append(reward)
        self._trim_buffer()
        self.t += 1
        self.history.append(
            {"t": self.t, "alpha": alpha, "reward": reward, "mode": mode}
        )
        if self.t % self.update_interval == 0:
            self.fit()

    def step(self, phi, reward_fn):
        """Select an arm, evaluate reward_fn(alpha), and observe."""
        alpha = self.select(phi)
        reward = float(reward_fn(alpha))
        self.observe(phi, alpha, reward)
        return alpha, reward

    def fit(self):
        """MSE fit of the reward model on the replay buffer for `epochs` epochs."""
        n = len(self.buffer_reward)
        if n == 0 or self.epochs < 1:
            return None
        cap = _bucket(n)
        xs = np.zeros((cap, self.context_dim + 1), dtype=np.float32)
        ys = np.zeros(cap, dtype=np.float32)
        mask = np.zeros(cap, dtype=np.float32)
        for i, (phi, alpha, reward) in enumerate(
            zip(self.buffer_phi, self.buffer_alpha, self.buffer_reward)
        ):
            xs[i] = model_input(phi, alpha, self.max_arm)
            ys[i] = reward
            mask[i] = 1.0
        params = self.params
        moment = jax.tree_util.tree_map(jnp.zeros_like, params)
        second = jax.tree_util.tree_map(jnp.zeros_like, params)
        step = jnp.array(1, dtype=jnp.int32)
        xs_j = jnp.asarray(xs)
        ys_j = jnp.asarray(ys)
        mask_j = jnp.asarray(mask)
        lr = jnp.asarray(self.lr, dtype=jnp.float32)
        last = None
        for _ in range(self.epochs):
            params, moment, second, last = adam_epoch(
                params, moment, second, step, xs_j, ys_j, mask_j, lr
            )
            step = step + 1
        self.params = params
        self.last_loss = float(last)
        self.n_fits += 1
        return self.last_loss
