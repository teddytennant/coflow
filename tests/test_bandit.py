"""NeuralUCB cold start, design matrix, beta, and periodic MSE fits."""

import numpy as np

from coflow.bandit import NeuralUCB


def _tiny(**kwargs):
    defaults = dict(
        arms=[4, 6, 8],
        context_dim=3,
        hidden=4,
        seed=0,
        epochs=2,
        update_interval=2,
        epsilon_steps=0,
        lambda_reg=1.0,
    )
    defaults.update(kwargs)
    return NeuralUCB(**defaults)


def test_cold_start_each_arm_once_and_design_update():
    bandit = _tiny()
    n = bandit.V.shape[0]
    assert np.allclose(bandit.V, np.eye(n))
    assert np.allclose(bandit.V_inv, np.eye(n))
    chosen = []
    for i, arm in enumerate(bandit.arms):
        phi = np.zeros(3)
        phi[0] = 0.1 * (i + 1)
        alpha = bandit.select(phi)
        assert alpha == arm
        g = bandit._pending[2]
        before = bandit.V.copy()
        bandit.observe(phi, alpha, -0.1 * alpha)
        assert np.allclose(bandit.V, before + np.outer(g, g))
        chosen.append(alpha)
    assert chosen == [4, 6, 8]
    assert np.allclose(bandit.V_inv @ bandit.V, np.eye(n), atol=1e-8)
    assert bandit.n_fits == 1  # t = 2 triggers the first fit, t = 3 does not
    assert bandit.last_loss is not None
    assert np.isfinite(bandit.last_loss)


def test_beta_schedule_and_ucb_argmax():
    bandit = _tiny(beta0=0.5, epsilon_steps=0, epochs=0, update_interval=100)
    assert np.isclose(bandit.beta(5), 0.5 * np.sqrt(np.log(6)))
    for i in range(len(bandit.arms)):
        phi = np.full(3, 0.2 * (i + 1))
        alpha = bandit.select(phi)
        bandit.observe(phi, alpha, -1.0)
    phi = np.array([0.4, 0.1, 0.0], dtype=np.float32)
    stats = bandit.arm_stats(phi)
    alpha = bandit.select(phi)
    assert alpha == bandit.arms[int(np.argmax(stats["ucb"]))]
    bandit.observe(phi, alpha, -0.2)
    assert bandit.history[-1]["mode"] == "ucb"


def test_epsilon_greedy_phase_and_buffer_cap():
    bandit = _tiny(
        epsilon=1.0,
        epsilon_steps=5,
        epochs=0,
        update_interval=100,
        buffer_size=2,
        seed=1,
    )
    for i in range(len(bandit.arms)):
        phi = np.ones(3) * (i + 1)
        alpha = bandit.select(phi)
        bandit.observe(phi, alpha, -0.3)
        assert bandit.history[-1]["mode"] == "cold"
    modes = []
    for i in range(4):
        phi = np.full(3, 0.5)
        alpha = bandit.select(phi)
        assert alpha in bandit.arms
        bandit.observe(phi, alpha, -0.4)
        modes.append(bandit.history[-1]["mode"])
    assert set(modes) == {"epsilon"}
    assert len(bandit.buffer_reward) == 2


def test_smaller_arm_set_runs():
    bandit = NeuralUCB(
        arms=[5, 10],
        context_dim=4,
        hidden=4,
        epochs=1,
        update_interval=2,
        seed=0,
    )
    phi = np.linspace(0.1, 0.4, 4)
    for _ in range(4):
        bandit.step(phi, lambda alpha: -0.01 * alpha)
    assert bandit.t == 4
    assert bandit.n_fits == 2
    assert bandit.greedy_action(phi) in bandit.arms
