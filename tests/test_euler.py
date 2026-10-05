"""Forward Euler step rule and the O(1/K) decay check."""

import numpy as np

from coflow.euler import forward_euler, rollout


def test_step_formula_and_times():
    calls = []

    def velocity(x, t):
        calls.append(float(t))
        return np.array([1.0, -2.0])

    x0 = np.array([0.0, 0.5])
    alpha = 4
    xs, vs, ts = rollout(velocity, x0, alpha)
    assert np.allclose(ts, [0.0, 0.25, 0.5, 0.75, 1.0])
    # alpha integration steps plus the terminal velocity sample.
    assert calls[0] == 0.0
    assert calls[-1] == 1.0
    assert len(calls) == alpha + 1
    h = 1.0 / alpha
    x = x0.astype(np.float64).copy()
    for k in range(alpha):
        v = np.array([1.0, -2.0])
        assert np.allclose(vs[k], v)
        x = x + h * v
        assert np.allclose(xs[k + 1], x)
    assert np.allclose(forward_euler(velocity, x0, alpha), xs[-1])
    assert np.allclose(xs[-1], x0 + np.array([1.0, -2.0]))


def test_decay_error_is_order_one_over_k():
    # dx/dt = -x on [0, 1]. Exact endpoint is x0 * exp(-1).
    # The paper's theorem gives an O(1/K) bound under a Lipschitz velocity
    # and a bounded second derivative. K * error(K) should stay flat.
    def velocity(x, t):
        return -x

    x0 = np.array([1.0])
    exact = float(np.exp(-1.0))
    ks = [5, 10, 20, 40]
    errors = []
    for k in ks:
        endpoint = float(forward_euler(velocity, x0, k)[0])
        errors.append(abs(endpoint - exact))
    for earlier, later in zip(errors, errors[1:]):
        assert later < earlier
    products = [k * err for k, err in zip(ks, errors)]
    assert max(products) / min(products) < 1.5
