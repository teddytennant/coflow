"""Reward MLP shapes and the concat input."""

import numpy as np
import jax

from coflow.model import forward, init_params, model_input, parameter_count


def test_two_layer_relu_mlp():
    context_dim = 40
    hidden = 64
    params = init_params(0, context_dim + 1, hidden)
    assert params["w1"].shape == (context_dim + 1, hidden)
    assert params["b1"].shape == (hidden,)
    assert params["w2"].shape == (hidden,)
    assert params["b2"].shape == ()
    assert set(params) == {"w1", "b1", "w2", "b2"}
    assert parameter_count(context_dim, hidden) == (context_dim + 1) * hidden + hidden + hidden + 1
    y = forward(params, jax.numpy.ones(context_dim + 1))
    assert y.shape == ()


def test_input_is_concat_of_phi_and_scaled_arm():
    phi = np.arange(4, dtype=np.float32)
    x = model_input(phi, alpha=10, max_arm=50)
    assert x.shape == (5,)
    assert np.allclose(x[:4], phi)
    assert np.isclose(x[-1], 10 / 50)
