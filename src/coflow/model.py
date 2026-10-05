"""Two-layer reward MLP.

One hidden layer of width 64 (configurable) with a ReLU, then a linear
scalar head. The input is concat(phi, alpha / max(A)). The head is linear
so the network can represent the scalar step reward.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

PARAM_KEYS = ("w1", "b1", "w2", "b2")
HIDDEN = 64


def parameter_count(context_dim, hidden=HIDDEN):
    in_dim = int(context_dim) + 1
    hidden = int(hidden)
    return in_dim * hidden + hidden + hidden + 1


def init_params(key, in_dim, hidden=HIDDEN):
    """He-normal weights, zero biases. key may be an int seed or a PRNGKey."""
    if isinstance(key, int):
        key = jax.random.PRNGKey(key)
    k1, k2 = jax.random.split(key)
    hidden = int(hidden)
    in_dim = int(in_dim)
    w1 = jax.random.normal(k1, (in_dim, hidden)) * jnp.sqrt(2.0 / in_dim)
    b1 = jnp.zeros((hidden,))
    w2 = jax.random.normal(k2, (hidden,)) * jnp.sqrt(2.0 / hidden)
    b2 = jnp.zeros(())
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def model_input(phi, alpha, max_arm):
    """concat(phi, alpha / max(A)) as float32."""
    phi = np.asarray(phi, dtype=np.float32).reshape(-1)
    frac = np.float32(float(alpha) / float(max_arm))
    return np.concatenate([phi, np.array([frac], dtype=np.float32)])


def forward(params, x):
    """Scalar prediction. x is concat(phi, alpha / max(A))."""
    x = jnp.asarray(x)
    hidden = jnp.maximum(x @ params["w1"] + params["b1"], 0.0)
    return hidden @ params["w2"] + params["b2"]


def flatten_params(params):
    return jnp.concatenate([jnp.ravel(params[k]) for k in PARAM_KEYS])


def predict_batch(params, xs):
    return jax.vmap(lambda x: forward(params, x))(xs)


def predict_and_grads(params, xs):
    """Per-row prediction and flattened parameter gradient.

    g = grad_theta f(theta, x). Rows of xs are independent contexts.
    """

    def one_grad(x):
        grads = jax.grad(lambda p: forward(p, x))(params)
        return jnp.concatenate([jnp.ravel(grads[k]) for k in PARAM_KEYS])

    preds = jax.vmap(lambda x: forward(params, x))(xs)
    grads = jax.vmap(one_grad)(xs)
    return preds, grads


predict_batch = jax.jit(predict_batch)
predict_and_grads = jax.jit(predict_and_grads)


def mse_loss(params, xs, ys, mask):
    preds = jax.vmap(lambda x: forward(params, x))(xs)
    err = (preds - ys) ** 2
    return jnp.sum(err * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def adam_epoch(params, moment, second, step, xs, ys, mask, lr):
    """One full-batch Adam step. step is 1-indexed for bias correction."""
    loss, grads = jax.value_and_grad(mse_loss)(params, xs, ys, mask)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    moment = jax.tree_util.tree_map(
        lambda m, g: b1 * m + (1.0 - b1) * g, moment, grads
    )
    second = jax.tree_util.tree_map(
        lambda v, g: b2 * v + (1.0 - b2) * (g * g), second, grads
    )
    mhat = jax.tree_util.tree_map(lambda m: m / (1.0 - b1 ** step), moment)
    vhat = jax.tree_util.tree_map(lambda v: v / (1.0 - b2 ** step), second)
    params = jax.tree_util.tree_map(
        lambda p, m, v: p - lr * m / (jnp.sqrt(v) + eps),
        params,
        mhat,
        vhat,
    )
    return params, moment, second, loss


adam_epoch = jax.jit(adam_epoch)
