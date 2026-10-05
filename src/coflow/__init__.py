"""Contextual flow matching with adaptive Euler step selection."""

from coflow.bandit import DEFAULT_ARMS, NeuralUCB
from coflow.euler import forward_euler, rollout
from coflow.features import (
    CONTEXT_DIM,
    EMBED_DIM,
    FEATURE_NAMES,
    HANDCRAFTED_DIM,
    context_features,
    handcrafted_features,
    hash_embedding,
)
from coflow.model import forward, init_params, model_input
from coflow.reward import cost_weight, step_reward, trajectory_variation
from coflow.synthetic import (
    SyntheticEnv,
    SyntheticVelocity,
    difficulty_from_features,
    difficulty_from_prompt,
    run_synthetic_experiment,
)

__all__ = [
    "CONTEXT_DIM",
    "DEFAULT_ARMS",
    "EMBED_DIM",
    "FEATURE_NAMES",
    "HANDCRAFTED_DIM",
    "NeuralUCB",
    "SyntheticEnv",
    "SyntheticVelocity",
    "context_features",
    "cost_weight",
    "difficulty_from_features",
    "difficulty_from_prompt",
    "forward",
    "forward_euler",
    "handcrafted_features",
    "hash_embedding",
    "init_params",
    "model_input",
    "rollout",
    "run_synthetic_experiment",
    "step_reward",
    "trajectory_variation",
]
