"""Synthetic field, difficulty map, and the learned step ranking."""

import numpy as np

from coflow.features import context_features
from coflow.synthetic import (
    DEFAULT_ARMS,
    EASY_PROMPTS,
    HARD_PROMPTS,
    SyntheticEnv,
    SyntheticVelocity,
    difficulty_from_prompt,
    run_synthetic_experiment,
)


def test_velocity_formula():
    field = SyntheticVelocity(difficulty=1.5, freq=2.5)
    x = np.array([0.2, -0.4])
    t = 0.3
    expected = (field.target - x) + 1.5 * np.sin(2 * np.pi * 2.5 * t) * field.orthogonal
    assert np.allclose(field(x, t), expected)
    assert np.dot(field.target, field.orthogonal) == 0.0


def test_difficulty_tracks_word_count_or_numeric_fraction():
    for prompt in EASY_PROMPTS:
        assert difficulty_from_prompt(prompt) < 0.05
    for prompt in HARD_PROMPTS:
        assert difficulty_from_prompt(prompt) > 10.0
    short_numeric = "3 cats and 5 dogs and 2 birds"
    long_plain = " ".join(["lamp"] * 30)
    assert difficulty_from_prompt(short_numeric) > 10.0
    assert difficulty_from_prompt(long_plain) > 10.0
    assert difficulty_from_prompt("a cat") < difficulty_from_prompt(short_numeric)


def _best_higher_arm(env, difficulty, low):
    higher = [arm for arm in env.arms if arm > low]
    return max(higher, key=lambda arm: env.reward_from_difficulty(difficulty, arm))


def test_frozen_field_reward_prefers_more_steps_when_rough():
    env = SyntheticEnv(DEFAULT_ARMS)
    low = min(DEFAULT_ARMS)
    # mu * max(A) = 1, so the largest arm is always clipped to 0.
    assert env.reward_from_difficulty(0.0, low) > env.reward_from_difficulty(
        0.0, max(DEFAULT_ARMS)
    )
    assert env.reward_from_difficulty(0.0, low) == max(
        env.reward_from_difficulty(0.0, arm) for arm in DEFAULT_ARMS
    )
    rough = 12.0
    high = _best_higher_arm(env, rough, low)
    assert high > low
    assert high < max(DEFAULT_ARMS)
    assert env.reward_from_difficulty(rough, high) > env.reward_from_difficulty(rough, low)


def test_learned_policy_uses_fewer_steps_on_easy_prompts():
    result = run_synthetic_experiment()
    assert result["easy_mean"] < result["hard_mean"]
    assert result["hard_mean"] - result["easy_mean"] >= 10.0
    bandit = result["bandit"]
    env = result["env"]
    # Frozen hard field: the surrogate the bandit fits must rank a higher
    # step budget above a low one. The hard prompt is paired with that field.
    hard_prompt = HARD_PROMPTS[0]
    phi = context_features(hard_prompt)
    difficulty = difficulty_from_prompt(hard_prompt)
    field = SyntheticVelocity(difficulty)
    low = min(DEFAULT_ARMS)
    high = _best_higher_arm(env, field.difficulty, low)
    true_low = env.reward_from_difficulty(field.difficulty, low)
    true_high = env.reward_from_difficulty(field.difficulty, high)
    assert high > low
    assert true_high > true_low
    assert bandit.predict(phi, high) > bandit.predict(phi, low)
    easy_phi = context_features(EASY_PROMPTS[0])
    assert bandit.predict(easy_phi, low) > bandit.predict(easy_phi, high)
