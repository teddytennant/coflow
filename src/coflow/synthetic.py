"""Synthetic 2D velocity field for the step-budget bandit.

v(x, t; difficulty) = (target - x) + difficulty * sin(2 * pi * freq * t) * u

u is a fixed unit vector orthogonal to the default target. difficulty is
read from the prompt: a logistic gate on max(word-count feature, numeric
fraction), scaled so a rough prompt has a large oscillatory amplitude.
High word count or a high numeric fraction raises difficulty. A smooth
field (difficulty near 0) has relative variation about 1/alpha, so the
linear cost prefers a small budget. A rough field does not: variation at
a small budget stays large, and the reward prefers more steps.

freq defaults to 0.75. An integer frequency can alias to zero on a coarse
grid (sin(2 * pi * 5 * k / 10) = 0), which would hide the roughness. 0.75
is not an integer. It is also low enough that, under the paper reward, a
rough field has a higher reward at an intermediate budget than at 10 steps.
A higher frequency keeps relative variation above 1 - mu * alpha at every
arm, so the clipped reward is flat and cannot prefer more steps.
"""

from __future__ import annotations

import numpy as np

from coflow.bandit import DEFAULT_ARMS, NeuralUCB
from coflow.euler import rollout
from coflow.features import CONTEXT_DIM, context_features, handcrafted_features
from coflow.reward import cost_weight, step_reward, trajectory_variation

DEFAULT_FREQ = 0.75
DEFAULT_GAIN = 12.0
DEFAULT_THRESHOLD = 0.25
DEFAULT_SHARPNESS = 40.0
DEFAULT_TARGET = np.array([1.0, 0.0], dtype=np.float64)
DEFAULT_ORTHOGONAL = np.array([0.0, 1.0], dtype=np.float64)
DEFAULT_X0 = np.zeros(2, dtype=np.float64)

# Short prompts with no digits. Word-count and numeric features stay below
# the difficulty gate.
EASY_PROMPTS = (
    "a cat",
    "a red ball",
    "one small bird",
    "blue sky",
)

# Long prompts, numeric prompts, or both. Either signal opens the gate.
HARD_PROMPTS = (
    (
        "a scene with 3 red cubes and 5 blue spheres and 2 green cones "
        "and 4 yellow cylinders arranged in 7 rows beside 8 lamps and 6 chairs "
        "with 9 windows and 1 table under 12 lights near 11 doors"
    ),
    " ".join(["object"] * 40),
    "3 cats and 5 dogs and 2 birds and 4 trees and 8 lamps",
    "generate 6 boxes with 9 stripes and 4 dots beside 7 circles and 2 arrows",
)

DEMO_SEED = 0
DEMO_ROUNDS = 64
DEMO_UPDATE_INTERVAL = 8
DEMO_EPOCHS = 20


def difficulty_from_features(
    features,
    gain=DEFAULT_GAIN,
    threshold=DEFAULT_THRESHOLD,
    sharpness=DEFAULT_SHARPNESS,
):
    """Map handcrafted features to an oscillation amplitude.

    Uses feature 0 (word count) and feature 3 (numeric fraction). A logistic
    gate around 0.25 keeps short, non-numeric prompts near difficulty 0 and
    sends high word count or a high numeric fraction to `gain`.
    """
    feats = np.asarray(features, dtype=np.float64).reshape(-1)
    raw = max(float(feats[0]), float(feats[3]))
    z = float(np.clip(sharpness * (raw - threshold), -60.0, 60.0))
    gate = 1.0 / (1.0 + np.exp(-z))
    return float(gain) * gate


def difficulty_from_prompt(prompt, **kwargs):
    return difficulty_from_features(handcrafted_features(prompt), **kwargs)


class SyntheticVelocity:
    """Callable field v(x, t) for a frozen difficulty."""

    def __init__(
        self,
        difficulty,
        freq=DEFAULT_FREQ,
        target=None,
        orthogonal=None,
    ):
        self.difficulty = float(difficulty)
        self.freq = float(freq)
        self.target = np.array(
            DEFAULT_TARGET if target is None else target, dtype=np.float64
        )
        self.orthogonal = np.array(
            DEFAULT_ORTHOGONAL if orthogonal is None else orthogonal,
            dtype=np.float64,
        )

    def __call__(self, x, t):
        state = np.asarray(x, dtype=np.float64)
        osc = np.sin(2.0 * np.pi * self.freq * float(t))
        return (self.target - state) + self.difficulty * osc * self.orthogonal


class SyntheticEnv:
    """Prompt to (context, reward) for a fixed arm set."""

    def __init__(self, arms=None, freq=DEFAULT_FREQ, x0=None):
        self.arms = tuple(int(a) for a in (DEFAULT_ARMS if arms is None else arms))
        self.mu = cost_weight(self.arms)
        self.freq = float(freq)
        self.x0 = np.array(DEFAULT_X0 if x0 is None else x0, dtype=np.float64)

    def context(self, prompt):
        return context_features(prompt)

    def velocity(self, difficulty):
        return SyntheticVelocity(difficulty, freq=self.freq)

    def variation(self, difficulty, alpha):
        vel = self.velocity(difficulty)
        _, vs, _ = rollout(vel, self.x0, int(alpha))
        return trajectory_variation(vs)

    def reward_from_difficulty(self, difficulty, alpha):
        variation = self.variation(difficulty, alpha)
        return step_reward(variation, alpha, self.mu)

    def reward(self, prompt, alpha):
        return self.reward_from_difficulty(difficulty_from_prompt(prompt), alpha)


def run_synthetic_experiment(
    seed=DEMO_SEED,
    rounds=DEMO_ROUNDS,
    epochs=DEMO_EPOCHS,
    update_interval=DEMO_UPDATE_INTERVAL,
    arms=None,
    epsilon=0.1,
    epsilon_steps=None,
    beta0=0.5,
    hidden=64,
    lr=5e-3,
):
    """Online NeuralUCB on alternating easy and hard prompts.

    After training, mean steps are the greedy argmax of the learned reward
    on the prompt pool, not the exploratory UCB choices. With the default
    beta the confidence bonus is comparable to the easy-prompt reward gap,
    so exploitation is read from f.
    """
    arms = tuple(DEFAULT_ARMS if arms is None else (int(a) for a in arms))
    env = SyntheticEnv(arms)
    bandit = NeuralUCB(
        arms=arms,
        context_dim=CONTEXT_DIM,
        hidden=hidden,
        lr=lr,
        epochs=epochs,
        update_interval=update_interval,
        epsilon=epsilon,
        epsilon_steps=epsilon_steps,
        beta0=beta0,
        seed=seed,
    )
    pool = []
    for easy, hard in zip(EASY_PROMPTS, HARD_PROMPTS):
        pool.append(("easy", easy))
        pool.append(("hard", hard))
    for t in range(int(rounds)):
        _kind, prompt = pool[t % len(pool)]
        phi = env.context(prompt)
        bandit.step(phi, lambda alpha, prompt=prompt: env.reward(prompt, alpha))
    easy_alphas = [bandit.greedy_action(env.context(p)) for p in EASY_PROMPTS]
    hard_alphas = [bandit.greedy_action(env.context(p)) for p in HARD_PROMPTS]
    return {
        "bandit": bandit,
        "env": env,
        "easy_alphas": easy_alphas,
        "hard_alphas": hard_alphas,
        "easy_mean": float(np.mean(easy_alphas)),
        "hard_mean": float(np.mean(hard_alphas)),
        "seed": int(seed),
        "rounds": int(rounds),
    }
