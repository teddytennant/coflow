"""Variation sum, reward, and the smooth versus rough field."""

import numpy as np

from coflow.bandit import DEFAULT_ARMS
from coflow.euler import rollout
from coflow.reward import VARIATION_EPS, cost_weight, step_reward, trajectory_variation
from coflow.synthetic import SyntheticVelocity


def test_variation_is_l1_relative_change_from_k1():
    # alpha = 3. v_0 is ignored. k = 1, 2 contribute.
    vs = np.array(
        [
            [9.0, 9.0],
            [3.0, 4.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    # k=1: ||v2 - v1||_1 / max(||v1||_1, eps) = 7 / 7
    # k=2: 0
    paper = 0.5 * (7.0 / 7.0)
    included_v0 = (11.0 / 18.0 + 7.0 / 7.0 + 0.0) / 3.0
    assert np.isclose(trajectory_variation(vs), paper)
    assert not np.isclose(paper, included_v0)

    # Default path is L1, not L2. Denominator is max(||v||, eps), not ||v|| + eps.
    angled = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [0.0, 4.0],
            [0.0, 4.0],
        ],
        dtype=np.float64,
    )
    diff = angled[2] - angled[1]
    l1 = np.linalg.norm(diff, ord=1) / max(np.linalg.norm(angled[1], ord=1), VARIATION_EPS)
    l2 = np.linalg.norm(diff, ord=2) / max(np.linalg.norm(angled[1], ord=2), VARIATION_EPS)
    assert np.isclose(trajectory_variation(angled), 0.5 * l1)
    assert not np.isclose(l1, l2)
    assert not np.isclose(trajectory_variation(angled), 0.5 * l2)
    assert np.isclose(trajectory_variation(angled, ord=2), 0.5 * l2)


def test_near_zero_base_uses_eps():
    eps = VARIATION_EPS
    tiny = 1e-12
    vs = np.array(
        [
            [0.0, 0.0],
            [tiny, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    paper = 0.5 * (abs(1.0 - tiny) / eps)
    added = 0.5 * (abs(1.0 - tiny) / (tiny + eps))
    got = trajectory_variation(vs, eps=eps)
    assert np.isclose(got, paper)
    assert not np.isclose(paper, added)
    assert not np.isclose(got, added)


def test_reward_matches_paper_and_clips_at_zero():
    assert DEFAULT_ARMS == (10, 15, 20, 25, 30, 35, 40, 45, 50)
    mu = cost_weight(DEFAULT_ARMS)
    assert mu == 1.0 / 50.0

    # Nothing clips: +1 does not change the ranking versus -V - mu * alpha.
    variations = {10: 0.1, 20: 0.05, 30: 0.2}
    paper = {a: (1.0 - v) - mu * a for a, v in variations.items()}
    old = {a: -v - mu * a for a, v in variations.items()}
    assert all(value > 0.0 for value in paper.values())
    assert max(paper, key=paper.get) == max(old, key=old.get)
    for alpha, variation in variations.items():
        assert step_reward(variation, alpha, mu) == paper[alpha]

    # (1 - vhat) - mu * alpha is negative. Paper reward is 0. -V - mu * alpha is not.
    vhat = 2.0
    alpha = 10
    assert (1.0 - vhat) - mu * alpha < 0.0
    assert step_reward(vhat, alpha, mu) == 0.0
    old_negative = -vhat - mu * alpha
    assert old_negative < 0.0
    assert old_negative != 0.0

    # mu * max(A) = 1, so the largest arm cannot have a positive reward.
    assert step_reward(0.0, max(DEFAULT_ARMS), mu) == 0.0


def test_smooth_field_low_variation_rough_field_not():
    x0 = np.zeros(2)
    _, smooth_vs, _ = rollout(SyntheticVelocity(0.0), x0, 10)
    _, rough_vs, _ = rollout(SyntheticVelocity(12.0), x0, 10)
    smooth = trajectory_variation(smooth_vs)
    rough = trajectory_variation(rough_vs)
    assert smooth < 0.2
    assert abs(smooth - 0.1) < 1e-6
    # L1 relative change at the default frequency. Still far above the smooth field.
    assert rough > 1.0
    assert rough > smooth
