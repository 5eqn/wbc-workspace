from __future__ import annotations

import numpy as np
import pytest

from bfm_zero_fast_induced_eval.config import EvalConfig
from bfm_zero_fast_induced_eval.disturbance import attempt_seed, sample_velocity_disturbance
from bfm_zero_fast_induced_eval.metrics import trial_metrics, wilson_interval
from bfm_zero_fast_induced_eval.pose import require_upright, xyzw_to_wxyz
from bfm_zero_fast_induced_eval.schema import (
    HISTORY_COMPONENTS,
    OBSERVATION_FIELDS,
    REPLAY_STATE_FIELDS,
    actor_input,
    build_history_actor,
    estimated_raw_gib,
    raw_bytes_per_accepted_trial,
    transition_at,
    validate_transition_alignment,
)


def test_field_dimensions_and_actor_order() -> None:
    assert list(OBSERVATION_FIELDS.items()) == [
        ("state", 64),
        ("privileged_state", 463),
        ("last_action", 29),
        ("history_actor", 372),
    ]
    assert list(REPLAY_STATE_FIELDS.items()) == [("qpos", 36), ("qvel", 35)]
    observation = {
        "state": np.full((2, 64), 1, np.float32),
        "last_action": np.full((2, 29), 2, np.float32),
        "history_actor": np.full((2, 372), 3, np.float32),
    }
    value = actor_input(observation, np.full(256, 4, np.float32))
    np.testing.assert_array_equal(value[:, :64], 1)
    np.testing.assert_array_equal(value[:, 64:93], 2)
    np.testing.assert_array_equal(value[:, 93:465], 3)
    np.testing.assert_array_equal(value[:, 465:], 4)


def test_identity_xyzw_to_wxyz_conversion() -> None:
    converted = xyzw_to_wxyz(np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(converted, [1.0, 0.0, 0.0, 0.0])


def test_arbitrary_xyzw_to_wxyz_conversion() -> None:
    converted = xyzw_to_wxyz(np.asarray([0.1, -0.2, 0.3, 0.9], dtype=np.float32))
    expected = np.asarray([0.9, 0.1, -0.2, 0.3], dtype=np.float32)
    np.testing.assert_array_equal(converted, expected)


def test_rejects_observed_inverted_mujoco_quaternion() -> None:
    qpos = np.zeros((1, 36), dtype=np.float32)
    qpos[0, 3:7] = [0.00931729, 0.00546296, 0.999124, -0.04042879]
    with pytest.raises(RuntimeError, match="not upright"):
        require_upright(qpos, "post-kick")


def test_history_alignment_and_newest_first_order() -> None:
    dimensions = {"actions": 29, "base_ang_vel": 3, "dof_pos": 29, "dof_vel": 29, "projected_gravity": 3}
    frames = {}
    for component_index, (key, dim) in enumerate(sorted(dimensions.items()), start=1):
        frames[key] = np.stack(
            [np.full((1, dim), component_index * 10 + lag, np.float32) for lag in range(4)], axis=1
        )
    history = build_history_actor(frames)
    assert history.shape == (1, 372)
    for key in sorted(dimensions):
        start, end = HISTORY_COMPONENTS[key]
        expected = frames[key].reshape(1, -1)
        np.testing.assert_array_equal(history[:, start:end], expected)


def test_transition_alignment() -> None:
    action = np.arange(400 * 29, dtype=np.float32).reshape(1, 400, 29)
    last_action = np.zeros((1, 401, 29), dtype=np.float32)
    last_action[:, 1:] = action
    qpos = np.zeros((1, 401, 36), dtype=np.float32)
    qvel = np.zeros((1, 401, 35), dtype=np.float32)
    validate_transition_alignment(last_action, action, qpos, qvel)
    trajectory = {
        name: np.arange(401)[:, None].repeat(dim, axis=1) for name, dim in OBSERVATION_FIELDS.items()
    }
    trajectory.update(
        qpos=np.arange(401)[:, None].repeat(36, axis=1),
        qvel=np.arange(401)[:, None].repeat(35, axis=1),
        action=np.arange(400 * 29).reshape(400, 29),
        terminated=np.zeros(400, dtype=bool),
        truncated=np.zeros(400, dtype=bool),
    )
    transition = transition_at(trajectory, 9)
    assert transition["observation"]["state"][0] == 9
    assert transition["action"][0] == 9 * 29
    assert transition["next_observation"]["state"][0] == 10
    assert transition["qpos"][0] == 9
    assert transition["next_qvel"][0] == 10
    with pytest.raises(IndexError):
        transition_at(trajectory, 400)


def test_disturbance_is_deterministic_and_seeded_like_old_runs() -> None:
    config = EvalConfig()
    first = sample_velocity_disturbance(attempt_seed(0, 7), config)
    second = sample_velocity_disturbance(7000, config)
    assert first == second
    np.testing.assert_allclose(
        first.linear_velocity_delta, [0.21971071, 2.9769621, -3.46856], rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        first.angular_velocity_delta, [0.9871539, 1.640753, 5.0639343], rtol=0, atol=1e-6
    )
    assert config.linear_up_cos_min <= first.linear_velocity_delta[2] / first.linear_velocity_magnitude <= 0


def test_fall_and_recovery_boundaries_are_strict() -> None:
    heights = np.full(1600, 0.8, dtype=np.float32)
    heights[10] = 0.45
    metrics = trial_metrics(heights)
    assert not metrics.induced_fall
    heights[10] = np.nextafter(np.float32(0.45), np.float32(0))
    heights[-100:] = 0.75
    metrics = trial_metrics(heights)
    assert metrics.induced_fall and not metrics.recovered
    heights[-100:] = np.nextafter(np.float32(0.75), np.float32(1))
    assert trial_metrics(heights).recovered


def test_wilson_and_storage_estimate() -> None:
    low, high = wilson_interval(98, 100)
    assert low < 0.98 < high
    assert EvalConfig().minimum_free_gib == 22.0
    assert raw_bytes_per_accepted_trial() == 1_649_596
    assert 15.3 < estimated_raw_gib(10_000) < 15.4
    assert 19.6 < estimated_raw_gib(12_800) < 19.7
