import numpy as np

from heracles_planner.config import HeraclesConfig
from heracles_planner.debug_videos import (
    FEATURED_MOTIONS,
    FUTURE_FRAMES,
    GET_UP_FRAMES,
    HARDWARE_FROM_SONIC_POLICY,
    deterministic_noise,
    target_prefix_indices,
)


def test_policy_to_hardware_joint_mapping_is_permutation():
    assert sorted(HARDWARE_FROM_SONIC_POLICY.tolist()) == list(range(29))


def test_target_prefix_clamps_at_final_frame():
    np.testing.assert_array_equal(target_prefix_indices(98, 5, 100), np.full(11, 99))
    np.testing.assert_array_equal(target_prefix_indices(10, 2, 100)[:3], [12, 13, 14])


def test_checkpoint_noise_is_deterministic_and_keyed():
    config = HeraclesConfig()
    first = deterministic_noise("motion", 7, 15, config)
    second = deterministic_noise("motion", 7, 15, config)
    different = deterministic_noise("motion", 8, 15, config)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)
    np.testing.assert_array_equal(first[0], np.zeros(config.state_dim))


def test_debug_video_contract_constants():
    assert FEATURED_MOTIONS == (
        "dance_phony_c01_neutral2s",
        "squat_001__A359_neutral2s",
        "fallAndGetUp1_subject1",
        "fallAndGetUp1_subject4",
    )
    assert FUTURE_FRAMES == (2, 5, 10)
    assert GET_UP_FRAMES == 601
