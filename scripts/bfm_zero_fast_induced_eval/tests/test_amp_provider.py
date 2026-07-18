from __future__ import annotations

import numpy as np

from bfm_zero_fast_induced_eval.providers import (
    AMP_DEFAULT_JOINT_POSITION,
    AMP_TERM_DIMS,
    KEY_BODY_NAMES,
    POLICY_JOINT_NAMES,
    POLICY_TO_SDK,
    SDK_JOINT_NAMES,
    SDK_TO_POLICY,
    CanonicalSnapshot,
    LeggedLabAmpProvider,
    action_to_target,
    key_body_positions,
    resolve_provider_artifacts,
    yaw_local_rotation_6d,
)


def _snapshot(batch: int = 1) -> CanonicalSnapshot:
    qpos = np.zeros((batch, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    qpos[:, 7:] = AMP_DEFAULT_JOINT_POSITION
    qvel = np.zeros((batch, 35), dtype=np.float32)
    body_names = ("pelvis", *KEY_BODY_NAMES)
    body_positions = np.zeros((batch, len(body_names), 3), dtype=np.float32)
    for index in range(1, len(body_names)):
        body_positions[:, index, 0] = index
    return CanonicalSnapshot(qpos, qvel, body_positions, body_names)


def test_provider_selection_and_exact_training_orders() -> None:
    actor, checkpoint, goals = resolve_provider_artifacts("legged-lab-amp", None, None, None)
    assert actor.name == "policy.onnx"
    assert checkpoint is goals is None
    assert len(SDK_JOINT_NAMES) == 29 and SDK_JOINT_NAMES[12] == "waist_yaw_joint"
    assert POLICY_JOINT_NAMES[:3] == (
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "waist_yaw_joint",
    )
    assert KEY_BODY_NAMES == (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
    )


def test_yaw_removal_rotation_columns_and_key_body_frame() -> None:
    yaw = np.asarray([[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]], dtype=np.float32)
    np.testing.assert_allclose(yaw_local_rotation_6d(yaw), [[1, 0, 0, 0, 0, 1]], atol=1e-6)
    snapshot = _snapshot()
    np.testing.assert_array_equal(key_body_positions(snapshot).reshape(1, 6, 3)[0, :, 0], np.arange(1, 7))


def test_term_major_history_reset_and_post_action_injection() -> None:
    provider = LeggedLabAmpProvider()
    snapshot = _snapshot(2)
    snapshot.qvel[:, 3:6] = [1, 2, 3]
    provider.reset({}, snapshot)
    model_input = provider.build_input(snapshot, advance=False)
    assert model_input.shape == (2, 570)
    np.testing.assert_array_equal(model_input[:, :15], np.tile([1, 2, 3], (2, 5)))
    action_start = 5 * sum(list(AMP_TERM_DIMS.values())[:4])
    np.testing.assert_array_equal(model_input[:, action_start : action_start + 145], 0)

    class Session:
        def get_inputs(self):
            return [type("Input", (), {"name": "obs"})()]

        def run(self, _outputs, feed):
            return [np.full((len(feed["obs"]), 29), 0.5, dtype=np.float32)]

    provider.infer({}, snapshot, Session())
    following = provider.build_input(snapshot)
    np.testing.assert_array_equal(following[:, action_start + 4 * 29 : action_start + 5 * 29], 0.5)


def test_amp_target_mapping_has_no_hidden_times_five() -> None:
    raw = np.linspace(-2, 2, 29, dtype=np.float32)
    target = action_to_target(raw)
    np.testing.assert_allclose(target, AMP_DEFAULT_JOINT_POSITION + 0.25 * raw, rtol=0, atol=0)


def test_policy_joint_input_and_output_are_permuted_from_canonical_sdk_order() -> None:
    provider = LeggedLabAmpProvider()
    snapshot = _snapshot()
    snapshot.qpos[:, 7:] = np.arange(29, dtype=np.float32)
    snapshot.qvel[:, 6:] = 100 + np.arange(29, dtype=np.float32)
    provider.reset({}, snapshot)
    model_input = provider.build_input(snapshot, advance=False)
    joint_pos_start = 5 * (3 + 6)
    joint_vel_start = joint_pos_start + 5 * 29
    np.testing.assert_array_equal(
        model_input[0, joint_pos_start : joint_pos_start + 29],
        np.arange(29, dtype=np.float32)[list(SDK_TO_POLICY)],
    )
    np.testing.assert_array_equal(
        model_input[0, joint_vel_start : joint_vel_start + 29],
        (100 + np.arange(29, dtype=np.float32))[list(SDK_TO_POLICY)],
    )

    class Session:
        def get_inputs(self):
            return [type("Input", (), {"name": "obs"})()]

        def run(self, _outputs, _feed):
            return [np.arange(29, dtype=np.float32)[None]]

    _, canonical_action = provider.infer({}, snapshot, Session())
    np.testing.assert_array_equal(canonical_action[0], np.arange(29, dtype=np.float32)[list(POLICY_TO_SDK)])
