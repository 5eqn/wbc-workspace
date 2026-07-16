import runpy
from types import SimpleNamespace

import numpy as np
import torch

from heracles_planner.config import HeraclesConfig
from heracles_planner.data import split_for_motion
from heracles_planner.inference import PlannerRuntime
from heracles_planner.model import HeraclesPlanner, conditional_flow_sample
from heracles_planner.paths import REPO_ROOT
from heracles_planner.rotations import quat_xyzw_to_rot6d, rot6d_to_quat_xyzw
from heracles_planner.zmq_protocol import pack_pose_v1, unpack_message

pack_sim_state_message = runpy.run_path(REPO_ROOT / "scripts/sim_bridge.py")[
    "pack_sim_state_message"
]


def test_subject_split_is_fixed():
    assert split_for_motion("walk1_subject1") == "train"
    assert split_for_motion("walk1_subject4") == "validation"
    assert split_for_motion("walk1_subject5") == "test"


def test_rotation_round_trip():
    quaternion = np.asarray([[0.1, -0.2, 0.3, 0.9]], dtype=np.float32)
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)
    reconstructed = rot6d_to_quat_xyzw(quat_xyzw_to_rot6d(quaternion))
    assert abs(float(np.sum(quaternion * reconstructed))) > 1.0 - 1e-6


def test_model_shape_parameter_count_and_inpainting():
    config = HeraclesConfig()
    model = HeraclesPlanner(config)
    assert 22_000_000 <= model.parameter_count <= 23_500_000
    output = model(torch.randn(2, 8, 35), torch.randn(2, 35), torch.rand(2), torch.full((2,), 0.2))
    assert output.shape == (2, 8, 35)
    assert torch.equal(output[:, 0], torch.zeros_like(output[:, 0]))


def test_linear_path_and_first_token_inpainting():
    clean = torch.randn(3, 8, 35)
    clean[:, 0] = 0
    generator = torch.Generator().manual_seed(42)
    path, _, target = conditional_flow_sample(clean, generator)
    assert torch.equal(path[:, 0], torch.zeros_like(path[:, 0]))
    assert torch.equal(target[:, 0], torch.zeros_like(target[:, 0]))


def test_protocol_v1_round_trip():
    joints = np.arange(2 * 29, dtype=np.float32).reshape(2, 29)
    velocity = joints * 0.1
    quat_xyzw = np.asarray([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.float32)
    message = pack_pose_v1(joints, velocity, quat_xyzw, np.arange(2, dtype=np.int64))
    header, fields = unpack_message(message, "pose")
    assert header["v"] == 1
    np.testing.assert_array_equal(fields["joint_pos"], joints)
    np.testing.assert_array_equal(fields["body_quat_w"][:, 0], np.ones(2))


def test_simulator_state_feed_matches_host_decoder():
    class Bridge:
        @staticmethod
        def body_q():
            return np.arange(29, dtype=np.float32)

        @staticmethod
        def body_dq():
            return np.arange(29, dtype=np.float32) * 0.1

    data = SimpleNamespace(qpos=np.asarray([0, 0, 0, 1, 0, 0, 0], dtype=np.float32))
    _, fields = unpack_message(pack_sim_state_message("sim_state", Bridge(), data, 7), "sim_state")
    assert fields["joint_pos"].shape == (1, 29)
    np.testing.assert_array_equal(fields["body_quat_w"], [[1, 0, 0, 0]])
    np.testing.assert_array_equal(fields["frame_index"], [7])


class _ZeroVelocity:
    def __call__(self, trajectory, state, time_value, duration):
        return np.zeros_like(trajectory)


def test_window_keeps_untouched_tail(tmp_path):
    config = HeraclesConfig()
    np.savez(
        tmp_path / "normalization.npz",
        mean=np.zeros(35, dtype=np.float32),
        std=np.ones(35, dtype=np.float32),
        loss_weights=np.ones(35, dtype=np.float32),
    )
    runtime = PlannerRuntime(_ZeroVelocity(), tmp_path / "normalization.npz", config)
    generated_joint = np.ones((11, 29), dtype=np.float32)
    generated_root = np.tile([0, 0, 0, 1], (11, 1)).astype(np.float32)
    original_joint = np.arange(46 * 29, dtype=np.float32).reshape(46, 29)
    original_root = np.tile([0, 0, 0, 1], (46, 1)).astype(np.float32)
    joints, roots, velocities = runtime.build_sonic_window(
        generated_joint, generated_root, original_joint, original_root
    )
    np.testing.assert_array_equal(joints[11:], original_joint[11:])
    np.testing.assert_array_equal(roots[11:], original_root[11:])
    assert velocities.shape == joints.shape
