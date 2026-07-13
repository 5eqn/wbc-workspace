import numpy as np

from heracles_planner.evaluation import score_trial, summarize_trials


def _trial(path, method, height_error, joint_error):
    frames = 51
    reference_root = np.zeros((frames, 3), dtype=np.float32)
    simulated_root = reference_root.copy()
    simulated_root[:, 2] += height_error
    quat = np.tile([0, 0, 0, 1], (frames, 1)).astype(np.float32)
    reference_joint = np.zeros((frames, 29), dtype=np.float32)
    np.savez(
        path,
        time_s=np.arange(frames, dtype=np.float32) / 50,
        motion=np.asarray("walk1_subject5"),
        test_kind=np.asarray("disturbed"),
        method=np.asarray(method),
        seed=np.asarray(42),
        completed=np.asarray(True),
        reference_joint_pos=reference_joint,
        simulated_joint_pos=reference_joint + joint_error,
        reference_root_pos=reference_root,
        simulated_root_pos=simulated_root,
        reference_root_quat_xyzw=quat,
        simulated_root_quat_xyzw=quat,
        failure_reason=np.asarray(""),
    )


def test_disturbed_metrics_and_selection(tmp_path):
    sonic = tmp_path / "sonic.npz"
    ours = tmp_path / "ours.npz"
    _trial(sonic, "sonic", 0.4, 0.2)
    _trial(ours, "ours", 0.1, 0.3)
    assert score_trial(ours).stand_up_success is True
    summary = summarize_trials([sonic, ours], tmp_path / "summary.json")
    selected = summary["video_selections"]["disturbed"]["Ours > SONIC"]
    assert selected[0]["motion"] == "walk1_subject5"
