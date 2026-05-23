#!/usr/bin/env python3
from __future__ import annotations

from benchmark_common import *

def load_reference(policy: str, motion: str) -> tuple[np.ndarray, float]:
    import numpy as np

    if policy == "sonic":
        q = read_csv_matrix(ROOT / "assets" / "motions" / "sonic_motions" / motion / "joint_pos.csv")
        return q[:, HARDWARE_FROM_SONIC_POLICY], 50.0
    path = ROOT / "assets" / "motions" / "holomotion_motions" / f"{motion}_holomotion.npz"
    data = np.load(path)
    for key in ("ref_dof_pos", "dof_pos"):
        if key in data:
            q = np.asarray(data[key], dtype=np.float64)
            return q[:, :29], 50.0
    raise KeyError(f"{path}: no HoloMotion joint-position array")


def load_tracked(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    if run_dir.suffix == ".npz":
        data = np.load(run_dir, allow_pickle=False)
        if "q" not in data:
            raise KeyError(f"{run_dir}: missing q array")
        q = np.asarray(data["q"], dtype=np.float64)
        if "timestamps" in data:
            timestamps = np.asarray(data["timestamps"], dtype=np.float64)
            times = timestamps - timestamps[0]
        else:
            times = np.arange(q.shape[0], dtype=np.float64) / 50.0
        return times, q[:, :29]

    lowcmd = run_dir / "lowcmd.csv"
    sonic_q = run_dir / "csv" / "q.csv"
    holo_log = run_dir / "holomotion_policy_log.csv"
    if lowcmd.exists():
        header = lowcmd.read_text().splitlines()[0].split(",")
        q_cols = [i for i, name in enumerate(header) if name.startswith("q_") or name.startswith("measured_q_")]
        t_col = header.index("time_s") if "time_s" in header else 0
        rows = []
        times = []
        with lowcmd.open(newline="") as f:
            for row in csv.DictReader(f):
                try:
                    times.append(float(row.get("time_s") or row.get(header[t_col]) or len(times) / 50.0))
                    rows.append([float(row[header[i]]) for i in q_cols[:29]])
                except (KeyError, ValueError):
                    continue
        if rows:
            return np.asarray(times), np.asarray(rows, dtype=np.float64)
    if sonic_q.exists():
        q = read_csv_matrix(sonic_q, skip_prefix_cols=1)
        return np.arange(q.shape[0], dtype=np.float64) / 50.0, q[:, :29]
    if holo_log.exists():
        rows = []
        times = []
        with holo_log.open(newline="") as f:
            for row in csv.DictReader(f):
                try:
                    times.append(float(row.get("phase_time_s") or row.get("sim_time_s") or len(times) / 50.0))
                    rows.append([float(row[f"measured_q_{i}"]) for i in range(29)])
                except (KeyError, ValueError):
                    continue
        if rows:
            return np.asarray(times), np.asarray(rows, dtype=np.float64)
    raise FileNotFoundError(f"{run_dir}: no supported tracked joint log")


def interpolate_ref(ref: np.ndarray, ref_hz: float, query_t: np.ndarray, delay_s: float) -> np.ndarray:
    import numpy as np

    ref_t = np.arange(ref.shape[0], dtype=np.float64) / ref_hz
    phase_t = np.clip(query_t - delay_s, ref_t[0], ref_t[-1])
    cols = [np.interp(phase_t, ref_t, ref[:, j]) for j in range(ref.shape[1])]
    return np.stack(cols, axis=1)


def aligned_rmse(ref: np.ndarray, ref_hz: float, tracked_t: np.ndarray, tracked_q: np.ndarray) -> dict:
    import numpy as np

    n = min(tracked_q.shape[1], ref.shape[1], 29)
    tracked_q = tracked_q[:, :n]
    best = None
    for delay in np.arange(-0.2, 0.2001, 0.01):
        rq = interpolate_ref(ref[:, :n], ref_hz, tracked_t, float(delay))
        err = tracked_q - rq
        rmse = np.sqrt(np.mean(err * err, axis=0))
        mean = float(np.mean(rmse))
        if best is None or mean < best["mean_joint_rmse_rad"]:
            best = {
                "tracking_delay_s": round(float(delay), 4),
                "mean_joint_rmse_rad": mean,
                "joint_rmse_rad": rmse.tolist(),
                "samples": int(tracked_q.shape[0]),
            }
    assert best is not None
    return best

def tracking_window(run_dir: Path, tracked_t, tracked_q):
    try:
        events = load_events(run_dir)
        playback = None
        for row in events:
            if row.get("event") in PLAYBACK_EVENTS:
                playback = row
                break
        if playback is None:
            return tracked_t, tracked_q, None
        playback_t = event_sim_time(playback)
        if tracked_t[0] <= playback_t <= tracked_t[-1]:
            keep = tracked_t >= playback_t
            return tracked_t[keep] - playback_t, tracked_q[keep], playback_t
    except Exception:
        pass
    return tracked_t, tracked_q, None


def indexed_columns(header: list[str], prefix: str) -> list[str]:
    cols = [name for name in header if name.startswith(prefix)]
    return sorted(cols, key=lambda name: int(name.rsplit("_", 1)[1]))


def load_replay(run_dir: Path) -> dict:
    import numpy as np

    if run_dir.is_file():
        raise FileNotFoundError(f"{run_dir}: replay requires an expanded run directory")
    summary_path = run_dir / "sim_bridge_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"{run_dir}: missing sim_bridge_summary.json")
    summary = json.loads(summary_path.read_text())
    path = run_dir / "replay.csv"
    if not path.exists():
        raise FileNotFoundError(f"{run_dir}: missing replay.csv")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        qpos_cols = indexed_columns(header, "qpos_")
        qvel_cols = indexed_columns(header, "qvel_")
        measured_cols = indexed_columns(header, "measured_q_")[:29]
        if len(qpos_cols) < 7:
            raise ValueError(f"{path}: replay log has no full floating-root qpos")
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError(f"{path}: replay log has too few rows")

    def col_float(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in rows], dtype=np.float64)

    replay = {
        "path": path,
        "time_s": col_float("time_s"),
        "wall_time_s": col_float("wall_time_s"),
        "sim_time_s": col_float("sim_time_s"),
        "support_active": np.asarray([int(float(row["support_active"])) for row in rows], dtype=np.int32),
        "cmd_active": np.asarray([int(float(row.get("cmd_active") or 0)) for row in rows], dtype=np.int32),
        "base_z": col_float("base_z"),
        "qpos": np.asarray([[float(row[col]) for col in qpos_cols] for row in rows], dtype=np.float64),
        "qvel": np.asarray([[float(row[col]) for col in qvel_cols] for row in rows], dtype=np.float64),
        "measured_q": np.asarray([[float(row[col]) for col in measured_cols] for row in rows], dtype=np.float64),
        "qpos_cols": qpos_cols,
        "qvel_cols": qvel_cols,
        "measured_cols": measured_cols,
        "summary": summary,
    }
    if replay["measured_q"].shape[1] < 29:
        raise ValueError(f"{path}: measured joint replay has fewer than 29 DOF")
    joint_names = summary.get("body_joint_names")
    qpos_adrs = summary.get("body_qpos_addresses")
    if not isinstance(joint_names, list) or joint_names[:29] != BODY_JOINT_NAMES:
        raise ValueError(f"{summary_path}: body joint order does not match benchmark G1 29DOF order")
    if not isinstance(qpos_adrs, list) or len(qpos_adrs) < 29:
        raise ValueError(f"{summary_path}: missing body qpos address metadata")
    if max(int(adr) for adr in qpos_adrs[:29]) >= replay["qpos"].shape[1]:
        raise ValueError(f"{summary_path}: body qpos address exceeds replay qpos width")
    qpos_joint = replay["qpos"][:, [int(adr) for adr in qpos_adrs[:29]]]
    max_joint_diff = float(np.max(np.abs(qpos_joint - replay["measured_q"][:, :29])))
    if max_joint_diff > 1e-6:
        raise ValueError(f"{path}: measured_q columns do not match qpos body joint addresses")
    replay["max_joint_qpos_diff"] = max_joint_diff
    return replay


def playback_event(events: list[dict[str, str]]) -> dict[str, str]:
    for row in events:
        if row.get("event") in PLAYBACK_EVENTS:
            return row
    raise ValueError(f"missing playback event ({', '.join(PLAYBACK_EVENTS)})")


def timing_proof(run_dir: Path) -> dict:
    import numpy as np

    if run_dir.suffix == ".npz":
        data = np.load(run_dir, allow_pickle=False)
        timestamps = np.asarray(data["timestamps"], dtype=np.float64)
        phase_t = timestamps - timestamps[0]
        wall_t = timestamps
        sim_t = phase_t
        tag = str(data["tag"]) if "tag" in data else run_dir.stem
    else:
        replay = load_replay(run_dir)
        phase_t = replay["time_s"] - replay["time_s"][0]
        wall_t = replay["wall_time_s"] - replay["wall_time_s"][0]
        sim_t = replay["sim_time_s"] - replay["sim_time_s"][0]
        tag = str(run_dir)
    dt = np.diff(phase_t)
    wall_duration = float(wall_t[-1] - wall_t[0]) if wall_t.size else 0.0
    sim_duration = float(sim_t[-1] - sim_t[0]) if sim_t.size else 0.0
    phase_duration = float(phase_t[-1] - phase_t[0]) if phase_t.size else 0.0
    return {
        "tag": tag,
        "phase_dt_s": float(np.mean(dt)) if dt.size else 0.0,
        "phase_dt_std_s": float(np.std(dt)) if dt.size else 0.0,
        "phase_duration_s": phase_duration,
        "wall_duration_s": wall_duration,
        "sim_duration_s": sim_duration,
        "phase_eq_wall": bool(abs(phase_duration - wall_duration) <= 0.05),
        "phase_eq_sim": bool(abs(phase_duration - sim_duration) <= 0.05),
        "wall_eq_sim": bool(abs(wall_duration - sim_duration) <= 0.05),
    }

def validate_release_validation_logs(logs: Path) -> dict:
    root = logs / "release_validation"
    failures = []
    cases = {}

    no_control = root / "no_control_direct_release"
    try:
        events = load_events(no_control)
        release_t = event_sim_time(first_event(events, RELEASE_CONFIRMED_EVENT))
        rows = load_status_rows(no_control)
        fall_t = first_status_time(
            rows,
            lambda row: float(row["sim_time_s"]) >= release_t and float(row["base_z"]) < 0.25,
        )
        passed = fall_t is not None and fall_t - release_t <= 2.0
        if not passed:
            failures.append("release_validation/no_control_direct_release: robot did not fall within 2s after release")
        cases["no_control_direct_release"] = {
            "passed": passed,
            "event_log": str(find_event_log(no_control)),
            "status_log": str(no_control / "simulator_status.csv"),
            "replay_log": str(no_control / "replay.csv"),
            "release_sim_time_s": release_t,
            "fall_sim_time_s": fall_t,
            "fall_after_release_s": None if fall_t is None else fall_t - release_t,
        }
    except Exception as exc:
        failures.append(f"release_validation/no_control_direct_release: {exc}")
        cases["no_control_direct_release"] = {"passed": False, "error": str(exc)}

    control_case = root / "sonic_control_release_stop"
    try:
        events = load_events(control_case)
        release_t = event_sim_time(first_event(events, RELEASE_CONFIRMED_EVENT))
        stable_t = event_sim_time(first_event(events, "stable_released_control_elapsed"))
        stop_t = event_sim_time(first_event(events, "control_stopped"))
        rows = load_status_rows(control_case)
        stable_rows = status_window(rows, release_t, release_t + 5.0)
        stable_duration_ok = stable_t - release_t >= 5.0
        standing_ok = bool(stable_rows) and min(float(row["base_z"]) for row in stable_rows) >= 0.25
        cmd_active_ok = bool(stable_rows) and any(row.get("cmd_active") == "1" for row in stable_rows)
        fall_t = first_status_time(
            rows,
            lambda row: float(row["sim_time_s"]) >= stop_t and float(row["base_z"]) < 0.25,
        )
        stop_fall_ok = fall_t is not None and fall_t - stop_t <= 2.0
        passed = stable_duration_ok and standing_ok and cmd_active_ok and stop_fall_ok
        if not stable_duration_ok:
            failures.append("release_validation/sonic_control_release_stop: released CONTROL did not run for >5s")
        if not standing_ok:
            failures.append("release_validation/sonic_control_release_stop: robot fell during released CONTROL window")
        if not cmd_active_ok:
            failures.append("release_validation/sonic_control_release_stop: no active command evidence during stable window")
        if not stop_fall_ok:
            failures.append("release_validation/sonic_control_release_stop: robot did not fall within 2s after control stopped")
        cases["sonic_control_release_stop"] = {
            "passed": passed,
            "event_log": str(find_event_log(control_case)),
            "status_log": str(control_case / "simulator_status.csv"),
            "replay_log": str(control_case / "replay.csv"),
            "release_sim_time_s": release_t,
            "stable_elapsed_event_s": stable_t,
            "control_stopped_sim_time_s": stop_t,
            "fall_sim_time_s": fall_t,
            "fall_after_control_stop_s": None if fall_t is None else fall_t - stop_t,
            "stable_min_base_z": None if not stable_rows else min(float(row["base_z"]) for row in stable_rows),
            "stable_cmd_active_samples": sum(1 for row in stable_rows if row.get("cmd_active") == "1"),
        }
    except Exception as exc:
        failures.append(f"release_validation/sonic_control_release_stop: {exc}")
        cases["sonic_control_release_stop"] = {"passed": False, "error": str(exc)}

    return {
        "passed": not failures,
        "cases": cases,
        "failures": failures,
    }


def write_release_validation_artifact(logs: Path, artifacts: Path) -> tuple[dict, list[str]]:
    proof = validate_release_validation_logs(logs)
    (artifacts / "release_validation.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof, list(proof["failures"])


def validate_scene_contract(scene: Path) -> dict:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene))
    missing = []
    joint_qpos_addresses = []
    actuator_ids = []
    for joint_name in BODY_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            joint_name.removesuffix("_joint"),
        )
        if joint_id < 0 or actuator_id < 0:
            missing.append({"joint": joint_name, "joint_id": joint_id, "actuator_id": actuator_id})
            continue
        joint_qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        actuator_ids.append(int(actuator_id))
    scene_text = str(scene)
    failures = []
    if model.nq != 36 or model.nv != 35 or model.nu != 29:
        failures.append(f"scene has nq={model.nq}, nv={model.nv}, nu={model.nu}; expected 36/35/29")
    if missing:
        failures.append(f"scene is missing {len(missing)} expected 29DOF joints/actuators")
    if "43dof" in scene_text.lower() or "with_hand" in scene_text.lower():
        failures.append(f"scene path is not a pure 29DOF active benchmark plant: {scene}")
    return {
        "scene": str(scene),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "body_mass_kg": float(sum(model.body_mass)),
        "body_joint_names": BODY_JOINT_NAMES,
        "body_qpos_addresses": joint_qpos_addresses,
        "body_actuator_ids": actuator_ids,
        "missing": missing,
        "passed": not failures,
        "failures": failures,
    }


def validate_single_robot_interface(logs: Path, artifacts: Path) -> tuple[dict, list[str]]:
    failures = []
    run_scenes: dict[str, str] = {}
    run_joint_names: dict[str, list[str]] = {}
    run_qpos_addresses: dict[str, list[int]] = {}
    run_dirs = [expected_run_dir(logs, policy, motion) for motion in MOTIONS for policy in POLICIES]
    run_dirs.extend([
        logs / "release_validation" / "no_control_direct_release",
        logs / "release_validation" / "sonic_control_release_stop",
    ])
    for run_dir in run_dirs:
        summary_path = run_dir / "sim_bridge_summary.json"
        tag = str(run_dir.relative_to(logs)) if run_dir.is_relative_to(logs) else str(run_dir)
        if not summary_path.exists():
            failures.append(f"{tag}: missing sim_bridge_summary.json")
            continue
        summary = json.loads(summary_path.read_text())
        scene = str(summary.get("scene", ""))
        run_scenes[tag] = scene
        joint_names = summary.get("body_joint_names")
        qpos_addresses = summary.get("body_qpos_addresses")
        run_joint_names[tag] = joint_names if isinstance(joint_names, list) else []
        run_qpos_addresses[tag] = qpos_addresses if isinstance(qpos_addresses, list) else []
        if joint_names != BODY_JOINT_NAMES:
            failures.append(f"{tag}: simulator joint order metadata does not match Unitree G1 29DOF body order")
        if not isinstance(qpos_addresses, list) or len(qpos_addresses) != 29:
            failures.append(f"{tag}: simulator did not log 29 body qpos addresses")

        lowcmd_path = run_dir / "lowcmd.csv"
        if lowcmd_path.exists():
            header = lowcmd_path.read_text(errors="replace").splitlines()[0].split(",")
            if "cmd_q_28" not in header or "cmd_q_29" in header:
                failures.append(f"{tag}: low-level command log is not exactly 29 commanded body joints")

    unique_scenes = sorted(set(run_scenes.values()))
    if len(unique_scenes) != 1:
        failures.append(f"policy/release runs used multiple simulator scenes: {unique_scenes}")

    scene_contract = None
    if unique_scenes:
        try:
            scene_contract = validate_scene_contract(logged_scene_path({"summary": {"scene": unique_scenes[0]}}))
            failures.extend(scene_contract["failures"])
        except Exception as exc:
            failures.append(f"single 29DOF scene contract unavailable: {exc}")

    proof = {
        "passed": not failures,
        "expected_active_scene": str(COMMON_SIM_SCENE),
        "unique_logged_scenes": unique_scenes,
        "scene_contract": scene_contract,
        "run_scenes": run_scenes,
        "run_joint_names": run_joint_names,
        "run_qpos_addresses": run_qpos_addresses,
        "real_robot_swap_claim": (
            "Policies communicate through the same Unitree G1 29DOF low-level state/command "
            "surface; replacing the simulator with a real 29DOF endpoint must not change "
            "policy deploy code, joint order, or orchestration other than disabling "
            "simulator-only support/release controls."
        ),
        "failures": failures,
    }
    (artifacts / "single_robot_interface.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof, failures


def report(args: argparse.Namespace) -> int:
    logs = Path(args.logs)
    artifacts = Path(args.artifacts)
    clean_report_artifacts(artifacts)
    impossibility_path = logs / GLOBAL_IMPOSSIBILITY
    if impossibility_path.exists():
        raise RuntimeError(f"{impossibility_path}: documented-impossibility mode is disabled")
    rows = []
    failures = []
    timing_rows = []
    for motion in MOTIONS:
        row: dict[str, object] = {"motion": motion}
        for policy in POLICIES:
            run_dir = expected_run_dir(logs, policy, motion)
            try:
                ref, hz = load_reference(policy, motion)
                t, q = load_tracked(run_dir)
                t_eval, q_eval, playback_offset = tracking_window(run_dir, t, q)
                metrics = aligned_rmse(ref, hz, t_eval, q_eval)
                if playback_offset is not None:
                    metrics["playback_offset_s"] = playback_offset
                try:
                    metrics.update(validate_release_order(run_dir))
                except Exception as exc:
                    metrics["passed_release_gate"] = False
                    metrics["release_gate_error"] = str(exc)
                    failures.append(f"{policy}/{motion}: {exc}")
                metrics["passed_rmse_gate"] = metrics["mean_joint_rmse_rad"] < 0.2
                row[policy] = metrics
                if not metrics["passed_rmse_gate"]:
                    if policy == "sonic":
                        failures.append(f"{policy}/{motion}: RMSE {metrics['mean_joint_rmse_rad']:.4f}")
            except Exception as exc:
                row[policy] = {"error": str(exc), "passed_rmse_gate": False}
                if policy == "sonic":
                    failures.append(f"{policy}/{motion}: {exc}")
            try:
                timing_rows.append(timing_proof(run_dir))
            except Exception as exc:
                failures.append(f"{policy}/{motion}: timing proof unavailable: {exc}")
        rows.append(row)
    sonic_rmse_passed = all(row["sonic"].get("passed_rmse_gate") is True for row in rows)
    holomotion_rmse_passed_count = sum(
        1 for row in rows if row["holomotion"].get("passed_rmse_gate") is True
    )
    all_rmse_passed = sonic_rmse_passed and holomotion_rmse_passed_count >= 8
    all_release_gates_passed = all(
        row[policy].get("passed_release_gate") is True
        for row in rows
        for policy in POLICIES
    )
    if holomotion_rmse_passed_count < 8:
        failures.append(
            f"holomotion: {holomotion_rmse_passed_count}/10 motions passed RMSE gate, need at least 8"
        )
    summary = {
        "motions": MOTIONS,
        "policies": POLICIES,
        "results": rows,
        "all_rmse_passed": all_rmse_passed,
        "sonic_rmse_passed": sonic_rmse_passed,
        "holomotion_rmse_passed_count": holomotion_rmse_passed_count,
        "holomotion_rmse_required_count": 8,
        "all_release_gates_passed": all_release_gates_passed,
        "failures": failures,
    }
    (artifacts / "phase_time_proof.json").write_text(json.dumps(timing_rows, indent=2) + "\n")
    write_metric_csvs(artifacts, summary)
    _, release_failures = write_release_validation_artifact(logs, artifacts)
    failures.extend(release_failures)
    single_robot, single_robot_failures = validate_single_robot_interface(logs, artifacts)
    failures.extend(single_robot_failures)
    summary["single_robot_interface_passed"] = single_robot["passed"]
    failures.extend(make_all_comparison_videos(logs, artifacts, rows))
    summary["failures"] = failures
    (artifacts / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(artifacts / "report.md", summary)
    return 0 if not failures else 1


def write_metric_csvs(artifacts: Path, summary: dict) -> None:
    with (artifacts / "rmse_summary.csv").open("w", newline="") as f:
        fieldnames = [
            "tag",
            "policy",
            "motion",
            "tracking_delay_s",
            "mean_joint_rmse_rad",
            "samples",
            "passed_rmse_gate",
            "passed_release_gate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["results"]:
            for policy in POLICIES:
                item = row[policy]
                writer.writerow({
                    "tag": log_tag(policy, row["motion"]),
                    "policy": policy,
                    "motion": row["motion"],
                    "tracking_delay_s": item.get("tracking_delay_s", ""),
                    "mean_joint_rmse_rad": item.get("mean_joint_rmse_rad", ""),
                    "samples": item.get("samples", ""),
                    "passed_rmse_gate": item.get("passed_rmse_gate", False),
                    "passed_release_gate": item.get("passed_release_gate", False),
                })
    with (artifacts / "per_joint_rmse.csv").open("w", newline="") as f:
        fieldnames = ["tag", "policy", "motion", "metric", *[f"joint_{i}" for i in range(29)]]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["results"]:
            for policy in POLICIES:
                item = row[policy]
                if "joint_rmse_rad" not in item:
                    continue
                values = {
                    "tag": log_tag(policy, row["motion"]),
                    "policy": policy,
                    "motion": row["motion"],
                    "metric": "rmse_rad",
                }
                for i, value in enumerate(item["joint_rmse_rad"][:29]):
                    values[f"joint_{i}"] = value
                writer.writerow(values)


def clean_report_artifacts(artifacts: Path) -> None:
    artifacts.mkdir(parents=True, exist_ok=True)
    generated = [artifacts / name for name in REPORT_OUTPUT_FILES]
    generated.extend(artifacts / f"{motion}_comparison.mp4" for motion in MOTIONS)
    for path in generated:
        if path.is_file() or path.is_symlink():
            path.unlink()


def make_all_comparison_videos(logs: Path, artifacts: Path, rows: list[dict]) -> list[str]:
    failures = []
    videos = []
    for row in rows:
        motion = row["motion"]
        try:
            path = make_comparison_video(logs, artifacts, motion, row)
            videos.append({"motion": motion, "path": str(path)})
        except Exception as exc:
            failures.append(f"{motion}: comparison video unavailable: {exc}")
    (artifacts / "comparison_videos.json").write_text(json.dumps(videos, indent=2) + "\n")
    return failures


def overlay_text(image, lines: list[str]):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    pil = Image.fromarray(image)
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    line_h = 14
    pad = 6
    box_h = pad * 2 + line_h * len(lines)
    draw.rectangle((0, 0, pil.size[0], box_h), fill=(0, 0, 0, 170))
    y = pad
    for line in lines:
        draw.text((pad, y), line, fill=(255, 255, 255, 255), font=font)
        y += line_h
    return np.asarray(Image.alpha_composite(pil.convert("RGBA"), overlay).convert("RGB"))


def nearest_replay_index(replay: dict, sim_time_s: float) -> int:
    import numpy as np

    times = replay["sim_time_s"]
    index = int(np.searchsorted(times, sim_time_s, side="left"))
    if index <= 0:
        return 0
    if index >= times.size:
        return int(times.size - 1)
    before = index - 1
    return before if abs(times[before] - sim_time_s) <= abs(times[index] - sim_time_s) else index


def validate_replay_window(policy: str, motion: str, run_dir: Path, replay: dict, events: list[dict[str, str]]) -> dict:
    control = first_event(events, CONTROL_EVENT)
    release = first_event(events, RELEASE_CONFIRMED_EVENT)
    playback = playback_event(events)
    control_t = event_sim_time(control)
    release_t = event_sim_time(release)
    playback_t = event_sim_time(playback)
    sim_t = replay["sim_time_s"]
    support = replay["support_active"]
    if not (float(sim_t[0]) <= control_t <= float(sim_t[-1])):
        raise ValueError(f"{policy}/{motion}: CONTROL event outside replay interval")
    if not (float(sim_t[0]) <= release_t <= float(sim_t[-1])):
        raise ValueError(f"{policy}/{motion}: release event outside replay interval")
    if not (float(sim_t[0]) <= playback_t <= float(sim_t[-1])):
        raise ValueError(f"{policy}/{motion}: playback event outside replay interval")
    if not bool(((sim_t < release_t) & (support == 1)).any()):
        raise ValueError(f"{policy}/{motion}: replay lacks support-active frames before release")
    if not bool(((sim_t >= release_t) & (support == 0)).any()):
        raise ValueError(f"{policy}/{motion}: replay lacks released frames after release")
    if not bool(((sim_t >= control_t) & (sim_t <= release_t)).any()):
        raise ValueError(f"{policy}/{motion}: replay lacks CONTROL-to-release interval")
    return {
        "control_sim_time_s": control_t,
        "release_sim_time_s": release_t,
        "playback_sim_time_s": playback_t,
        "playback_event": playback.get("event"),
    }


def render_replay_image(mujoco, model, data, renderer, camera, qpos):
    qpos_width = min(len(qpos), model.nq)
    if qpos_width < model.nq:
        raise ValueError(f"replay qpos has {qpos_width} columns, model requires {model.nq}")
    data.qpos[:] = qpos[: model.nq]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    camera.lookat[:] = [float(data.qpos[0]), float(data.qpos[1]), max(0.7, float(data.qpos[2]))]
    renderer.update_scene(data, camera=camera)
    return renderer.render()


def validate_video_file(path: Path) -> None:
    import imageio.v2 as imageio
    import numpy as np

    if not path.exists() or path.stat().st_size <= 0:
        raise ValueError(f"{path}: video missing or empty")
    reader = imageio.get_reader(path)
    try:
        frame = reader.get_data(0)
    finally:
        reader.close()
    if frame.size == 0 or float(np.std(frame)) < 1.0:
        raise ValueError(f"{path}: video appears blank")


def make_comparison_video(logs: Path, artifacts: Path, motion: str, row: dict) -> Path:
    import imageio.v2 as imageio
    import numpy as np
    import mujoco

    sides = []
    for policy in ["holomotion", "sonic"]:
        run_dir = expected_run_dir(logs, policy, motion)
        replay = load_replay(run_dir)
        events = load_events(run_dir)
        window = validate_replay_window(policy, motion, run_dir, replay, events)
        scene = logged_scene_path(replay)
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 3.0
        camera.azimuth = 135.0
        camera.elevation = -16.0
        sides.append({
            "policy": policy,
            "run_dir": run_dir,
            "replay": replay,
            "events": events,
            "window": window,
            "model": model,
            "data": data,
            "renderer": renderer,
            "camera": camera,
            "metrics": row[policy],
        })

    motion_duration = policy_motion_duration("sonic", motion)
    pre_roll = max(
        side["window"]["playback_sim_time_s"] - side["window"]["release_sim_time_s"] + 0.50
        for side in sides
    )
    pre_roll = max(pre_roll, 1.0)
    max_available_pre = min(
        side["window"]["playback_sim_time_s"] - float(side["replay"]["sim_time_s"][0])
        for side in sides
    )
    if max_available_pre + 1e-6 < pre_roll:
        raise ValueError(
            f"{motion}: replay logs start too late for support-release preroll "
            f"(need {pre_roll:.3f}s, have {max_available_pre:.3f}s)"
        )

    fps = 20
    frame_times = np.arange(-pre_roll, motion_duration + 1e-9, 1.0 / fps)
    out = artifacts / f"{motion}_comparison.mp4"
    with imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=16) as writer:
        for phase_t in frame_times:
            rendered = []
            for side in sides:
                replay = side["replay"]
                sim_time = side["window"]["playback_sim_time_s"] + float(phase_t)
                idx = nearest_replay_index(replay, sim_time)
                image = render_replay_image(
                    mujoco,
                    side["model"],
                    side["data"],
                    side["renderer"],
                    side["camera"],
                    replay["qpos"][idx],
                )
                metrics = side["metrics"]
                release_phase = side["window"]["release_sim_time_s"] - side["window"]["playback_sim_time_s"]
                support = "support" if int(replay["support_active"][idx]) else "released"
                event = "pre-motion"
                if abs(phase_t) <= 0.05:
                    event = "ACTING START"
                elif phase_t >= motion_duration - 0.05:
                    event = "ACTING END"
                elif abs(phase_t - release_phase) <= 0.05:
                    event = "SUPPORT RELEASE"
                lines = [
                    f"{side['policy']} | {motion}",
                    (
                        f"phase={phase_t:+.2f}s sim={replay['sim_time_s'][idx]:.2f}s "
                        f"{support} {event}"
                    ),
                    (
                        f"RMSE={metrics.get('mean_joint_rmse_rad', float('nan')):.3f} "
                        f"delay={metrics.get('tracking_delay_s', float('nan')):.3f}s "
                        f"release={release_phase:+.2f}s"
                    ),
                ]
                rendered.append(overlay_text(image, lines))
            writer.append_data(np.concatenate(rendered, axis=1))
    for side in sides:
        side["renderer"].close()
    validate_video_file(out)
    return out


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# WBC Benchmark Report",
        "",
        "| Motion | SONIC RMSE | SONIC Delay | SONIC Release | HoloMotion RMSE | HoloMotion Delay | HoloMotion Release |",
        "| --- | ---: | ---: | :---: | ---: | ---: | :---: |",
    ]
    for row in summary["results"]:
        vals = []
        for policy in POLICIES:
            item = row[policy]
            vals.extend([
                f"{item.get('mean_joint_rmse_rad', float('nan')):.4f}" if "mean_joint_rmse_rad" in item else "missing",
                f"{item.get('tracking_delay_s', float('nan')):.3f}" if "tracking_delay_s" in item else "missing",
                "pass" if item.get("passed_release_gate") else "fail",
            ])
        lines.append(
            f"| {row['motion']} | {vals[0]} | {vals[1]} | {vals[2]} | "
            f"{vals[3]} | {vals[4]} | {vals[5]} |"
        )
    lines.extend([
        "",
        f"All RMSE gates passed: `{summary['all_rmse_passed']}`",
        f"All release-order gates passed: `{summary['all_release_gates_passed']}`",
        f"Single 29DOF robot interface passed: `{summary.get('single_robot_interface_passed', False)}`",
        "",
    ])
    if summary["failures"]:
        lines.append("## Failures")
        lines.extend(f"- {failure}" for failure in summary["failures"])
    path.write_text("\n".join(lines) + "\n")
