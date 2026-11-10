#!/usr/bin/env python3
"""Interactively replay SAC and Jacobian-DLS rollouts side by side.

This viewer only assigns recorded joint angles and calls ``mj_forward``.  It
does not run physics, infer an RL policy, or communicate with robot hardware.
On macOS, launch it with ``mjpython`` so MuJoCo can create the GUI window.
"""
from __future__ import annotations

import argparse
import csv
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

import real_to_sim as r2s


POLICIES = ("sac", "jacobian_dls")
PREFIXES = {"sac": "sac_", "jacobian_dls": "dls_"}
COLORS = {
    "sac": np.array([0.12, 0.47, 0.85, 1.0], dtype=np.float32),
    "jacobian_dls": np.array([0.95, 0.42, 0.10, 1.0], dtype=np.float32),
}
ROBOT_RGBA = np.array([0.90, 0.90, 0.90, 1.0], dtype=np.float32)
Y_OFFSETS = {"sac": 0.22, "jacobian_dls": -0.22}
JOINT_COLUMNS = tuple(f"joint_{i}" for i in range(1, 6))


def load_rollouts(path: Path) -> dict[str, list[dict[str, float]]]:
    """Load and strictly validate the two recorded, zero-based trajectories."""
    rows = {policy: [] for policy in POLICIES}
    required = {"policy", "step", *JOINT_COLUMNS,
                "tcp_x", "tcp_y", "tcp_z",
                "target_x", "target_y", "target_z"}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")
        for line, raw in enumerate(reader, start=2):
            policy = raw["policy"]
            if policy not in rows:
                continue
            try:
                row = {name: float(raw[name]) for name in
                       (*JOINT_COLUMNS, "tcp_x", "tcp_y", "tcp_z",
                        "target_x", "target_y", "target_z")}
                row["step"] = int(raw["step"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value on CSV line {line}") from exc
            if not all(math.isfinite(value) for value in row.values()):
                raise ValueError(f"Non-finite value on CSV line {line}")
            rows[policy].append(row)

    for policy, trajectory in rows.items():
        if not trajectory:
            raise ValueError(f"CSV has no {policy} rows")
        trajectory.sort(key=lambda row: row["step"])
        steps = [row["step"] for row in trajectory]
        if steps != list(range(len(trajectory))):
            raise ValueError(f"{policy} steps must be unique and contiguous from zero")

    initial = lambda policy: np.array(
        [rows[policy][0][name] for name in JOINT_COLUMNS])
    if not np.array_equal(initial("sac"), initial("jacobian_dls")):
        raise ValueError("SAC and DLS initial joint angles do not match")
    targets = lambda policy: np.array(
        [rows[policy][0][f"target_{axis}"] for axis in "xyz"])
    if not np.array_equal(targets("sac"), targets("jacobian_dls")):
        raise ValueError("SAC and DLS target positions do not match")
    return rows


def build_scene(rollouts, urdf: Path):
    """Attach two prefixed copies of the visual URDF to one MuJoCo scene."""
    root = mujoco.MjSpec()
    root.worldbody.add_geom(
        name="viewer_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0, 0, -0.026], size=[0.75, 0.65, 0.01],
        rgba=[0.88, 0.88, 0.88, 1], contype=0, conaffinity=0)

    for policy in POLICIES:
        y_offset = Y_OFFSETS[policy]
        frame = root.worldbody.add_frame(
            name=f"{PREFIXES[policy]}origin", pos=[0, y_offset, 0])
        robot = mujoco.MjSpec.from_file(str(urdf))
        robot.compiler.discardvisual = False
        root.attach(robot, prefix=PREFIXES[policy], frame=frame)
        color = COLORS[policy]
        root.worldbody.add_geom(
            name=f"{PREFIXES[policy]}platform",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.08, y_offset, -0.015], size=[0.25, 0.16, 0.01],
            rgba=[*color[:3], 0.35], contype=0, conaffinity=0)
        target = np.array(
            [rollouts[policy][0][f"target_{axis}"] for axis in "xyz"])
        target[1] += y_offset
        root.worldbody.add_geom(
            name=f"{PREFIXES[policy]}target",
            type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=target,
            size=[0.012, 0, 0], rgba=[*color[:3], 0.9],
            contype=0, conaffinity=0)

    model = root.compile()
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))
        if body_name and body_name.startswith(tuple(PREFIXES.values())):
            # Detach any imported material/texture so it cannot override white.
            model.geom_matid[geom_id] = -1
            model.geom_rgba[geom_id] = ROBOT_RGBA
    data = mujoco.MjData(model)
    qpos_addresses = {}
    tcp_body_ids = {}
    for policy in POLICIES:
        prefix = PREFIXES[policy]
        addresses = []
        for joint_name in r2s.JOINTS:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, prefix + joint_name)
            if joint_id < 0:
                raise ValueError(f"Attached model is missing {prefix + joint_name}")
            addresses.append(int(model.jnt_qposadr[joint_id]))
        qpos_addresses[policy] = np.array(addresses)
        tcp_body_ids[policy] = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, prefix + "gripper_link")
        if tcp_body_ids[policy] < 0:
            raise ValueError(f"Attached model is missing {prefix}gripper_link")
    return model, data, qpos_addresses, tcp_body_ids


def apply_frame(model, data, qpos_addresses, rollouts, frame: int):
    """Apply one synchronized frame, holding a completed policy at its end."""
    for policy in POLICIES:
        trajectory = rollouts[policy]
        row = trajectory[min(frame, len(trajectory) - 1)]
        data.qpos[qpos_addresses[policy]] = [row[name] for name in JOINT_COLUMNS]
    mujoco.mj_forward(model, data)


def validate_kinematics(model, data, qpos_addresses, tcp_body_ids, rollouts):
    """Check every replayed TCP against the coordinates recorded in the CSV."""
    maximum_error = 0.0
    max_frame = max(len(rows) for rows in rollouts.values()) - 1
    for frame in range(max_frame + 1):
        apply_frame(model, data, qpos_addresses, rollouts, frame)
        for policy in POLICIES:
            row = rollouts[policy][min(frame, len(rollouts[policy]) - 1)]
            expected = np.array([row[f"tcp_{axis}"] for axis in "xyz"])
            expected[1] += Y_OFFSETS[policy]
            actual = np.asarray(data.xpos[tcp_body_ids[policy]])
            maximum_error = max(maximum_error,
                                float(np.linalg.norm(actual - expected)))
    apply_frame(model, data, qpos_addresses, rollouts, 0)
    if maximum_error > 1e-8:
        raise ValueError(
            f"Reconstructed TCP differs from CSV by {maximum_error:.3g} m")
    return maximum_error


@dataclass
class PlaybackState:
    frame: int = 0
    paused: bool = False
    speed: float = 1.0

    def __post_init__(self):
        self.lock = threading.Lock()


def make_key_callback(state: PlaybackState, max_frame: int):
    def on_key(key: int):
        with state.lock:
            if key == 32:  # Space
                state.paused = not state.paused
            elif key in (ord("R"), ord("r")):
                state.frame = 0
                state.paused = False
            elif key == 262:  # Right arrow
                state.frame = min(state.frame + 1, max_frame)
                state.paused = True
            elif key == 263:  # Left arrow
                state.frame = max(state.frame - 1, 0)
                state.paused = True
            elif key in (ord("+"), ord("="), 334):
                state.speed = min(state.speed * 1.25, 8.0)
            elif key in (ord("-"), ord("_"), 333):
                state.speed = max(state.speed / 1.25, 0.125)
    return on_key


def add_trajectory_lines(viewer, rollouts):
    """Add thin, static TCP path lines to the viewer's user scene."""
    segments = sum(max(0, len(rows) - 1) for rows in rollouts.values())
    if segments > viewer.user_scn.maxgeom:
        print("Warning: viewer user-scene capacity is too small; paths are hidden")
        return
    viewer.user_scn.ngeom = 0
    for policy in POLICIES:
        points = np.array([[row[f"tcp_{axis}"] for axis in "xyz"]
                           for row in rollouts[policy]])
        points[:, 1] += Y_OFFSETS[policy]
        for start, end in zip(points[:-1], points[1:]):
            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(
                geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                np.eye(3).ravel(), COLORS[policy])
            mujoco.mjv_connector(
                geom, mujoco.mjtGeom.mjGEOM_LINE, 2.0, start, end)
            viewer.user_scn.ngeom += 1


def run_viewer(model, data, qpos_addresses, rollouts, fps: float):
    import mujoco.viewer

    max_frame = max(len(rows) for rows in rollouts.values()) - 1
    state = PlaybackState()
    callback = make_key_callback(state, max_frame)
    print("\nControls:")
    print("  Mouse left/right/middle drag: rotate / move; scroll: zoom")
    print("  Space: play/pause    R: restart and play")
    print("  Left/Right arrows: previous/next step (and pause)")
    print("  +/-: slower/faster  Esc: close")
    print("  Blue left robot: SAC    Orange right robot: Jacobian DLS\n")

    with mujoco.viewer.launch_passive(
            model, data, key_callback=callback,
            show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.lookat[:] = [0.13, 0.0, 0.13]
        viewer.cam.distance = 0.72
        # View from -X so +Y (SAC) appears on the left of the screen.
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -18
        with viewer.lock():
            add_trajectory_lines(viewer, rollouts)
        accumulator = 0.0
        previous_time = time.monotonic()
        previous_frame = -1
        while viewer.is_running():
            now = time.monotonic()
            elapsed = min(now - previous_time, 0.25)
            previous_time = now
            with state.lock:
                if not state.paused and state.frame < max_frame:
                    accumulator += elapsed * fps * state.speed
                    advances = int(accumulator)
                    if advances:
                        state.frame = min(state.frame + advances, max_frame)
                        accumulator -= advances
                elif state.frame >= max_frame:
                    state.paused = True
                frame, paused, speed = state.frame, state.paused, state.speed
            if frame != previous_frame:
                with viewer.lock():
                    apply_frame(model, data, qpos_addresses, rollouts, frame)
                previous_frame = frame
            viewer.set_texts((
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "SAC (blue, left)  |  Jacobian DLS (orange, right)",
                f"step {frame}/{max_frame}  "
                f"{'PAUSED' if paused else 'PLAYING'}  speed {speed:.2f}x",
            ))
            viewer.sync()
            time.sleep(1 / 120)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=r2s.DEFAULT_URDF)
    parser.add_argument("--fps", type=float, default=4.0,
                        help="recorded steps per second (default: 4)")
    parser.add_argument("--validate-only", action="store_true",
                        help="validate the full replay without opening a GUI")
    args = parser.parse_args()
    if args.fps <= 0 or not math.isfinite(args.fps):
        parser.error("--fps must be a finite positive number")
    if not args.csv.is_file():
        parser.error(f"CSV was not found: {args.csv}")
    if not args.urdf.is_file():
        parser.error(f"URDF was not found: {args.urdf}")

    rollouts = load_rollouts(args.csv)
    model, data, addresses, tcp_ids = build_scene(
        rollouts, args.urdf.resolve(strict=True))
    error = validate_kinematics(model, data, addresses, tcp_ids, rollouts)
    print(f"Validated {model.nq} qpos values and all recorded frames; "
          f"maximum TCP reconstruction error={error:.3g} m")
    if args.validate_only:
        return
    run_viewer(model, data, addresses, rollouts, args.fps)


if __name__ == "__main__":
    main()
