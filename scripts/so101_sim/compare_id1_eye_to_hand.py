#!/usr/bin/env python3
"""Offline real-photo vs MuJoCo Side camera view for one captured SO-101 pose.

Uses the preliminary ID1 center-only calibration and existing manifest PNG.
NO robot/OBS connections. DOES NOT apply calibration to the real robot.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "so101_capture"))
import real_to_sim as r2s


def mujoco_camera_pose(T_base_cv):
    """Convert OpenCV optical (x right,y down,z forward) to MuJoCo (x right,y up,-z forward)."""
    matrix = np.asarray(T_base_cv, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected finite T_base_camera 4x4")
    R = matrix[:3, :3]
    if (not np.allclose(R.T @ R, np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(R), 1, atol=1e-3)
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)):
        raise ValueError("Invalid camera rigid transform")
    R_mj = R @ np.diag([1., -1., -1.])
    q_xyzw = Rotation.from_matrix(R_mj).as_quat()
    return matrix[:3, 3].tolist(), q_xyzw[[3, 0, 1, 2]].tolist()


def fovy_from_intrinsics(fy, height):
    if not (math.isfinite(fy) and fy > 0 and height > 0):
        raise ValueError("Invalid focal length/height")
    return math.degrees(2 * math.atan(height / (2 * fy)))


def marker_center_px(marker):
    points = np.asarray(marker["corners_px"], dtype=float).reshape(4, 2)
    return np.mean(points, axis=0)


def add_simple_studio_light(spec):
    """Add one fixed soft-ish white light for easier visual comparison.

    This is a lightweight display aid, not measured real-world illumination.
    The source URDF, camera transform, and recorded data are unchanged.
    """
    spec.worldbody.add_light(
        name="r2s_preview_fill_light",
        pos=[0.1, -0.2, 1.2],
        dir=[-0.1, 0.2, -1.0],
        ambient=[0.30, 0.30, 0.30],
        diffuse=[0.85, 0.85, 0.85],
        specular=[0.10, 0.10, 0.10],
        castshadow=False,
    )


def apply_white_follower_preview(model):
    """Change only the compiled preview model's yellow printed-plastic material.

    The source URDF remains unchanged; motors (dark material) remain dark.
    """
    import mujoco

    white = np.array([0.92, 0.92, 0.90, 1.0], dtype=float)
    mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "3d_printed")
    if mid >= 0:
        model.mat_rgba[mid] = white
    # MuJoCo's URDF importer can instead bake colors into individual geoms.
    yellow = np.array([1.0, 0.82, 0.12], dtype=float)
    for idx in range(model.ngeom):
        color = model.geom_rgba[idx]
        if np.allclose(color[:3], yellow, atol=0.025):
            model.geom_rgba[idx] = white


def add_backdrop(spec, T_base_cv):
    """Large pale wall behind the arm in camera view, to avoid black void.

    This wall is a simple visual backdrop, not part of the real workcell.
    """
    import mujoco

    T = np.asarray(T_base_cv, dtype=float)
    center = T[:3, 3] + T[:3, :3] @ np.array([0., 0., 1.8])
    R_mj = T[:3, :3] @ np.diag([1., -1., -1.])
    quat_xyzw = Rotation.from_matrix(R_mj).as_quat()
    spec.worldbody.add_geom(
        name="r2s_preview_pale_backdrop",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=center.tolist(),
        quat=quat_xyzw[[3, 0, 1, 2]].tolist(),
        size=[3.0, 3.0, 0.01],
        rgba=[0.73, 0.76, 0.79, 1.0],
        contype=0, conaffinity=0,
    )


def workcell_shift_left_front(T_base_cv, left_m=0.08, front_m=0.10):
    """Ground-plane translation left/toward camera, not arbitrary base X/Y.

    Project camera right and forward onto base XY; negative camera right is
    screen-left, negative camera forward is toward the viewer. Preserve Z.
    """
    if not (math.isfinite(left_m) and math.isfinite(front_m)):
        raise ValueError("Workcell offsets must be finite")
    R = np.asarray(T_base_cv, dtype=float)[:3, :3]
    if R.shape != (3, 3) or not np.isfinite(R).all():
        raise ValueError("Invalid camera rotation")
    left = -R[:2, 0].copy()
    front = -R[:2, 2].copy()
    for direction in (left, front):
        norm = np.linalg.norm(direction)
        if norm < 1e-7:
            raise ValueError("Camera direction cannot be projected onto workcell XY")
        direction /= norm
    return np.r_[left_m * left + front_m * front, 0.]


def add_approximate_workcell(spec, shift_xyz=(0., 0., 0.),
                             table_extra_xyz=(0., 0., 0.), table_yaw_deg=0.):
    """Photo-approximate Muhi, clear cup and 65x60cm table from earlier Colab scene.

    These placements are historic, unmeasured approximations, not calibration.
    No grasp physics: all geoms are static and collision-disabled.
    """
    import mujoco

    world = spec.worldbody
    shift = np.asarray(shift_xyz, dtype=float)
    if shift.shape != (3,) or not np.isfinite(shift).all():
        raise ValueError("Invalid workcell shift")
    extra = np.asarray(table_extra_xyz, dtype=float)
    if extra.shape != (3,) or not np.isfinite(extra).all() or not math.isfinite(table_yaw_deg):
        raise ValueError("Invalid table-only translation/yaw")
    yaw_rad = math.radians(table_yaw_deg)
    c, sn = math.cos(yaw_rad), math.sin(yaw_rad)
    yaw_rot = np.array([[c, -sn], [sn, c]], dtype=float)
    yaw_quat = [math.cos(yaw_rad / 2), 0., 0., math.sin(yaw_rad / 2)]

    def geom(name, shape, pos, size, rgba, *, quat=None, table_part=False):
        p = np.asarray(pos, dtype=float).copy()
        if table_part:
            p[:2] = yaw_rot @ p[:2]
            p += extra
            if quat is not None:
                raise ValueError("Table parts should not have their own rotation")
            quat = yaw_quat
        kwargs = dict(name=name, type=shape, pos=(p + shift).tolist(), size=size, rgba=rgba,
                      contype=0, conaffinity=0)
        if quat is not None:
            kwargs["quat"] = quat
        world.add_geom(**kwargs)

    # The earlier notebook used a 0.65x0.60x0.04m desk with its top at z=-0.005m.
    geom("r2s_preview_table", mujoco.mjtGeom.mjGEOM_BOX,
         [0, 0, -.025], [.325, .30, .02], [.63, .46, .33, 1.0], table_part=True)
    # Contrast between the tabletop and the background helps identify the edge.
    for x in (-.29, .29):
        for y in (-.26, .26):
            geom(f"r2s_table_leg_{'p' if x>0 else 'm'}{'p' if y>0 else 'm'}",
                 mujoco.mjtGeom.mjGEOM_BOX, [x, y, -.225],
                 [.018, .018, .18], [.44, .35, .30, 1.0], table_part=True)

    # Earlier photo-based workcell: bottle ~100mm long, 38mm diameter.
    # Notebook placed bottle horizontally along the y-axis.
    x, y = .18, .11
    side_quat = [math.sqrt(.5), math.sqrt(.5), 0., 0.]
    geom("r2s_muhi_body", mujoco.mjtGeom.mjGEOM_CYLINDER,
         [x, y-.0175, .019], [.019, .0325, 0],
         [.95, .95, .94, 1.], quat=side_quat)
    geom("r2s_muhi_cap", mujoco.mjtGeom.mjGEOM_CYLINDER,
         [x, y+.0325, .019], [.019, .0175, 0],
         [.82, .89, .96, 1.], quat=side_quat)

    # Earlier transparent container: diameter ~100mm, height ~120mm.
    cx, cy, radius, h = .18, -.12, .05, .12
    geom("r2s_cup_base", mujoco.mjtGeom.mjGEOM_CYLINDER,
         [cx, cy, .001], [radius, .001, 0], [.70, .85, .92, .55])
    for i in range(16):
        a = 2 * math.pi * i / 16
        px, py = cx+(radius-.001)*math.cos(a), cy+(radius-.001)*math.sin(a)
        quat = [math.cos((a+math.pi/2)/2), 0., 0.,
                math.sin((a+math.pi/2)/2)]
        geom(f"r2s_cup_wall_{i:02d}", mujoco.mjtGeom.mjGEOM_BOX,
             [px, py, .055], [.010, .0015, h/2],
             [.70, .86, .92, .62], quat=quat)


def project_reference_points(T_base_cv, cam, shift_xyz=(0., 0., 0.),
                             table_extra_xyz=(0., 0., 0.)):
    """Project approximate workcell reference points; diagnostic only."""
    T = np.asarray(T_base_cv, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    points = {
        "table_center": [0., 0., -.005],
        "muhi_center": [.18, .11, .019],
        "cup_center": [.18, -.12, .06],
    }
    result = {}
    shift = np.asarray(shift_xyz, dtype=float)
    table_extra = np.asarray(table_extra_xyz, dtype=float)
    for name, p in points.items():
        point = np.asarray(p, dtype=float) + shift
        if name == "table_center":
            point = point + table_extra
        xyz = R.T @ (point - t)
        if xyz[2] <= .01:
            result[name] = {"in_front": False}
            continue
        u = float(cam["fx"] * xyz[0] / xyz[2] + cam["cx"])
        v = float(cam["fy"] * xyz[1] / xyz[2] + cam["cy"])
        result[name] = {
            "pixel_approx": [round(u, 1), round(v, 1)],
            "in_frame": 0 <= u < cam["width"] and 0 <= v < cam["height"],
            "depth_m": round(float(xyz[2]), 3),
        }
    return result


def render_one(manifest_path, extrinsics_path, intrinsics_path, urdf_path, output_dir,
               marker_id=1, sample_number=None, lighting='simple', workcell='simple',
               workcell_left_m=0.08, workcell_front_m=0.10,
               table_extra_left_m=0.04, table_extra_front_m=0.05, table_yaw_deg=15.):
    import mujoco

    if output_dir.exists():
        raise ValueError("Output exists; use a fresh directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    extr = json.loads(extrinsics_path.read_text(encoding="utf-8"))
    cam = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    if (cam.get("width"), cam.get("height")) != (640, 480):
        raise ValueError("Camera intrinsics must be 640x480")
    if int(extr["marker_id"]) != marker_id or extr.get("marker_link") != "gripper_link":
        raise ValueError("Extrinsics are not the expected ID1/gripper_link fit")
    if not extr.get("identifiable_9dof", False):
        raise ValueError("Fit is rank-deficient; camera pose is not reliable even as an initial visualization")
    if extr.get("status") != "provisional_not_applied_to_mujoco":
        raise ValueError("Unexpected extrinsic format")
    if manifest.get("marker_sizes_mm_by_id", {}).get(str(marker_id)) != 25.0:
        raise ValueError("Expected the 25mm ID1 capture")
    valid = [s for s in manifest["samples"]
             if any(int(m["id"]) == marker_id for m in s["markers"])]
    if not valid:
        raise ValueError("Selected marker not visible in any manifest sample")
    selected = valid[len(valid)//2] if sample_number is None else next(
        (s for s in valid if int(s["frame_index"]) == sample_number), None)
    if selected is None:
        raise ValueError("Selected frame has no ID1 observation")

    position, quaternion = mujoco_camera_pose(extr["T_base_camera"])
    fovy = fovy_from_intrinsics(float(cam["fy"]), int(cam["height"]))

    spec = mujoco.MjSpec.from_file(str(urdf_path.resolve()))
    # MuJoCo defaults discardvisual=true when importing URDF. In that mode
    # ALL added display-only geoms (contype=conaffinity=0) are discarded on
    # compile, including our table, Muhi and cup. Keep them for visualization.
    # This does not change the source URDF or enable contacts.
    if workcell == "simple":
        spec.compiler.discardvisual = False
    if workcell not in ('simple', 'none'):
        raise ValueError('workcell must be simple or none')
    shift = workcell_shift_left_front(extr['T_base_camera'], workcell_left_m, workcell_front_m)
    table_extra = workcell_shift_left_front(extr['T_base_camera'],
                                            table_extra_left_m, table_extra_front_m)
    if workcell == 'simple':
        add_approximate_workcell(spec, shift, table_extra, table_yaw_deg)
        add_backdrop(spec, extr['T_base_camera'])
    if lighting not in ("simple", "original"):
        raise ValueError("lighting must be simple or original")
    if lighting == "simple":
        add_simple_studio_light(spec)
    spec.worldbody.add_camera(name="r2s_side_calibrated_candidate",
                              pos=position, quat=quaternion, fovy=fovy)
    model = spec.compile()
    if workcell == "simple":
        import mujoco
        for name in ("r2s_preview_table", "r2s_muhi_body", "r2s_cup_base"):
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) < 0:
                raise RuntimeError(f"MuJoCo omitted expected workcell geom: {name}")
    apply_white_follower_preview(model)
    data = mujoco.MjData(model)
    state = [float(selected["joints"][key]) for key in (*r2s.JOINTS, "gripper")]
    config = {"joints": {
        "sign": {key: 1 for key in r2s.JOINTS},
        "offset_deg": {key: 0. for key in r2s.JOINTS},
    }}
    r2s.set_pose(model, data, state, config)

    width, height = int(cam["width"]), int(cam["height"])
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera="r2s_side_calibrated_candidate")
        simulated = Image.fromarray(renderer.render().copy()).convert("RGB")
    real_path = manifest_path.parent / selected["image"]
    with Image.open(real_path) as photo:
        if photo.size != (width, height):
            raise ValueError("Real PNG dimensions differ from intrinsics")
        real = photo.convert("RGB")

    # Show an observed ID1 center on the real image only. The robot URDF does
    # not contain a corresponding marker mesh, so no simulated marker is drawn.
    observed_marker = next(m for m in selected["markers"] if int(m["id"]) == marker_id)
    observed_x, observed_y = marker_center_px(observed_marker)
    annotated_real = real.copy()
    d = ImageDraw.Draw(annotated_real)
    d.ellipse((observed_x-5, observed_y-5, observed_x+5, observed_y+5),
              outline="lime", width=2)
    d.text((observed_x+7, observed_y+3), "ID1", fill="lime")

    canvas = Image.new("RGB", (width*2, height+28), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), "REAL: captured Side (ID1 green)", fill="white")
    draw.text((width+8, 7), "SIM: preliminary Side camera / FK", fill="white")
    canvas.paste(annotated_real, (0, 28))
    canvas.paste(simulated, (width, 28))

    output_dir.mkdir(parents=True, exist_ok=False)
    canvas.save(output_dir / "real_vs_mujoco.png")
    simulated.save(output_dir / "mujoco_only.png")
    report = {
        "status": "VISUAL_CHECK_ONLY_PROVISIONAL",
        "frame_index": int(selected["frame_index"]),
        "episode": int(selected["episode_index"]),
        "marker_id": marker_id,
        "observed_marker_center_px": [float(observed_x), float(observed_y)],
        "estimated_camera_pos_base_m": position,
        "estimated_camera_quat_mujoco_wxyz": quaternion,
        "approx_fovy_deg": fovy,
        "fit_holdout_rms_mm": extr["holdout_error_mm"]["rms"],
        "lighting_preset": lighting,
        "workcell_preset": workcell,
        "workcell_offset_m_in_base": shift.tolist() if workcell == "simple" else None,
        "workcell_left_m": workcell_left_m,
        "workcell_front_m": workcell_front_m,
        "table_extra_offset_m_in_base": table_extra.tolist() if workcell == "simple" else None,
        "table_yaw_deg_in_base": table_yaw_deg,
        "urdf_discardvisual_disabled_for_preview": workcell == "simple",
        "workcell_note": "Approximate 65x60cm tabletop and historically modeled Muhi bottle / transparent cylindrical cup. No measured physical placement.",
        "workcell_projection": project_reference_points(extr["T_base_camera"], cam, shift, table_extra) if workcell == "simple" else None,
        "visual_material_note": "Preview-only printed-plastic yellow was changed to off-white; source URDF and black servo parts unchanged.",
        "lighting_note": "Simple preset adds one fixed world light; no image postprocessing or photometric calibration.",
        "image": str(real_path),
        "output": str(output_dir / "real_vs_mujoco.png"),
        "caveats": [
            "Joint zero/sign conventions and URDF/geometry not yet verified physically.",
            "Marker itself is not rendered in URDF; judge overall robot silhouette/position.",
            "MuJoCo fovy approximates fy; principal point offset and distortion are not simulated.",
            "Virtual tabletop, Muhi bottle and cup are historical approximations; they may not match objects visible in this capture.",
            "No contact or motor simulation and no robot/camera device connections.",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--extrinsics", type=Path, required=True)
    p.add_argument("--intrinsics", type=Path,
                   default=root / "assets/calibration/side_charuco_20261107_restricted_intrinsics_candidate.json")
    p.add_argument("--urdf", type=Path,
                   default=root / "data/urdf/so101/so101_new_calib.urdf")
    p.add_argument("--marker-id", type=int, default=1)
    p.add_argument("--frame", type=int, default=None,
                   help="Optional known manifest frame index containing marker ID1")
    p.add_argument("--lighting", choices=("simple", "original"), default="simple",
                   help="simple adds a fixed white fill light; original preserves original MuJoCo lighting")
    p.add_argument("--workcell", choices=("simple", "none"), default="simple",
                   help="Render approximate photo-based Muhi/cup/table, or none for robot alone")
    p.add_argument("--workcell-left-cm", type=float, default=8.0,
                   help="Shift all illustrative props to image-left on base XY plane (default: 8 cm)")
    p.add_argument("--workcell-front-cm", type=float, default=10.0,
                   help="Shift props toward the camera on base XY plane (default: 10 cm)")
    p.add_argument("--table-extra-left-cm", type=float, default=4.,
                   help="Additional tabletop-only image-left shift; Muhi and cup stay put (default: 4 cm)")
    p.add_argument("--table-extra-front-cm", type=float, default=5.,
                   help="Additional tabletop-only camera-facing shift (default: 5 cm)")
    p.add_argument("--table-yaw-deg", type=float, default=15.,
                   help="Tabletop/legs yaw around table center in base XY; approximate (default: +15 deg)")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = render_one(a.manifest, a.extrinsics, a.intrinsics, a.urdf,
                   a.output, a.marker_id, a.frame, a.lighting, a.workcell,
                   a.workcell_left_cm / 100., a.workcell_front_cm / 100.,
                   a.table_extra_left_cm / 100., a.table_extra_front_cm / 100.,
                   a.table_yaw_deg)
    print("Saved:", r["output"])
    print("Frame:", r["frame_index"], "FOVy:", round(r["approx_fovy_deg"], 2))


if __name__ == "__main__":
    main()
