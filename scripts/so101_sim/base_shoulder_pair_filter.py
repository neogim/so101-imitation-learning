"""Optional SO-101 base-link/shoulder-link collision filtering.

This is an isolated MuJoCo *experiment*, not a general self-collision policy.
It excludes ONLY the base-vs-shoulder link pair in the contact bitmask logic.
All object/table/finger contacts remain eligible. No URDF is modified.
"""
from __future__ import annotations

import mujoco

BASE_BODY = 0  # fixed base_link is merged into world by MuJoCo URDF compiler
SHOULDER_LINK = "shoulder_link"


def _geom(model, gid):
    return {
        "geom_id": int(gid),
        "body_id": int(model.geom_bodyid[gid]),
        "contype": int(model.geom_contype[gid]),
        "conaffinity": int(model.geom_conaffinity[gid]),
    }


def apply_base_shoulder_pair_filter(model, *, object_geoms=(), table_geom=None):
    """Filter base/shoulder mutual contacts while keeping 1/1 other contacts.

    The two group collision masks are (2,1) and (4,1), respectively:
      base vs shoulder  : (2 & 1) | (4 & 1) == 0  => no collision
      base vs normal 1/1: (2 & 1) | (1 & 1) != 0  => collision eligible
      shoulder vs 1/1  : (4 & 1) | (1 & 1) != 0  => collision eligible

    Fail closed if this exact mask scheme's preconditions are violated.
    """
    shoulder = int(mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, SHOULDER_LINK))
    if shoulder < 0:
        raise ValueError("SO-101 shoulder_link body missing")

    # A static table attached to a fixed body is merged into MuJoCo body 0
    # just like the URDF base. Never infer "base" from world/body 0 alone:
    # that would classify the table as a robot collision mesh, and cause a
    # false "Object/table collision geom missing or nonstandard" failure.
    protected = tuple(int(g) for g in object_geoms) + (
        (int(table_geom),) if table_geom is not None else ())
    if not protected or any(g < 0 or g >= model.ngeom for g in protected):
        raise ValueError("Missing/invalid protected object or table geom IDs")
    if len(set(protected)) != len(protected):
        raise ValueError("Duplicate protected object/table geom ID")
    if any((int(model.geom_contype[g]), int(model.geom_conaffinity[g])) != (1, 1)
           for g in protected):
        raise ValueError("Object/table collision geom uses unexpected mask")

    # The SO-101 URDF base has four named collision meshes. Other world
    # geoms (including the collidable tabletop) must never be remasked.
    base_mesh_stems = (
        "base_motor_holder_so101_v1",
        "base_so101_v2",
        "sts3215_03a_v1",
        "waveshare_mounting_plate_so101_v2",
    )
    def is_base_geom(g):
        if int(model.geom_bodyid[g]) != BASE_BODY or (
                int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH)):
            return False
        mesh_id = int(model.geom_dataid[g])
        mesh_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or ""
        return any(mesh_name.startswith(stem) for stem in base_mesh_stems)

    base = tuple(g for g in range(model.ngeom)
                 if g not in protected and is_base_geom(g)
                 and (model.geom_contype[g] or model.geom_conaffinity[g]))
    upper = tuple(g for g in range(model.ngeom)
                  if int(model.geom_bodyid[g]) == shoulder
                  and (model.geom_contype[g] or model.geom_conaffinity[g]))
    if len(base) < 3 or not upper:
        raise ValueError(
            "Cannot identify original SO-101 base/shoulder collision meshes")
    related = set(base + upper)
    if any(g in related for g in protected):
        raise ValueError("Protected object/table geom misidentified as robot")
    remaining = tuple(g for g in range(model.ngeom)
                      if g not in related
                      and (model.geom_contype[g] or model.geom_conaffinity[g]))
    if any((int(model.geom_contype[g]), int(model.geom_conaffinity[g])) != (1, 1)
           for g in (*base, *upper, *remaining)):
        raise ValueError("Unexpected non-1/1 collision mask; refusing diagnostic filter")

    for g in base:
        model.geom_contype[g] = 2
        model.geom_conaffinity[g] = 1
    for g in upper:
        model.geom_contype[g] = 4
        model.geom_conaffinity[g] = 1

    assert all(
        (int(model.geom_contype[g]), int(model.geom_conaffinity[g])) == (1, 1)
        for g in protected)
    return {
        "mode": "SO101_BASE_SHOULDER_PAIR_FILTER_DIAGNOSTIC",
        "base_world_geom_ids": list(base),
        "shoulder_geom_ids": list(upper),
        "protected_object_and_table_geom_ids": [int(g) for g in protected],
        "world_geoms_excluded_from_base": [int(g) for g in protected
                                          if int(model.geom_bodyid[g]) == 0],
        "base_mesh_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH,
                        int(model.geom_dataid[g])) for g in base],
        "other_geoms_remain_1_1": True,
        "source_urdf_modified": False,
        "production_setting": False,
        "warning": (
            "This alters only current compiled-model collision bits, "
            "and is not a validated real-hardware self-collision design."),
    }
