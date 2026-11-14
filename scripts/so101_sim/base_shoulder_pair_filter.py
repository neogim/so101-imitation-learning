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

    base = tuple(g for g in range(model.ngeom)
                 if int(model.geom_bodyid[g]) == BASE_BODY
                 and (model.geom_contype[g] or model.geom_conaffinity[g]))
    upper = tuple(g for g in range(model.ngeom)
                  if int(model.geom_bodyid[g]) == shoulder
                  and (model.geom_contype[g] or model.geom_conaffinity[g]))
    if not base or not upper:
        raise ValueError("Missing base or shoulder collidable geoms")
    related = set(base + upper)
    remaining = tuple(g for g in range(model.ngeom)
                      if g not in related
                      and (model.geom_contype[g] or model.geom_conaffinity[g]))
    if any((int(model.geom_contype[g]), int(model.geom_conaffinity[g])) != (1, 1)
           for g in (*base, *upper, *remaining)):
        raise ValueError("Unexpected non-1/1 collision mask; refusing diagnostic filter")

    protected = list(object_geoms)
    if table_geom is not None:
        protected.append(table_geom)
    if not protected or any(
        int(g) in related or
        (int(model.geom_contype[g]), int(model.geom_conaffinity[g])) != (1, 1)
        for g in protected):
        raise ValueError("Object/table collision geom missing or nonstandard")

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
        "other_geoms_remain_1_1": True,
        "source_urdf_modified": False,
        "production_setting": False,
        "warning": (
            "This alters only current compiled-model collision bits, "
            "and is not a validated real-hardware self-collision design."),
    }
