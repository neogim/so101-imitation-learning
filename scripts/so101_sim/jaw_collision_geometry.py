#!/usr/bin/env python3
"""Create a simulation-only compound-convex collision approximation for SO-101 jaws.

Never changes the source URDF or STL. Original CAD mesh remains the visual.
Each axial slab is an independent convex mesh, preserving more of the original
grip-surface profile than one convex hull of the entire jaw. This is an
approximation, NOT a claim that sub-mm teeth or compliance are faithfully modeled.
"""
from __future__ import annotations

import copy
import json
import os
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError

TARGETS = {
    'gripper_link': 'wrist_roll_follower_so101_v1.stl',
    'moving_jaw_so101_v1_link': 'moving_jaw_so101_v1.stl',
}


def read_stl_triangles(path: Path):
    """Read binary or ASCII STL, preserving source mesh-unit coordinates."""
    blob = path.read_bytes()
    if len(blob) >= 84 and 84 + 50 * struct.unpack_from('<I', blob, 80)[0] == len(blob):
        rec = np.frombuffer(blob, dtype=np.dtype([
            ('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attr', '<u2')
        ]), offset=84)
        return np.array(rec['vertices'], dtype=np.float64)
    import re
    data = np.array([list(map(float, xyz)) for xyz in re.findall(
        rb'vertex\s+([-+\deE.]+)\s+([-+\deE.]+)\s+([-+\deE.]+)', blob
    )], dtype=float)
    if len(data) < 12 or len(data) % 3:
        raise ValueError(f'Unrecognized STL or insufficient triangles: {path}')
    return data.reshape(-1, 3, 3)


def clip_polygon(poly, axis, threshold, keep_above):
    """Clip convex polygon at a slab boundary, including edge intersections."""
    if not len(poly):
        return []
    result = []
    prev = np.asarray(poly[-1], dtype=float)
    prev_s = float(np.dot(prev, axis) - threshold)
    prev_in = prev_s >= -1e-10 if keep_above else prev_s <= 1e-10
    for pt in poly:
        cur = np.asarray(pt, dtype=float)
        cur_s = float(np.dot(cur, axis) - threshold)
        cur_in = cur_s >= -1e-10 if keep_above else cur_s <= 1e-10
        if cur_in != prev_in:
            den = prev_s - cur_s
            if abs(den) > 1e-15:
                result.append(prev + (cur - prev) * (prev_s / den))
        if cur_in:
            result.append(cur)
        prev, prev_s, prev_in = cur, cur_s, cur_in
    return result


def write_hull_stl(path, points):
    points = np.unique(np.round(points, 8), axis=0)
    if len(points) < 4:
        raise ValueError('Insufficient vertices in convex slab')
    hull = ConvexHull(points)
    if hull.volume <= 1e-12:
        raise ValueError('Degenerate convex slab')
    with path.open('wb') as f:
        f.write(b'SO101 simulation collision slab'.ljust(80, b' '))
        f.write(struct.pack('<I', len(hull.simplices)))
        for tri in hull.simplices:
            v = points[tri]
            # Qhull equations provide the outward normal for each triangular facet.
            # MuJoCo handles collision using its compiled convex geometry.
            normal = np.cross(v[1] - v[0], v[2] - v[0])
            norm = float(np.linalg.norm(normal))
            normal = normal / norm if norm > 1e-15 else np.zeros(3)
            f.write(struct.pack('<12fH', *normal, *v[0], *v[1], *v[2], 0))
    return {'vertices': int(len(points)), 'volume_m3': float(hull.volume)}


def split_convex_slabs(src: Path, dst_dir: Path, prefix: str, slabs: int):
    if not 2 <= slabs <= 32:
        raise ValueError('jaw slab count must be between 2 and 32')
    triangles = read_stl_triangles(src)
    all_vertices = triangles.reshape(-1, 3)
    unique = np.unique(np.round(all_vertices, 7), axis=0)
    # PCA on unique points, rather than triangle frequency, avoids over-weighting
    # a highly tessellated area. Along the main extension axis the whole-mesh
    # convex hull otherwise fills the concave interior grip profile.
    _, _, vt = np.linalg.svd(unique - unique.mean(axis=0), full_matrices=False)
    axis = vt[0]
    projections = all_vertices @ axis
    start, stop = float(projections.min()), float(projections.max())
    edges = np.linspace(start, stop, slabs + 1)
    records = []
    for i in range(slabs):
        lo, hi = float(edges[i]), float(edges[i+1])
        overlaps = (np.max(triangles @ axis, axis=1) >= lo-1e-10) & (
            np.min(triangles @ axis, axis=1) <= hi+1e-10)
        clipped = []
        for tri in triangles[overlaps]:
            poly = clip_polygon(tri, axis, lo, True)
            poly = clip_polygon(poly, axis, hi, False)
            clipped.extend(poly)
        if not clipped:
            continue
        file = dst_dir / f'{prefix}_{i:02d}.stl'
        try:
            stats = write_hull_stl(file, np.asarray(clipped))
        except (ValueError, QhullError):
            # An empty or planar end-slab has no collision volume.
            continue
        records.append({'file': file.name, 'slab': i,
                        'axial_range_mesh_units': [lo, hi], **stats})
    if len(records) < 2:
        raise RuntimeError(f'Cannot make compound jaw collision from {src}: {len(records)} valid slabs')
    return records, axis.tolist()


def build_split_jaw_urdf(original_urdf: Path, output_dir: Path, slabs: int = 12):
    """Emit isolated URDF + STLs. Preserve all original visuals/joints/inertias."""
    original_urdf = original_urdf.resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_link = output_dir / 'assets'
    if not asset_link.exists():
        asset_link.symlink_to(original_urdf.parent / 'assets', target_is_directory=True)
    mesh_dir = output_dir / 'jaw_collision'
    mesh_dir.mkdir(exist_ok=True)
    tree = ET.parse(original_urdf)
    root = tree.getroot()
    diagnostics = {'mode': 'axial_convex_slabs',
                   'source_urdf': str(original_urdf),
                   'visual_meshes_unchanged': True,
                   'jaw_collision_slabs_requested': slabs,
                   'warning': 'Slab hulls are approximate; inspect contact normals and tooth/concavity fidelity before claiming real-world equivalence.',
                   'jaw_parts': {}}
    for link in root.findall('link'):
        body_name = link.get('name')
        if body_name not in TARGETS:
            continue
        target = TARGETS[body_name]
        found = False
        for collision in list(link.findall('collision')):
            mesh = collision.find('./geometry/mesh')
            if mesh is None or Path(mesh.get('filename', '')).name != target:
                continue
            if found:
                raise ValueError('Ambiguous duplicate jaw collision element: '+body_name)
            found = True
            source = (original_urdf.parent / mesh.get('filename')).resolve(strict=True)
            parts, axis = split_convex_slabs(source, mesh_dir, body_name, slabs)
            at = list(link).index(collision)
            link.remove(collision)
            for j, part in enumerate(parts):
                item = copy.deepcopy(collision)
                item.set('name', f'{body_name}_collision_slab_{j:02d}')
                item.find('./geometry/mesh').set(
                    'filename', 'jaw_collision/'+part['file'])
                link.insert(at+j, item)
            diagnostics['jaw_parts'][body_name] = {
                'visual_stl': target, 'collision_mesh_count': len(parts),
                'principal_axis_mesh_local': axis, 'parts': parts}
        if not found:
            raise ValueError(f'Jaw collision mesh {target} not found in {body_name}')
    if set(diagnostics['jaw_parts']) != set(TARGETS):
        raise ValueError('Both fixed and moving jaws must be processed')
    new_urdf = output_dir / original_urdf.name
    tree.write(new_urdf, encoding='utf-8', xml_declaration=True)
    (output_dir / 'jaw_collision_build.json').write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return new_urdf, diagnostics
