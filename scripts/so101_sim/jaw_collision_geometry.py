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


def split_convex_slabs(src: Path, dst_dir: Path, prefix: str,
                       slabs: int, cross_bands: int = 1):
    """Generate non-overlapping *axial and transverse* local convex patches.

    An axial-only convex hull can span both walls of a concavity within one
    slice, filling the real root aperture. Transverse segmentation reduces
    that bridging without inventing a wider jaw gap or altering the visuals.
    Triangle polygons are clipped to both local axes, not just assigned by
    vertex position, so neighboring cells meet consistently.
    """
    if not 2 <= slabs <= 32:
        raise ValueError('jaw slab count must be between 2 and 32')
    if not 1 <= cross_bands <= 8:
        raise ValueError('jaw transverse band count must be between 1 and 8')
    triangles = read_stl_triangles(src)
    all_vertices = triangles.reshape(-1, 3)
    unique = np.unique(np.round(all_vertices, 7), axis=0)
    _, _, vt = np.linalg.svd(unique - unique.mean(axis=0), full_matrices=False)
    primary, transverse = vt[0], vt[1]
    longitudinal = all_vertices @ primary
    crosswise = all_vertices @ transverse
    axial_edges = np.linspace(float(longitudinal.min()),float(longitudinal.max()),slabs+1)
    band_edges = np.linspace(float(crosswise.min()),float(crosswise.max()),cross_bands+1)
    longitudinal_tri = triangles @ primary
    crosswise_tri = triangles @ transverse
    records=[]
    for i in range(slabs):
        lo,hi=float(axial_edges[i]),float(axial_edges[i+1])
        ax_mask=(longitudinal_tri.max(axis=1)>=lo-1e-10)&(
                 longitudinal_tri.min(axis=1)<=hi+1e-10)
        for j in range(cross_bands):
            blo,bhi=float(band_edges[j]),float(band_edges[j+1])
            mask=ax_mask & (crosswise_tri.max(axis=1)>=blo-1e-10)&(
                                crosswise_tri.min(axis=1)<=bhi+1e-10)
            clipped=[]
            for tri in triangles[mask]:
                poly=clip_polygon(tri,primary,lo,True)
                poly=clip_polygon(poly,primary,hi,False)
                poly=clip_polygon(poly,transverse,blo,True)
                poly=clip_polygon(poly,transverse,bhi,False)
                clipped.extend(poly)
            if len(clipped)<4:continue
            file=dst_dir/f'{prefix}_{i:02d}_{j:02d}.stl'
            try:
                stats=write_hull_stl(file,np.asarray(clipped))
            except (ValueError,QhullError):
                continue
            records.append({'file':file.name,'slab':i,'band':j,
                            'axial_range_mesh_units':[lo,hi],
                            'transverse_range_mesh_units':[blo,bhi],**stats})
    if len(records)<2:
        raise RuntimeError(f'Cannot make jaw collision patches from {src}')
    # Label thicker end as root to support aperture checks; this is a heuristic,
    # not a measured distance from the tip on a calibrated physical gripper.
    volumes=[sum(p['volume_m3'] for p in records if p['slab']==k) for k in range(slabs)]
    n=max(2,slabs//4)
    root_side='low_index' if sum(volumes[:n])>sum(volumes[-n:]) else 'high_index'
    return records, primary.tolist(), transverse.tolist(), root_side


def build_split_jaw_urdf(original_urdf: Path, output_dir: Path, slabs: int = 12, cross_bands: int = 1):
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
                   'jaw_collision_cross_bands_requested': cross_bands,
                   'root_label_method': 'heuristic: thicker end of original jaw STL; check real-jaw orientation',
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
            parts, axis, cross_axis, root_side = split_convex_slabs(source, mesh_dir, body_name, slabs, cross_bands)
            at = list(link).index(collision)
            link.remove(collision)
            for j, part in enumerate(parts):
                item = copy.deepcopy(collision)
                item.set('name', f'{body_name}_collision_patch_{j:03d}')
                item.find('./geometry/mesh').set(
                    'filename', 'jaw_collision/'+part['file'])
                link.insert(at+j, item)
            diagnostics['jaw_parts'][body_name] = {
                'visual_stl': target, 'collision_mesh_count': len(parts),
                'principal_axis_mesh_local': axis,
                'transverse_axis_mesh_local': cross_axis,
                'root_side_heuristic':root_side,
                'parts': parts}
        if not found:
            raise ValueError(f'Jaw collision mesh {target} not found in {body_name}')
    if set(diagnostics['jaw_parts']) != set(TARGETS):
        raise ValueError('Both fixed and moving jaws must be processed')
    new_urdf = output_dir / original_urdf.name
    tree.write(new_urdf, encoding='utf-8', xml_declaration=True)
    (output_dir / 'jaw_collision_build.json').write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return new_urdf, diagnostics
