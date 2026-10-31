# MuJoCo workcell from physical photos (2026-10-31)

Photo-based approximations, not calibrated measurements:

- MUHI bottle, horizontal: length ~100 mm, diameter ~38 mm (white body + rounded cap)
- Transparent plastic container: open cylindrical shell with side cutout; approximate height ~120 mm and outer diameter ~100 mm
- SO-101 mounted near table edge, bottle and container in front of the arm.
- Position initial guesses in SO-101 base frame: MUHI x=0.18 m, y=+0.11 m; cup x=0.18 m, y=-0.12 m. **These coordinates are not derived from calibration.**
- Table is represented as a primitive box; actual table edge and base mount are not yet calibrated.

The local Google Colab notebook `SO101_Cartesian7_MuJoCo_Colab_Workcell.ipynb` adds these approximated rigid props to the URDF via fixed joints while preserving mesh asset-relative paths. It is a visualization only. It does not model real grasp contact, physical object motion, or closed-loop ACT execution.

Existing contacts in robot URDF and joint convention mismatches remain unresolved. **Not approved for real hardware control.**
