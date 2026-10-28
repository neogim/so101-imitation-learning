# SO-101 side camera / FK comparison (read-only)

**Not camera-calibrated.** The rendered FK world XZ plots cannot be overlaid quantitatively on side image coordinates. No definitive sign ranking.

## Frame 0

- shoulder+_elbow+: TCP (x,y,z) = (177.8, 13.2, 8.3) mm; horizontal reach 178.3 mm.
- shoulder-_elbow+: TCP (x,y,z) = (16.4, 6.1, 261.0) mm; horizontal reach 17.5 mm.
- shoulder+_elbow-: TCP (x,y,z) = (-291.1, -7.3, 176.4) mm; horizontal reach 291.2 mm.
- shoulder-_elbow-: TCP (x,y,z) = (369.2, 21.6, -90.9) mm; horizontal reach 369.8 mm.

## Frame 240

- shoulder+_elbow+: TCP (x,y,z) = (226.6, 207.5, 52.9) mm; horizontal reach 307.3 mm.
- shoulder-_elbow+: TCP (x,y,z) = (228.8, 209.8, 166.4) mm; horizontal reach 310.4 mm.
- shoulder+_elbow-: TCP (x,y,z) = (220.9, 201.6, 43.7) mm; horizontal reach 299.1 mm.
- shoulder-_elbow-: TCP (x,y,z) = (226.4, 207.3, 154.6) mm; horizontal reach 307.0 mm.

## Frame 480

- shoulder+_elbow+: TCP (x,y,z) = (266.5, -11.7, 99.7) mm; horizontal reach 266.7 mm.
- shoulder-_elbow+: TCP (x,y,z) = (183.5, -4.6, -45.4) mm; horizontal reach 183.5 mm.
- shoulder+_elbow-: TCP (x,y,z) = (286.9, -13.4, 318.4) mm; horizontal reach 287.2 mm.
- shoulder-_elbow-: TCP (x,y,z) = (363.3, -19.9, 79.7) mm; horizontal reach 363.9 mm.

The base frame position, camera intrinsics/extrinsics, model zero offsets, and reliable image motor-axis correspondences are not calibrated. A visual check only disfavors some extreme shapes; no safe hardware mapping inferred.

## Figure

The matching comparison image is `so101_side_fk_comparison.png`. Store it next to this Markdown file. The PNG is an illustration and not a camera-calibrated metric comparison.

![Side camera versus four FK hypotheses](so101_side_fk_comparison.png)
