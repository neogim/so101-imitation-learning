# Phase 6G: Is reverse J4 wrist bend required for the middle Muhi grasp?

## Hypothesis
After improving the original SO-101 jaw collision geometry (12 longitudinal x 3 transverse convex patches), the middle grasp was much better even with J4 (wrist_flex) at approximately **-94.994 degrees**. The next question is whether the same grasp target can be reached and held **without the negative J4 bend**.

## Design
The default Phase 6G run now adds a fourth candidate, **middle_j4_nonnegative**, alongside root, middle and tip. It uses the **same provisional local anchor depth (-70 mm)**, Muhi world position (0.220, 0, 0.0485 m), collision meshes, friction, arm and J6 controllers as **middle**. The modified IK imposes **J4 >= 0 degrees**, without forcing other arm joints to stay unchanged. The other four arm joints are free to adjust to preserve the same grasp-anchor target. This is an alternate wrist posture IK comparison, not a single-joint J4 motion.

- Existing successful setup remains separately saved under `middle/`.
- New result: `middle_j4_nonnegative/result.json`, `middle_j4_nonnegative/grasp.mp4` (if geometry screening permits dynamics), `middle_j4_nonnegative/jaw_push_direction.png` (if physics executes).
- CSV adds `j4_solved_deg` and `j4_constraint`. Summary adds `j4_sign_comparison_enabled`.
- `--no-j4-compare` removes the extra candidate. `--j4-nonnegative-min-deg 5` can demand at least +5 degrees, but the default allows straight J4 at 0 degrees.
- Note: positive/negative J4 are *joint-angle signs*, not photographic proof of which direction is biomechanically normal. Judge rendered posture visually.

## Run and inspect
Use the existing Colab notebook, section 5 (fresh Git clone), then 6 (execute), 7 and 8 (save). Notebook section 7 currently displays videos only for `root,middle,tip`; the **fourth candidate is saved but not automatically displayed**. Add this cell in Colab to watch the new comparison video:

```python
from IPython.display import display, Video
from pathlib import Path
new_video = PHASE6G_OUT/'middle_j4_nonnegative'/'grasp.mp4'
print('New candidate:', new_video, 'exists:', new_video.is_file())
if new_video.is_file():
    display(Video(str(new_video), embed=True, width=700))
```

Evaluation should first check `comparison.csv` for J4 solution and IK error, then geometry skip versus actual physics, then hold contact duration, XY displacement, and video. Do not classify a motion as successful solely because two jaw geoms touch. The previous middle baseline had 500/500 hold steps of bilateral contact but still moved up to 27.550 mm and had penetration >3 mm.

## Status
Implementation committed in GitHub but **Colab has not been run from this change**. The historical baseline data are measured, the nonnegative J4 result is not. Previous snapshot is preserved at `backup/phase6g-before-j4-sign-comparison-20261118`. No physical robot operation.
