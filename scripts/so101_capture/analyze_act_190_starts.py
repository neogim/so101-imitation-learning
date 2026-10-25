"""Summarize recorded starting poses; propose examples without moving hardware."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


NAMES = ('shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
         'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--historical-report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    with a.selection.open(newline='') as file:
        selected = list(csv.DictReader(file))
    data = sorted((a.dataset / 'data').rglob('*.parquet'))
    if len(data) != 1 or len(selected) != 190:
        raise ValueError('Expected one Parquet and 190 selected episodes')
    table = pq.read_table(data[0], columns=['episode_index', 'frame_index', 'observation.state'])
    starts = [(int(e), np.asarray(s, dtype=float)) for e,f,s in zip(
        table['episode_index'].to_pylist(), table['frame_index'].to_pylist(),
        table['observation.state'].to_pylist(), strict=True) if int(f) == 0]
    if [e for e,_ in starts] != list(range(190)):
        raise ValueError('Missing or duplicated first frame')
    values = np.stack([s for _,s in starts])
    if values.shape != (190,6) or not np.isfinite(values).all():
        raise ValueError('Invalid start state')
    report = json.loads(a.historical_report.read_text())
    event = next(x for x in report['events'] if x['event'] == 'first_motion_action_check')
    historical_goal = np.asarray([event['joints'][k]['goal'] for k in NAMES[:5]] +
                                 [event['gripper_held_goal']], dtype=float)
    median = np.median(values, axis=0)
    scale = np.maximum(np.quantile(values, .75, axis=0)-np.quantile(values, .25, axis=0), 1)
    medoid = int(np.argmin(np.linalg.norm((values-median)/scale, axis=1)))
    # Rotation joints share degrees; keep the normalized gripper difference separate.
    historical_nearest = int(np.argmin(np.linalg.norm(values[:,:5]-historical_goal[:5], axis=1)))
    stats = {name: dict(min=float(np.min(values[:,i])), median=float(median[i]),
                        max=float(np.max(values[:,i]))) for i,name in enumerate(NAMES)}
    by_group = {}
    for group in sorted({r['group'] for r in selected}):
        group_values = values[[i for i,r in enumerate(selected) if r['group'] == group]]
        by_group[group] = {name: dict(min=float(group_values[:,i].min()),
                                      median=float(np.median(group_values[:,i])),
                                      max=float(group_values[:,i].max())) for i,name in enumerate(NAMES)}
    def candidate(index):
        row = selected[index]
        return dict(merged_episode_index=index, episode_id=row['episode_id'],
                    source=row['source'], source_episode_index=int(row['episode_index']),
                    split=row['split'], group=row['group'], subgroup=row['subgroup'],
                    state={name:float(values[index,i]) for i,name in enumerate(NAMES)},
                    historical_goal_difference={name:float(values[index,i]-historical_goal[i])
                                                for i,name in enumerate(NAMES)})
    result = dict(episodes=190, units={name:('degrees' if i<5 else 'normalized_0_100')
                                       for i,name in enumerate(NAMES)},
                  all_starts=stats, by_group=by_group,
                  historical_goal={name:float(historical_goal[i]) for i,name in enumerate(NAMES)},
                  historical_goal_source=str(a.historical_report),
                  representative_recorded_start=candidate(medoid),
                  nearest_to_historical_goal=candidate(historical_nearest),
                  note='Historical Goal is not a fresh hardware observation. Candidates are recorded examples, not automatic position commands.')
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'representative':selected[medoid]['episode_id'],
                      'historical_nearest':selected[historical_nearest]['episode_id']}))


if __name__ == '__main__':
    main()
