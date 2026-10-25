"""Metadata-only audit of SO-101 sessions. Never edits source datasets."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq

from offline_labels import validate_labels


GROUP_DIRS = {
    'original': 'so101-data',
    'yaw': 'so101-yaw-experiment',
    'xy': 'so101-upright-xy-experiment',
    'recovery': 'so101-recovery-experiment',
}
JOINTS = ('shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
          'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos')
NORMAL = {'manual', 'time_limit'}
FIELDS = ('group', 'subgroup', 'source_folder', 'session', 'episode_index',
          'status', 'adopt', 'reason', 'label_source', 'end_reason', 'pick',
          'place', 'overall', 'frames', 'duration_s', 'fps', 'format',
          'action_requested_present', 'numeric_sha256')


def subgroup(group, session):
    if group == 'yaw':
        found = re.search(r'_yaw(0|45|90|135)(?:_|$)', session)
        return found.group(1) + '°' if found else 'unclassified'
    if group == 'xy':
        found = re.search(r'_(left|right|front|back)_3cm(?:_|$)', session)
        return {'left': 'left', 'right': 'right', 'front': 'front',
                'back': 'back'}.get(found.group(1), 'unclassified') if found else 'unclassified'
    return 'recovery' if group == 'recovery' else 'fixed'


def audit_session(source, group, labels_dir):
    info = json.loads((source / 'meta/info.json').read_text())
    count = int(info.get('total_episodes', 0))
    if not count:
        return [], 'empty_dataset'
    episode_rows = []
    for path in sorted((source / 'meta/episodes').rglob('*.parquet')):
        episode_rows.extend(pq.read_table(path, columns=['episode_index', 'length']).to_pylist())
    lengths = {int(x['episode_index']): int(x['length']) for x in episode_rows}
    session_errors = []
    if len(episode_rows) != count or set(lengths) != set(range(count)):
        session_errors.append('episode_metadata_mismatch')
    features = info.get('features', {})
    expected = {'observation.state', 'action', 'observation.images.iphone'}
    if info.get('codebase_version') != 'v3.0' or info.get('fps') != 20:
        session_errors.append('format_or_fps_mismatch')
    if not expected <= features.keys() or any(
        features.get(k, {}).get('names') != list(JOINTS) or features[k].get('shape') != [6]
        for k in ('observation.state', 'action') if k in features
    ) or features.get('observation.images.iphone', {}).get('shape') != [480, 640, 3]:
        session_errors.append('feature_mismatch')
    data_files = sorted((source / 'data').rglob('*.parquet'))
    video_files = sorted((source / 'videos/observation.images.iphone').rglob('*.mp4'))
    if len(data_files) != 1 or len(video_files) != 1 or not video_files[0].stat().st_size:
        session_errors.append('missing_or_unexpected_data_video')
    rows_by_ep = defaultdict(list)
    if len(data_files) == 1:
        table = pq.read_table(data_files[0], columns=[
            'episode_index', 'frame_index', 'timestamp', 'observation.state', 'action'])
        if len(table) != info.get('total_frames'):
            session_errors.append('parquet_total_frames_mismatch')
        for row in table.to_pylist():
            rows_by_ep[int(row['episode_index'])].append(row)
    labels_path = labels_dir / f'{source.name}.json'
    label_source = 'offline_labels' if labels_path.exists() else 'episode_outcomes'
    labels_error = None
    try:
        if labels_path.exists():
            labels = validate_labels(json.loads(labels_path.read_text()), source)
        elif (source / 'episode_outcomes.json').exists():
            labels = json.loads((source / 'episode_outcomes.json').read_text())
        else:
            labels, label_source = [], 'missing'
    except Exception as exc:
        labels, labels_error = [], f'invalid_labels:{type(exc).__name__}:{exc}'
    by_label = {int(x['episode_index']): x for x in labels}
    if len(by_label) != len(labels):
        labels_error = 'duplicate_label_episode_index'
    output = []
    for ep in range(count):
        label = by_label.get(ep, {})
        rows = rows_by_ep.get(ep, [])
        reasons = list(session_errors)
        if len(rows) != lengths.get(ep, -1):
            reasons.append('episode_frame_count_mismatch')
        signature = hashlib.sha256()
        for i, row in enumerate(rows):
            if row['frame_index'] != i or abs(float(row['timestamp']) - i / 20) > .002:
                reasons.append('frame_index_or_timestamp_mismatch')
                break
            state = np.asarray(row['observation.state'], dtype=np.float32)
            action = np.asarray(row['action'], dtype=np.float32)
            if state.shape != (6,) or action.shape != (6,) or not (
                np.isfinite(state).all() and np.isfinite(action).all()
            ):
                reasons.append('invalid_state_or_action')
                break
            signature.update(state.tobytes())
            signature.update(action.tobytes())
        if label and label.get('frames') != lengths.get(ep):
            reasons.append('label_frame_count_mismatch')
        if label and label.get('episode_index') != ep:
            reasons.append('label_index_mismatch')
        ending = label.get('end_reason', 'unknown')
        picks = [label.get(k, 'unknown') for k in ('pick', 'place', 'overall')]
        adopted = label.get('adopt') is True
        if reasons:
            status, reason = 'excluded', ';'.join(sorted(set(reasons)))
        elif ending == 'interrupted' or 'failure' in picks:
            status, reason = 'excluded', 'interrupted_or_failure_flag'
        elif labels_error:
            status, reason = 'review', labels_error
        elif not label:
            status, reason = 'review', 'no_saved_adoption_flag'
        elif ending in NORMAL and picks == ['success'] * 3 and adopted:
            status, reason = 'adopted', 'normal_end_and_explicit_adopt'
        else:
            status, reason = 'review', 'not_explicitly_adopted_or_outcome_unknown'
        output.append(dict(group=group, subgroup=subgroup(group, source.name),
                           source_folder=str(source.parent), session=source.name,
                           episode_index=ep, status=status, adopt=adopted, reason=reason,
                           label_source=label_source, end_reason=ending,
                           pick=picks[0], place=picks[1], overall=picks[2],
                           frames=len(rows), duration_s=round(len(rows) / 20, 3),
                           fps=info.get('fps'), format=info.get('codebase_version'),
                           action_requested_present='action_requested' in features,
                           numeric_sha256=signature.hexdigest() if rows else ''))
    return output, None


def summarize(rows, key):
    result = []
    for label in sorted({r[key] for r in rows}):
        subset = [r for r in rows if r[key] == label]
        statuses = Counter(r['status'] for r in subset)
        result.append(dict(category=label, total_episodes=len(subset),
                           a_success=statuses['adopted'],
                           failed_or_interrupted=statuses['excluded'],
                           review=statuses['review'], adopted=statuses['adopted'],
                           frames=sum(r['frames'] for r in subset),
                           duration_s=round(sum(r['duration_s'] for r in subset), 3)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.home() / 'workspace')
    parser.add_argument('--labels-dir', type=Path, default=Path(__file__).parent / 'labels')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    rows, skipped = [], []
    for group, directory in GROUP_DIRS.items():
        root = args.workspace / directory
        if not root.is_dir():
            raise FileNotFoundError(root)
        for source in sorted(root.iterdir()):
            if not source.is_dir():
                continue
            if not (source / 'meta/info.json').is_file():
                skipped.append(dict(group=group, session=source.name, reason='no_dataset_metadata'))
                continue
            selected, reason = audit_session(source, group, args.labels_dir)
            rows.extend(selected)
            if reason:
                skipped.append(dict(group=group, session=source.name, reason=reason))
    # Identical numeric trajectories are held for review; video content is not inferred.
    hashes = defaultdict(list)
    for row in rows:
        if row['status'] == 'adopted':
            hashes[(row['frames'], row['numeric_sha256'])].append(row)
    for copies in hashes.values():
        if len(copies) > 1:
            for row in copies:
                row.update(status='review', reason='identical_numeric_trajectory_check_video')
    with (args.output / 'episodes.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / 'skipped_sessions.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=('group', 'session', 'reason'))
        writer.writeheader()
        writer.writerows(skipped)
    summary = {group: summarize(rows, group) for group in ('group', 'subgroup')}
    summary['total'] = dict(episodes=len(rows), adopted=sum(r['status'] == 'adopted' for r in rows),
                            excluded=sum(r['status'] == 'excluded' for r in rows),
                            review=sum(r['status'] == 'review' for r in rows),
                            adopted_frames=sum(r['frames'] for r in rows if r['status'] == 'adopted'))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (args.output / 'summary.csv').open('w', newline='') as file:
        fields = ('type', 'category', 'total_episodes', 'a_success',
                  'failed_or_interrupted', 'review', 'adopted', 'frames', 'duration_s')
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for kind in ('group', 'subgroup'):
            for item in summary[kind]:
                writer.writerow(dict(type=kind, **item))
    (args.output / 'selected.json').write_text(json.dumps([
        dict(source=str(Path(r['source_folder']) / r['session']), episode_index=r['episode_index'],
             frames=r['frames'], group=r['group'], subgroup=r['subgroup'])
        for r in rows if r['status'] == 'adopted'], indent=2) + '\n')
    lines = ['| Group | Total | A success | Failed/interrupted | Review | Adopted | Frames |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary['group']:
        lines.append('| {category} | {total_episodes} | {a_success} | {failed_or_interrupted} | '
                     '{review} | {adopted} | {frames} |'.format(**row))
    lines.extend(['', '| Yaw / XY / Recovery | Total | A success | Failed/interrupted | Review | Adopted | Frames |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for row in summary['subgroup']:
        if row['category'] == 'fixed':
            continue
        lines.append('| {category} | {total_episodes} | {a_success} | {failed_or_interrupted} | '
                     '{review} | {adopted} | {frames} |'.format(**row))
    lines.append(f"\nTotal: {summary['total']['episodes']} episodes; "
                 f"{summary['total']['adopted']} adopted; "
                 f"{summary['total']['adopted_frames']} adopted frames at 20 FPS.\n")
    (args.output / 'summary.md').write_text('\n'.join(lines))
    print(json.dumps(summary['total']))


if __name__ == '__main__':
    main()
