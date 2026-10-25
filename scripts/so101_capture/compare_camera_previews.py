"""Summarize robot-free camera preview trials; never opens cameras or motors."""
import argparse
import csv
import json
from pathlib import Path


def flatten(label, trial, directory):
    path = directory / 'preview.json'
    if not path.is_file():
        return dict(configuration=label, trial=trial, path=str(directory),
                    completed=False, abnormal_stop=True,
                    fault='preview.json missing (startup failure or test not run)')
    report = json.loads(path.read_text())
    row = dict(configuration=label, trial=trial, path=str(directory),
               completed=bool(report.get('completed')),
               abnormal_stop=not bool(report.get('completed')),
               fault=report.get('fault') or report.get('first_failed_sample') or '',
               failed_samples=report.get('failed_samples', 0),
               elapsed_s=report.get('elapsed_s'),
               samples_side=report.get('samples', {}).get('side'),
               samples_wrist=report.get('samples', {}).get('wrist'))
    for name in ('side', 'wrist'):
        cam = report.get('camera', {}).get(name, {})
        row[f'{name}_provider'] = cam.get('provider')
        row[f'{name}_jpeg_quality'] = cam.get('jpeg_quality') if name == 'side' else None
        for metric in ('request_ms', 'image_decode_ms', 'image_age_at_sample_ms'):
            for stat in ('mean', 'p95', 'max'):
                row[f'{name}_{metric}_{stat}'] = (cam.get(metric) or {}).get(stat)
    for metric in ('camera_response_skew_ms', 'loop_interval_ms'):
        for stat in ('p95', 'max'):
            row[f'{metric}_{stat}'] = (report.get(metric) or {}).get(stat)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', type=Path, nargs=3, required=True,
                        help='Three separate side=OBS, wrist=AVFoundation preview directories')
    parser.add_argument('--b', type=Path, nargs=3, required=True,
                        help='Three separate side=OBS, wrist=OBS preview directories')
    parser.add_argument('--label-a', default='A', help='First condition label')
    parser.add_argument('--label-b', default='B', help='Second condition label')
    parser.add_argument('--output', type=Path, required=True, help='New summary CSV path')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite summary CSV')
    if args.label_a == args.label_b:
        parser.error('Condition labels must differ')
    rows = [flatten(label, trial, directory) for label, paths in
            ((args.label_a, args.a), (args.label_b, args.b))
            for trial, directory in enumerate(paths, 1)]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for label in (args.label_a, args.label_b):
        group = [row for row in rows if row['configuration'] == label]
        print(f'{label}: completed={sum(row["completed"] for row in group)}/3; '
              f'faults={[row.get("fault") for row in group if not row["completed"]]}; '
              f'max_side_age_ms={max((row.get("side_image_age_at_sample_ms_max") or 0) for row in group):.3f}; '
              f'max_skew_ms={max((row.get("camera_response_skew_ms_max") or 0) for row in group):.3f}')
    print(args.output)


if __name__ == '__main__':
    main()
