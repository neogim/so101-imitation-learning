"""Read an active LeRobot train log and write a small CSV/SVG report."""
import argparse
import csv
import json
from pathlib import Path
import re


TRAIN = re.compile(r'\bstep:([\d.]+[KM]?)\s+smpl:.*?\bloss:([\d.eE+-]+).*?\bstep_s:([\d.eE+-]+)')
EVAL = re.compile(r'\bstep\s+(\d+):\s+eval_loss=([\d.eE+-]+)')
PROGRESS = re.compile(r'\b(\d+)/(\d+)\s*\[')
FAILURE = re.compile(r'Traceback|out of memory|MPS backend out of memory|RuntimeError:')


def svg(rows, latest_step):
    width, height = 980, 560
    xmax = max(latest_step, *(r['step'] for r in rows), 1)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="60" y="30" font-size="20">ACT training: train and validation loss</text>']
    for key, title, top, bottom, color in (
        ('train_loss', 'Train loss', 65, 285, '#2364aa'),
        ('eval_loss', 'Validation loss', 325, 535, '#d1495b'),
    ):
        values = [(r['step'], r[key]) for r in rows if r[key] is not None]
        ymax = max((v for _, v in values), default=1) * 1.05
        ymin = min((v for _, v in values), default=0)
        ymin = min(0, ymin * .95)
        parts += [f'<text x="60" y="{top+12}" font-size="15">{title}</text>',
                  f'<path d="M 60 {bottom} H 950 M 60 {top+25} V {bottom}" stroke="#888" fill="none"/>',
                  f'<text x="62" y="{top+43}" font-size="11">{ymax:.3g}</text>',
                  f'<text x="930" y="{bottom+17}" font-size="11">{xmax}</text>']
        if values:
            points = ' '.join(f'{60+890*x/xmax:.1f},{bottom-(bottom-top-25)*(v-ymin)/max(ymax-ymin,1e-9):.1f}' for x,v in values)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
            if len(values) == 1:
                parts.append(f'<circle cx="{points.split(",")[0]}" cy="{points.split(",")[1]}" r="4" fill="{color}"/>')
    parts.append('</svg>')
    return '\n'.join(parts) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--checkpoints', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    lines = args.log.read_text(errors='replace').replace('\r', '\n').splitlines()
    train, validation, progress = {}, {}, 0
    failures = []
    for line in lines:
        match = TRAIN.search(line)
        if match:
            step = (len(train) + 1) * 50  # this run logs at 50-step intervals
            train[step] = (float(match[2]), float(match[3]))
        match = EVAL.search(line)
        if match:
            validation[int(match[1])] = float(match[2])
        match = PROGRESS.search(line)
        if match and int(match[2]) == 20000:
            progress = max(progress, int(match[1]))
        if FAILURE.search(line):
            failures.append(line[-300:])
    checkpoints = sorted(int(p.name) for p in args.checkpoints.glob('[0-9]*')
                         if p.name.isdigit() and (p / 'pretrained_model/model.safetensors').is_file())
    all_steps = sorted(train.keys() | validation.keys())
    rows = [dict(step=s, train_loss=train.get(s, (None, None))[0],
                 eval_loss=validation.get(s), step_s=train.get(s, (None, None))[1],
                 checkpoint=s in checkpoints) for s in all_steps]
    args.output.mkdir(parents=True)
    with (args.output / 'loss.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=('step','train_loss','eval_loss','step_s','checkpoint'))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / 'loss.svg').write_text(svg(rows, progress))
    eligible = {s:v for s,v in validation.items() if s in checkpoints}
    best = min(eligible, key=eligible.get) if eligible else None
    report = dict(current_step=max(progress, max(train, default=0)), target_steps=20000,
                  latest_train_step=max(train, default=None),
                  latest_train_loss=train[max(train)][0] if train else None,
                  latest_step_s=train[max(train)][1] if train else None,
                  validation_losses=validation, checkpoints=checkpoints,
                  best_validation_checkpoint=best,
                  best_validation_loss=eligible[best] if best is not None else None,
                  failure_lines=failures[-5:])
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
