"""Sparse, timestamp-aligned offline ACT evaluation; never connects to hardware."""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch


JOINTS = ('shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
          'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos')


def sampled_positions(length):
    phases = {'start': range(min(5, length)),
              'middle': range(max(0, length // 2 - 2), min(length, length // 2 + 3)),
              'end': range(max(0, length - 5), length)}
    seen = set()
    for phase, values in phases.items():
        for index in values:
            if index not in seen:
                seen.add(index)
                yield phase, index


def summarize(samples):
    by_key = defaultdict(list)
    for sample in samples:
        error = np.asarray(sample['absolute_error'], dtype=float)
        for kind, category in (('overall','all'), ('group',sample['group']),
                               ('subgroup',sample['subgroup']), ('phase',sample['phase'])):
            by_key[(sample['split'], kind, category)].append((error, sample['gripper_outside_0_100']))
    return [dict(split=split, kind=kind, category=category, samples=len(values),
                 joint_mae={name: float(x) for name, x in zip(JOINTS,
                     np.mean([v[0] for v in values], axis=0), strict=True)},
                 gripper_outside_0_100=sum(v[1] for v in values),
                 gripper_outside_0_100_rate=sum(v[1] for v in values)/len(values))
            for (split, kind, category), values in sorted(by_key.items())]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=('eval', 'train', 'both'), default='eval')
    p.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    p.add_argument('--max-episodes', type=int, default=0, help='For a small offline smoke run')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for name in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[name] = '1'
    os.environ.setdefault('HF_HOME', '/tmp/so101-act-hf190-eval')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable in this process')
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.modeling_act import ACTPolicy

    selection = list(csv.DictReader(args.selection.open()))
    if len(selection) != 190 or [int(x['merged_episode_index']) for x in selection] != list(range(190)):
        raise ValueError('Selection must contain merged episodes 0..189 in order')
    train_sessions = {x['source'] for x in selection if x['split'] == 'train'}
    eval_sessions = {x['source'] for x in selection if x['split'] == 'eval'}
    if len([x for x in selection if x['split'] == 'train']) != 165 or len([x for x in selection if x['split'] == 'eval']) != 25 or train_sessions & eval_sessions:
        raise ValueError('Train/eval split mismatch or session leakage')
    checkpoint = args.checkpoint.resolve()
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    if (config.type, config.chunk_size, config.n_action_steps, config.temporal_ensemble_coeff) != ('act', 50, 50, None):
        raise ValueError('Unexpected ACT checkpoint configuration')
    if set(config.input_features) != {'observation.images.iphone', 'observation.state'} or set(config.output_features) != {'action'}:
        raise ValueError('Unexpected ACT features')
    if tuple(config.input_features['observation.images.iphone'].shape) != (3, 168, 224):
        raise ValueError('Image preprocessing shape mismatch')
    config.device = args.device
    config.n_action_steps = 1  # select first action from each newly predicted 50-action chunk
    policy = ACTPolicy.from_pretrained(checkpoint, config=config, local_files_only=True, strict=True)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(checkpoint),
        preprocessor_overrides={'device_processor': {'device': args.device}})
    ds = LeRobotDataset('local/muhi_act_success_190_168x224', root=args.dataset.resolve(), video_backend='pyav')
    if ds.num_episodes != 190 or len(ds) != 98977:
        raise ValueError('Dataset count mismatch')
    starts = ds.meta.episodes['dataset_from_index']
    stops = ds.meta.episodes['dataset_to_index']
    selected = [x for x in selection if args.split == 'both' or x['split'] == args.split]
    if args.max_episodes:
        selected = selected[:args.max_episodes]
    samples = []
    with torch.inference_mode():
        for row in selected:
            ep = int(row['merged_episode_index'])
            length = int(stops[ep]) - int(starts[ep])
            if length != int(row['frames']):
                raise ValueError(f'Episode length mismatch: {ep}')
            for phase, frame_index in sampled_positions(length):
                item = ds[int(starts[ep]) + frame_index]
                obs = {key: item[key].unsqueeze(0).to(args.device) for key in
                       ('observation.images.iphone', 'observation.state')}
                policy.reset()  # no queued action can shift prediction time
                prediction = postprocessor(policy.select_action(preprocessor(obs)))
                predicted = prediction.detach().cpu().numpy().reshape(-1)
                target = item['action'].numpy().reshape(-1)
                if predicted.shape != (6,) or target.shape != (6,) or not np.isfinite(predicted).all():
                    raise ValueError(f'Invalid prediction: episode {ep}, frame {frame_index}')
                samples.append(dict(episode_id=row['episode_id'], merged_episode_index=ep,
                                    frame_index=frame_index, phase=phase, split=row['split'],
                                    group=row['group'], subgroup=row['subgroup'],
                                    predicted=predicted.tolist(), target=target.tolist(),
                                    absolute_error=np.abs(predicted-target).tolist(),
                                    gripper_outside_0_100=bool(predicted[5] < 0 or predicted[5] > 100)))
            print(f'Evaluated {len(samples)} samples through episode {ep}', flush=True)
    args.output.mkdir(parents=True)
    with (args.output / 'predictions.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['episode_id','merged_episode_index','frame_index','phase','split','group','subgroup',
                         *[f'pred_{x}' for x in JOINTS], *[f'target_{x}' for x in JOINTS],
                         *[f'abs_error_{x}' for x in JOINTS], 'gripper_outside_0_100'])
        for x in samples:
            writer.writerow([x['episode_id'],x['merged_episode_index'],x['frame_index'],x['phase'],
                             x['split'],x['group'],x['subgroup'],*x['predicted'],*x['target'],
                             *x['absolute_error'],x['gripper_outside_0_100']])
    report = dict(dataset=str(args.dataset.resolve()), checkpoint=str(checkpoint),
                  samples=len(samples), episodes=len(selected), split=args.split,
                  first_action_same_timestamp=True, chunk_size=50, n_action_steps=1,
                  metrics=summarize(samples),
                  note='Teacher-forced offline MAE does not guarantee closed-loop or real-robot success.')
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'samples': len(samples), 'episodes': len(selected), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
