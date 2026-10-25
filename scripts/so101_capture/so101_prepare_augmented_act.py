#!/usr/bin/env python3
"""Make a separate LeRobot ACT training config with mild photometric augmentation.

Does not train, change datasets, edit the baseline config, or access robot hardware.
Use --run only after reviewing the new JSON and confirming LeRobot compatibility.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-config', type=Path, required=True,
                   help='train_config.json from the completed 53-episode baseline')
    p.add_argument('--output-config', type=Path, required=True,
                   help='New JSON file, must not already exist')
    p.add_argument('--output-dir', type=Path, required=True,
                   help='New training output directory, must not already exist')
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--run', action='store_true', help='Start training; default is config generation only')
    args = p.parse_args()

    source = args.baseline_config.resolve()
    dest = args.output_config.resolve()
    run_output = args.output_dir.resolve()
    if not source.is_file():
        p.error(f'baseline config not found: {source}')
    if dest.exists() or run_output.exists():
        p.error('output config or output training directory already exists; refusing overwrite')
    if args.steps <= 0:
        p.error('steps must be positive')
    cfg = json.loads(source.read_text(encoding='utf-8'))
    dataset = cfg.get('dataset')
    policy = cfg.get('policy')
    if not isinstance(dataset, dict) or not isinstance(policy, dict):
        p.error('Expected a LeRobot train_config.json with dataset and policy objects')
    if policy.get('type') != 'act':
        p.error('Source policy is not ACT')
    data_root = Path(dataset.get('root') or '').expanduser()
    info_path = data_root / 'meta' / 'info.json'
    if not info_path.is_file():
        p.error(f'Dataset meta/info.json not found: {info_path}')
    info = json.loads(info_path.read_text(encoding='utf-8'))
    cameras = {'observation.images.side', 'observation.images.wrist'}
    features = info.get('features', {})
    if info.get('total_episodes') != 53 or info.get('total_frames') != 30846 or not cameras.issubset(features):
        p.error('Expected 53 episodes, 30846 frames, side and wrist cameras')
    for camera in cameras:
        if features[camera].get('shape') != [168, 224, 3]:
            p.error(f'Unexpected camera shape for {camera}: {features[camera].get("shape")}')
    if dataset.get('image_transforms', {}).get('enable'):
        p.error('Baseline config already has image transforms enabled')

    new = copy.deepcopy(cfg)
    new['dataset']['image_transforms'] = {
        'enable': True,
        'max_num_transforms': 2,
        'random_order': False,
        'tfs': {
            'brightness': {'weight': 1.0, 'type': 'ColorJitter', 'kwargs': {'brightness': [0.85, 1.15]}},
            'contrast': {'weight': 1.0, 'type': 'ColorJitter', 'kwargs': {'contrast': [0.85, 1.15]}},
            'saturation': {'weight': 1.0, 'type': 'ColorJitter', 'kwargs': {'saturation': [0.90, 1.10]}},
            'hue': {'weight': 1.0, 'type': 'ColorJitter', 'kwargs': {'hue': [-0.02, 0.02]}},
        },
    }
    new['output_dir'] = str(run_output)
    new['resume'] = False
    new['steps'] = args.steps
    # Never inherit a trained policy as initialization for a fair from-scratch comparison.
    for k in ('pretrained_path',):
        if new['policy'].get(k) not in (None, ''):
            p.error(f'Baseline policy has {k} set; inspect manually instead of continuing')
    if new['dataset'].get('eval_split', 0) <= 0:
        p.error('Source config has no validation split')
    if new.get('wandb', {}).get('enable'):
        p.error('Source config enables wandb; disable explicitly before continuing')
    new['policy']['push_to_hub'] = False
    new['save_checkpoint_to_hub'] = False

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(new, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    command = ['lerobot-train', '--config_path', str(dest)]
    print('Saved augmented ACT config:', dest)
    print('Dataset:', data_root)
    print('Training output:', run_output)
    print('Augmentations: brightness, contrast, saturation, hue ONLY (up to 2 per frame)')
    print('Command:', ' '.join(command))
    print('IMPORTANT: Check the generated config and validate with a short run first.')
    if args.run:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
