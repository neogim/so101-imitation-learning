"""Check Dataset camera keys and print the matching ACT training command; never trains."""
import argparse
import json
from pathlib import Path
import shlex

from multi_camera import image_keys, load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera-config', type=Path, required=True)
    p.add_argument('--enable-cameras')
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--split-manifest', type=Path)
    args = p.parse_args()
    cfg = load_config(args.camera_config, args.enable_cameras)
    info = json.loads((args.dataset / 'meta/info.json').read_text())
    actual = {k for k in info['features'] if k.startswith('observation.images.')}
    expected = set(image_keys(cfg))
    if actual != expected or args.output.exists() or args.steps <= 0:
        raise ValueError(f'Dataset cameras {actual} != configured {expected}, output exists, or steps invalid')
    for key in expected:
        if info['features'][key]['shape'] != [168, 224, 3]:
            raise ValueError(f'{key}: expected a separate 168x224 RGB video')
    if info['features']['observation.state']['shape'] != [6] or info['features']['action']['shape'] != [6]:
        raise ValueError('Expected SO-101 six-axis State and Action')
    count = info['total_episodes']
    eval_split = 0.1
    episode_order = None
    if args.split_manifest is not None:
        split = json.loads(args.split_manifest.read_text())
        train = [int(x) for x in split['train_episode_indices']]
        validation = [int(x) for x in split['validation_episode_indices']]
        if (set(train) & set(validation) or set(train) | set(validation) != set(range(count))
                or not train or not validation):
            raise ValueError('Split manifest overlaps, is incomplete, or has an empty split')
        episode_order = train + validation
        eval_split = len(validation) / count
    repo_id = (f'local/muhi_act_success_{count:03d}_168x224' if len(expected) == 1
               else f'local/muhi_act_success_{count:03d}_{len(expected)}cam_168x224')
    cmd = [
        'lerobot-train', f'--dataset.repo_id={repo_id}',
        f'--dataset.root={args.dataset.resolve()}', '--dataset.video_backend=pyav',
        f'--dataset.eval_split={eval_split}', '--dataset.use_imagenet_stats=false',
        '--policy.type=act', '--policy.device=mps', '--policy.push_to_hub=false',
        '--policy.pretrained_backbone_weights=null', '--policy.chunk_size=50',
        '--policy.n_action_steps=50', '--policy.dim_model=128', '--policy.n_heads=4',
        '--policy.dim_feedforward=512', '--policy.n_encoder_layers=2',
        '--policy.n_vae_encoder_layers=2', '--policy.optimizer_lr=1e-5',
        f'--steps={args.steps}', '--batch_size=2', '--num_workers=0',
        '--eval_steps=1000', '--env_eval_freq=0', '--log_freq=50',
        '--save_freq=1000', '--save_checkpoint_to_hub=false',
        '--wandb.enable=false', f'--output_dir={args.output.resolve()}',
    ]
    if episode_order is not None:
        cmd.insert(3, '--dataset.episodes=[' + ','.join(map(str, episode_order)) + ']')
    print('Dataset cameras:', ', '.join(sorted(expected)))
    print('Episodes:', info['total_episodes'], 'Frames:', info['total_frames'])
    if episode_order is not None:
        print('Split:', len(train), 'train,', len(validation), 'validation')
    print('Command (not executed):\n' + shlex.join(cmd))


if __name__ == '__main__':
    main()
