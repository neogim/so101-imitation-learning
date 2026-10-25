"""Offline CPU ACT forward/backward/checkpoint smoke; NOT a deployable policy."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    for k in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY'):
        os.environ[k]='1'
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    torch.set_num_threads(2)
    torch.manual_seed(42)
    ds=LeRobotDataset('local/act_smoke', root=a.root.resolve(), video_backend='pyav',
                      delta_timestamps={'action':[i/20 for i in range(4)]})
    if 'action_requested' in ds.features:
        raise ValueError('Use derived dataset with only the ACT action feature')
    cfg=ACTConfig(device='cpu', input_features={
        'observation.images.iphone':PolicyFeature(FeatureType.VISUAL,(3,480,640)),
        'observation.state':PolicyFeature(FeatureType.STATE,(6,))},
        output_features={'action':PolicyFeature(FeatureType.ACTION,(6,))},
        pretrained_backbone_weights=None, chunk_size=4, n_action_steps=4,
        dim_model=64, n_heads=4, dim_feedforward=128, n_encoder_layers=1,
        n_decoder_layers=1,n_vae_encoder_layers=1,latent_dim=8)
    policy=ACTPolicy(cfg)
    opt=torch.optim.AdamW(policy.parameters(), lr=1e-5)
    item=ds[0]
    batch={k:item[k].unsqueeze(0) for k in [*cfg.input_features,'action','action_is_pad']}
    # Same feature-wise mean/std convention, explicitly saved with smoke artifact.
    norms={}
    for k in [*cfg.input_features,'action']:
        mean=torch.as_tensor(ds.meta.stats[k]['mean'],dtype=torch.float32)
        std=torch.as_tensor(ds.meta.stats[k]['std'],dtype=torch.float32).clamp_min(1e-8)
        batch[k]=(batch[k]-mean)/std
        norms[k]={'mean':mean.tolist(),'std':std.tolist()}
    start=time.perf_counter()
    policy.train()
    loss,details=policy(batch)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite loss')
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None):
        raise ValueError('Nonfinite gradients')
    opt.step()
    elapsed=time.perf_counter()-start
    a.output.mkdir(parents=True)
    policy.save_pretrained(a.output/'checkpoint')
    restored=ACTPolicy.from_pretrained(a.output/'checkpoint',local_files_only=True)
    restored.eval()
    with torch.no_grad():
        action=restored.select_action({k:v for k,v in batch.items() if k in cfg.input_features})
    assert tuple(action.shape)==(1,6) and torch.isfinite(action).all()
    (a.output/'normalization.json').write_text(json.dumps(norms,indent=2))
    result=dict(loss=float(loss.detach()),details=details,step_s=elapsed,device='cpu',
        image_shape=list(batch['observation.images.iphone'].shape),action_shape=list(batch['action'].shape),
        checkpoint_reload=True, action_output_shape=list(action.shape),
        warning='Reduced random-init architecture, synthetic data, one step. NOT for hardware deployment. Action is normalized.')
    (a.output/'report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
