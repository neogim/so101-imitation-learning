"""Real LeRobot storage with fake hardware: interrupt/fault recovery and filtering."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    for k in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY'):
        os.environ[k]='1'
    from capture_session import create_dataset,DatasetSink,Session,SimulationIO,atomic_json
    from export_act_dataset import export
    from offline_labels import update_labels
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    results=[]
    for fault in ('interrupt','communication','position_recovered','torque_failed'):
        root=a.output/fault
        io=SimulationIO()
        sink=DatasetSink(create_dataset(root,True),root)
        s=Session(io,sink,seconds=1,min_seconds=.1,countdown=0,num_episodes=3)
        # First complete episode.
        for i in range(12):
            s.tick(i*.05)
        s.tick(.6,'r')
        s.tick(.65)
        s.tick(.8,'e')
        s.tick(.85,'h')
        for j,k in enumerate('plsa'):
            s.tick(.9+j*.05,k)
        # Wait for asynchronous save outside this deterministic fake control clock.
        deadline=time.monotonic()+30
        while sink.saved<1:
            sink.check()
            if time.monotonic()>deadline:
                raise TimeoutError('save')
            time.sleep(.02)
        s.tick(1.2); s.tick(1.4,'r'); s.tick(1.45)
        try:
            if fault=='communication':
                def broken():
                    raise OSError('injected serial read failure')
                io.request=broken
                s.tick(1.5)
            elif fault in ('position_recovered','torque_failed'):
                from unittest.mock import Mock
                if fault=='position_recovered':
                    io.state=Mock(side_effect=[ConnectionError('injected position failure'),io.state()])
                else:
                    io.torque_ok=Mock(side_effect=ConnectionError('injected torque failure'))
                s.tick(1.5)
            else:
                raise KeyboardInterrupt('simulated Ctrl-C')
        except (KeyboardInterrupt,OSError):
            sent_before=len(io.sent)
            errors=s.stop(1.5)
            if fault=='interrupt':
                assert not errors,errors
            else:
                assert errors and s.io.latched
                assert len(io.sent)==sent_before
            atomic_json(root/'communication_diagnostics.json',s.io.report())
            report=json.loads((root/'communication_diagnostics.json').read_text())
            assert report['latched']==(fault!='interrupt')
        sink.close()
        s.outcomes[0].update(pick='unknown',place='unknown',overall='unknown',adopt=False)
        atomic_json(root/'episode_outcomes.json',s.outcomes)
        assert len(s.outcomes)==2 and s.outcomes[1]['end_reason']=='interrupted'
        assert s.outcomes[1]['adopt'] is False
        verify=Path(__file__).with_name('verify_dataset.py')
        subprocess.run([sys.executable,str(verify),str(root),'--output',str(a.output/(fault+'-verify')),
                        '--body-unit','degrees'],check=True,stdout=subprocess.DEVNULL)
        before=digest(root)
        labels_file=a.output/(fault+'-labels.json')
        update_labels(root,labels_file,0,pick='success',place='success',overall='success',adopt=True)
        try:
            update_labels(root,labels_file,1,pick='success',place='success',overall='success',adopt=True)
        except ValueError:
            pass
        else:
            raise AssertionError('Interrupted episode was adopted')
        derived=a.output/(fault+'-act')
        lineage=export(root,derived,labels_file)
        assert len(lineage)==1 and lineage[0]['source_episode_index']==0
        assert digest(root)==before,'source changed'
        reader=LeRobotDataset('local/derived',root=derived,video_backend='pyav')
        assert 'action_requested' not in reader.features
        assert reader.meta.total_episodes==1 and len(reader)==s.outcomes[0]['frames']
        derived_check=a.output/(fault+'-act-verify')
        subprocess.run([sys.executable,str(verify),str(derived),'--output',str(derived_check),
                        '--body-unit','degrees'],check=True,stdout=subprocess.DEVNULL)
        assert json.loads((derived_check/'report.json').read_text())['errors']==[]
        # Existing output must be rejected before source/target mutation.
        try:
            export(root,derived,labels_file)
        except FileExistsError:
            pass
        else:
            raise AssertionError('overwrite accepted')
        results.append(dict(fault=fault,raw_episodes=2,derived_episodes=1,source_unchanged=True,
                            raw_frames=sum(r['frames'] for r in s.outcomes)))
    (a.output/'report.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
