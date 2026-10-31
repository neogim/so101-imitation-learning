#!/usr/bin/env python3
"""Experimental one-axis SO-101 Leader->Follower teleop; explicit physical enable.

No camera, calibration writes, alignment or ACT. Default simulation ONLY.
Hardware mode writes Goal_Position and Torque_Enable: physically hazardous.
Review wiring, mechanical clearance, gravity support, E-stop before enabling.
"""
from __future__ import annotations
import argparse,csv,math,sys,time
from pathlib import Path
from datetime import datetime,timezone

NAMES=("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper")
AXES=("shoulder_lift","elbow_flex")
APPROVAL="I HAVE SECURED THE ARM AND VERIFIED EMERGENCY STOP"
MAX_S=8.0
MAX_TOTAL=1.0
MAX_STEP=.15
MIN_MARGIN_TICKS=12

def check_map(x):
    if set(x)!=set(NAMES):raise RuntimeError("Incomplete joint snapshot")
    if any(not math.isfinite(float(v)) for v in x.values()):raise RuntimeError("Nonfinite motor reading")

def run_sim(axis):
    leader={name:0.0 for name in NAMES}; follower=leader.copy()
    for cycle in range(20):
        leader[axis]=min(.7,cycle*.05)
        proposed=bounded_step(follower[axis],leader[axis],0.,MAX_STEP,MAX_TOTAL)
        follower[axis]=proposed
    assert abs(follower[axis]-.7)<1e-8
    print("SIMULATION PASS",axis,"result",follower[axis],"NO hardware connected")

def bounded_step(command,requested,base,step,total):
    if abs(requested-base)>total:raise RuntimeError("Leader exceeded excursion limit")
    candidate=command+max(-step,min(step,requested-command))
    if abs(candidate-base)>total:raise RuntimeError("Goal exceeded excursion limit")
    return candidate

def run_hardware(a):
    from lerobot.robots.so_follower import SO101Follower,SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader,SO101LeaderConfig
    from record_episode import FOLLOWER,LEADER
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION
    leader=SO101Leader(SO101LeaderConfig(port=a.leader_port or LEADER,id="my_leader_arm",use_degrees=True))
    follower=SO101Follower(SO101FollowerConfig(port=a.follower_port or FOLLOWER,
                                                id="my_follower_arm",use_degrees=True,
                                                disable_torque_on_disconnect=False))
    buses=[]
    goal_written=False
    torque_enabled=False
    started=None
    lines=[]
    fail=None
    try:
        for device in (leader,follower):
            device.bus.connect()
            buses.append(device)
        if not leader.is_calibrated or not follower.is_calibrated:raise RuntimeError("Calibration missing/mismatch")
        for device in (leader,follower):
            if set(device.bus.motors)!=set(NAMES):raise RuntimeError("Unexpected joint inventory")
        start_l=leader.bus.sync_read("Present_Position",num_retry=0)
        start_f=follower.bus.sync_read("Present_Position",num_retry=0)
        check_map(start_l);check_map(start_f)
        tl=leader.bus.sync_read("Torque_Enable",normalize=False,num_retry=0)
        tf=follower.bus.sync_read("Torque_Enable",normalize=False,num_retry=0)
        if set(tl)!=set(NAMES) or set(tf)!=set(NAMES) or any(x!=0 for x in list(tl.values())+list(tf.values())):
            raise RuntimeError("All motor torque must already be OFF")
        # Demand absolute poses near identical BEFORE switching torque on.
        if any(abs(start_l[j]-start_f[j])>1.5 for j in NAMES):
            raise RuntimeError("Leader/Follower initial pose difference >1.5 degrees/units")
        limits=follower.bus.calibration
        if set(limits)!=set(NAMES):raise RuntimeError("Missing calibration limits")
        raw=follower.bus.sync_read("Present_Position",normalize=False,num_retry=0)
        if set(raw)!=set(NAMES):raise RuntimeError("Missing raw joint readings")
        for name in NAMES:
            item=limits[name]
            low,high=item.range_min,item.range_max
            if not low<=raw[name]<=high:raise RuntimeError(f"{name} outside calibrated raw range")
        chosen=limits[a.axis]
        available=min(raw[a.axis]-chosen.range_min,chosen.range_max-raw[a.axis])
        if available<MIN_MARGIN_TICKS:raise RuntimeError(f"{a.axis} close to calibrated raw endpoint ({available} ticks)")
        # Cross-check that a +/-1 degree motion would remain inside the motor
        # calibration raw interval; not proof of mechanical collision clearance.
        max_ticks=math.ceil(MAX_TOTAL*4095/360)
        if available<max_ticks+2:raise RuntimeError("No symmetric +/-1deg calibrated margin")
        print("LIVE start positions", {j:(round(start_l[j],2),round(start_f[j],2)) for j in NAMES},flush=True)
        print("Calibrated margin for",a.axis,available,"raw ticks",flush=True)
        # Freeze ALL axes at current raw encoder position BEFORE enabling torque.
        follower.bus.sync_write("Goal_Position",raw,normalize=False,num_retry=0)
        goal_written=True
        time.sleep(.03)
        check_goal=follower.bus.sync_read("Goal_Position",normalize=False,num_retry=0)
        if any(abs(check_goal[j]-raw[j])>2 for j in NAMES):
            raise RuntimeError("Goal readback mismatch; torque NOT enabled")
        # This is actual actuation. Torque may cause mechanical motion.
        follower.bus.enable_torque()
        torque_enabled=True
        read_t=follower.bus.sync_read("Torque_Enable",normalize=False,num_retry=0)
        if set(read_t)!=set(NAMES) or any(x!=1 for x in read_t.values()):
            raise RuntimeError("Torque enable inconsistent; stop")
        hold=follower.bus.sync_read("Present_Position",num_retry=0)
        if any(abs(hold[j]-start_f[j])>.5 for j in NAMES):
            raise RuntimeError("Unexpected movement immediately after torque ON")
        command=start_f.copy()
        started=time.monotonic()
        previous=started
        # User can interrupt with Ctrl-C, but communication failure/stop does
        # not remove power. Never leave the arm unsupported.
        while time.monotonic()-started<MAX_S:
            tick=time.monotonic()
            if tick-previous>.2:raise RuntimeError("Stale control loop")
            previous=tick
            l=leader.bus.sync_read("Present_Position",num_retry=0)
            f=follower.bus.sync_read("Present_Position",num_retry=0)
            check_map(l);check_map(f)
            if any(abs(l[j]-start_l[j])>.5 for j in NAMES if j!=a.axis):
                raise RuntimeError("Nonselected Leader joint moved")
            if any(abs(f[j]-start_f[j])>.5 for j in NAMES if j!=a.axis):
                raise RuntimeError("Nonselected Follower joint moved")
            if abs(l[a.axis]-start_l[a.axis])>MAX_TOTAL:
                raise RuntimeError("Leader excursion >1 degree")
            if abs(f[a.axis]-command[a.axis])>1.0:
                raise RuntimeError("Follower tracking deviation >1 degree")
            desired=start_f[a.axis]+(l[a.axis]-start_l[a.axis])
            command[a.axis]=bounded_step(command[a.axis],desired,start_f[a.axis],MAX_STEP,MAX_TOTAL)
            # Other five goals stay at their initial observed positions.
            follower.bus.sync_write("Goal_Position",command,num_retry=0)
            lines.append({"utc":datetime.now(timezone.utc).isoformat(),"t_s":tick-started,
                          "joint":a.axis,"leader":l[a.axis],"follower":f[a.axis],
                          "sent_goal":command[a.axis]})
            time.sleep(max(0,.05-(time.monotonic()-tick)))
        print("8 second movement window completed",flush=True)
    except BaseException as exc:
        fail=f"{type(exc).__name__}: {exc}"
        print("STOP:",fail,flush=True)
    finally:
        # Do not send emergency motion or reset goal when faults occur.
        # Closing WITHOUT torque-off may leave servos holding, or torque may
        # already have been lost. Physical support and power-stop needed.
        for device in reversed(buses):
            try:device.bus.disconnect(disable_torque=False)
            except Exception as exc:print("Close warning:",exc,flush=True)
        if lines:
            a.output.parent.mkdir(parents=True,exist_ok=True)
            if a.output.exists():raise FileExistsError(a.output)
            with a.output.open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(lines[0]));w.writeheader();w.writerows(lines)
            print("Saved",len(lines),"rows to",a.output,flush=True)
        print("Torque was enabled:",torque_enabled,"; last torque state NOT verified. SUPPORT ARM.",flush=True)
        print("Goal write occurred:",goal_written,flush=True)
    if fail:raise RuntimeError(fail)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--axis",choices=AXES,default="shoulder_lift")
    p.add_argument("--hardware",action="store_true")
    p.add_argument("--approval",default="")
    p.add_argument("--leader-port")
    p.add_argument("--follower-port")
    p.add_argument("--output",type=Path,default=Path("outputs/so101-live-j2-mapping.csv"))
    a=p.parse_args()
    if not a.hardware:
        run_sim(a.axis)
        print("No device opened. --hardware is separate, explicitly approved operator procedure.")
        return
    if a.approval!=APPROVAL:p.error("Missing exact physical-safety approval phrase")
    if not sys.stdin.isatty():p.error("Real interactive terminal required")
    if a.output.exists():p.error("Output exists; will not overwrite")
    run_hardware(a)
if __name__=="__main__":main()
