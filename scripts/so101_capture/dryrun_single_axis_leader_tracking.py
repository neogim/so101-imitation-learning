#!/usr/bin/env python3
"""Offline SO-101 single-axis Leader-Follower tracking-limit dry run.

No hardware imports or connections. The 'command' produced here is only a
numeric proposal; it is never sent to the robot. Tests a conservative
single-axis command policy before any hardware integration.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

ROTARY=("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll")
JOINTS=(*ROTARY,"gripper")
AXES=("shoulder_lift","elbow_flex")

class Unsafe(RuntimeError):
    pass

def guard(initial, measured, leader, previous, axis, *, dt=.05, limit=2.0, rate=5.0, tracking=1.0):
    if axis not in AXES:raise Unsafe("Only J2 or J3 are permitted")
    if not (0<dt<=.15 and 0<limit<=2 and 0<rate<=5 and 0<tracking<=1):raise Unsafe("Bad safety settings")
    if any(set(data)!=set(JOINTS) for data in (initial,measured,leader,previous)):
        raise Unsafe("Incomplete joint snapshot")
    for data in (initial,measured,leader,previous):
        if not all(math.isfinite(float(v)) for v in data.values()):raise Unsafe("Nonfinite joint")
    # Detect unexpected motion of any unselected axis (including the gripper).
    for joint in JOINTS:
        if joint!=axis and abs(measured[joint]-initial[joint])>.5:
            raise Unsafe(f"Nonselected joint moved: {joint}")
    if abs(measured[axis]-previous[axis])>tracking:
        raise Unsafe("Selected joint tracking error too large")
    if abs(leader[axis]-initial[axis])>limit:
        raise Unsafe("Leader moved beyond maximum allowed displacement")
    if abs(leader[axis]-measured[axis])>tracking:
        raise Unsafe("Leader-Follower deviation too large")
    for joint in JOINTS:
        if joint!=axis and abs(leader[joint]-initial[joint])>.5:
            raise Unsafe(f"Leader moved unexpected axis: {joint}")
    delta=max(-rate*dt,min(rate*dt,leader[axis]-previous[axis]))
    candidate=previous.copy()
    candidate[axis]=previous[axis]+delta
    if abs(candidate[axis]-initial[axis])>limit:
        raise Unsafe("Cumulative excursion too large")
    return candidate

def simulate(axis,output):
    initial=dict(zip(JOINTS,(-5.,-40.,20.,10.,-15.,10.)))
    previous=initial.copy()
    rows=[]
    fault=None
    for i in range(30):
        leader=initial.copy()
        leader[axis]+=min(1.0,i*.1)
        measured=previous.copy()
        try:
            command=guard(initial,measured,leader,previous,axis)
        except Unsafe as exc:
            fault=str(exc);break
        rows.append(dict(step=i,axis=axis,leader_deg=leader[axis],
                         measured_deg=measured[axis],proposed_deg=command[axis],
                         command_delta_deg=command[axis]-previous[axis]))
        previous=command
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    with output.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    return dict(mode="SIMULATION_ONLY",axis=axis,cycles=len(rows),
                max_proposed_offset_deg=max(abs(r["proposed_deg"]-initial[axis]) for r in rows),
                fault=fault,output=str(output))

def selftest():
    origin=dict(zip(JOINTS,(-5.,-40.,20.,10.,-15.,10.)))
    axis="shoulder_lift"
    wanted=origin.copy();wanted[axis]+=.2
    assert abs(guard(origin,origin,wanted,origin,axis)[axis]-(-39.8))<1e-6
    for data,needle in [
        ({**wanted,axis:origin[axis]+2.1},"maximum"),
        ({**wanted,"wrist_roll":origin["wrist_roll"]+1},"unexpected")]:
        try:guard(origin,origin,data,origin,axis)
        except Unsafe as exc:assert needle in str(exc)
        else:raise AssertionError("Missing rejection")
    try:guard(origin,{**origin,"wrist_flex":origin["wrist_flex"]+1},wanted,origin,axis)
    except Unsafe as exc:assert "Nonselected" in str(exc)
    else:raise AssertionError("Nonselected motion not rejected")
    print("Self-test PASS: 4 checks; zero hardware access")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--axis",choices=AXES,default="shoulder_lift")
    p.add_argument("--output",type=Path,default=Path("outputs/so101-j2j3-teleop-dryrun.csv"))
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:selftest();return
    print(json.dumps(simulate(a.axis,a.output),indent=2))
    print("NO motor commands were sent. Do not replay proposed positions directly.")
if __name__=="__main__":
    main()
