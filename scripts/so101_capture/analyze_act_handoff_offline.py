#!/usr/bin/env python3
"""Analyze saved SO-101 handoff events offline. Never connects to hardware."""
import argparse,json,statistics,collections
from pathlib import Path
JOINTS=("shoulder_pan.pos","shoulder_lift.pos","elbow_flex.pos",
        "wrist_flex.pos","wrist_roll.pos","gripper.pos")
def stats(v):
 if not v:return None
 return {"min":round(min(v),3),"max":round(max(v),3),
         "mean":round(statistics.mean(v),3),"first":round(v[0],3),"last":round(v[-1],3)}
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--report",type=Path,required=True)
 p.add_argument("--output",type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():p.error("Output exists: "+str(a.output))
 report=json.loads(a.report.read_text())
 events=[e for e in report.get("events",[]) if e.get("event")=="act_sent"]
 if not events:raise SystemExit("No act_sent events in report")
 result={"source":str(a.report),"status":"OFFLINE_NO_HARDWARE",
         "act_sent_events":len(events),"event_counts":dict(collections.Counter(e.get("event") for e in report.get("events",[]))),
         "caution":"act_sent contains post-guard command; raw ACT prediction is not stored for every ACT cycle. Cannot separate model prediction from guard clipping without additional traces."}
 for name in JOINTS:
  records={}
  for field in ("sent","pre_send_present","pre_send_goal"):
   vals=[]
   for e in events:
    dic=e.get(field)
    if not isinstance(dic,dict) or name not in dic:continue
    vals.append(float(dic[name]))
   records[field]=stats(vals)
  pairs=[(e["sent"][name],e["pre_send_present"][name]) for e in events
         if isinstance(e.get("sent"),dict) and isinstance(e.get("pre_send_present"),dict)
         and name in e["sent"] and name in e["pre_send_present"]]
  if pairs:
   records["sent_minus_present_abs"]=stats([abs(float(s)-float(v)) for s,v in pairs])
  result[name]=records
 reasons=collections.Counter()
 for e in events:
  for reason in e.get("reasons",[]) or []:
   reasons[str(reason)]+=1
 result["guard_reasons"]=dict(reasons)
 timing=report.get("cycle_timings",[])
 result["cycle_over_50ms"]=report.get("cycle_over_50ms")
 result["cycle_interval_over_60ms"]=report.get("cycle_interval_over_60ms")
 result["continuous_completion"]=report.get("continuous_act_completion")
 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+"\n")
 print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
if __name__=="__main__":main()
