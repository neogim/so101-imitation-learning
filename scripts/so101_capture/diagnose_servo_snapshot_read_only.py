#!/usr/bin/env python3
"""Measure six-axis Follower register-read latency without any motor writes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time

REGISTERS = ("Present_Position", "Goal_Position", "Torque_Enable", "Status")


def measure_once(bus, *, clock=time.perf_counter) -> dict:
    started = clock()
    row = {"started_at_monotonic": started, "register_reads": [], "result": "pending"}
    for register in REGISTERS:
        read_started = clock()
        previous = row["register_reads"][-1] if row["register_reads"] else None
        try:
            values = bus.sync_read(register, normalize=register in ("Present_Position", "Goal_Position"),
                                   num_retry=0)
        except BaseException as exc:
            completed = clock()
            row["register_reads"].append({"register": register,
                "duration_ms": (completed-read_started)*1000, "result": "error",
                "error": f"{type(exc).__name__}: {exc}"})
            row.update(result="error", failed_register=register,
                       completed_at_monotonic=completed,
                       total_duration_ms=(completed-started)*1000)
            return row
        returned = clock()
        joint_count = len(values)
        completed = clock()
        row["register_reads"].append({"register": register,
            "inter_register_gap_ms": (None if previous is None else
                (read_started-previous["completed_at_monotonic"])*1000),
            "sync_read_ms": (returned-read_started)*1000,
            "python_processing_ms": (completed-returned)*1000,
            "duration_ms": (completed-read_started)*1000, "result": "ok",
            "completed_at_monotonic": completed, "joint_count": joint_count})
    row.update(result="ok", completed_at_monotonic=completed,
               total_duration_ms=(completed-started)*1000,
               threshold_ms=100.0,
               threshold_exceeded=(completed-started) > .10)
    return row


def distribution(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    percentile = lambda q: ordered[min(len(ordered)-1, math.ceil(q*len(ordered))-1)]
    return {"count": len(values), "min": min(values), "mean": statistics.fmean(values),
            "p50": percentile(.50), "p95": percentile(.95),
            "p99": percentile(.99), "max": max(values)}


def summarize(samples: list[dict]) -> dict:
    successful = [x for x in samples if x["result"] == "ok"]
    return {
        "samples": len(samples), "successful": len(successful),
        "errors": len(samples)-len(successful),
        "over_100ms": sum(x.get("threshold_exceeded", False) for x in successful),
        "total_duration_ms": distribution([x["total_duration_ms"] for x in successful]),
        "register_duration_ms": {register: distribution([
            read["duration_ms"] for sample in successful for read in sample["register_reads"]
            if read["register"] == register]) for register in REGISTERS},
        "sync_read_ms": {register: distribution([
            read["sync_read_ms"] for sample in successful for read in sample["register_reads"]
            if read["register"] == register]) for register in REGISTERS},
        "python_processing_ms": {register: distribution([
            read["python_processing_ms"] for sample in successful for read in sample["register_reads"]
            if read["register"] == register]) for register in REGISTERS},
        "inter_register_gap_ms": distribution([
            read["inter_register_gap_ms"] for sample in successful
            for read in sample["register_reads"] if read["inter_register_gap_ms"] is not None]),
        "sample_start_lateness_ms": distribution([
            x["sample_start_lateness_ms"] for x in samples
            if x.get("sample_start_lateness_ms") is not None]),
    }


def camera_summary(rig) -> dict | None:
    if rig is None:
        return None
    result = {}
    for name, camera in rig.cameras.items():
        rows = camera.timing_snapshot()
        result[name] = {
            "identity": camera.identity,
            "fetches": len(rows),
            "total_fetch_ms": distribution([x["total_ms"] for x in rows]),
            "request_ms": distribution([x["request_ms"] for x in rows]),
            "image_decode_ms": distribution([x["image_decode_ms"] for x in rows]),
            "capture_to_host_ms": distribution([
                x["capture_to_host_ms"] for x in rows if "capture_to_host_ms" in x]),
            "native_diagnostics": list(getattr(camera, "native_diagnostics", ())),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--interval-ms", type=float, default=50.0)
    parser.add_argument("--camera-config", type=Path,
                        help="Optional camera load; still performs servo reads only")
    parser.add_argument("--enable-cameras",
                        help="Comma-separated subset such as side, wrist, or side,wrist")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if not 1 <= args.samples <= 10000 or not 0 <= args.interval_ms <= 1000:
        parser.error("samples must be 1..10000 and interval-ms 0..1000")

    if args.enable_cameras and not args.camera_config:
        parser.error("--enable-cameras requires --camera-config")
    from probe_goal_broadcast import HardwareIO
    io = HardwareIO()
    samples = []
    metadata = None
    rig = None
    try:
        if args.camera_config:
            from multi_camera import CameraRig, load_config
            rig = CameraRig(load_config(args.camera_config, args.enable_cameras),
                            collect_timings=True)
            rig.connect()  # cameras only; no robot object or motor write
        metadata = io.open()  # bus handshake and reads only
        if not io.calibration_matches():
            raise RuntimeError("Follower calibration mismatch; no writes attempted")
        started = time.perf_counter()
        for index in range(args.samples):
            due = started + index*args.interval_ms/1000
            time.sleep(max(0.0, due-time.perf_counter()))
            tick = time.perf_counter()
            sample = measure_once(io.robot.bus)
            sample["index"] = index
            sample["scheduled_at_monotonic"] = due
            sample["sample_start_lateness_ms"] = (tick-due)*1000
            samples.append(sample)
    finally:
        io.close_without_writes()
        if rig is not None:
            rig.disconnect()

    report = {
        "mode": "read_only", "metadata": metadata,
        "camera_condition": ("off" if rig is None else args.enable_cameras or
                             ",".join(rig.cameras)),
        "camera": camera_summary(rig),
        "register_order": list(REGISTERS), "nominal_threshold_ms": 100.0,
        "summary": summarize(samples), "samples": samples,
        "safety": "No Goal, torque, configure, calibration, or motor command is issued.",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
