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
        completed = clock()
        row["register_reads"].append({"register": register,
            "duration_ms": (completed-read_started)*1000, "result": "ok",
            "joint_count": len(values)})
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
            "p50": percentile(.50), "p95": percentile(.95), "max": max(values)}


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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--interval-ms", type=float, default=50.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if not 1 <= args.samples <= 10000 or not 0 <= args.interval_ms <= 1000:
        parser.error("samples must be 1..10000 and interval-ms 0..1000")

    from probe_goal_broadcast import HardwareIO
    io = HardwareIO()
    samples = []
    metadata = None
    try:
        metadata = io.open()  # bus handshake and reads only
        if not io.calibration_matches():
            raise RuntimeError("Follower calibration mismatch; no writes attempted")
        for index in range(args.samples):
            tick = time.perf_counter()
            sample = measure_once(io.robot.bus)
            sample["index"] = index
            samples.append(sample)
            time.sleep(max(0.0, args.interval_ms/1000-(time.perf_counter()-tick)))
    finally:
        io.close_without_writes()

    report = {
        "mode": "read_only", "metadata": metadata,
        "register_order": list(REGISTERS), "nominal_threshold_ms": 100.0,
        "summary": summarize(samples), "samples": samples,
        "safety": "No Goal, torque, configure, calibration, or motor command is issued.",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
