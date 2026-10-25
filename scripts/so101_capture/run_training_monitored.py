"""Run an approved local training command and sample its process memory."""

import argparse
import json
from pathlib import Path
import subprocess
import time

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.log.exists() or args.memory.exists():
        raise ValueError("Provide a command and unused output paths")
    start = time.monotonic()
    peak_rss = samples = 0
    peak_footprint = None
    with args.log.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        target = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                memory = target.memory_info()
                peak_rss = max(peak_rss, memory.rss)
                try:
                    footprint = getattr(target.memory_full_info(), "footprint", None)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    footprint = None
                if footprint is not None:
                    peak_footprint = max(peak_footprint or 0, footprint)
                samples += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(2)
        result = dict(exit_code=process.returncode, elapsed_seconds=time.monotonic() - start,
                      peak_rss_bytes=peak_rss, peak_footprint_bytes=peak_footprint,
                      memory_samples=samples, command=command)
        args.memory.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
