#!/usr/bin/env python3
"""Compare read-only servo snapshot reports from conditions A-D; no hardware access."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def flatten(label: str, path: Path) -> dict:
    report = json.loads(path.read_text())
    summary = report["summary"]
    total = summary.get("total_duration_ms") or {}
    row = {"condition": label, "path": str(path),
           "camera": report.get("camera_condition", "unknown"),
           "samples": summary.get("samples"), "successful": summary.get("successful"),
           "errors": summary.get("errors"), "over_100ms": summary.get("over_100ms")}
    for key in ("mean", "p50", "p95", "p99", "max"):
        row[f"snapshot_{key}_ms"] = total.get(key)
    for register, stats in summary.get("register_duration_ms", {}).items():
        for key in ("mean", "p50", "p95", "p99", "max"):
            row[f"{register}_{key}_ms"] = (stats or {}).get(key)
    identities = {}
    for name, camera in (report.get("camera") or {}).items():
        identity = camera.get("identity") or {}
        identities[name] = {k: identity.get(k) for k in
                            ("localized_name", "unique_id", "source_name", "source_uuid")
                            if identity.get(k) is not None}
    row["camera_identities_json"] = json.dumps(identities, ensure_ascii=False, sort_keys=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True,
                        help="Repeat LABEL=report.json, for example A=outputs/a.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    rows = []
    for item in args.input:
        label, separator, value = item.partition("=")
        if not separator or not label or not Path(value).is_file():
            parser.error(f"Invalid LABEL=path: {item}")
        rows.append(flatten(label, Path(value)))
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
