#!/usr/bin/env python3
"""Prevent accidental publication of machine-local usernames and metadata.

Reads Git objects scheduled for push; does not change commits or files.
Triggered by .githooks/pre-push. This is a guardrail, not a guarantee of privacy.
"""
from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import re
import subprocess
import sys
import zlib

ZERO = "0" * 40
WINDOWS_HOME = re.compile(r"(?i)[a-z]:[\\/]+users[\\/]+([^\\/\s\"']+)")
UNIX_HOME = re.compile(r"(?i)/(?:users|home)/([^/\s\"']+)")
SENSITIVE_KEYS = re.compile(r"(?i)(gps|location|latitude|longitude|author|creator|comment|description)")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".heic", ".tif", ".tiff")
MAX_BYTES = 12 * 1024 * 1024  # Bigger files require separate review, rather than silently skipped.


def git(*args: str, input_text: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], input=input_text, text=True, capture_output=True, check=check,
    )


def local_names() -> set[str]:
    # Names are determined on the user's machine and never hard-coded into the repo.
    names = {Path.home().name, os.environ.get("USER", ""), os.environ.get("USERNAME", "")}
    try:
        names.add(getpass.getuser())
    except OSError:
        pass
    names.update(v.strip() for v in os.environ.get("PRIVACY_EXTRA_IDENTIFIERS", "").split(","))
    return {v.lower() for v in names if v and len(v) >= 4 and v.lower() not in {"root", "user", "test", "admin", "runner"}}


def find_issues(path: str, content: bytes, *, names: set[str]) -> list[str]:
    issues: list[str] = []
    name = path.lower()
    for pattern in (UNIX_HOME, WINDOWS_HOME):
        for match in pattern.finditer(path):
            issues.append(f"username in tracked filename ({match.group(1)[:2]}***)")
    if any(v in name for v in names):
        issues.append("local username/identifier in tracked filename")

    # PNG metadata chunks: reject ancillary text & EXIF metadata outright.
    # PNG pixels themselves are unaffected; text chunks may contain timestamps or GPS.
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        i = 8
        while i + 12 <= len(content):
            size = int.from_bytes(content[i:i+4], "big")
            if size < 0 or i + 12 + size > len(content):
                issues.append("malformed PNG chunk structure")
                break
            typ = content[i+4:i+8]
            if typ in (b"eXIf", b"tEXt", b"iTXt", b"zTXt"):
                issues.append(f"PNG metadata chunk {typ.decode('ascii')}: manual review/removal required")
            i += 12 + size
            if typ == b"IEND":
                break
        return issues
    if content.startswith(b"\xff\xd8") and b"Exif\x00\x00" in content[:65536]:
        issues.append("JPEG EXIF metadata: manual review/removal required")
        return issues

    if name.endswith(IMAGE_SUFFIXES):
        # Non-PNG images can contain embedded metadata requiring manual verification.
        issues.append("non-PNG image: manually check EXIF/GPS before push")
        return issues

    if b"\0" in content[:8192]:
        return issues  # Other binary formats require their own audit; no claim made.
    try:
        value = content.decode("utf-8")
    except UnicodeError:
        return issues

    for m in UNIX_HOME.finditer(value):
        issues.append(f"absolute home path containing username ({m.group(1)[:2]}***)")
    for m in WINDOWS_HOME.finditer(value):
        issues.append(f"Windows user home path ({m.group(1)[:2]}***)")

    # Inspect literal user-specific identifiers, not a hard-coded account name.
    for v in names:
        if re.search(r"(?<![A-Za-z0-9_.-])" + re.escape(v) + r"(?![A-Za-z0-9_.-])", value, re.I):
            issues.append("local username/identifier found in text")
            break
    return list(dict.fromkeys(issues))


def parse_push_input(lines: str) -> list[tuple[str, str]]:
    ranges = []
    for line in lines.splitlines():
        columns = line.split()
        if len(columns) != 4:
            raise ValueError("Unexpected pre-push hook input")
        _local_ref, local_sha, _remote_ref, remote_sha = columns
        if local_sha == ZERO or local_sha.startswith(ZERO):  # delete ref
            continue
        if not re.fullmatch(r"[0-9a-f]{40,64}", local_sha):
            raise ValueError("Unrecognized Git object ID")
        if remote_sha == ZERO or remote_sha.startswith(ZERO):
            # New branch: scan objects not already reachable from any local remote-tracking ref.
            ranges.append((local_sha, None))
        else:
            if not re.fullmatch(r"[0-9a-f]{40,64}", remote_sha):
                raise ValueError("Unrecognized remote Git object ID")
            ranges.append((local_sha, remote_sha))
    return ranges


def candidate_blobs(pushes: list[tuple[str, str]]) -> dict[str, str]:
    objects: dict[str, str] = {}
    for local, remote in pushes:
        args = ["rev-list", "--objects", local]
        if remote:
            args.append("^" + remote)
        else:
            args.extend(["--not", "--remotes"])
        proc = git(*args)
        for line in proc.stdout.splitlines():
            sha, sep, path = line.partition(" ")
            if not sep:
                continue
            objects[sha] = path
    return objects


def scan(push_input: str) -> tuple[int, list[str]]:
    pushes = parse_push_input(push_input)
    if not pushes:
        return 0, ["No new commits to scan (ref deletion or empty push)."]
    names = local_names()
    candidates = candidate_blobs(pushes)
    issues = []
    inspected = 0
    for sha, path in sorted(candidates.items(), key=lambda kv: kv[1]):
        kind = git("cat-file", "-t", sha).stdout.strip()
        if kind != "blob":
            continue
        size = int(git("cat-file", "-s", sha).stdout.strip())
        if size > MAX_BYTES:
            issues.append(f"{path}: large file ({size} bytes) requires manual privacy review")
            continue
        content = subprocess.run(["git", "cat-file", "blob", sha], capture_output=True, check=True).stdout
        inspected += 1
        for issue in find_issues(path, content, names=names):
            issues.append(f"{path}: {issue}")
    if issues:
        return 1, [f"Privacy pre-push check FAILED ({inspected} blobs inspected):", *issues,
                   "Remove sensitive text/metadata from the commit, or inspect and remediate history before retrying."]
    return 0, [f"Privacy pre-push check passed ({inspected} blobs inspected).",
               "This scans committed blobs and image metadata only; visible scene details, archives and all binary formats are not fully audited."]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check-working-file", type=Path,
                   help="test a working-tree file separately, without pushing")
    args = p.parse_args()
    if args.check_working_file:
        b = args.check_working_file.read_bytes()
        issues = find_issues(str(args.check_working_file), b, names=local_names())
        for issue in issues:
            print("BLOCK:", issue)
        if not issues:
            print("No supported privacy issues found; not a comprehensive privacy audit.")
        return int(bool(issues))
    try:
        status, lines = scan(sys.stdin.read())
    except (subprocess.CalledProcessError, OSError, ValueError) as exc:
        print("Privacy pre-push check ERROR; blocking push:", exc, file=sys.stderr)
        return 2
    print("\n".join(lines), file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
