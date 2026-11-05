#!/usr/bin/env python3
"""Install the repository privacy pre-push hook without replacing existing hooks.

Usage: python3 scripts/privacy/install_pre_push.py
Installed only in this working copy's .git/hooks. No global Git config changes.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> int:
    root = Path(git("rev-parse", "--show-toplevel"))
    configured = subprocess.run(
        ["git", "config", "--get", "core.hooksPath"], capture_output=True, text=True
    )
    if configured.returncode == 0 and configured.stdout.strip():
        print(f"Refusing: core.hooksPath already set to {configured.stdout.strip()!r}. "
              "Please integrate this check into your existing pre-push hook instead.",
              file=sys.stderr)
        return 2

    source = root / ".githooks" / "pre-push"
    if not source.is_file():
        print(f"Missing versioned hook: {source}", file=sys.stderr)
        return 2

    hooks_path = Path(git("rev-parse", "--git-path", "hooks"))
    if not hooks_path.is_absolute():
        hooks_path = root / hooks_path
    target = hooks_path / "pre-push"
    if target.exists() or target.is_symlink():
        if target.is_file() and target.read_bytes() == source.read_bytes():
            target.chmod(target.stat().st_mode | 0o111)
            print(f"Already installed: {target}")
            return 0
        print(f"Refusing to overwrite existing hook: {target}", file=sys.stderr)
        return 2
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    target.chmod(0o755)
    print(f"Installed: {target}")
    print("Pushes in THIS repository will be checked. Other repositories are unaffected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
