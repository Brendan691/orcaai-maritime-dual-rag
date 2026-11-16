#!/usr/bin/env python3
"""Verify every SHA-256 entry in checksums/SHA256SUMS.txt."""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = ROOT / "checksums" / "SHA256SUMS.txt"

ok = fail = 0
for line in manifest.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    expected, rel = line.split(maxsplit=1)
    path = ROOT / rel
    if not path.is_file():
        print(f"MISSING {rel}")
        fail += 1
        continue
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual == expected:
        ok += 1
    else:
        print(f"MISMATCH {rel}")
        fail += 1
print(f"checked={ok + fail} ok={ok} fail={fail}")
sys.exit(0 if fail == 0 else 1)
