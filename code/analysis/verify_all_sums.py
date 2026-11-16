#!/usr/bin/env python3
"""Verify every SHA256SUMS* manifest under experiments/results (recursive).

Evidence packages in this repo use two path conventions:
  - dir-local bare filenames (the ticket 22/24 convention)
  - project-root-relative paths (some later packages)

Each entry is resolved against the manifest's own directory first and the
repository root second, so no per-manifest base detection is needed.
`logs/experiments-history.md` is a mutable global log that two historical
readiness sub-manifests registered as a frozen input; this tool counts it as
a known-mutable skip (neither pass nor failure) and reports it separately.

Exits non-zero if any entry fails.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

KNOWN_MUTABLE = {"logs/experiments-history.md"}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_target(manifest: Path, relative: str) -> Path | None:
    for base in (manifest.parent, REPO_ROOT):
        candidate = base / relative
        if candidate.is_file():
            return candidate
    return None


def verify(manifest: Path) -> tuple[int, int, int, list[str]]:
    ok = fail = skipped = 0
    failures: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        expected, relative = parts[0].lower(), parts[1].strip().lstrip("*")
        if relative in KNOWN_MUTABLE:
            skipped += 1
            continue
        target = resolve_target(manifest, relative)
        if target is None:
            fail += 1
            failures.append(f"{relative} (missing)")
            continue
        if sha256_of(target) == expected:
            ok += 1
        else:
            fail += 1
            failures.append(relative)
    return ok, fail, skipped, failures


def main() -> int:
    manifests = sorted(
        path for path in RESULTS_DIR.rglob("SHA256SUMS*") if path.is_file()
    )
    if not manifests:
        print("no SHA256SUMS manifests found", file=sys.stderr)
        return 2
    total_ok = total_fail = total_skipped = 0
    for manifest in manifests:
        ok, fail, skipped, failures = verify(manifest)
        total_ok += ok
        total_fail += fail
        total_skipped += skipped
        status = "OK" if fail == 0 else "FAIL"
        label = str(manifest.parent.relative_to(RESULTS_DIR))
        print(
            f"[{status:4}] {label:58} ok={ok:<4} fail={fail:<4} "
            f"skipped={skipped:<3}"
        )
        for name in failures[:5]:
            print(f"         - {name}")
    print(
        f"\ntotal: ok={total_ok} fail={total_fail} skipped={total_skipped} "
        f"packages={len(manifests)}"
    )
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
