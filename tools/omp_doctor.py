#!/usr/bin/env python3
"""
omp_doctor.py — diagnose & repair the macOS "multiple OpenMP runtimes" crash
that kills the `mlenv` Jupyter kernel (SIGSEGV inside libomp.dylib).

ROOT CAUSE
----------
PyTorch, scikit-learn, LightGBM and XGBoost each pull in their *own* copy of
libomp.dylib. When two or more different builds load into one Python process
their OpenMP thread pools corrupt shared global state and a worker thread
segfaults in __kmp_suspend_64 / __kmp_launch_worker. The kernel dies with
"ExitCode: undefined". It is intermittent and NOT project-specific — any
notebook that imports enough of these libraries can trigger it.

THE FIX
-------
Collapse everything onto ONE OpenMP runtime. LightGBM/XGBoost hard-require the
Homebrew libomp via their embedded rpath (/opt/homebrew/opt/libomp/lib), which
we cannot cleanly change, so we make that the single hub and replace the
vendored copies shipped by torch/sklearn (and any other package) with symlinks
to it. All OpenMP 5.0-ABI builds are mutually compatible, and Homebrew's is the
newest superset.

This is idempotent and reversible (originals saved as *.orig-bak). Re-run it
after any `pip install/upgrade` of torch, scikit-learn, etc.

USAGE
-----
    python tools/omp_doctor.py            # diagnose + repair (default)
    python tools/omp_doctor.py --check    # diagnose only, exit 1 if unhealthy
    python tools/omp_doctor.py --restore  # undo: put vendored copies back
"""
from __future__ import annotations
import os, sys, glob, shutil, argparse, subprocess

ENV = sys.prefix   # conda/venv root, e.g. .../miniforge3/envs/mlenv
SITE = glob.glob(os.path.join(ENV, "lib", "python*", "site-packages"))
SITE = SITE[0] if SITE else None

HUB_CANDIDATES = [
    "/opt/homebrew/opt/libomp/lib/libomp.dylib",   # Homebrew (preferred: lgbm/xgb rpath)
    os.path.join(ENV, "lib", "libomp.dylib"),       # conda env fallback
]


def find_hub() -> str | None:
    for h in HUB_CANDIDATES:
        if os.path.exists(h):
            return os.path.realpath(h)
    return None


def vendored_copies() -> list[str]:
    # os.walk (not glob) so we also descend hidden dirs like sklearn/.dylibs/
    if not SITE:
        return []
    found = []
    for root, _dirs, files in os.walk(SITE):
        for fn in files:
            if fn.startswith("libomp") and fn.endswith(".dylib"):
                found.append(os.path.join(root, fn))
    return sorted(found)


def runtimes_in_one_process() -> set[str]:
    """Import the usual suspects together and report distinct libomp images mapped."""
    code = (
        "import os,subprocess\n"
        "try:\n import numpy,sklearn\nexcept Exception:pass\n"
        "try:\n import torch\nexcept Exception:pass\n"
        "try:\n import lightgbm,xgboost\nexcept Exception:pass\n"
        "o=subprocess.run(['vmmap','-w',str(os.getpid())],capture_output=True,text=True).stdout\n"
        "p=set(l.split('  ')[-1].strip() for l in o.splitlines() if l.strip().endswith('libomp.dylib'))\n"
        "print('\\n'.join(sorted(os.path.realpath(x) for x in p)))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return set(l for l in out.stdout.splitlines() if l.strip())


def diagnose() -> tuple[set[str], list[str]]:
    print(f"env:  {ENV}")
    print(f"site: {SITE}")
    hub = find_hub()
    print(f"hub:  {hub}")
    print("\nvendored libomp files under site-packages:")
    for f in vendored_copies():
        tag = "symlink->" + os.path.realpath(f) if os.path.islink(f) else "REAL FILE (duplicate runtime)"
        print(f"  {f}\n      {tag}")
    rts = runtimes_in_one_process()
    print(f"\ndistinct libomp runtimes that load into ONE process: {len(rts)}")
    for r in sorted(rts):
        print(f"  {r}")
    return rts, vendored_copies()


def repair() -> int:
    hub = find_hub()
    if not hub:
        print("ERROR: no libomp hub found. Install one:  brew install libomp")
        return 2
    changed = 0
    for f in vendored_copies():
        if os.path.islink(f):
            if os.path.realpath(f) != hub:
                os.remove(f); os.symlink(hub, f); changed += 1
                print(f"re-pointed symlink -> hub:  {f}")
            continue
        bak = f + ".orig-bak"
        if not os.path.exists(bak):
            shutil.copy2(f, bak)
            print(f"backed up: {bak}")
        os.remove(f); os.symlink(hub, f); changed += 1
        print(f"redirected -> hub:  {f}")
    print(f"\n{changed} file(s) changed." if changed else "\nalready healthy, nothing to change.")
    rts = runtimes_in_one_process()
    print(f"verification: {len(rts)} libomp runtime(s) now load into one process.")
    return 0 if len(rts) <= 1 else 1


def restore() -> int:
    n = 0
    for f in vendored_copies():
        bak = f + ".orig-bak"
        if os.path.islink(f) and os.path.exists(bak):
            os.remove(f); shutil.move(bak, f); n += 1
            print(f"restored original: {f}")
    print(f"{n} file(s) restored.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="diagnose only; exit 1 if unhealthy")
    ap.add_argument("--restore", action="store_true", help="undo the fix")
    a = ap.parse_args()

    if a.restore:
        sys.exit(restore())

    print("=" * 70 + "\nOpenMP doctor — mlenv kernel health check\n" + "=" * 70)
    rts, _ = diagnose()
    if a.check:
        ok = len(rts) <= 1
        print("\nHEALTHY ✅" if ok else "\nUNHEALTHY ❌ — run:  python tools/omp_doctor.py")
        sys.exit(0 if ok else 1)
    if len(rts) <= 1:
        print("\nHEALTHY ✅ — single OpenMP runtime, no repair needed.")
        sys.exit(0)
    print("\n" + "-" * 70 + "\nrepairing...\n" + "-" * 70)
    sys.exit(repair())
