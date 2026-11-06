"""Check MojoDEC's doc strings against a per-file baseline.

`mojo doc --diagnose-missing-doc-strings` warns about public symbols without a
doc string and about doc strings without the Args, Parameters, Returns, or
Raises sections their function needs. This script counts the warnings per
file and fails when any file has more than `docs/docstring-baseline.txt`
records, so the count can only go down.

    pixi run doc-check                      # compare with the baseline
    pixi run doc-check --update             # record the current counts
    pixi run doc-check --list core/cochain  # the warnings in matching files

Run from the repository root, after `pixi run numojo`.
"""

import argparse
import collections
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(ROOT, "docs", "docstring-baseline.txt")
WARNING = re.compile(r"^(?P<path>[^:\s]+\.mojo):(?P<line>\d+):(?P<col>\d+): warning: (?P<msg>.*)$")


def run_mojo_doc():
    """The warnings, as (relative path, line, message) tuples."""
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            [
                "mojo", "doc", "-I", "build", "--diagnose-missing-doc-strings",
                "-o", os.path.join(tmp, "api.json"), "mojodec",
            ],
            cwd=ROOT, capture_output=True, text=True,
        )
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        sys.exit(out + "\nmojo doc failed (did `pixi run numojo` run?)")
    warnings = []
    for line in out.splitlines():
        m = WARNING.match(line.strip())
        if m:
            path = os.path.relpath(m["path"], ROOT)
            warnings.append((path, int(m["line"]), m["msg"]))
    return warnings


def read_baseline():
    counts = {}
    if os.path.exists(BASELINE):
        with open(BASELINE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    n, path = line.split(maxsplit=1)
                    counts[path] = int(n)
    return counts


def write_baseline(counts):
    with open(BASELINE, "w") as f:
        f.write(
            "# Doc string warnings per file, from `pixi run doc-check --update`.\n"
            "# `pixi run doc-check` fails when a file has more than this.\n"
            f"# total {sum(counts.values())}\n"
        )
        for path in sorted(counts):
            f.write(f"{counts[path]:5d} {path}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--update", action="store_true", help="record the current counts as the baseline")
    parser.add_argument("--list", metavar="PATTERN", help="print the warnings in files whose path contains PATTERN")
    args = parser.parse_args()

    warnings = run_mojo_doc()
    counts = collections.Counter(path for path, _, _ in warnings)

    if args.list is not None:
        shown = [w for w in warnings if args.list in w[0]]
        for path, line, msg in sorted(shown):
            print(f"{path}:{line}: {msg}")
        print(f"{len(shown)} warnings")
        return

    if args.update:
        write_baseline(counts)
        print(f"baseline: {sum(counts.values())} warnings in {len(counts)} files")
        return

    baseline = read_baseline()
    worse = {p: n for p, n in counts.items() if n > baseline.get(p, 0)}
    better = {p: n for p, n in counts.items() if n < baseline.get(p, 0)}
    better.update({p: 0 for p in baseline if p not in counts})
    total, before = sum(counts.values()), sum(baseline.values())
    print(f"doc string warnings: {total} (baseline {before})")
    for path in sorted(better):
        print(f"  fewer: {path}: {baseline[path]} -> {better[path]}")
    if worse:
        for path in sorted(worse):
            print(f"  MORE:  {path}: {baseline.get(path, 0)} -> {worse[path]}")
            for p, line, msg in sorted(warnings):
                if p == path:
                    print(f"    {p}:{line}: {msg}")
        sys.exit("doc-check failed: document the new code, see docs-plan.md for the format")
    if better:
        print("run `pixi run doc-check --update` to lower the baseline")


if __name__ == "__main__":
    main()
