"""Run the cross-library comparison and write a results table.

    pixi run bench-compare [--repeats N] [--libs mojodec,pydec,...]
                           [--meshes tri_100,tet_10,...] [--timeout S]

Steps (each skipped when its output exists):
1. write the meshes (meshes.mojo) to build/compare/meshes;
2. build mojodec_bench.mojo (release, -D ASSERT=none) and the Rust crate
   (dexterior, formoniq; pinned in rust/Cargo.toml);
3. clone PyDEC at PYDEC_REV into build/compare/pydec and apply
   pydec-numpy2.patch, unless PYDEC_PATH names a checkout to use instead;
4. run each library on each mesh in its own process, single-threaded and,
   for the multithreaded libraries (MojoDEC, formoniq), on all cores;
5. write build/compare/results/<time>.jsonl (every measurement) and
   <time>.md (tables), and print the tables.

See README.md here for what each phase measures and how to read `check`.
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "benchmarks" / "compare"
BUILD = ROOT / "build" / "compare"
MESHES = BUILD / "meshes"
RESULTS = BUILD / "results"
MOJO_BIN = BUILD / "mojodec_bench"
TARGET = BUILD / "target"
RUST_BIN = TARGET / "release" / "dec-compare"

PYDEC_URL = "https://github.com/hirani/pydec.git"
PYDEC_REV = "c181202c70e79794dcc8fa4577537a4eb4987b2a"

LIBS = ["mojodec", "pydec", "dexterior", "formoniq"]
# PyDEC's geometry is pure Python and takes minutes on the large meshes,
# so it runs only when asked for (--libs mojodec,pydec,...)
DEFAULT_LIBS = ["mojodec", "dexterior", "formoniq"]
MULTITHREADED = {"mojodec", "formoniq"}
LABELS = {"mojodec": "MojoDEC", "pydec": "PyDEC", "dexterior": "dexterior",
          "formoniq": "formoniq"}

# a check agrees when within this relative difference of MojoDEC's
CHECK_RTOL = 1e-9


def sh(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), file=sys.stderr, flush=True)
    subprocess.run(cmd, check=True, **kw)


def prepare(libs):
    if not (MESHES / "meshes.txt").exists():
        sh(["mojo", "run", "-I", ".", "-I", "build",
            "benchmarks/compare/meshes.mojo", str(MESHES)], cwd=ROOT)
    if "mojodec" in libs:
        sh(["mojo", "build", "-D", "ASSERT=none", "-I", ".", "-I", "build",
            "benchmarks/compare/mojodec_bench.mojo", "-o", str(MOJO_BIN)],
           cwd=ROOT)
    if {"dexterior", "formoniq"} & set(libs):
        sh(["cargo", "build", "--release", "--quiet"], cwd=HERE / "rust",
           env={**os.environ, "CARGO_TARGET_DIR": str(TARGET)})
    if "pydec" in libs:
        return pydec_path()
    return None


def pydec_path():
    """The directory to put on PYTHONPATH for `import pydec`."""
    if os.environ.get("PYDEC_PATH"):
        return Path(os.environ["PYDEC_PATH"]).resolve()
    repo = BUILD / "pydec"
    if not repo.exists():
        sh(["git", "clone", "--quiet", PYDEC_URL, str(repo)])
        sh(["git", "checkout", "--quiet", PYDEC_REV], cwd=repo)
        sh(["git", "apply", str(HERE / "pydec-numpy2.patch")], cwd=repo)
    return repo / "pydec"


def command(lib, mesh, repeats, pydec):
    if lib == "mojodec":
        return [str(MOJO_BIN), str(MESHES), str(repeats), mesh]
    if lib == "pydec":
        return ["pixi", "run", "--quiet", "-e", "bench", "python", "-W",
                "ignore", str(HERE / "pydec_bench.py"), str(MESHES),
                str(repeats), mesh]
    return [str(RUST_BIN), lib, str(MESHES), str(repeats), mesh]


def thread_env(threads):
    """Environment capping every library's (and BLAS's) thread count;
    `threads=None` leaves them at their defaults (all cores)."""
    env = dict(os.environ)
    for var in ["MOJODEC_NUM_THREADS", "RAYON_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
        env.pop(var, None)
        if threads is not None:
            env[var] = str(threads)
    return env


def run_one(lib, mesh, threads, repeats, pydec, timeout):
    env = thread_env(threads)
    if pydec is not None:
        env["PYTHONPATH"] = str(pydec)
    tag = "1" if threads == 1 else "all"
    print(f"{LABELS[lib]} {mesh} threads={tag}", file=sys.stderr, flush=True)
    try:
        out = subprocess.run(command(lib, mesh, repeats, pydec), env=env,
                             capture_output=True, text=True, timeout=timeout,
                             cwd=ROOT)
    except subprocess.TimeoutExpired:
        return [{"lib": lib, "mesh": mesh, "threads": tag,
                 "error": f"timeout after {timeout} s"}]
    records = []
    for line in out.stdout.splitlines():
        if line.startswith("{"):
            rec = json.loads(line)  # Python's json accepts Rust's bare NaN
            rec["threads"] = tag
            records.append(rec)
    if out.returncode != 0:
        records.append({"lib": lib, "mesh": mesh, "threads": tag,
                        "error": out.stderr.strip()[-500:]})
    return records


def machine():
    cpu = platform.processor()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass

    def version(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=ROOT).stdout.strip().splitlines()[0]
        except (OSError, IndexError):
            return "?"

    return {
        "cpu": cpu,
        "cores": os.cpu_count(),
        "mojodec": version(["git", "rev-parse", "--short", "HEAD"]),
        "mojo": version(["mojo", "--version"]),
        "rustc": version(["rustc", "--version"]),
        "pydec": PYDEC_REV[:7] if not os.environ.get("PYDEC_PATH")
        else "PYDEC_PATH",
    }


# ===------------------------------------------------------------------=== #
# Tables
# ===------------------------------------------------------------------=== #

def agrees(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x != x or y != y:  # NaN
            return False
        if abs(x - y) > CHECK_RTOL * max(abs(x), abs(y), 1e-300):
            return False
    return True


def check_sources(by, lib, mesh, phase, t, r):
    """The (record, MojoDEC phase) pairs whose checks back a table cell.
    dexterior times only `build` and reports the topology and geometry
    checks separately."""
    if lib == "dexterior" and phase == "build":
        pairs = [(by.get((lib, mesh, "topology_check", t)), "topology"),
                 (by.get((lib, mesh, "geometry_check", t)), "geometry")]
    else:
        pairs = [(r, phase)]
    return [(rec, ph) for rec, ph in pairs if rec and rec.get("check")]


def columns(libs):
    cols = []
    for lib in libs:
        cols.append((lib, "1"))
        if lib in MULTITHREADED:
            cols.append((lib, "all"))
    return cols


def fmt_ms(ms):
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    if ms >= 10:
        return f"{ms:.0f} ms"
    return f"{ms:.2f} ms"


def tables(records, libs, meta):
    by = {}
    errors = []
    for r in records:
        if "error" in r:
            errors.append(r)
            continue
        by[(r["lib"], r["mesh"], r["phase"], r["threads"])] = r
    cols = columns(libs)
    lines = [
        f"Machine: {meta['cpu']}, {meta['cores']} logical cores. "
        f"MojoDEC {meta['mojodec']} ({meta['mojo']}), {meta['rustc']}, "
        f"PyDEC {meta['pydec']}.",
        "",
        "Best of the repeats, in each library's own build. `1` = one "
        "thread, `all` = all cores (MojoDEC and formoniq only; PyDEC and "
        "dexterior are single-threaded). In brackets: time relative to "
        "MojoDEC with the same threads. "
        f"`†`: checks differ from MojoDEC's by more than {CHECK_RTOL:g} "
        "relative; `NaN`: the library's own checks contain NaN. "
        "formoniq's `geometry` is squared edge lengths only (it has no "
        "circumcentric dual), so it is not comparable with the others'.",
        "",
    ]
    meshes = sorted({r["mesh"] for r in records},
                    key=lambda m: (m.split("_")[0] != "tri",
                                   int(m.split("_")[1])))
    for mesh in meshes:
        n = 2 if mesh.startswith("tri") else 3
        phases = ["topology", "geometry", "build"] + \
            [f"whitney_{k}" for k in range(n + 1)]
        lines.append(f"### {mesh}")
        lines.append("")
        lines.append("| phase | " + " | ".join(
            f"{LABELS[l]} {t}" for l, t in cols) + " |")
        lines.append("|---|" + "---:|" * len(cols))
        for phase in phases:
            row = [phase]
            for lib, t in cols:
                r = by.get((lib, mesh, phase, t))
                if r is None:
                    row.append("—")
                    continue
                cell = fmt_ms(r["ms"])
                base = by.get(("mojodec", mesh, phase, t))
                if lib != "mojodec" and base is not None and base["ms"] > 0:
                    cell += f" ({r['ms'] / base['ms']:.0f}×)" \
                        if r["ms"] / base["ms"] >= 10 else \
                        f" ({r['ms'] / base['ms']:.1f}×)"
                for rec, ph in check_sources(by, lib, mesh, phase, t, r):
                    vals = rec["check"]
                    base_rec = by.get(("mojodec", mesh, ph, t))
                    if any(v != v for v in vals):
                        cell += " NaN"
                    elif lib != "mojodec" and base_rec is not None and \
                            not agrees(vals, base_rec["check"]):
                        cell += " †"
                row.append(cell)
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    if errors:
        lines.append("### Errors")
        lines.append("")
        for e in errors:
            where = f"{LABELS[e['lib']]} {e['mesh']} {e.get('phase', '')} " \
                f"threads={e['threads']}"
            lines.append(f"- {where}: {e['error'].splitlines()[-1]}")
        lines.append("")
    return "\n".join(lines)


def load(path):
    """The metadata and records of a results file."""
    with open(path) as f:
        first, *rest = f.read().splitlines()
    return json.loads(first), [json.loads(l) for l in rest]


def comparison(old, new):
    """MojoDEC's times in `new` against `old`, phase by phase: new / old,
    so below 1 is faster. Checks that changed are marked `check!`."""
    def index(records):
        return {(r["mesh"], r["phase"], r["threads"]): r for r in records
                if r.get("lib") == "mojodec" and "error" not in r}
    a, b = index(old), index(new)
    lines = ["### MojoDEC against the earlier run (new / old)", "",
             "| mesh | phase | threads | old | new | ratio |",
             "|---|---|---|---:|---:|---:|"]
    for key in sorted(b, key=lambda k: (k[0].split("_")[0] != "tri",
                                        int(k[0].split("_")[1]), k[1], k[2])):
        if key not in a:
            continue
        ra, rb = a[key], b[key]
        ratio = rb["ms"] / ra["ms"] if ra["ms"] > 0 else float("nan")
        flag = "" if ra.get("check") == rb.get("check") else " check!"
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {fmt_ms(ra['ms'])} "
                     f"| {fmt_ms(rb['ms'])} | {ratio:.2f}{flag} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--libs", default=",".join(DEFAULT_LIBS),
                    help="comma-separated; pydec is not run unless named "
                    "here (a full PyDEC run takes about 12 minutes)")
    ap.add_argument("--meshes", default="")
    ap.add_argument("--timeout", type=float, default=3600,
                    help="seconds per library and mesh")
    ap.add_argument("--compare", metavar="OLD_JSONL",
                    help="after the run (or with --report, for that run), "
                    "print MojoDEC's times against an earlier run's")
    ap.add_argument("--report", metavar="JSONL",
                    help="only print the tables of an earlier run")
    args = ap.parse_args()
    libs = [l for l in args.libs.split(",") if l]
    for lib in libs:
        if lib not in LIBS:
            sys.exit(f"unknown library {lib}; choose from {LIBS}")

    if args.report:
        meta, records = load(args.report)
        print(tables(records, meta.get("libs", LIBS), meta))
        if args.compare:
            print(comparison(load(args.compare)[1], records))
        return

    pydec = prepare(libs)
    meshes = [m for m in args.meshes.split(",") if m] or \
        [l.split()[0] for l in open(MESHES / "meshes.txt") if l.strip()]
    meta = machine()
    meta["libs"] = libs
    meta["repeats"] = args.repeats
    stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    RESULTS.mkdir(parents=True, exist_ok=True)
    jsonl = RESULTS / f"{stamp}.jsonl"
    records = []
    with open(jsonl, "w") as out:
        out.write(json.dumps(meta) + "\n")
        for mesh in meshes:
            for lib in libs:
                for threads in ([1, None] if lib in MULTITHREADED else [1]):
                    for rec in run_one(lib, mesh, threads, args.repeats,
                                       pydec if lib == "pydec" else None,
                                       args.timeout):
                        records.append(rec)
                        out.write(json.dumps(rec) + "\n")
                        out.flush()
    text = tables(records, libs, meta)
    if args.compare:
        text += "\n" + comparison(load(args.compare)[1], records)
    (RESULTS / f"{stamp}.md").write_text(text + "\n")
    print(text)
    print(f"\nwrote {jsonl} and {jsonl.with_suffix('.md')}", file=sys.stderr)


if __name__ == "__main__":
    main()
