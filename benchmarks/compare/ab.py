"""A/B timing of two mojodec_bench builds, run alternately (best of each).

    python benchmarks/compare/ab.py OLD_BIN NEW_BIN ROUNDS MESH...

Runs that are minutes apart differ by up to 30% on a desktop; alternating
the two binaries in one session is what makes a 10% change visible.
Build OLD_BIN from a git worktree of the earlier commit with
`mojo build -D ASSERT=none -I . -I build benchmarks/compare/mojodec_bench.mojo`.
"""
import json
import os
import subprocess
import sys

old, new, rounds = sys.argv[1], sys.argv[2], int(sys.argv[3])
rest = sys.argv[4:]
threads_list = ["1", "all"]
meshes = [m for m in rest]
MESHES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "build", "compare", "meshes")
best = {}
checks = {}
for threads in threads_list:
    env = dict(os.environ)
    env.pop("MOJODEC_NUM_THREADS", None)
    if threads == "1":
        env["MOJODEC_NUM_THREADS"] = "1"
    for _ in range(rounds):
        for tag, binary in (("old", old), ("new", new)):
            for mesh in meshes:
                out = subprocess.run([binary, MESHES, "3", mesh], env=env,
                                     capture_output=True, text=True, check=True)
                for line in out.stdout.splitlines():
                    r = json.loads(line)
                    key = (mesh, r["phase"], threads, tag)
                    best[key] = min(best.get(key, 1e300), r["ms"])
                    checks[key] = r["check"]
print("| mesh | phase | threads | old ms | new ms | new/old |")
print("|---|---|---|---:|---:|---:|")
for (mesh, phase, threads, tag) in sorted(best):
    if tag != "old":
        continue
    o = best[(mesh, phase, threads, "old")]
    n = best[(mesh, phase, threads, "new")]
    same = checks[(mesh, phase, threads, "old")] == checks[(mesh, phase, threads, "new")]
    print(f"| {mesh} | {phase} | {threads} | {o:.2f} | {n:.2f} | {n / o:.2f}"
          f"{'' if same else ' CHECK DIFFERS'} |")
