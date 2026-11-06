#!/bin/bash
# Run MojoDEC's examples, each from its own directory.
#
#   pixi run examples                       # all of them; figures saved as PNG
#   pixi run examples DarcyFlow             # one of them
#   pixi run examples DarcyFlow --show      # ... and open its figures in windows
#   pixi run examples --show                # all of them, showing every figure
#
# Without pixi, run `bash examples/run.sh ...` inside the pixi environment
# after `pixi run numojo`.
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
all=(Ranking RipsComplex HodgeDecomposition DarcyFlow HeatConduction ResonantCavity Sphere Neighborhoods Neighborhoods3D MakeMesh)

run() {
    local name=$1
    shift
    local dir="$root/examples/$name"
    local main=driver.mojo
    if [ "$name" = MakeMesh ]; then
        main=make_mesh.mojo
    fi
    if [ ! -f "$dir/$main" ]; then
        echo "unknown example '$name'; the examples are: ${all[*]}" >&2
        exit 1
    fi
    echo "=== $name"
    (cd "$dir" && mojo run -I "$root" -I "$root/build" "$main" "$@")
}

if [ $# -eq 0 ] || [[ $1 == -* ]]; then
    for name in "${all[@]}"; do
        run "$name" "$@"
    done
else
    run "$@"
fi
