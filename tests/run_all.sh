#!/bin/bash
# Run every MojoDEC test file. Run from the repository root: pixi run test
# MOJO_FLAGS is passed to `mojo run` (e.g. "-D ASSERT=none").
set -e
for f in tests/test_*.mojo; do
    mojo run $MOJO_FLAGS -I . -I build "$f"
done
