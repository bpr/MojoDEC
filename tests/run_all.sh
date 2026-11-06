#!/bin/bash
# Run every MojoDEC test file. Run from the repository root: pixi run test
set -e
for f in tests/test_*.mojo; do
    mojo run -I . -I build "$f"
done
