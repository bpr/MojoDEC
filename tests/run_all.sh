#!/bin/bash
# Run every MojoDEC test file. Run from the repository root: pixi run test
# MOJO_FLAGS is passed to `mojo run` (e.g. "-D ASSERT=none").
#
# tests/test_properties*.mojo need proptest (a port of Hypothesis), found at
# $PROPTEST_PATH (default: ../proptest, a sibling checkout); they are skipped
# with a warning if proptest isn't there. proptest's example database is off
# unless PROPTEST_DATABASE is set (e.g. to .proptest/examples).
set -e
PROPTEST_PATH=${PROPTEST_PATH:-../proptest}
export PROPTEST_DATABASE=${PROPTEST_DATABASE:-none}
for f in tests/test_*.mojo; do
    if [[ $f == tests/test_properties*.mojo ]]; then
        if [ ! -d "$PROPTEST_PATH/proptest" ]; then
            echo "warning: skipping $f: proptest not found at $PROPTEST_PATH (set PROPTEST_PATH)" >&2
            continue
        fi
        mojo run $MOJO_FLAGS -I . -I build -I "$PROPTEST_PATH" "$f"
    else
        mojo run $MOJO_FLAGS -I . -I build "$f"
    fi
done
