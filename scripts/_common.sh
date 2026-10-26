#!/usr/bin/env bash

# Keep wrappers independent of the directory they are called from. An
# editable install is still recommended, but this also makes a local checkout
# usable before installing the package itself.
GSAR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$GSAR_ROOT"
export PYTHONPATH="$GSAR_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
