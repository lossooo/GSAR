#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  exec python -m gsar.verification.reward --help
fi

METHOD="${1:-goalstate}"
MODEL="${2:-${GSAR_MODEL:-gpt-4o}}"
STAGE="${3:-both}"

exec python -m gsar.verification.reward --method "$METHOD" --model "$MODEL" --stage "$STAGE"
