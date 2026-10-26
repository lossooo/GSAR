#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

STEP="${1:-}"
if [[ "$STEP" == "-h" || "$STEP" == "--help" ]]; then
  echo "Usage: $0 {filter|postprocess|qwen2p5|uitars1|statistics|analyze} [arguments...]"
  exit 0
fi
if [[ -z "$STEP" ]]; then
  echo "Usage: $0 {filter|postprocess|qwen2p5|uitars1|statistics|analyze} [arguments...]" >&2
  exit 2
fi
shift

case "$STEP" in
  filter)      exec python -m gsar.synthesis.filter_traj "$@" ;;
  postprocess) exec python -m gsar.sft.postprocess "$@" ;;
  qwen2p5)     exec python -m gsar.sft.convert_traj_to_qwen2p5 "$@" ;;
  uitars1)     exec python -m gsar.sft.convert_traj_to_uitars1 "$@" ;;
  statistics)  exec python -m gsar.sft.statistics "$@" ;;
  analyze)     exec python -m gsar.synthesis.analyze_results "$@" ;;
  *)
    echo "Unknown step: $STEP" >&2
    exit 2
    ;;
esac
