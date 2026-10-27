#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$project_dir"
if [[ -x "$project_dir/.venv/bin/python" ]]; then
  project_python="$project_dir/.venv/bin/python"
elif [[ -x "$project_dir/../venv/bin/python" ]]; then
  project_python="$project_dir/../venv/bin/python"
else
  echo 'Create the environment first: python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt' >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
command_name="${1:-demo}"
if [[ $# -gt 0 ]]; then shift; fi
case "$command_name" in
  demo) exec "$project_python" python/infer_record_fusion90.py "$@" ;;
  demo-software) exec "$project_python" python/infer_record_software.py "$@" ;;
  rtl) exec "$project_python" python/run_fusion90_checks.py "$@" ;;
  board-sim) exec "$project_python" python/run_fusion_board_checks.py "$@" ;;
  results) exec "$project_python" python/show_latest_results.py "$@" ;;
  plots) exec "$project_python" python/report_followup.py "$@" ;;
  verify) exec "$project_python" python/check_provenance.py "$@" ;;
  synth)
    export YOWASP_CACHE_DIR="$project_dir/experiments/yowasp_cache"
    if [[ -d "$project_dir/tools/yosys_stable" ]]; then export PYTHONPATH="$project_dir/tools/yosys_stable${PYTHONPATH:+:$PYTHONPATH}"; fi
    exec "$project_python" python/synthesize_fusion90.py --board "$@" ;;
  *) echo 'Usage: ./run.sh {demo|demo-software|rtl|board-sim|results|plots|verify|synth} [options]' >&2; exit 2 ;;
esac
