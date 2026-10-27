#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
./run.sh results
./run.sh demo
printf '\nFinished. Full instructions: RUN_GUIDE.md\n'
read -r -p 'Press Enter to close.' answer
