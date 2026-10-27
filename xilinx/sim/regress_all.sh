#!/bin/bash
# Re-run the five RTL-vs-golden cases on the current RTL (run from project root).
cd "$(dirname "$0")/../.."
for t in rA rB rC rD rE; do
  ./xilinx/sim/run_case.sh $t 2>&1 | grep -E "TB done|BIT-EXACT|FAIL|MISMATCH|cycles" | sed "s/^/[$t] /"
done
echo REGRESSION-DONE
