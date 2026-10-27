#!/bin/bash
set -euo pipefail
if ! command -v octave-cli >/dev/null 2>&1; then
  echo "BLOCKED: octave-cli is required to regenerate per-case ROMs. No regression was run." >&2
  exit 2
fi
# Isolated per-case regression: each case regenerates its OWN params/ROMs,
# then compiles and runs the RTL against its OWN golden log.
cd "$(dirname "$0")/../.."
run () {  # TAG REC T0 DUR SEED GAIN NOISE ART TAU
  T=$1
  octave-cli --quiet --eval "REC=$2;T0=$3;DUR=$4;TAG='$T';SEED=$5;GAIN=$6;NOISE=$7;ART=$8;TAU=$9;source('matlab/sim/make_vectors.m')" >/dev/null 2>&1
  D=xilinx/sim/runs/$T; rm -rf $D; mkdir -p $D && cp xilinx/rtl/gen/params.vh xilinx/rtl/gen/wrom.mem xilinx/rtl/gen/tree.mem $D/
  NS=$(cat xilinx/sim/${T}_ns.txt)
  iverilog -g2012 -o $D/a.vvp \
    -P tb_accel.NS=$NS -P tb_accel.STREAM=\"xilinx/sim/${T}_stream.mem\" -P tb_accel.LOG=\"$D/rtl.txt\" \
    -P tb_accel.CYCLOG=\"$D/cyc.txt\" -P tb_accel.WROM=\"$D/wrom.mem\" -P tb_accel.TREE=\"$D/tree.mem\" \
    -I$D xilinx/rtl/ecg_afe_accel_top.v xilinx/rtl/ecg_div24.v xilinx/sim/tb_accel.v
  vvp $D/a.vvp | grep "TB done" | sed "s/^/[$T] /"
  python3 xilinx/sim/compare.py xilinx/sim/${T}_golden.txt $D/rtl.txt | tail -1 | sed "s/^/[$T] /"
}
run rA 100 0   120 3  38 0  0 100
run rB 119 300 150 5  38 8  1 0
run rC 119 100 150 9  45 3  0 256
run rD 100 600 100 21 30 15 1 180
run rE 119 0   300 13 38 5  1 150
echo ISO-DONE
