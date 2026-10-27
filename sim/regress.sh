#!/bin/bash
# Full RTL-vs-golden regression. Run from the project root. Needs octave-cli + iverilog.
run_one () {  # TAG REC T0 DUR SEED GAIN NOISE ART TAU
  TAG=$1
  octave-cli --quiet --eval "REC=$2;T0=$3;DUR=$4;TAG='$TAG';SEED=$5;GAIN=$6;NOISE=$7;ART=$8;TAU=$9;source('sim/make_vectors.m')" >/dev/null 2>&1
  # each case needs its own params/weights -> copy gen dir per case
  mkdir -p sim/runs/$TAG && cp rtl/gen/* sim/runs/$TAG/
  NS=$(cat sim/${TAG}_ns.txt)
  iverilog -g2012 -Isim/runs/$TAG -o sim/runs/$TAG/a.vvp rtl/ecg_afe_accel_top.v rtl/ecg_div24.v sim/tb_accel.v \
    -P tb_accel.NS=$NS -P tb_accel.STREAM=\"sim/${TAG}_stream.mem\" -P tb_accel.LOG=\"sim/runs/$TAG/rtl.txt\" \
    -P tb_accel.CYCLOG=\"sim/runs/$TAG/cyc.txt\" -P tb_accel.WROM=\"sim/runs/$TAG/wrom.mem\" -P tb_accel.TREE=\"sim/runs/$TAG/tree.mem\" 2>/dev/null
  vvp sim/runs/$TAG/a.vvp | grep "TB done" | sed "s/^/[$TAG] /"
  python3 sim/compare.py sim/${TAG}_golden.txt sim/runs/$TAG/rtl.txt | tail -1 | sed "s/^/[$TAG] /"
}
run_one rA 100 0   120 3  38 0  0 100
run_one rB 119 300 150 5  38 8  1 0
run_one rC 119 100 150 9  45 3  0 256
run_one rD 100 600 100 21 30 15 1 180
run_one rE 119 0   300 13 38 5  1 150
echo REGRESSION-COMPLETE
