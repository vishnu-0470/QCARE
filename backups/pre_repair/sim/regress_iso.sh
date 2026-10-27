#!/bin/bash
# Isolated per-case regression: each case regenerates its OWN params/ROMs,
# then compiles and runs the RTL against its OWN golden log.
cd "$(dirname "$0")/.."
run () {  # TAG REC T0 DUR SEED GAIN NOISE ART TAU
  T=$1
  octave-cli --quiet --eval "REC=$2;T0=$3;DUR=$4;TAG='$T';SEED=$5;GAIN=$6;NOISE=$7;ART=$8;TAU=$9;source('sim/make_vectors.m')" >/dev/null 2>&1
  D=sim/runs/$T; rm -rf $D; mkdir -p $D && cp rtl/gen/params.vh rtl/gen/wrom.mem rtl/gen/tree.mem $D/
  NS=$(cat sim/${T}_ns.txt)
  iverilog -g2012 -o $D/a.vvp rtl/ecg_afe_accel_top.v rtl/ecg_div24.v sim/tb_accel.v \
    -P tb_accel.NS=$NS -P tb_accel.STREAM=\"sim/${T}_stream.mem\" -P tb_accel.LOG=\"$D/rtl.txt\" \
    -P tb_accel.CYCLOG=\"$D/cyc.txt\" -P tb_accel.WROM=\"$D/wrom.mem\" -P tb_accel.TREE=\"$D/tree.mem\" \
    -I$D 2>/dev/null
  vvp $D/a.vvp | grep "TB done" | sed "s/^/[$T] /"
  python3 sim/compare.py sim/${T}_golden.txt $D/rtl.txt | tail -1 | sed "s/^/[$T] /"
}
run rA 100 0   120 3  38 0  0 100
run rB 119 300 150 5  38 8  1 0
run rC 119 100 150 9  45 3  0 256
run rD 100 600 100 21 30 15 1 180
run rE 119 0   300 13 38 5  1 150
echo ISO-DONE
