#!/bin/bash
set -euo pipefail
# usage: xilinx/sim/run_case.sh TAG   (expects xilinx/sim/TAG_stream.mem / _golden.txt / _ns.txt; run from project root)
TAG=$1; NS=$(cat xilinx/sim/${TAG}_ns.txt)
iverilog -g2012 -Ixilinx/rtl/gen -o xilinx/sim/${TAG}.vvp \
  -P tb_accel.NS=$NS -P tb_accel.STREAM=\"xilinx/sim/${TAG}_stream.mem\" -P tb_accel.LOG=\"xilinx/sim/${TAG}_rtl.txt\" \
  -P tb_accel.CYCLOG=\"xilinx/sim/${TAG}_cycles.txt\" xilinx/rtl/ecg_afe_accel_top.v xilinx/rtl/ecg_div24.v xilinx/sim/tb_accel.v
vvp xilinx/sim/${TAG}.vvp | grep -v "WARNING\|finish"
python3 xilinx/sim/compare.py xilinx/sim/${TAG}_golden.txt xilinx/sim/${TAG}_rtl.txt
python3 - <<PY
import collections
g=[l.split() for l in open("xilinx/sim/${TAG}_golden.txt")]
c=collections.Counter()
for r in g:
    if r[0]=='D': c['drop']+=1
    else:
        c['cls%s'%r[3]]+=1; c['early' if r[4]=='1' else 'cnn']+=1
print("coverage:",dict(c))
cy=[tuple(map(int,l.split())) for l in open("xilinx/sim/${TAG}_cycles.txt")]
if cy:
    allc=[x[0] for x in cy]; cnn=[x[0] for x in cy if x[1]!=3 and x[2]==0]
    print("cycles/beat: mean=%d max=%d | CNN beats mean=%d"%(sum(allc)/len(allc),max(allc),(sum(cnn)/len(cnn)) if cnn else 0))
PY
