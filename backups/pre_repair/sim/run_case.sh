#!/bin/bash
# usage: sim/run_case.sh TAG   (expects sim/TAG_stream.mem / _golden.txt / _ns.txt; run from project root)
TAG=$1; NS=$(cat sim/${TAG}_ns.txt)
iverilog -g2012 -Irtl/gen -o sim/${TAG}.vvp rtl/ecg_afe_accel_top.v rtl/ecg_div24.v sim/tb_accel.v \
  -P tb_accel.NS=$NS -P tb_accel.STREAM=\"sim/${TAG}_stream.mem\" -P tb_accel.LOG=\"sim/${TAG}_rtl.txt\" \
  -P tb_accel.CYCLOG=\"sim/${TAG}_cycles.txt\" 2>&1 | grep -v "timescale\|sensitive to all\|truncated"
vvp sim/${TAG}.vvp | grep -v "WARNING\|finish"
python3 sim/compare.py sim/${TAG}_golden.txt sim/${TAG}_rtl.txt
python3 - <<PY
import collections
g=[l.split() for l in open("sim/${TAG}_golden.txt")]
c=collections.Counter()
for r in g:
    if r[0]=='D': c['drop']+=1
    else:
        c['cls%s'%r[3]]+=1; c['early' if r[4]=='1' else 'cnn']+=1
print("coverage:",dict(c))
cy=[tuple(map(int,l.split())) for l in open("sim/${TAG}_cycles.txt")]
if cy:
    allc=[x[0] for x in cy]; cnn=[x[0] for x in cy if x[1]!=3 and x[2]==0]
    print("cycles/beat: mean=%d max=%d | CNN beats mean=%d"%(sum(allc)/len(allc),max(allc),(sum(cnn)/len(cnn)) if cnn else 0))
PY
