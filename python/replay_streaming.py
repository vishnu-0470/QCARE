from pathlib import Path
import subprocess,json
r=Path(__file__).resolve().parents[1];tag='rE';ns=int((r/'sim/rE_ns.txt').read_text());out=r/'reports/baseline'
cmd=['iverilog','-g2012','-Irtl/gen','-o',str(out/'streaming.vvp'),'rtl/ecg_afe_accel_top.v','rtl/ecg_div24.v','sim/tb_accel.v',f'-Ptb_accel.NS={ns}',f'-Ptb_accel.STREAM="sim/rE_stream.mem"',f'-Ptb_accel.LOG="reports/baseline/streaming_rE.txt"',f'-Ptb_accel.CYCLOG="reports/baseline/streaming_rE_cycles.txt"']
cmd=[piece for arg in cmd for piece in (['-P',arg[2:]] if arg.startswith('-P') else [arg])]
# Put all options before source filenames for POSIX getopt portability.
files=['rtl/ecg_afe_accel_top.v','rtl/ecg_div24.v','sim/tb_accel.v'];cmd=[v for v in cmd if v not in files]+files
c=subprocess.run(cmd,cwd=r,capture_output=True,text=True);(out/'streaming_compile.log').write_text(c.stdout+c.stderr)
if c.returncode:raise RuntimeError(c.stderr)
c=subprocess.run(['vvp',str(out/'streaming.vvp')],cwd=r,capture_output=True,text=True);(out/'streaming_rE.log').write_text(c.stdout+c.stderr)
golden=[s.split() for s in (r/'sim/rE_golden.txt').read_text().splitlines()];actual=[s.split() for s in (out/'streaming_rE.txt').read_text().splitlines()]
report={'status':'PASS' if golden==actual else 'FAIL','case':'rE','beats':len(actual),'expected_beats':len(golden),'comparison':'existing frozen vector/ROM pair only','not_5class_QNN':True,'not_accuracy':True}
(out/'streaming_frozen_replay.json').write_text(json.dumps(report,indent=2));print(report)
