from pathlib import Path
import json,numpy as np
from qcare.program import read_hex,run_program
from qcare.frontend import cwt_maps
r=Path(__file__).resolve().parents[1];p=r/'xilinx/rtl/qnn'
ops=read_hex(p/'gen/qnn_prog.mem',16).reshape(-1,18);w=read_hex(p/'gen/qnn_w.mem',16);b=read_hex(p/'gen/qnn_b.mem',16)
x=read_hex(p/'vec/qnn_vec_in.mem',8).reshape(-1,1,60,60);expected=np.loadtxt(p/'vec/qnn_vec_exp.txt',dtype=int)
for i,v in enumerate(x):
 lg=run_program(v,ops,w,b);np.testing.assert_array_equal(lg,expected[i,1:]);assert int(lg.argmax())==expected[i,0]
beat=np.loadtxt(r/'xilinx/sim/stage/beat_rec100_beat10.txt');saved=np.loadtxt(r/'xilinx/sim/stage/map_rec100_beat10.txt')
actual=cwt_maps(beat)[0,0];error=float(np.max(np.abs(actual-saved)))
assert error<6e-6,(error,saved.shape)
report={'qnn_program_oracle':{'status':'PASS','vectors':len(x),'all_predicted_F':bool(np.all(expected[:,0]==3))},'cwt_saved_matlab_map':{'status':'PASS','maximum_absolute_error':error,'tolerance':6e-6,'tolerance_reason':'archived text map has six significant digits'},'cadence_original_testbench':{'status':'FAIL','reason':'200000-cycle timeout < 883717 engine cycles'},'stage2_matlab_tests':{'status':'BLOCKED','reason':'MATLAB/Octave unavailable'},'stage3_synthetic_memorization':{'status':'NOT TESTED','reason':'No MATLAB/Octave and new training restricted to real annotated ECG'}}
(r/'reports/baseline/software_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
