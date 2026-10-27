from pathlib import Path
import argparse,subprocess,json
r=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--export',default='experiments/accuracy90_fusion_mlp/export');p.add_argument('--model-export',default='experiments/accuracy90_fusion_mlp/export');p.add_argument('--vectors',type=int,default=10);p.add_argument('--log',default='reports/accuracy90/RTL_fusion.log');a=p.parse_args()
source=r/a.export;model=r/a.model_export;out=r/a.log;out.parent.mkdir(parents=True,exist_ok=True);exe=out.with_suffix('.vvp')
cmd=['iverilog','-g2012','-s','tb_qnn_fusion_checked','-o',str(exe)]
contract=__import__('json').loads((model/'model_contract.json').read_text())
args={'NOPS':str(contract['NOPS']),'NW':str(contract['NW']),'NB':str(contract['NB']),'NV':str(a.vectors),'PROG_FILE':'"'+str(model/'gen/qnn_prog.mem')+'"','WROM_FILE':'"'+str(model/'gen/qnn_w.mem')+'"','BROM_FILE':'"'+str(model/'gen/qnn_b.mem')+'"','INPUT_FILE':'"'+str(source/'vec/qnn_vec_in.mem')+'"','EXPECTED_FILE':'"'+str(source/'vec/qnn_vec_exp.txt')+'"'}
for k,v in args.items():cmd+=['-P','tb_qnn_fusion_checked.'+k+'='+v]
cmd +=[str(r/'rtl/qnn/qnn_engine.v'),str(r/'sim/tb_qnn_fusion_checked.v')]
c=subprocess.run(cmd,cwd=r,capture_output=True,text=True)
if c.returncode:raise RuntimeError(c.stdout+c.stderr)
c=subprocess.run(['vvp',str(exe)],cwd=r,capture_output=True,text=True);out.write_text(c.stdout+c.stderr)
if c.returncode or f'PASS {a.vectors}/{a.vectors}' not in c.stdout:raise RuntimeError(c.stdout+c.stderr)
print(c.stdout)
