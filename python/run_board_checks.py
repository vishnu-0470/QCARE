from pathlib import Path
import shutil,subprocess,json
r=Path(__file__).resolve().parents[1];model=r/'experiments/ds1_current/export';out=r/'experiments/ds1_current/board_export';(out/'gen').mkdir(parents=True,exist_ok=True)
for p in (model/'gen').iterdir():shutil.copy2(p,out/'gen'/p.name)
tokens=(model/'vec/qnn_vec_in.mem').read_text().split();(out/'gen/beats.mem').write_text('\n'.join(tokens[:14400])+'\n');rows=(model/'vec/qnn_vec_exp.txt').read_text().splitlines();(out/'expected.txt').write_text('\n'.join(rows[:4])+'\n')
exe=r/'reports/trained_board_checked.vvp';cmd=['iverilog','-g2012','-s','tb_board_checked','-o',str(exe),str(r/'rtl/qnn/qnn_engine.v'),str(r/'rtl/basys3/basys3_qnn_top.v'),str(r/'sim/tb_board_checked.v')]
p=subprocess.run(cmd,cwd=out,capture_output=True,text=True)
if p.returncode:raise RuntimeError(p.stdout+p.stderr)
p=subprocess.run(['vvp',str(exe)],cwd=out,capture_output=True,text=True);(r/'reports/trained_board_checked.log').write_text(p.stdout+p.stderr)
if p.returncode or 'PASS 4/4' not in p.stdout:raise RuntimeError(p.stdout+p.stderr)
(r/'reports/trained_board_checked.json').write_text(json.dumps({'status':'PASS','scope':'digital wrapper simulation only; no physical board programming','switches':[0,1,2,3],'identities':json.loads((model/'vec/identities.json').read_text())[:4],'checks':['signed logits','one-hot class LEDs','class letter','beat index','display scan initialized'],'source_directory':str(out.relative_to(r))},indent=2));print(p.stdout)
