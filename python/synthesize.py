"""Run an available Xilinx synthesis flow; require final reports, not exit code alone."""
from pathlib import Path
import argparse,shutil,sys,subprocess,json,hashlib,os
r=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--board',action='store_true');p.add_argument('--yosys');a=p.parse_args()
exe=a.yosys or shutil.which('yosys') or shutil.which('yowasp-yosys')
if not exe:
 candidate=Path(sys.executable).with_name('yowasp-yosys');exe=str(candidate) if candidate.exists() else None
if not exe:raise SystemExit('BLOCKED: install yosys or yowasp-yosys and pass --yosys')
name='trained_board' if a.board else 'trained_core';out=r/'reports/synthesis'/name;out.mkdir(exist_ok=True,parents=True)
cwd=r/'experiments/ds1_current'/('board_export' if a.board else 'export');top='basys3_qnn_top' if a.board else 'qnn_engine';sources=[r/'xilinx/rtl/qnn/qnn_engine.v']+([r/'xilinx/rtl/basys3/basys3_qnn_top.v'] if a.board else [])
outrel=Path(os.path.relpath(out,cwd)).as_posix()
quote=lambda v:'"'+str(v).replace('\\','/').replace('"','\\"')+'"'
script='read_verilog '+' '.join(map(quote,sources))+'\n'
script+=f'synth_xilinx -family xc7 -top {top}\n'
script+=f'tee -o {outrel}/statistics.json stat -json\n'
script+=f'tee -o {outrel}/check.txt check\n'
script+=f'write_edif {outrel}/netlist.edif\n'
script+=f'write_json {outrel}/netlist.json\n'
(out/'run.ys').write_text(script)
proc=subprocess.run([exe,'-l',str(out/'yosys.log'),'-s',str(out/'run.ys')],cwd=cwd,capture_output=True,text=True)
(out/'console.log').write_text(proc.stdout+proc.stderr)
status={'status':'FAIL','returncode':proc.returncode,'command':[exe,'-l',str(out/'yosys.log'),'-s',str(out/'run.ys')],'cwd':str(cwd.relative_to(r)),'scope':'generic Xilinx 7-series synthesis; no placement/routing, timing or power measurement','memory_files_sha256':{str(p.relative_to(cwd)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (cwd/'gen').iterdir() if p.is_file()},'source_sha256':{str(s.relative_to(r)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}}
try:
 stats=json.loads((out/'statistics.json').read_text());cells=stats.get('design',stats['modules']['\\'+top])['num_cells_by_type'];check=(out/'check.txt').read_text()
 if proc.returncode or 'Found and reported 0 problems.' not in check:raise ValueError('Synthesis/check did not finish cleanly')
 if any(c.startswith('$') for c in cells):raise ValueError('Unmapped generic cells remain')
 status.update(status='PASS',tool=stats['creator'],top=top,cells=cells,LUT_primitives=sum(v for k,v in cells.items() if k.startswith('LUT')),FF=sum(v for k,v in cells.items() if k.startswith('FD')),RAMB36=cells.get('RAMB36E1',0),RAMB18=cells.get('RAMB18E1',0),DSP48=cells.get('DSP48E1',0),CARRY4=cells.get('CARRY4',0),timing_status='NOT TESTED',power_status='NOT TESTED')
except Exception as e:status['reason']=str(e)
(out/'result.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2))
if status['status']!='PASS':raise SystemExit(1)
