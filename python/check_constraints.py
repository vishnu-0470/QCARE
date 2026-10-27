"""Check all used Basys3 pin assignments against the reviewed Digilent master."""
from pathlib import Path
import json,re
r=Path(__file__).resolve().parents[1];text=(r/'rtl/basys3/basys3.xdc').read_text()
expected={'clk':'W5','btnC':'U18','sw[0]':'V17','sw[1]':'V16'}
expected.update(dict(zip([f'led[{i}]' for i in range(5)],'U16 E19 U19 V19 W18'.split())))
expected.update(dict(zip([f'seg[{i}]' for i in range(7)],'W7 W6 U8 V8 U5 V5 U7'.split())))
expected['dp']='V7';expected.update(dict(zip([f'an[{i}]' for i in range(4)],'U2 U4 V4 W4'.split())))
actual={}
for line in text.splitlines():
 if line.lstrip().startswith('#') or 'PACKAGE_PIN' not in line:continue
 pin=re.search(r'PACKAGE_PIN\s+(\w+)',line).group(1);port=re.search(r'get_ports\s+(?:\{([^}]+)\}|([^\]\s]+))',line);name=port.group(1) or port.group(2);actual[name]=pin
assert actual==expected,(actual,expected)
assert '-period 10.00' in text
report={'status':'PASS','used_pin_count':len(actual),'pins':actual,'clock_constraint_ns':10.,'reference':'https://github.com/Digilent/digilent-xdc/blob/master/Basys-3-Master.xdc','scope':'digital pin mapping and declared clock only; no timing closure/electrical/physical check'}
(r/'reports/constraints_check.json').write_text(json.dumps(report,indent=2));print('PASS',len(actual),'pin assignments')
