from pathlib import Path
import json
r=Path(__file__).resolve().parents[1]
for name,file in [('Original quantized IRN','reports/ds2_metrics.json'),('4W/8A fusion network','reports/accuracy90/DS2_metrics.json'),('Stronger software classifier','reports/accuracy98/DS2_metrics.json')]:
 d=json.loads((r/file).read_text());m=d['quantized_current_irn'] if name=='Original quantized IRN' else d
 print(f'{name}: accuracy={m["accuracy"]:.4%}, macro-F1={m["macro_f1"]:.6f}, beats={m.get("samples",m.get("test_beats"))}')
latest=json.loads((r/'reports/LATEST_RESULTS.json').read_text())
print('90% quantized-model target:',latest['quantized_fusion']['target_90_status'],'| 98% software target:',latest['software']['target_98_status'])
