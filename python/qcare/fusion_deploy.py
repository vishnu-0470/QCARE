"""Bit-exact export of the frozen nonlinear rhythm/CWT fusion head."""
from pathlib import Path
import json,numpy as np
from .program import read_hex,write_hex,shift,clip
from .provenance import digest

def quant_head(state,ex,ew1,eh,ew2):
 W1=np.clip(np.floor(state['l1.weight'].detach().numpy()/2.**ew1+.5),-8,7).astype(np.int64)
 B1=np.floor(state['l1.bias'].detach().numpy()/2.**(ex+ew1)+.5).astype(np.int64)
 W2=np.clip(np.floor(state['l2.weight'].detach().numpy()/2.**ew2+.5),-8,7).astype(np.int64)
 B2=np.floor(state['l2.bias'].detach().numpy()/2.**(eh+ew2)+.5).astype(np.int64)
 if max(np.abs(B1).max(),np.abs(B2).max())>32767:raise OverflowError('B16 overflow')
 return W1,B1,W2,B2

def head_forward(gap_codes,aux_codes,params,sh):
 W1,B1,W2,B2=params;X=np.concatenate([np.asarray(gap_codes),np.asarray(aux_codes)],axis=-1).astype(np.int64);H=clip(shift(X@W1.T+B1,sh));logits=H@W2.T+B2
 if np.abs(logits).max()>=2**23:raise OverflowError('Signed24 output overflow')
 return logits,H

def auxiliary_codes(raw,config):
 return np.clip(np.floor((np.asarray(raw)-np.array(config['aux_mean']))/np.array(config['aux_std'])*config['aux_gain']/2.**config['aux_exponent']+.5),-128,127).astype(np.int64)

def export_fusion(root,checkpoint,out):
 root=Path(root);out=Path(out);(out/'gen').mkdir(exist_ok=True,parents=True);(out/'vec').mkdir(exist_ok=True)
 head=checkpoint['head'];config=checkpoint['config'];ex=config['aux_exponent'];ew1=head['ew1'];eh=head['eh'];ew2=head['ew2'];params=quant_head(head['state'],ex,ew1,eh,ew2);W1,B1,W2,B2=params
 gen=root/'experiments/ds1_current/export/gen';oldops=read_hex(gen/'qnn_prog.mem',16).reshape(-1,18);oldw=read_hex(gen/'qnn_w.mem',16);oldb=read_hex(gen/'qnn_b.mem',16);weights=np.r_[oldw[:-80],W1.ravel(),W2.ravel()];biases=np.r_[oldb[:-5],B1,B2]
 WB=len(oldw)-80;BB=len(oldb)-5;sh=eh-ex-ew1;hidden=32080
 conv=[1,32048,0,hidden,32,32,1,1,1,1,1,1,0,WB,BB,sh,0,0]
 fc=[5,hidden,0,0,32,5,0,0,0,0,0,0,0,WB+1024,BB+32,0,0,0]
 ops=np.vstack([oldops[:-2],conv,fc,[0]*18]);write_hex(out/'gen/qnn_prog.mem',ops);write_hex(out/'gen/qnn_w.mem',weights);write_hex(out/'gen/qnn_b.mem',biases)
 (out/'gen/qnn_prog.mem.wide').write_text(''.join(''.join(f'{int(v)&65535:04x}' for v in row[::-1])+'\n' for row in ops))
 (out/'gen/qnn_params.vh').write_text(f'localparam NOPS={len(ops)};\nlocalparam NW={len(weights)};\nlocalparam NB={len(biases)};\nlocalparam AUX_N=16;\nlocalparam AUX_BASE=32064;\n')
 contract={'class_order':['N','S','V','F','Q'],'original_backbone_convolutions':14,'added_pointwise_convolutions':1,'head_hidden_channels':32,'FC_input_channels':32,'NOPS':len(ops),'NW':len(weights),'NB':len(biases),'weight_bits':4,'bias_bits':16,'input_bits':8,'auxiliary_features':16,'auxiliary_RAM_base':32064,'hidden_RAM_base':hidden,'activation_RAM_used':32112,'input_memory_tokens_per_vector':3616,'input_layout':'3600 CWT row-major signed bytes, then16 signed auxiliary feature bytes; loader maps latter to32064..32079','input_exponent':ex,'hidden_exponent':eh,'weight_exponents':[ew1,ew2],'hidden_requant_shift':sh,'logit_exponent':eh+ew2,'rounding':'floor(x+0.5), post-ReLU saturation0..127','host_preprocessing':['CWT','causal RR/morphology proxies'],'baseline_weight_backbone_sha256':digest(gen/'qnn_w.mem')}
 (out/'model_contract.json').write_text(json.dumps(contract,indent=2));return ops,weights,biases,params,contract
