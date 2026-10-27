"""Independent integer oracle and compatible 18-word QNN program exporter."""
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch

def read_hex(path,bits):
 x=np.array([int(s,16) for s in Path(path).read_text().split()],dtype=np.int64)
 return np.where(x>=(1<<(bits-1)),x-(1<<bits),x)
def write_hex(path,x,bits=16):
 Path(path).write_text(''.join(f'{int(v)&((1<<bits)-1):0{(bits+3)//4}x}\n' for v in np.asarray(x).ravel()))
def shift(x,s):
 if s>0:return (x+(1<<(s-1)))>>s
 return x<<(-s)
def clip(x):return np.clip(x,0,127)

def run_program(qin,ops,weights,biases,return_trace=False):
 ram=np.zeros(65536,dtype=np.int64);ram[:3600]=np.asarray(qin).ravel();trace=[];logits=None
 for row in ops:
  op,src,src2,dst,ci,co,h,w,ho,wo,k,s,p,wb,bb,sh,sa,half=map(int,row)
  if sh>=32768:sh-=65536
  if op==0:break
  if op==1:
   x=ram[src:src+ci*h*w].reshape(ci,h,w);x=np.pad(x,((0,0),(p,p),(p,p)))
   win=sliding_window_view(x,(k,k),axis=(1,2))[:,::s,::s]
   W=weights[wb:wb+co*ci*k*k].reshape(co,ci,k,k)
   acc=np.einsum('cyxij,ocij->oyx',win,W,optimize=True)+biases[bb:bb+co,None,None]
   if np.abs(acc).max()>=2**31:raise OverflowError('Int32 accumulator')
   y=clip(shift(acc,sh));ram[dst:dst+y.size]=y.ravel()
  elif op==2:
   x=ram[src:src+co*h*w].reshape(co,h,w);y=x.reshape(co,ho,2,wo,2).max(axis=(2,4));ram[dst:dst+y.size]=y.ravel()
  elif op==3:
   A=ram[sa:sa+co*h*w].reshape(co,h,w);B=np.concatenate([ram[src:src+half*h*w],ram[src2+half*h*w:src2+co*h*w]]).reshape(co,h,w)
   y=clip(A+B);ram[dst:dst+y.size]=y.ravel()
  elif op==4:
   x=ram[src:src+co*h*w].reshape(co,h*w);y=clip((x.sum(1)+(h*w)//2)//(h*w));ram[dst:dst+co]=y
  elif op==5:
   g=ram[src:src+ci];W=weights[wb:wb+co*ci].reshape(co,ci);logits=W@g+biases[bb:bb+co];y=logits
   if np.any(logits<-(1<<23)) or np.any(logits>=(1<<23)):raise OverflowError('Signed int24 output')
  else:raise ValueError('Unknown QNN instruction')
  trace.append(y.copy())
 if logits is None:raise ValueError('Program has no FC')
 return (logits,trace) if return_trace else logits

def export_model(model,outdir):
 outdir=Path(outdir);(outdir/'gen').mkdir(parents=True,exist_ok=True);(outdir/'vec').mkdir(exist_ok=True)
 if not model.qat:raise ValueError('Model must have frozen QAT scales')
 E=model.E;WE=model.WE;ops=[];weights=[];biases=[];top=0;addresses={}
 def alloc(n,name):
  nonlocal top
  start=top;top+=n;addresses[name]=start;return start
 def conv(src,dst,ci,co,h,ho,k,s,p,layer,wn,bn,key,ex,ey):
  ew=WE[key];w=np.clip(np.floor(getattr(layer,wn).detach().cpu().numpy()/2.0**ew+0.5),-8,7).astype(np.int64)
  b=np.floor(getattr(layer,bn).detach().cpu().numpy()/2.0**(ex+ew)+0.5).astype(np.int64)
  if b.min()<-32768 or b.max()>32767:raise OverflowError('Bias B16 overflow; do not silently change scales')
  wb=len(weights);bb=len(biases);weights.extend(w.ravel());biases.extend(b.ravel())
  sh=ey-ex-ew
  if not -16<=sh<=24:raise ValueError('Unsupported shift range')
  if ci*k*k*128*8+np.abs(b).max()>=2**23:raise OverflowError('Unsafe dot-product bound')
  ops.append([1,src,0,dst,ci,co,h,h,ho,ho,k,s,p,wb,bb,sh,0,0])
 def block(src,name,c,h,ex,eb):
  layer=getattr(model,name);hw=h*h
  A=alloc(c*hw,name+'_A');BM=alloc(c*hw,name+'_BM');BP=alloc(c//2*hw,name+'_B1P');OUT=alloc(c*hw,name)
  conv(src,A,c,c,h,h,1,1,0,layer,'W1','b1',name+'_W1',ex,eb)
  conv(A,BM,c,c,h,h,3,1,1,layer,'W2','b2',name+'_W2',eb,eb)
  conv(BM,BP,c//2,c//2,h,h,3,1,1,layer,'W3','b3',name+'_W3',eb,eb)
  ops.append([3,BP,BM,OUT,0,c,h,h,0,0,0,0,0,0,0,0,A,c//2]);return OUT
 IN=alloc(3600,'input');D1=alloc(3200,'ds1');conv(IN,D1,1,8,60,20,3,3,0,model.ds1,'W','b','ds1',-7,E['ds1'])
 U1=block(D1,'irn1',8,20,E['ds1'],E['blk1']);U2=block(U1,'irn2',8,20,E['blk1'],E['blk2'])
 P=alloc(800,'pool');ops.append([2,U2,0,P,8,8,20,20,10,10,0,0,0,0,0,0,0,0])
 D2=alloc(256,'ds2');conv(P,D2,8,16,10,4,3,2,0,model.ds2,'W','b','ds2',E['blk2'],E['ds2'])
 U3=block(D2,'irn3',16,4,E['ds2'],E['blk3']);U4=block(U3,'irn4',16,4,E['blk3'],E['blk4'])
 G=alloc(16,'gap');ops.append([4,U4,0,G,0,16,4,4,0,0,5,0,0,0,0,0,0,0])
 ew=WE['fc'];w=np.clip(np.floor(model.fc.W.detach().cpu().numpy()/2.0**ew+0.5),-8,7).astype(np.int64)
 b=np.floor(model.fc.b.detach().cpu().numpy()/2.0**(E['blk4']+ew)+0.5).astype(np.int64)
 if np.abs(b).max()>32767:raise OverflowError('FC B16 overflow')
 wb=len(weights);bb=len(biases);weights.extend(w.ravel());biases.extend(b.ravel())
 ops.extend([[5,G,0,0,16,5,0,0,0,0,0,0,0,wb,bb,0,0,0],[0]*18])
 ops=np.array(ops,dtype=np.int64);weights=np.array(weights,dtype=np.int64);biases=np.array(biases,dtype=np.int64)
 write_hex(outdir/'gen/qnn_prog.mem',ops);(outdir/'gen/qnn_prog.mem.wide').write_text(''.join(''.join(f'{int(v)&65535:04x}' for v in row[::-1])+'\n' for row in ops));write_hex(outdir/'gen/qnn_w.mem',weights);write_hex(outdir/'gen/qnn_b.mem',biases)
 (outdir/'gen/qnn_params.vh').write_text(f'// Compatible current Q-CARE export\nlocalparam NOPS={len(ops)};\nlocalparam NW={len(weights)};\nlocalparam NB={len(biases)};\nlocalparam IN_BASE=0;\n')
 import json
 (outdir/'model_contract.json').write_text(json.dumps({'activation_exponents':E,'weight_exponents':WE,'class_order':['N','S','V','F','Q'],'activation_ram_used':top,'addresses':addresses,'weight_bits':4,'weight_rom_container_bits':16,'bias_bits':16,'activation_range':[0,127],'logit_bits':24,'rounding':'floor(x+0.5)','row_major':True},indent=2))
 return ops,weights,biases

def export_vectors(outdir,maps,ops,weights,biases,identities=None):
 outdir=Path(outdir);(outdir/'vec').mkdir(exist_ok=True,parents=True)
 write_hex(outdir/'vec/qnn_vec_in.mem',maps,8);lines=[]
 for x in maps:
  lg=run_program(x,ops,weights,biases);lines.append(' '.join(map(str,[int(lg.argmax()),*lg.tolist()])))
 (outdir/'vec/qnn_vec_exp.txt').write_text('\n'.join(lines)+'\n')
 if identities is not None:
  import json
  (outdir/'vec/identities.json').write_text(json.dumps(identities,indent=2))
