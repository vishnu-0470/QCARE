"""Same ds1 + IRN1/2 + pool + ds2 + IRN3/4 + GAP + FC as MATLAB.

No batch normalization or replacement architecture. Training-only dropout is
applied at the FC input as stated in the reference paper. Integer inference
matches the existing 4W / post-ReLU signed-A8 / accumulator-B16 engine.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
NAMES=['ds1','irn1','irn2','ds2','irn3','irn4','fc']

def round_ste(x):return x+(torch.floor(x+0.5)-x).detach()
def quant(x,ex,bits=8,positive=False):
 s=2.0**int(ex);lo=0 if positive else -(1<<(bits-1));hi=(1<<(bits-1))-1
 return round_ste(x/s).clamp(lo,hi)*s

def weight_exp(w):
 maximum=float(w.detach().abs().amax())
 return (math.floor(math.log2(maximum)) if maximum else -10)-2

class IRN(nn.Module):
 def __init__(self,channels):
  super().__init__();h=channels//2
  self.W1=nn.Parameter(torch.randn(channels,channels,1,1)*math.sqrt(2/channels));self.b1=nn.Parameter(torch.zeros(channels))
  self.W2=nn.Parameter(torch.randn(channels,channels,3,3)*math.sqrt(2/(9*channels)));self.b2=nn.Parameter(torch.zeros(channels))
  self.W3=nn.Parameter(torch.randn(h,h,3,3)*math.sqrt(2/(9*h)));self.b3=nn.Parameter(torch.zeros(h))

class Layer(nn.Module):
 def __init__(self,shape,fanin):
  super().__init__();self.W=nn.Parameter(torch.randn(*shape)*math.sqrt(2/fanin));self.b=nn.Parameter(torch.zeros(shape[0]))

class CurrentIRN(nn.Module):
 def __init__(self,seed=26,dropout=0.5):
  super().__init__();torch.manual_seed(seed)
  self.ds1=Layer((8,1,3,3),9);self.irn1=IRN(8);self.irn2=IRN(8)
  self.ds2=Layer((16,8,3,3),72);self.irn3=IRN(16);self.irn4=IRN(16);self.fc=Layer((5,16),16)
  self.dropout=dropout;self.qat=False;self.E=None;self.WE=None

 def weight_exponents(self):
  result={}
  for name in NAMES:
   layer=getattr(self,name)
   for w in ('W',) if name in ('ds1','ds2','fc') else ('W1','W2','W3'):
    result[name if w=='W' else name+'_'+w]=weight_exp(getattr(layer,w))
  return result

 def configure_qat(self,E,WE=None):self.E=dict(E);self.WE=self.weight_exponents() if WE is None else dict(WE);self.qat=True

 def conv(self,x,layer,wn,bn,key,ex,ey,stride=1,pad=0):
  w=getattr(layer,wn);b=getattr(layer,bn)
  if self.qat:
   ew=self.WE[key];w=quant(w,ew,4);b=quant(b,ex+ew,16)
  y=F.conv2d(x,w,b,stride,pad)
  return quant(y,ey,8,True) if self.qat else F.relu(y)

 def block(self,x,name,ex,eb,trace):
  b=getattr(self,name);a=self.conv(x,b,'W1','b1',name+'_W1',ex,eb)
  y=self.conv(a,b,'W2','b2',name+'_W2',eb,eb,pad=1);h=y.shape[1]//2
  r=self.conv(y[:,:h],b,'W3','b3',name+'_W3',eb,eb,pad=1)
  z=torch.cat([r,y[:,h:]],dim=1)+a
  out=quant(z,eb,8,True) if self.qat else F.relu(z)
  trace[name+'_A']=a;trace[name+'_BM']=y;trace[name+'_B1P']=r;trace[name]=out
  return out

 def forward(self,x,return_trace=False):
  E=self.E or dict.fromkeys(['ds1','blk1','blk2','ds2','blk3','blk4'],-7);trace={}
  x=self.conv(x,self.ds1,'W','b','ds1',-7,E['ds1'],stride=3);trace['ds1']=x
  x=self.block(x,'irn1',E['ds1'],E['blk1'],trace);x=self.block(x,'irn2',E['blk1'],E['blk2'],trace)
  x=F.max_pool2d(x,2);trace['pool']=x
  x=self.conv(x,self.ds2,'W','b','ds2',E['blk2'],E['ds2'],stride=2);trace['ds2']=x
  x=self.block(x,'irn3',E['ds2'],E['blk3'],trace);x=self.block(x,'irn4',E['blk3'],E['blk4'],trace)
  g=x.mean(dim=(2,3))
  if self.qat:g=quant(g,E['blk4'],8,True)
  trace['gap']=g
  # Dropout is training-only and requantized to the same activation grid.
  g=F.dropout(g,p=self.dropout,training=self.training)
  if self.qat:g=quant(g,E['blk4'],8,True)
  w=self.fc.W;b=self.fc.b
  if self.qat:w=quant(w,self.WE['fc'],4);b=quant(b,E['blk4']+self.WE['fc'],16)
  logits=F.linear(g,w,b);trace['logits']=logits
  return (logits,trace) if return_trace else logits

 @torch.no_grad()
 def clip_latent(self):
  if not self.qat:return
  for name in NAMES:
   layer=getattr(self,name)
   for w in ('W',) if name in ('ds1','ds2','fc') else ('W1','W2','W3'):
    ex=self.WE[name if w=='W' else name+'_'+w];getattr(layer,w).clamp_(-8*2.0**ex,7*2.0**ex)

@torch.no_grad()
def calibrate(model,maps,batch=64):
 was=model.training;wasqat=model.qat;model.eval();model.qat=False
 vals={k:[] for k in ['ds1','blk1','blk2','ds2','blk3','blk4']}
 for start in range(0,len(maps),batch):
  x=torch.as_tensor(maps[start:start+batch],dtype=torch.float32)/128
  _,t=model(x,True)
  vals['ds1'].append(t['ds1'].ravel());vals['ds2'].append(t['ds2'].ravel())
  for i in range(1,5):
   vals['blk'+str(i)].extend([t['irn'+str(i)+suffix].ravel() for suffix in ['_A','_BM','_B1P','']])
 result={}
 for name,arrays in vals.items():
  v=torch.cat(arrays);k=max(1,math.ceil(0.999*len(v)));p=float(torch.kthvalue(v,k).values)
  result[name]=max(-20,min(8,math.ceil(math.log2(p/127)) if p>0 else -7))
 model.train(was);model.qat=wasqat;return result
