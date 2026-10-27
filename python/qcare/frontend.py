"""Exact current Stage-1 linear CWT operator, not the earlier replacement model."""
from functools import lru_cache
import numpy as np

@lru_cache(maxsize=1)
def cwt_operator():
 a=np.zeros((3600,360),dtype=np.float64);samples=np.arange(360)
 for si,j in enumerate(range(1,61)):
  tau=np.arange(-4*j,4*j+1,dtype=np.float64)
  g=np.cos(np.pi/2*tau/j)*np.exp(-(tau/j)**2/2)/np.sqrt(j)
  H=np.zeros(len(g)+4)
  for r in range(3):H[2*r:2*r+len(g)]+=g/3
  for q in range(60):
   idx=samples-6*q+4*j;valid=(idx>=0)&(idx<len(H));a[si*60+q,valid]=H[idx[valid]]
 return a

def quantize_map(m):
 # Canonical QNN e_in=-7, matching QAT/export (not the legacy 127 helper).
 return np.clip(np.floor(np.asarray(m)*128+0.5),-128,127).astype(np.int8)

def cwt_maps(beats):
 beats=np.asarray(beats,dtype=np.float64).reshape(-1,360)
 maps=beats@cwt_operator().T
 norm=np.max(np.abs(maps),axis=1,keepdims=True)
 maps=np.divide(maps,norm,out=np.zeros_like(maps),where=norm!=0)
 return maps.reshape(-1,1,60,60)

def centered_beat(signal,peak,shift=0):
 start=int(peak)-140+int(shift)
 if start<0 or start+360>len(signal):raise ValueError('Incomplete 140-pre/219-post beat')
 return np.asarray(signal[start:start+360])
