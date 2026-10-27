from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
root=Path(__file__).resolve().parents[1]
project=root;out=root/'reports/plots/CWT_outputs';out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titleweight':'semibold','axes.spines.top':False,'axes.spines.right':False})
items=[
 {'beat':project/'experiments/beat_trace/annotated_beat.csv','map':project/'experiments/beat_trace/cwt_map.csv','column':2,'ylabel':'ECG (mV)','title':'MIT-BIH record 100 | MLII lead | annotated class N','peak':'Annotated R peak','file':'MITBIH_record100_CWT.png'},
 {'beat':project/'experiments/afe_adc_trained/segment_360.csv','map':project/'experiments/afe_adc_trained/cwt_map.csv','column':1,'ylabel':'Differential voltage (V)','title':'Cadence 2.csv | ADC-reconstructed segment','peak':'Heuristic peak','file':'Cadence_2csv_CWT.png'}]
for item in items:
 beat=np.loadtxt(item['beat'],delimiter=',',skiprows=1);M=np.loadtxt(item['map'],delimiter=',')
 assert len(beat)==360 and M.shape==(60,60)
 t=(np.arange(360)-140)/360
 # Each temporal bin averages three translations at offsets 6q,6q+2,6q+4.
 # Its center is 6q+2, with spacing six samples at the 360 Hz model rate.
 time_edges=(np.arange(61)*6-1-140)/360
 scale_edges=np.arange(.5,61.5)
 fig=plt.figure(figsize=(10,6.2),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1,2.3],width_ratios=[1,.035],hspace=.10)
 ax=fig.add_subplot(gs[0,0]);im_ax=fig.add_subplot(gs[1,0],sharex=ax);cax=fig.add_subplot(gs[1,1])
 fig.suptitle(item['title'],fontsize=16,fontweight='bold')
 ax.plot(t,beat[:,item['column']],color='#163b5c',lw=1.6)
 ax.axvline(0,color='#bf5a22',ls='--',lw=1.1,label=item['peak']);ax.legend(loc='upper right',frameon=False,fontsize=9)
 ax.set_ylabel(item['ylabel']);ax.grid(alpha=.16);ax.tick_params(labelbottom=False);ax.set_title('360-sample input beat',loc='left',fontsize=11,pad=6)
 mesh=im_ax.pcolormesh(time_edges,scale_edges,M,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1),shading='flat',rasterized=True)
 im_ax.axvline(0,color='#242424',ls='--',lw=.8,alpha=.55)
 im_ax.set(xlabel='Time relative to selected peak (s)',ylabel='Wavelet scale',ylim=(.5,60.5),xlim=(t[0],t[-1]),yticks=[1,10,20,30,40,50,60])
 im_ax.set_title('Cosine-Morlet CWT | 60 scales × 60 time bins',loc='left',fontsize=11,pad=7)
 cb=fig.colorbar(mesh,cax=cax);cb.set_label('Signed coefficient / beat maximum')
 fig.savefig(out/item['file'],dpi=180,facecolor='white');plt.close(fig)
print('Saved two CWT figures from the actual exported CSV maps.')
