from pathlib import Path
import json,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[1];metrics=json.loads((r/'reports/ds2_metrics.json').read_text());out=r/'reports/plots';out.mkdir(parents=True,exist_ok=True)
labels=['N','S','V','F','Q']
for name,m in metrics.items():
 cm=np.array(m['confusion_matrix']);norm=np.divide(cm,cm.sum(1)[:,None],out=np.zeros_like(cm,dtype=float),where=cm.sum(1)[:,None]!=0)
 fig,ax=plt.subplots(figsize=(6.3,5.4));im=ax.imshow(norm,vmin=0,vmax=1,cmap='Blues');fig.colorbar(im,ax=ax,label='Row fraction')
 ax.set(xticks=np.arange(5),yticks=np.arange(5),xticklabels=labels,yticklabels=labels,xlabel='Predicted class',ylabel='Annotated class',title=f'{name}\nAccuracy {m["accuracy"]:.4%} | Macro-F1 {m["macro_f1"]:.4f}')
 for i in range(5):
  for j in range(5):ax.text(j,i,f'{cm[i,j]}\n{norm[i,j]:.1%}',ha='center',va='center',color='white' if norm[i,j]>.55 else 'black',fontsize=9)
 fig.tight_layout();fig.savefig(out/(name+'_confusion.png'),dpi=180);plt.close(fig)
h=json.loads((r/'experiments/ds1_current/history.json').read_text());fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].plot([v['epoch']+1 for v in h],[v['train_loss'] for v in h]);ax[0].set(xlabel='Epoch',ylabel='Training cross-entropy',title='Real DS1 training')
ax[1].plot([v['epoch']+1 for v in h],[v['validation_macro_f1'] for v in h]);ax[1].set(xlabel='Epoch',ylabel='Macro-F1 (all five)',title='DS1 record-held-out validation')
for a in ax:a.axvline(30,color='#c04c36',ls='--',label='QAT begins');a.grid(alpha=.2)
fig.tight_layout();fig.savefig(out/'training.png',dpi=180);plt.close(fig)
