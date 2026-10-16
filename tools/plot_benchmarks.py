"""Plot recorded results only; consistency is deliberately not labeled mAP."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
bench=json.loads((ROOT/'docs/predictions/latency.json').read_text())
consistency=json.loads((ROOT/'docs/predictions/consistency.json').read_text())
manifest=json.loads((ROOT/'docs/predictions/manifest.json').read_text())
out=ROOT/'docs/figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})

def chart(names,values,title,ylabel,filename):
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    ax.bar(names,values,color=['#778594','#376886','#c18653','#ac4a2d'][:len(names)])
    ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for index,value in enumerate(values):ax.annotate(f'{value:.2f}',(index,value),ha='center',va='bottom')
    ax.margins(y=.18)
    fig.savefig(out/filename,dpi=160);plt.close(fig)

names=list(bench['evaluations'])
chart(names,[bench['evaluations'][n]['mean_ms'] for n in names],
      'Sequential real-image predict() latency; host-specific, not browser speed','Mean milliseconds',
      'latency_throughput_comparison.png')
chart(list(manifest['models']),[m['size_bytes']/1048576 for m in manifest['models'].values()],
      'Stored artifact footprint (not inference precision)','MiB','model_size_compression.png')
metrics=consistency['evaluations']
chart(list(metrics),[m['f1_at_50_percent'] for m in metrics.values()],
      'Cross-model agreement at IoU 0.5 on 10 illustrative holdout images; NOT mAP',
      'F1 agreement with PyTorch (%)','accuracy_retention_comparison.png')
print('Rendered plots from versioned measurements, without hardcoded result values')
