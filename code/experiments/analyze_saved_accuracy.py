"""Analyze saved accuracy logs without training or API calls."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/hitl-matplotlib')
from pathlib import Path
import re,json,csv,textwrap,html
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/accuracy_analysis'; OUT.mkdir(parents=True,exist_ok=True)
pattern=re.compile(r'^(.*?) - Epoch (\d+) completed\. (Validation|Test) Accuracy: ([\d.]+)%, F1_Score: ([\d.]+)')
runs=[]
for folder in [ROOT/'user_study_logs',ROOT/'code/user_study_logs']:
 for path in sorted(folder.glob('*/model_scores*.log')):
  rows=[]
  for line in path.read_text(errors='replace').splitlines():
   m=pattern.match(line)
   if m: rows.append(dict(time=m[1],epoch=int(m[2]),split=m[3],accuracy=float(m[4]),f1=float(m[5])))
  if not rows: continue
  # Separate restarted epoch sequences rather than silently merging them.
  segments=[]; current=[]; prev=0
  for row in rows:
   if row['split']=='Validation' and row['epoch']<=prev:
    segments.append(current);current=[];prev=0
   current.append(row)
   if row['split']=='Validation':prev=row['epoch']
  if current: segments.append(current)
  for i,segment in enumerate(segments):
   tests=[r for r in segment if r['split']=='Test']
   if not tests:continue
   llm=next(path.parent.glob('llm_suggestions*.log'),None)
   content=llm.read_text(errors='replace') if llm else ''
   runs.append(dict(session=path.parent.name,segment=i+1,source=str(path.relative_to(ROOT)),rows=segment,first_epoch=tests[0]['epoch'],last_epoch=tests[-1]['epoch'],first_test=tests[0]['accuracy'],last_test=tests[-1]['accuracy'],change_pp=round(tests[-1]['accuracy']-tests[0]['accuracy'],2),best_test=max(r['accuracy'] for r in tests),llm_applied_records=len(re.findall(r' - APPLIED,',content))))
fields=['session','segment','source','first_epoch','last_epoch','first_test','last_test','change_pp','best_test','llm_applied_records']
with (OUT/'session_accuracy_summary.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in runs)
(OUT/'accuracy_data.json').write_text(json.dumps(runs,indent=2))
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'svg.fonttype':'none'})
figures=[]
def save(fig,name,title,explanation):
 fig.tight_layout();fig.canvas.draw(); bottom=fig.get_tightbbox(fig.canvas.get_renderer()).y0/fig.get_figheight()
 fig.text(.025,bottom-.05,textwrap.fill(explanation,width=int(fig.get_figwidth()*12)),va='top',fontsize=10,color='#334155',linespacing=1.5)
 for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig);figures.append((name,title,explanation))
def series(run,split):return [r for r in run['rows'] if r['split']==split]
latest=max(runs,key=lambda r:max(x['time'] for x in r['rows']))
fig,ax=plt.subplots(figsize=(10,5.4))
for split,color in [('Validation','#2563eb'),('Test','#e28316')]:
 rows=series(latest,split);ax.plot([r['epoch'] for r in rows],[r['accuracy'] for r in rows],'o-',color=color,label=split)
 for r in rows:ax.annotate(f"{r['accuracy']:.2f}%",(r['epoch'],r['accuracy']),xytext=(0,9 if split=='Validation' else -17),textcoords='offset points',ha='center')
ax.set(xlabel='Completed training epoch',ylabel='Accuracy (%)',title=f"Latest logged session: {latest['session']}",xticks=[r['epoch'] for r in series(latest,'Test')]);ax.legend();ax.margins(y=.3)
explanation=f"Higher means more correct predictions. Test accuracy goes from {latest['first_test']:.2f}% at epoch {latest['first_epoch']} to {latest['last_test']:.2f}% at epoch {latest['last_epoch']}: an increase of {latest['change_pp']:.2f} percentage points. This compares the first and last recorded epochs, not the model before training. Normal training can also improve accuracy, so this curve alone cannot show how much the LLM helped."
save(fig,'09_latest_accuracy','Accuracy during the latest session',explanation)

llm=next(r for r in runs if r['session']=='12_test_cifar_llm_loss')
ce=next(r for r in runs if r['session']=='13_ce_loss')
fig,axes=plt.subplots(1,2,figsize=(13,5.4))
for ax,split in zip(axes,['Validation','Test']):
 for run,label,color in [(llm,'Run named LLM loss','#2563eb'),(ce,'Run named CE loss','#e28316')]:
  rows=series(run,split);ax.plot([r['epoch'] for r in rows],[r['accuracy'] for r in rows],label=label,color=color)
 ax.set(title=split+' accuracy',xlabel='Completed training epoch',ylabel='Accuracy (%)');ax.legend(fontsize=9)
explanation=f"These are two older runs, labelled by their saved folder names. Higher is better. At epoch 29, test accuracy is {llm['last_test']:.2f}% for the run named LLM loss and {ce['last_test']:.2f}% for CE loss. The LLM-labelled run is {abs(llm['last_test']-ce['last_test']):.2f} percentage points lower. This is one run of each, and matching settings, starting weights and data splits are not recorded. It does not establish that the LLM causes better or worse accuracy."
save(fig,'10_historical_accuracy','Older LLM-labelled and CE-labelled runs',explanation)
fig,ax=plt.subplots(figsize=(10,5.3))
differences={}
for split,color in [('Validation','#2563eb'),('Test','#e28316')]:
 a={r['epoch']:r['accuracy'] for r in series(llm,split)};b={r['epoch']:r['accuracy'] for r in series(ce,split)}
 epochs=sorted(a.keys()&b.keys()); delta=[a[e]-b[e] for e in epochs];differences[split]=dict(zip(epochs,delta));ax.plot(epochs,delta,label=split,color=color)
ax.axhline(0,color='black',ls='--',lw=1);ax.set(title='Observed difference between the two older runs',xlabel='Completed training epoch',ylabel='LLM-labelled minus CE-labelled accuracy\n(percentage points)');ax.legend()
save(fig,'11_historical_difference','Accuracy difference at each epoch','Above zero means the LLM-labelled run scored higher at that epoch. Below zero means the CE-labelled run scored higher. The curves cross, so the observed difference changes during training. Equal epoch numbers do not make these a controlled comparison. These lines compare two historical runs; they do not measure the effect caused by LLM suggestions.')
findings=dict(latest={k:latest[k] for k in fields},historical=dict(llm={k:llm[k] for k in fields},ce={k:ce[k] for k in fields},final_test_difference_pp=round(llm['last_test']-ce['last_test'],2)),controlled_strategy_results_found=False)
(OUT/'findings.json').write_text(json.dumps(findings,indent=2))
warning='These are observed accuracy changes in saved sessions. There are no saved matched strategy-experiment results, so an accuracy benefit caused by the LLM is not established. Older run names are labels, not verified training configurations. No new training or paid LLM calls were performed.'
sections=''.join(f'<section><h2>{title}</h2><img src="{name}.png" alt="{title}"><p>{html.escape(explanation)}</p><a href="{name}.png">PNG</a> · <a href="{name}.svg">SVG</a></section>' for name,title,explanation in figures)
(OUT/'index.html').write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Accuracy checks</title><style>body{{font:16px/1.6 system-ui;max-width:1200px;margin:40px auto;padding:24px;color:#172033}}img{{width:100%}}section{{margin:36px 0}}a{{color:#2563eb}}</style><h1>Accuracy checks from saved logs</h1><p>{warning}</p>{sections}</html>')
(OUT/'README.md').write_text('# Accuracy analysis\n\n'+warning+'\n\n'+ '\n\n'.join(f'## {title}\n\n{explanation}' for _,title,explanation in figures)+'\n\nSources and epoch values are in accuracy_data.json; session summaries are in session_accuracy_summary.csv.\n')
# Link the additional plots from existing galleries, without regenerating them.
for folder in ['experiment_plots','supervisor_plots']:
 path=ROOT/'reports'/folder/'index.html'
 if path.exists():
  content=path.read_text();marker='<!-- accuracy-analysis-link -->'
  if marker not in content: path.write_text(content+f'{marker}<section><h2>Accuracy improvement</h2><p>Three extra plots check the saved training logs and explain what can be concluded.</p><a href="../accuracy_analysis/index.html">Open accuracy plots and findings</a></section>')
print(json.dumps(findings,indent=2));print('Saved',OUT)
