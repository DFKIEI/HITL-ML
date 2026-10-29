"""Offline analysis of completed controlled strategy runs; no API calls."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/hitl-matplotlib')
import argparse,json,csv,html,textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results',type=Path,default=ROOT/'code/experiments/results/strategies_CIFAR10_60ep_gpt5mini_controlled_v2.jsonl')
args=parser.parse_args()
rows=[json.loads(l) for l in args.results.read_text().splitlines() if l.strip()]
conditions=['ce_only','no_moves','llm_2d','llm_high_dim'];seeds=[0,1,2]
names={'ce_only':'Ordinary training','no_moves':'Extra loss, no moves','llm_2d':'LLM in 2D','llm_high_dim':'LLM in full space'}
colors=['#64748b','#e28316','#2563eb','#b43c68']
lookup={(r['condition'],r['seed']):r for r in rows}
if len(rows)!=12 or len(lookup)!=12 or set(lookup)!={(c,s) for c in conditions for s in seeds}:raise ValueError('Expected exactly 12 unique condition/seed runs')
for r in rows:
 if [e['epoch'] for e in r['epochs']]!=list(range(1,61)):raise ValueError('Incomplete or duplicate epochs')
 if r.get('protocol')!=dict(epochs=60,pause=5,batch=512,split_seed=20261028,version=2):raise ValueError('Unexpected protocol')
 if any(not np.isfinite(e[k]) for e in r['epochs'] for k in ['test_accuracy','val_accuracy','ce_loss','interaction_loss']):raise ValueError('Nonfinite metrics')
for s in seeds:
 fingerprints={lookup[c,s].get('initial_weights_sha256') for c in conditions}
 if len(fingerprints)!=1 or None in fingerprints:raise ValueError('Starting weights differ')
OUT=ROOT/'reports/controlled_accuracy';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.17,'svg.fonttype':'none'})
figures=[]
def save(fig,name,title,caption):
 fig.tight_layout();fig.canvas.draw();bottom=fig.get_tightbbox(fig.canvas.get_renderer()).y0/fig.get_figheight()
 fig.text(.02,bottom-.05,textwrap.fill(caption,width=int(fig.get_figwidth()*12)),va='top',fontsize=10,color='#334155',linespacing=1.5)
 for ext in ['png','svg']:
  p=OUT/f'{name}.{ext}';fig.savefig(p,dpi=180,bbox_inches='tight')
  if ext=='svg':p.write_text('\n'.join(l.rstrip() for l in p.read_text().splitlines())+'\n')
 plt.close(fig);figures.append((name,title,caption))
finals={c:np.array([lookup[c,s]['epochs'][-1]['test_accuracy'] for s in seeds]) for c in conditions}
summary={c:dict(test_accuracy_by_seed=finals[c].tolist(),mean_test_accuracy=float(finals[c].mean()),sample_sd=float(finals[c].std(ddof=1)),mean_minutes=float(np.mean([lookup[c,s]['minutes'] for s in seeds])),request_errors=sum(bool(q.get('error')) for s in seeds for q in lookup[c,s]['llm_requests'])) for c in conditions}
comparisons={}
for a,b in [('llm_2d','ce_only'),('llm_2d','no_moves'),('no_moves','ce_only'),('llm_high_dim','ce_only')]:
 d=finals[a]-finals[b];comparisons[f'{a} minus {b}']=dict(differences_pp=d.tolist(),mean_pp=float(d.mean()),sample_sd_pp=float(d.std(ddof=1)))
fig,ax=plt.subplots(figsize=(11,5.5))
for i,c in enumerate(conditions):
 ax.scatter(i+np.array([-.11,0,.11]),finals[c],s=55,color=colors[i])
 ax.errorbar(i,finals[c].mean(),yerr=finals[c].std(ddof=1),fmt='_',ms=25,mew=2,color='black',capsize=5)
 for j,v in enumerate(finals[c]):ax.annotate(f's{j}: {v:.2f}',(i+[-.11,0,.11][j],v),xytext=(6,(-15 if j==1 else 7)),textcoords='offset points',fontsize=8)
ax.set(xticks=range(4),xticklabels=[names[c] for c in conditions],ylabel='Final test accuracy (%)',title='Accuracy after 60 epochs: all three seeds')
save(fig,'01_final_accuracy','Final accuracy','Each coloured dot is one training run. Black marks show the mean and one sample standard deviation, not confidence intervals. Higher is better. The 2D LLM method has the highest mean; the full-space method performs much worse and varies strongly between runs. Only three seeds were tested.')
fig,axes=plt.subplots(1,2,figsize=(13,5.4))
for ax,key,title in zip(axes,['test_accuracy','val_accuracy'],['Test accuracy','Validation accuracy']):
 for i,c in enumerate(conditions):
  vals=np.array([[e[key] for e in lookup[c,s]['epochs']] for s in seeds]);epochs=np.arange(1,61)
  for v in vals:ax.plot(epochs,v,color=colors[i],alpha=.18,lw=.8)
  ax.plot(epochs,vals.mean(axis=0),color=colors[i],label=names[c],lw=2)
 ax.axvline(5,color='gray',ls='--',lw=1);ax.set(title=title,xlabel='Epoch',ylabel='Accuracy (%)');ax.legend(fontsize=8)
save(fig,'02_learning_curves','Accuracy during training','Thick lines show averages across three runs; thin lines show the individual runs. The first LLM suggestions arrive after epoch 5, marked by the dashed line. The 2D method finishes slightly above ordinary training on average. The full-space method falls behind after interventions begin. This timing does not by itself identify the cause.')
fig,ax=plt.subplots(figsize=(12,5.4))
for i,(key,value) in enumerate(comparisons.items()):
 d=np.array(value['differences_pp']);ax.scatter(i+np.array([-.1,0,.1]),d,s=55,color=colors[i]);ax.plot(i,d.mean(),'_',ms=24,mew=2,color='black')
 for j,v in enumerate(d):ax.annotate(f'{v:+.2f}',(i+[-.1,0,.1][j],v),xytext=(5,6),textcoords='offset points',fontsize=9)
ax.axhline(0,color='black',lw=1);ax.set(xticks=range(4),xticklabels=['2D LLM minus\nordinary training','2D LLM minus\nextra loss, no moves','Extra loss minus\nordinary training','Full-space LLM minus\nordinary training'],ylabel='Final test accuracy difference (percentage points)',title='Compare methods using the same seed')
save(fig,'03_paired_differences','Accuracy differences','Above zero means the first method did better. The 2D LLM method beats ordinary training in all three seeds: +0.70, +1.56 and +0.62 points. Against the extra-loss control it gains +0.90, +0.64 and -0.08 points. Some of the overall benefit may therefore come from the extra loss. These three runs do not establish a reliable general advantage.')
fig,axes=plt.subplots(1,2,figsize=(13,5.5))
for i,c in enumerate(conditions):
 for s in seeds:
  r=lookup[c,s];eps=np.arange(1,61)
  axes[0].plot(eps,[e['ce_loss'] for e in r['epochs']],color=colors[i],alpha=.5,label=names[c] if s==0 else None)
  axes[1].plot(eps,[e['interaction_loss'] for e in r['epochs']],color=colors[i],alpha=.5)
axes[0].set(title='Classification loss',xlabel='Epoch',ylabel='Cross-entropy loss');axes[0].legend(fontsize=8)
axes[1].set(title='Interaction loss',xlabel='Epoch',ylabel='Loss (symmetric log scale)',yscale='symlog');axes[1].axvline(5,color='gray',ls='--',lw=1)
save(fig,'04_loss_diagnostics','Loss diagnostics','Each line is one run. Lower classification loss usually means a better fit to the training labels. Interaction loss measures disagreement with the requested layout. In two full-space runs it grows sharply, reaching about 70,167 and 5,486 at epoch 60. This is evidence of instability that needs investigation; loss values alone do not establish the gradient cause. The right axis compresses large values so all runs remain visible.')
errors=[dict(condition=r['condition'],seed=r['seed'],after_epoch=5*(i+1),error=q['error']) for r in rows for i,q in enumerate(r['llm_requests']) if q.get('error')]
request_count=sum(len(r['llm_requests']) for r in rows)
checks=dict(complete_runs=12,epochs_per_run=60,starting_fingerprints_match=True,recorded_protocol_matches=True,planned_requests=request_count,failed_requests=len(errors),split_archive_present=Path(str(args.results)+'.splits.npz').exists())
result=dict(source=str(args.results.relative_to(ROOT)),checks=checks,conditions=summary,paired_comparisons=comparisons,request_errors=errors,total_minutes=sum(r['minutes'] for r in rows))
(OUT/'summary.json').write_text(json.dumps(result,indent=2))
with (OUT/'final_accuracy.csv').open('w',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['condition','seed','test_accuracy','validation_accuracy','minutes','failed_requests'])
 for r in rows:w.writerow([r['condition'],r['seed'],r['epochs'][-1]['test_accuracy'],r['epochs'][-1]['val_accuracy'],r['minutes'],sum(bool(q.get('error')) for q in r['llm_requests'])])
report='''# Controlled accuracy study results

All 12 runs completed 60 epochs. Starting model fingerprints match within each seed and protocol metadata agree. The exact split archive was not supplied, so the recorded split seed is checked but saved split indices cannot be independently audited here.

| Method | Final test accuracy, mean ± sample SD | Seeds 0 / 1 / 2 |
|---|---|---|
'''
for c in conditions:report+=f"| {names[c]} | {summary[c]['mean_test_accuracy']:.2f}% ± {summary[c]['sample_sd']:.2f} | {' / '.join(f'{x:.2f}' for x in finals[c])} |\n"
report+='''
The 2D LLM method gains 0.96 percentage points over ordinary training, averaged across matched seeds. Against the no-moves interaction-loss control, the gain is 0.49 points; one seed is slightly worse. These are preliminary results from three seeds on one fixed data split.

The full-space method loses 20.61 points against ordinary training on average. All three seeds are worse, including the two with no API errors. The interaction loss grows markedly in two runs. This warrants checking target construction, loss scale and weighting before repeating the full study; it does not prove which component causes the failure.

There were 66 LLM request records, 63 without a recorded error. Seed 2 of the full-space condition had token-limit errors after epochs 45, 50 and 55. The experiment retains previous suggestions on request failure, so report this condition as executed with those failures.

The separate legacy strategies_CIFAR10_60ep.jsonl file is not pooled: it lacks protocol and initial-weights metadata and makes 12 requests per LLM run rather than 11. Suggestion-level geometric gains were not recorded in this study, so a gain-versus-accuracy correlation cannot be calculated.

Final accuracy is taken at epoch 60, not selected by the best test epoch. No confidence claim is made from three seeds. Raw loss magnitude is diagnostic, not proof of gradient dominance. There is no full-space no-moves control.
'''
(OUT/'README.md').write_text(report)
sections=''.join(f'<section><h2>{title}</h2><img src="{name}.png" alt="{title}"><p>{html.escape(caption)}</p><a href="{name}.png">PNG</a> · <a href="{name}.svg">SVG</a></section>' for name,title,caption in figures)
(OUT/'index.html').write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Controlled accuracy results</title><style>body{{font:16px/1.6 system-ui;max-width:1250px;margin:40px auto;padding:24px;color:#172033}}img{{width:100%}}section{{margin:36px 0}}a{{color:#2563eb}}</style><h1>Controlled accuracy study</h1><p>12 complete runs, 60 epochs each, three matched seeds. 2D LLM: 83.42% mean test accuracy; ordinary training: 82.46%; full-space LLM: 61.85%.</p><p>Three request failures in full-space seed 2. The split archive is missing; the metadata record a common split seed. Results are preliminary.</p><a href="README.md">Full findings and limitations</a>{sections}</html>')
print(json.dumps(result,indent=2))
