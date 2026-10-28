"""Offline figures for the supervisor meeting; no API calls or training.
Run from any directory with the project's Python environment.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/hitl-matplotlib')
from pathlib import Path
import json
from collections import defaultdict
import html
import textwrap
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
OUT = ROOT / 'reports/experiment_plots'
OUT.mkdir(parents=True, exist_ok=True)

def load(name):
    return [json.loads(l) for l in (ROOT/'code/experiments/results'/name).read_text().splitlines() if l.strip()]
comparison = load('llm_comparison.jsonl')
variants = load('llm_prompt_variants.jsonl')
valid = lambda rows: [r for r in rows if not r.get('error') and r.get('suggestions')]
models = sorted({r['model'] for r in comparison}, key=lambda m: -np.mean([r['combined_impact']['separation_gain'] for r in valid(comparison) if r['model']==m]))
short = lambda m: m.split('/')[-1]
colors = dict(zip(models, ['#2563eb','#e28316','#14967f','#ad52aa','#64748b']))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','axes.grid':True,'grid.alpha':.16,'svg.fonttype':'none'})
figures=[]
summary={}

EXPLANATIONS = {
    '01_cost_vs_gain': "Each dot is one LLM. Farther left means cheaper answers. Higher up means its suggestions separate the classes better. GPT-5-mini is much cheaper than Opus and has a similar average gain here. The vertical lines show uncertainty in the averages. These scores describe point layouts before retraining; they do not show better prediction accuracy.",
    '02_gain_by_checkpoint': "Left: each coloured dot is one answer. The black dot shows the average, and its line shows uncertainty. Higher means more improvement in class separation. Right: the same models are compared at four saved stages of training. Results change with the dataset and training stage, so one overall average does not tell the whole story.",
    '03_prompt_effects': "This shows what happens when we change the instructions given to the LLM. Zero means the same gain as the normal prompt. Right of zero means more gain; left means less. The lines show uncertainty. 'Focus' means the goal typed by the user. Asking for only the worst pair or minimal changes lowers the total gain here. That alone does not tell us whether the advice is more useful to a person.",
    '04_prompt_tradeoffs': "The four panels ask: How much does separation improve? How useful is each suggestion on its own? How far do points move? How many suggestions are returned? Higher means more of the named quantity, not always better. Asking for the worst pair gives fewer suggestions and less total gain, but stronger individual suggestions. Small lines show uncertainty in the averages.",
    '05_targeting_and_moves': "Left: taller bars mean the LLM more often chooses a pair that the classifier frequently confuses. The first suggestion is usually better targeted than the full list. Right: taller bars mean that type of action is used more often. Models mostly push classes apart; none pulls classes together here. Tightening can happen at the same time as another move.",
    '06_pair_agreement': "These tables show how often the same class pairs are chosen. A value of 1 means exactly the same set of pairs; 0 means no shared pairs. Darker blue means more agreement. Left compares different LLMs. Right compares changed instructions with the normal prompt for the same LLM. A lower value means different choices; it does not mean worse choices.",
    '07_benefit_calibration': "The LLM labels each suggestion as low, medium or high expected benefit. Each dot shows how much that suggestion actually improves class separation when tested alone. Black marks show the averages; n counts the suggestions. If the labels are useful, high should usually sit above medium and low. Empty groups have no evidence. This tests layout improvement, not prediction accuracy.",
    '08_before_after_layout': "Each dot is a real example, and each colour is a class. Left shows a saved layout. Right shows where the same points would go after one suggestion. The separation score rises from about -0.21 to -0.16, although many colours still mix. This is an illustration using a suggestion from another saved experiment, not a recorded before-and-after training result. The model has not been retrained here."
}

def save(fig, name, title, caption):
    explanation = EXPLANATIONS[name]
    fig.canvas.draw()
    bottom = fig.get_tightbbox(fig.canvas.get_renderer()).y0 / fig.get_figheight()
    wrapped = textwrap.fill(explanation, width=int(fig.get_figwidth() * 12))
    fig.text(.02, bottom - .045, wrapped, ha='left', va='top', fontsize=10,
             linespacing=1.5, color='#334155')
    fig.savefig(OUT/(name+'.png'), dpi=180, bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'), bbox_inches='tight')
    figures.append((name,title,caption))
    plt.close(fig)

def interval(rows, fn, seed=42):
    """Stratified bootstrap: fixed checkpoints; resample draws within each.
    Uncertainty is conditional on these checkpoints, not new datasets/models.
    """
    groups=defaultdict(list)
    for r in rows:
        v=fn(r)
        if v is not None and np.isfinite(v): groups[r['checkpoint']].append(v)
    if not groups: return np.nan,np.nan,np.nan
    rng=np.random.default_rng(seed)
    samples=[rng.choice(v,(4000,len(v)),replace=True) for v in groups.values()]
    boot=np.concatenate(samples,axis=1).mean(axis=1)
    return np.mean([x for v in groups.values() for x in v]), *np.quantile(boot,[.025,.975])

def pairs(r): return {tuple(sorted((s['class_i'],s['class_j']))) for s in r['suggestions']}
def gain(r): return r['combined_impact'].get('separation_gain')
def point_ci(ax,x,stats,color):
    mean,lo,hi=stats
    ax.errorbar(x,mean,yerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt='o',color=color,capsize=4,ms=7)

# 1: cost and gain
fig,ax=plt.subplots(figsize=(10,5.8))
for m in models:
    rows=valid([r for r in comparison if r['model']==m]); stats=interval(rows,gain)
    cost=np.mean([r['usage']['cost'] for r in rows if r.get('usage',{}).get('cost') is not None])
    point_ci(ax,cost,stats,colors[m]); ax.annotate(short(m),(cost,stats[0]),xytext=(8,9),textcoords='offset points',fontsize=10)
    summary[m]={'valid':len(rows),'attempted':sum(r['model']==m for r in comparison),'mean_combined_gain':stats[0],'gain_ci':stats[1:],'mean_cost_successful_answer':cost}
ax.set(xscale='log',xlabel='Recorded cost per successful answer (USD; logarithmic scale)',ylabel='Combined silhouette gain',title='Model comparison: cost and geometric benefit')
ax.margins(x=.25,y=.25)
save(fig,'01_cost_vs_gain','Cost versus benefit','Means and 95% stratified bootstrap intervals over repeated draws within the four fixed checkpoints. Gain applies the complete answer jointly. Costs are recorded experiment costs, not current price quotes; failed answers excluded.')

# 2: per checkpoint and distributions
fig,axes=plt.subplots(1,2,figsize=(13,5))
for i,m in enumerate(models):
    rows=valid([r for r in comparison if r['model']==m]); vals=[gain(r) for r in rows]
    jitter=np.random.default_rng(i).uniform(-.13,.13,len(vals))
    axes[0].scatter(i+jitter,vals,color=colors[m],alpha=.65,s=26)
    point_ci(axes[0],i,interval(rows,gain),'black')
checkpoints=sorted({r['checkpoint'] for r in comparison}, key=lambda c:(c.split('/')[1],int(Path(c).stem.split('_')[-1])))
for m in models:
    means=[np.mean([gain(r) for r in valid(comparison) if r['model']==m and r['checkpoint']==c]) for c in checkpoints]
    axes[1].plot(range(4),means,'o-',label=short(m),color=colors[m])
axes[0].set(xticks=range(5),xticklabels=[short(m) for m in models],ylabel='Combined silhouette gain',title='Individual answers and mean intervals')
axes[0].tick_params(axis='x',rotation=30)
axes[1].set(xticks=range(4),xticklabels=[c.split('/')[1]+'\nepoch '+Path(c).stem.split('_')[-1] for c in checkpoints],ylabel='Mean combined silhouette gain',title='Results depend on the checkpoint')
axes[1].legend(fontsize=8)
fig.tight_layout()
save(fig,'02_gain_by_checkpoint','Gain distributions and checkpoints','Each dot is one successful answer. Intervals resample draws within fixed checkpoints. The right panel separates the two datasets and two training stages each.')

# 3: prompt paired differences
prompts=['baseline','always_five','diverse_moves','plan_first','terse','goal_tighten','goal_worst_pair','goal_keep_layout']
labels=['Baseline','Always five','Diverse moves','Plan first','Short prompt','Goal: tighten','Goal: worst pair','Goal: minimal change']
vmodels=[m for m in models if any(r['model']==m for r in variants)]
fig,axes=plt.subplots(1,len(vmodels),figsize=(14,6),sharey=True,sharex=True)
for ax,m in zip(axes,vmodels):
    base={(r['checkpoint'],r['draw']):r for r in valid(variants) if r['model']==m and r['prompt']=='baseline'}
    for i,p in enumerate(prompts):
        rows=[]
        for r in valid(variants):
            key=(r['checkpoint'],r['draw'])
            if r['model']==m and r['prompt']==p and key in base:
                rows.append(dict(r,delta=gain(r)-gain(base[key])))
        mean,lo,hi=interval(rows,lambda r:r['delta'])
        ax.errorbar(mean,i,xerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt='o',color=colors[m],capsize=3)
        ax.annotate(' n='+str(len(rows)),(mean,i),xytext=(6,5),textcoords='offset points',fontsize=8)
    ax.axvline(0,color='#475569',lw=1); ax.axhline(4.5,color='#94a3b8',ls='--'); ax.set(title=short(m),xlabel='Combined gain minus matched baseline')
axes[0].set(yticks=range(8),yticklabels=labels); axes[0].invert_yaxis()
fig.suptitle('Prompt effects on identical checkpoint–draw states',fontweight='bold'); fig.tight_layout()
save(fig,'03_prompt_effects','Prompt effects relative to baseline','Differences are paired by checkpoint and draw within the prompt study, with 95% stratified bootstrap intervals. Failed or missing pairs excluded; n is shown. Dashed line separates wording variants from operator goals. Intervals do not cover uncertainty about new checkpoints.')

# 4 prompt tradeoffs
metrics=[('Total gain',lambda r:gain(r)),('Gain per suggestion (each alone)',lambda r:np.mean([s['impact']['separation_gain'] for s in r['suggestions']])),('Relative displacement',lambda r:r['combined_impact']['shift_ratio']),('Number of suggestions',lambda r:len(r['suggestions']))]
fig,axes=plt.subplots(2,2,figsize=(14,10))
for ax,(title,fn) in zip(axes.flat,metrics):
    for j,m in enumerate(vmodels):
        for i,p in enumerate(prompts):
            rows=[r for r in valid(variants) if r['model']==m and r['prompt']==p]
            point_ci(ax,i+(j-1)*.2,interval(rows,fn),colors[m])
        ax.plot([],[], 'o',color=colors[m],label=short(m))
    ax.set(title=title,xticks=range(8),xticklabels=labels); ax.tick_params(axis='x',rotation=38)
axes[0,0].legend(fontsize=9)
fig.suptitle('What changes when the prompt changes?',fontweight='bold'); fig.tight_layout()
save(fig,'04_prompt_tradeoffs','Prompt benefit, movement and answer length','Mean and 95% stratified bootstrap intervals. Per-suggestion gain evaluates each suggestion independently; it is not total gain divided by suggestion count. Displacement is the recorded shift_ratio.')

# 5 targeting and strategies
fig,axes=plt.subplots(1,2,figsize=(14,5.5))
for i,m in enumerate(models):
    rows=valid([r for r in comparison if r['model']==m]); ss=[s for r in rows for s in r['suggestions']]
    vals=[np.mean([r['suggested_in_top']/len(r['suggestions']) for r in rows]),np.mean([r['first_in_top'] for r in rows])]
    axes[0].bar(np.array([0,1])+(i-2)*.15,vals,width=.14,color=colors[m],label=short(m))
    props=[np.mean([s['direction']==d for s in ss]) for d in ['away_from_j','toward_j']]+[np.mean([s['direction'] not in ['away_from_j','toward_j'] for s in ss]),np.mean([s['tighten_i'] or s['tighten_j'] for s in ss])]
    axes[1].bar(np.arange(4)+(i-2)*.15,props,width=.14,color=colors[m])
axes[0].set(xticks=[0,1],xticklabels=['All suggestions\n(mean answer share)','First suggestion'],title='Targeting the five most confused pairs')
axes[1].set(xticks=range(4),xticklabels=['Push apart','Pull together','Open space','Also tighten'],title='Suggested intervention types')
for ax in axes: ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.set_ylim(0,1.12)
axes[0].legend(fontsize=8,loc='upper left'); fig.tight_layout()
save(fig,'05_targeting_and_moves','Targeting and intervention types','Descriptive proportions from successful model-comparison answers. Tightening can accompany a movement, so the four intervention bars do not sum to 100%.')

# 6 agreement
fig,axes=plt.subplots(1,2,figsize=(14,6))
lookup={(r['checkpoint'],r['draw'],r['model']):r for r in valid(comparison)}
mat=np.eye(len(models))
for i,a in enumerate(models):
    for j,b in enumerate(models):
        vals=[]
        for c,d,m in lookup:
            if m==a and (c,d,b) in lookup:
                x,y=pairs(lookup[c,d,a]),pairs(lookup[c,d,b]); vals.append(len(x&y)/len(x|y) if x|y else 1)
        mat[i,j]=np.mean(vals)
im=axes[0].imshow(mat,vmin=0,vmax=1,cmap='Blues')
axes[0].set(xticks=range(5),yticks=range(5),xticklabels=[short(m) for m in models],yticklabels=[short(m) for m in models],title='Agreement between models')
for i in range(5):
    for j in range(5): axes[0].text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',color='white' if mat[i,j]>.6 else 'black')
vm=np.zeros((3,8))
for j,m in enumerate(vmodels):
    base={(r['checkpoint'],r['draw']):r for r in valid(variants) if r['model']==m and r['prompt']=='baseline'}
    for i,p in enumerate(prompts):
        vals=[]
        for r in valid(variants):
            key=r['checkpoint'],r['draw']
            if r['model']==m and r['prompt']==p and key in base:
                x,y=pairs(r),pairs(base[key]); vals.append(len(x&y)/len(x|y) if x|y else 1)
        vm[j,i]=np.mean(vals)
axes[1].imshow(vm,vmin=0,vmax=1,cmap='Blues',aspect='auto'); axes[1].set(xticks=range(8),xticklabels=labels,yticks=range(3),yticklabels=[short(m) for m in vmodels],title='Pair overlap with matched baseline prompt')
for j in range(3):
    for i in range(8): axes[1].text(i,j,f'{vm[j,i]:.2f}',ha='center',va='center',color='white' if vm[j,i]>.6 else 'black',fontsize=8)
for ax in axes: ax.grid(False); ax.tick_params(axis='x',rotation=48)
fig.subplots_adjust(left=.15,right=.88,bottom=.33,top=.9,wspace=.7); fig.colorbar(im,ax=list(axes),shrink=.65,label='Mean Jaccard similarity')
save(fig,'06_pair_agreement','Agreement about which pairs to change','Pairs are unordered class-ID pairs, matched by checkpoint and draw. Failed answers excluded. Diagonal and baseline self-comparisons are 1 by construction. Lower values indicate different selected pairs, not necessarily worse suggestions.')

# 7 calibration
fig,axes=plt.subplots(1,5,figsize=(16,4),sharey=True)
for ax,m in zip(axes,models):
    ss=[s for r in valid(comparison) if r['model']==m for s in r['suggestions']]
    for i,level in enumerate(['low','medium','high']):
        vals=[s['impact']['separation_gain'] for s in ss if s['expected_benefit']==level]
        if vals:
            ax.scatter(i+np.random.default_rng(i).uniform(-.15,.15,len(vals)),vals,s=13,alpha=.35,color=colors[m]); ax.plot(i,np.mean(vals),'_',color='black',ms=18,mew=2)
        ax.text(i,-.17,f'n={len(vals)}',ha='center',transform=ax.get_xaxis_transform(),fontsize=9)
    ax.set(title=short(m),xticks=range(3),xticklabels=['Low','Medium','High']); ax.axhline(0,lw=.7,color='gray')
axes[0].set_ylabel('Individual-suggestion silhouette gain'); fig.suptitle('Does predicted benefit track measured geometric benefit?',fontweight='bold'); fig.tight_layout()
save(fig,'07_benefit_calibration','Expected versus measured benefit','Each dot is a suggestion, evaluated alone; black marks indicate means. Suggestions from the same answer share an input state and are not independent observations. Empty categories are labelled n=0; no calibration fit is inferred for them.')

# 8 illustrative real layout, applying a recorded suggestion on a separate saved state
from llm.suggestions import parse_suggestions
from llm.impact import apply_to_points, silhouette
snapshot=ROOT/'user_study_logs/18_test new/spaces/round_002.npz'
if snapshot.exists():
    data=np.load(snapshot); points=data['features_2d']; ys=data['labels']
    names=dict(enumerate(['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']))
    candidates=[r for r in valid(comparison) if r['model']=='openai/gpt-5-mini' and r['dataset']=='CIFAR10']
    record=min(candidates,key=lambda r:abs(gain(r)-np.median([gain(x) for x in candidates])))
    _,suggestions=parse_suggestions(record['raw'],names,set(int(c) for c in np.unique(ys)))
    s=suggestions[0]; target=apply_to_points(points,ys,[s]); before=silhouette(points,ys); after=silhouette(target,ys)
    fig,axes=plt.subplots(1,2,figsize=(12,5.8),sharex=True,sharey=True)
    for ax,arr,title in zip(axes,[points,target],['Saved session layout','Proposed layout before retraining']):
        for c in np.unique(ys):
            mask=ys==c; ax.scatter(arr[mask,0],arr[mask,1],s=25,alpha=.8,label=names[int(c)],color=plt.get_cmap('tab10')(int(c)))
        ax.set(title=title,xlabel='Saved plot coordinate 1',aspect='equal')
    allpoints=np.concatenate([points,target]); pad=np.ptp(allpoints,axis=0)*.1
    axes[0].set(xlim=(allpoints[:,0].min()-pad[0],allpoints[:,0].max()+pad[0]),ylim=(allpoints[:,1].min()-pad[1],allpoints[:,1].max()+pad[1]),ylabel='Saved plot coordinate 2')
    axes[1].legend(loc='center left',bbox_to_anchor=(1, .5),fontsize=8)
    fig.suptitle(f'Example: {s.class_i_name} / {s.class_j_name} | 2D silhouette {before:.3f} → {after:.3f}',fontweight='bold')
    fig.tight_layout()
    save(fig,'08_before_after_layout','Before and after: a real saved layout','Illustration, not a replay of the comparison experiment: the first suggestion from a median-gain GPT-5-mini CIFAR10 answer is applied to a separate real saved session snapshot (round 002). Identical points, colours and axis limits; metrics here are recomputed in 2D. No training occurs. The original experiment measured higher-dimensional geometry.')
    np.savez_compressed(OUT/'before_after_layout.npz',before=points,after=target,labels=ys)
    (OUT/'before_after_provenance.json').write_text(json.dumps({'snapshot':str(snapshot.relative_to(ROOT)),'source_checkpoint':record['checkpoint'],'source_draw':record['draw'],'model':record['model'],'suggestion':record['suggestions'][0],'silhouette_2d_before':before,'silhouette_2d_after':after},indent=2))

# Gallery, provenance and reproducibility
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
notes='Intervals describe repeated draws at four fixed checkpoints; they do not establish generalisation to other datasets or training runs. Two failed prompt-study answers are excluded where needed. No strategy-training result file was found: the plot of suggestion gain versus held-out accuracy improvement cannot yet be calculated. Session logs are not a substitute for matched strategy experiments.'
sections=''.join(f'<section><h2>{html.escape(title)}</h2><img src="{name}.png" alt="{html.escape(title)}"><p>{html.escape(EXPLANATIONS[name])}</p><details><summary>Technical details</summary><p>{html.escape(caption)}</p></details><p><a href="{name}.png">PNG</a> · <a href="{name}.svg">SVG</a></p></section>' for name,title,caption in figures)
(OUT/'index.html').write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Supervisor meeting figures</title><style>body{{font:16px/1.6 system-ui;max-width:1250px;margin:40px auto;padding:0 24px;color:#172033;background:#f8fafc}}section{{background:white;padding:24px;margin:26px 0;border:1px solid #dde3eb;border-radius:12px}}img{{width:100%;height:auto}}h1,h2{{line-height:1.2}}p{{max-width:1050px}}a{{color:#2563eb}}</style><h1>Supervisor meeting figures</h1><p>Based on {len(comparison)} model-comparison calls and {len(variants)} prompt-study calls. Generated offline from saved data.</p><p>{html.escape(notes)}</p>{sections}</html>')
(OUT/'README.md').write_text('# Supervisor meeting figures\n\nOpen index.html for the complete gallery. Every figure is available as PNG and editable SVG.\n\nRegenerate from the repository root:\n\n```sh\n/Users/vishalbanwari/miniconda3/envs/pamap/bin/python code/experiments/plot_supervisor_results.py\n```\n\n'+notes+'\n\n'+ '\n\n'.join(f'## {title}\n\n{EXPLANATIONS[name]}\n\nTechnical details: {caption}' for name,title,caption in figures))
print(f'Created {len(figures)} figures in {OUT}')
print('Comparison failures:',sum(bool(r.get('error')) for r in comparison))
print('Prompt failures:',sum(bool(r.get('error')) for r in variants))
