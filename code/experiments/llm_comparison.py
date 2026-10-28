"""Which LLM gives the best latent-space suggestions, and what strategies do
they propose? Same prompt, same latent states, several models.

For every checkpoint and random 10-per-class draw (as in the app), the
semantic state and prompt are built once and sent to every model, so the
models answer exactly the same question. Per answer it records:

- reliability and cost: valid JSON, latency, tokens, OpenRouter cost;
- targeting (as real_latent_check.py): suggestions on one of the 5 most
  confused pairs, whether the first one is, and harmful "move closer" on a
  confused pair;
- measured impact on the real latent features (llm/impact.py): silhouette
  gain, Fisher ratio, shift, neighbours kept - per suggestion and for all of
  them together;
- calibration: the model's own expected_benefit vs the measured gain;
- strategy: the global plan text and the mix of moves it uses (push apart /
  pull together / open space / tighten), and how much the models agree on
  which pairs to touch.

    python experiments/llm_comparison.py                               # models from openrouter.RECOMMENDED_MODELS
    python experiments/llm_comparison.py --models openai/gpt-5-mini google/gemini-2.5-flash --draws 3
    python experiments/llm_comparison.py --summarize experiments/results/llm_comparison.jsonl --strategies
    python experiments/llm_comparison.py --prompts baseline always_five diverse_moves plan_first terse   # what if the prompt changes?
    python experiments/llm_comparison.py --dry-run                     # no API calls, checks the pipeline

Results are appended to --out one line per answer; answers already in it are
skipped, so an interrupted run can be restarted with the same command.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import load_dataset  # noqa: E402
from experiments.real_latent_check import (CHECKPOINTS, PROJECT_DIR, TOP_K, balanced_sample,  # noqa: E402
                                           confused_pairs, device, latent_features)
from llm import openrouter  # noqa: E402
from llm.impact import BENEFIT_LEVELS, measured_benefit_level, suggestion_impact  # noqa: E402
from llm.prompt_variants import VARIANTS, build_variant_prompt  # noqa: E402
from llm.suggestions import build_semantic_state, parse_suggestions  # noqa: E402
from model import get_model  # noqa: E402
from ui.ui_display import create_sequential_mapping, get_label_names  # noqa: E402

DEFAULT_OUT = os.path.join('experiments', 'results', 'llm_comparison.jsonl')
IMPACT_KEYS = ('separation_gain', 'shift_ratio', 'knn_preserved', 'fisher_before', 'fisher_after',
               'subspace_rotation_deg', 'loss_target', 'pair_overlap_before', 'pair_overlap_after')


def ask(model_id, prompt, dry_run, state):
    """(content, usage, latency). ``dry_run`` answers from the prompt's own
    flagged pairs instead of calling OpenRouter."""
    started = time.monotonic()
    if dry_run:
        flagged = [p for p in state['pairs'] if p['overlap_level'] in ('medium', 'high', 'very_high')][:5]
        content = json.dumps({'global': {'issue': 'dry run', 'strategy': 'push the flagged pairs apart'},
                              'suggestions': [{'class_i': p['class_i'], 'class_j': p['class_j'],
                                               'suggestion': 'dry run', 'direction': 'away_from_j',
                                               'scale': 'medium', 'tighten_i': False, 'tighten_j': False,
                                               'expected_benefit': 'medium'} for p in flagged]})
        return content, {}, time.monotonic() - started
    messages = [{'role': 'system', 'content': 'Return only JSON that matches the requested schema.'},
                {'role': 'user', 'content': prompt}]
    content, body = openrouter.chat_completion(messages, model=model_id, extra={'usage': {'include': True}})
    usage = body.get('usage') or {}
    return content, {k: usage.get(k) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cost')}, \
        time.monotonic() - started


def evaluate(content, points, labels, names, ranking, classes):
    top = {pair for pair, _ in ranking[:TOP_K]}
    rank_of = {pair: r for r, (pair, _) in enumerate(ranking)}
    summary, suggestions = parse_suggestions(content, names, set(classes))
    rows = []
    for s in suggestions:
        pair = frozenset((s.class_i, s.class_j))
        impact = suggestion_impact(points, labels, s)
        rows.append({
            'class_i': s.class_i, 'class_j': s.class_j, 'pair': f"{s.class_i_name}/{s.class_j_name}",
            'confusion_rank': rank_of.get(pair, -1) + 1, 'direction': s.direction, 'scale': s.scale,
            'tighten_i': s.tighten_i, 'tighten_j': s.tighten_j, 'expected_benefit': s.expected_benefit,
            'measured_benefit': measured_benefit_level(impact), 'text': s.suggestion,
            'impact': {k: impact.get(k) for k in IMPACT_KEYS},
        })
    combined = suggestion_impact(points, labels, list(suggestions)) if suggestions else {}
    pairs = [frozenset((r['class_i'], r['class_j'])) for r in rows]
    return {
        'issue': summary.issue, 'strategy': summary.strategy, 'suggestions': rows,
        'suggested_in_top': sum(p in top for p in pairs),
        'first_in_top': bool(pairs) and pairs[0] in top,
        'toward_on_top_pair': sum(r['direction'] == 'toward_j' and p in top for p, r in zip(pairs, rows)),
        'combined_impact': {k: combined.get(k) for k in IMPACT_KEYS},
    }


def run(args):
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = {(r['checkpoint'], r['draw'], r['model'], r.get('prompt', 'baseline')) for r in load_results(out_path)}
    checkpoints = [tuple(c.split(':', 1)) for c in args.checkpoint] if args.checkpoint else CHECKPOINTS
    rng = np.random.default_rng(args.seed)
    dev = device()
    print(f"models: {', '.join(args.models)}\nresults: {out_path}\n", flush=True)

    os.chdir(PROJECT_DIR)  # the loaders read 'dataset/...' relative to the project root
    loaded = {}
    with open(out_path, 'a') as out:
        for dataset, checkpoint_path in checkpoints:
            if not os.path.exists(checkpoint_path):
                print(f"skipping {checkpoint_path}: not found", flush=True)
                continue
            if dataset not in loaded:
                loaded[dataset] = load_dataset(dataset, 512)
            _, valloader, _, num_classes, input_shape = loaded[dataset]
            model = get_model(f'CNN_{dataset}', input_shape, num_classes)
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=False)['model_state_dict'])
            features, labels, predictions = latent_features(model.to(dev), valloader, dev)
            classes = sorted(int(c) for c in np.unique(labels))
            label_names = get_label_names(valloader.dataset)
            sequential = create_sequential_mapping(label_names) if label_names else {}
            names = {c: sequential.get(c, f'class_{c}') for c in classes}
            ranking = confused_pairs(labels, predictions, classes)
            print(f"=== {checkpoint_path} (val accuracy {float((labels == predictions).mean()):.1%})", flush=True)

            for draw in range(args.draws):
                # Drawn even when every model is done, so later draws stay the same.
                sample = balanced_sample(labels, args.per_class, rng)
                points, sample_labels = features[sample], labels[sample]
                state, _ = build_semantic_state(points, sample_labels, names)
                for variant, model_id in ((v, m) for v in args.prompts for m in args.models):
                    tag = f"{model_id}" if variant == 'baseline' and len(args.prompts) == 1 else f"{model_id} [{variant}]"
                    if (checkpoint_path, draw, model_id, variant) in done:
                        print(f"  draw {draw} {tag}: already in results, skipped", flush=True)
                        continue
                    prompt = build_variant_prompt(variant, state, dataset)
                    record = {'checkpoint': checkpoint_path, 'dataset': dataset, 'draw': draw,
                              'per_class': args.per_class, 'model': model_id, 'prompt': variant,
                              'dry_run': args.dry_run,
                              'top_confused': ['/'.join(names[c] for c in sorted(p)) for p, _ in ranking[:TOP_K]]}
                    try:
                        content, usage, latency = ask(model_id, prompt, args.dry_run, state)
                        record.update(usage=usage, latency_s=latency, raw=content)
                        record.update(evaluate(content, points, sample_labels, names, ranking, classes))
                        record['error'] = None
                    except Exception as e:  # report and keep going
                        record['error'] = f"{type(e).__name__}: {e}"
                    out.write(json.dumps(record) + '\n')
                    out.flush()
                    if record['error']:
                        print(f"  draw {draw} {tag}: ERROR {record['error'][:200]}", flush=True)
                    else:
                        gain = record['combined_impact'].get('separation_gain')
                        print(f"  draw {draw} {tag}: {len(record['suggestions'])} suggestion(s), "
                              f"{record['suggested_in_top']} on a top-{TOP_K} pair, separation gain "
                              f"{gain if gain is None else round(gain, 3)}, {record['latency_s']:.1f}s", flush=True)


def load_results(path):
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else float('nan')


def summarize(results, show_strategies=False):
    multi_prompt = len({r.get('prompt', 'baseline') for r in results}) > 1
    by_model = defaultdict(list)
    for r in results:
        by_model[f"{r['model']} [{r.get('prompt', 'baseline')}]" if multi_prompt else r['model']].append(r)

    print(f"{'model [prompt]' if multi_prompt else 'model':44s} {'ok':>6s} {'latency':>8s} {'cost $':>8s} {'#sugg':>6s} {'on top5':>8s} "
          f"{'1st top5':>8s} {'closer!':>7s} {'sep gain':>9s} {'per sugg':>9s} {'calib rho':>9s} {'calib ==':>8s}")
    profiles = {}
    for model_id, rows in sorted(by_model.items()):
        ok = [r for r in rows if not r.get('error')]
        suggestions = [s for r in ok for s in r['suggestions']]
        expected = [(BENEFIT_LEVELS.index(s['expected_benefit']), s['impact']['separation_gain'])
                    for s in suggestions if s.get('expected_benefit') in BENEFIT_LEVELS
                    and s['impact'].get('separation_gain') is not None]
        rho = (spearmanr(*zip(*expected)).correlation
               if len(expected) > 2 and len({e for e, _ in expected}) > 1 else float('nan'))
        agree = _mean([s['expected_benefit'] == s['measured_benefit'] for s in suggestions
                       if s.get('expected_benefit')])
        costs = [(r.get('usage') or {}).get('cost') for r in ok]
        print(f"{model_id[:44]:44s} {len(ok):>3d}/{len(rows):<2d} {_mean([r['latency_s'] for r in ok]):>7.1f}s "
              f"{_mean(costs):>8.4f} {_mean([len(r['suggestions']) for r in ok]):>6.1f} "
              f"{_mean([r['suggested_in_top'] / max(len(r['suggestions']), 1) for r in ok]):>8.0%} "
              f"{_mean([r['first_in_top'] for r in ok if r['suggestions']]):>8.0%} "
              f"{sum(r['toward_on_top_pair'] for r in ok):>7d} "
              f"{_mean([r['combined_impact'].get('separation_gain') for r in ok]):>+9.3f} "
              f"{_mean([s['impact'].get('separation_gain') for s in suggestions]):>+9.3f} "
              f"{rho:>9.2f} {agree:>8.0%}")
        moves = Counter()
        for s in suggestions:
            moves[{'away_from_j': 'push apart', 'toward_j': 'pull together'}.get(s['direction'], 'open space')] += 1
            moves['tighten'] += bool(s['tighten_i'] or s['tighten_j'])
        profiles[model_id] = (moves, len(suggestions), [r['strategy'] for r in ok if r.get('strategy')])

    print("\non top5 = share of suggestions on one of the 5 most confused pairs; closer! = 'move closer' on such a "
          "pair (harmful);\nsep gain = silhouette gain of all suggestions together on the real latent features, "
          "per sugg = of each alone;\ncalib = the model's expected_benefit vs the measured gain (Spearman rho, and "
          "share of equal levels).")

    print("\nStrategy profile (share of suggestions):")
    for model_id, (moves, total, _) in profiles.items():
        cells = ', '.join(f"{k} {moves[k] / max(total, 1):.0%}"
                          for k in ('push apart', 'pull together', 'open space', 'tighten'))
        print(f"  {model_id[:44]:44s} {cells}")

    # Do the models pick the same pairs for the same latent state?
    by_state = defaultdict(dict)
    for r in results:
        if not r.get('error'):
            by_state[(r['checkpoint'], r['draw'], r.get('prompt', 'baseline'))][r['model']] = {
                frozenset((s['class_i'], s['class_j'])) for s in r['suggestions']}
    overlap = defaultdict(list)
    for answers in by_state.values():
        for a, b in combinations(sorted(answers), 2):
            union = answers[a] | answers[b]
            overlap[(a, b)].append(len(answers[a] & answers[b]) / len(union) if union else 1.0)
    if overlap:
        print("\nAgreement on which pairs to change (mean Jaccard over the same latent states):")
        for (a, b), values in sorted(overlap.items()):
            print(f"  {a[:30]:30s} vs {b[:30]:30s} {np.mean(values):.2f} (n={len(values)})")

    if multi_prompt:
        # How much does the prompt change which pairs a model picks?
        answers = {(r['checkpoint'], r['draw'], r['model'], r.get('prompt', 'baseline')):
                   {frozenset((s['class_i'], s['class_j'])) for s in r['suggestions']}
                   for r in results if not r.get('error')}
        print("\nPrompt sensitivity: same pairs as with the baseline prompt (mean Jaccard, same model and latent state):")
        for model_id in sorted({r['model'] for r in results}):
            cells = []
            for variant in sorted({r.get('prompt', 'baseline') for r in results} - {'baseline'}):
                values = []
                for (checkpoint, draw, m, v), pairs in answers.items():
                    base = answers.get((checkpoint, draw, m, 'baseline'))
                    if m == model_id and v == variant and base is not None:
                        union = pairs | base
                        values.append(len(pairs & base) / len(union) if union else 1.0)
                if values:
                    cells.append(f"{variant} {np.mean(values):.2f}")
            print(f"  {model_id[:34]:34s} {', '.join(cells)}")

    if show_strategies:
        print("\nPlans the models proposed (global strategy):")
        for model_id, (_, _, plans) in profiles.items():
            print(f"  {model_id}:")
            for plan, count in Counter(plans).most_common():
                print(f"    {count}x {plan}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--models', nargs='*', default=list(openrouter.RECOMMENDED_MODELS))
    parser.add_argument('--prompts', nargs='*', default=['baseline'], choices=VARIANTS,
                        help='prompt variants to send (llm/prompt_variants.py); default: the app\'s own prompt')
    parser.add_argument('--checkpoint', nargs='*', default=None, metavar='DATASET:PATH',
                        help='checkpoints to use (default: real_latent_check.CHECKPOINTS that exist)')
    parser.add_argument('--draws', type=int, default=2, help='random draws per checkpoint')
    parser.add_argument('--per-class', type=int, default=10, dest='per_class')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--dry-run', action='store_true', dest='dry_run', help='no API calls')
    parser.add_argument('--summarize', default=None, metavar='RESULTS', help='only print the summary')
    parser.add_argument('--strategies', action='store_true', help='also print every proposed plan')
    args = parser.parse_args()
    if args.summarize:
        summarize(load_results(args.summarize), args.strategies)
        return
    if args.dry_run and args.out == DEFAULT_OUT:
        args.out = DEFAULT_OUT.replace('.jsonl', '_dry_run.jsonl')
    args.out = os.path.abspath(args.out)  # before run() changes directory
    run(args)
    print()
    summarize([r for r in load_results(args.out) if r['model'] in args.models
               and r.get('prompt', 'baseline') in args.prompts], args.strategies)


if __name__ == '__main__':
    main()
