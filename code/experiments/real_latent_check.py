"""Do the LLM's suggestions point at the classes a real model confuses?

Loads trained checkpoints, takes the model's real latent features on the
validation split, and compares three things per checkpoint:

- the confusion matrix: which class pairs the classifier actually mixes up
  (ground truth for "which pair needs fixing");
- the prompt's own signal: which pairs build_semantic_state reports with
  overlap_level medium or higher (is the data we send the LLM right?);
- the LLM's suggestions for that prompt (does the model pick the right pairs,
  and never pull a confused pair closer?).

The state is built twice: from 10 random samples per class, as the app does
(plots/plots.py), and from 100 per class. Calls OpenRouter.

    python experiments/real_latent_check.py
    python experiments/real_latent_check.py --draws 3 --out results/real_latent.jsonl
"""

import argparse
import json
import os
import sys
from itertools import combinations

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import load_dataset  # noqa: E402
from llm import openrouter  # noqa: E402
from llm.suggestions import ask_for_suggestions, build_semantic_state  # noqa: E402
from model import get_model  # noqa: E402
from ui.ui_display import create_sequential_mapping, get_label_names  # noqa: E402

CHECKPOINTS = [
    ('CIFAR10', 'saved_checkpoints/CIFAR10/checkpoint_epoch_20.pt'),
    ('CIFAR10', 'saved_checkpoints/CIFAR10/checkpoint_epoch_80.pt'),
    ('PAMAP2', 'saved_checkpoints/PAMAP2/checkpoint_epoch_5.pt'),
    ('PAMAP2', 'saved_checkpoints/PAMAP2/checkpoint_epoch_40.pt'),
]
TOP_K = 5


def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


@torch.no_grad()
def latent_features(model, loader, dev):
    model.eval()
    features, labels, predictions = [], [], []
    for inputs, targets in loader:
        outputs, _, latent = model(inputs.to(dev))
        features.append(latent.reshape(latent.size(0), -1).cpu().numpy())
        labels.append(np.asarray(targets).ravel())
        predictions.append(outputs.argmax(dim=1).cpu().numpy())
    return np.concatenate(features), np.concatenate(labels), np.concatenate(predictions)


def confused_pairs(labels, predictions, classes):
    """Pairs ranked by how often the classifier mixes them up, in either
    direction, as a share of both classes' samples."""
    index = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)))
    for t, p in zip(labels, predictions):
        if p in index:
            matrix[index[t], index[p]] += 1
    counts = matrix.sum(axis=1)
    rates = {}
    for a, b in combinations(classes, 2):
        i, j = index[a], index[b]
        rates[frozenset((a, b))] = (matrix[i, j] + matrix[j, i]) / max(counts[i] + counts[j], 1)
    return sorted(rates.items(), key=lambda kv: -kv[1])


def balanced_sample(labels, per_class, rng):
    chosen = []
    for c in np.unique(labels):
        indices = np.where(labels == c)[0]
        chosen.extend(rng.choice(indices, min(per_class, len(indices)), replace=False))
    return np.sort(np.array(chosen))


def check(features, labels, ranking, names, dataset, per_class, rng, model_id):
    sample = balanced_sample(labels, per_class, rng)
    points, sample_labels = features[sample], labels[sample]
    top = {pair for pair, _ in ranking[:TOP_K]}
    rank_of = {pair: r for r, (pair, _) in enumerate(ranking)}

    state, _ = build_semantic_state(points, sample_labels, names)
    flagged = [frozenset((p['class_i'], p['class_j'])) for p in state['pairs']
               if p['overlap_level'] in ('medium', 'high', 'very_high')]

    summary, suggestions, _, raw = ask_for_suggestions(points, sample_labels, names, dataset, model=model_id)
    suggested = [frozenset((s.class_i, s.class_j)) for s in suggestions]
    pair_suggestions = [(p, s) for p, s in zip(suggested, suggestions) if len(p) == 2]

    def pair_name(pair):
        return '/'.join(names.get(c, str(c)) for c in sorted(pair))

    return {
        'per_class': per_class,
        'prompt_flagged': [pair_name(p) for p in flagged],
        'prompt_flagged_in_top': sum(p in top for p in flagged),
        'suggestions': [
            {'pair': pair_name(p), 'confusion_rank': rank_of.get(p, -1) + 1, 'direction': s.direction,
             'scale': s.scale, 'tighten': [names.get(c) for c, f in ((s.class_i, s.tighten_i),
                                                                      (s.class_j, s.tighten_j)) if f],
             'text': s.suggestion}
            for p, s in pair_suggestions],
        'suggested_in_top': sum(p in top for p, _ in pair_suggestions),
        'first_in_top': bool(pair_suggestions) and pair_suggestions[0][0] in top,
        'toward_on_top_pair': sum(s.direction == 'toward_j' and p in top for p, s in pair_suggestions),
        'issue': summary.issue,
        'raw': raw,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model', default=None, help='OpenRouter model id (default: llm_config.txt)')
    parser.add_argument('--draws', type=int, default=2, help='random 10-per-class draws per checkpoint')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=None, help='write every result as JSON lines here')
    args = parser.parse_args()
    model_id = args.model or openrouter.get_default_model()
    dev = device()
    rng = np.random.default_rng(args.seed)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out = open(os.path.abspath(args.out), 'w') if args.out else None  # before the chdir below
    print(f"LLM: {model_id}\n")

    os.chdir(PROJECT_DIR)  # the loaders read 'dataset/...' relative to the project root
    loaded = {}
    for dataset, checkpoint_path in CHECKPOINTS:
        if dataset not in loaded:
            loaded[dataset] = load_dataset(dataset, 512)
        _, valloader, _, num_classes, input_shape = loaded[dataset]
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model = get_model(f'CNN_{dataset}', input_shape, num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(dev)

        features, labels, predictions = latent_features(model, valloader, dev)
        classes = sorted(int(c) for c in np.unique(labels))
        label_names = get_label_names(valloader.dataset)
        sequential = create_sequential_mapping(label_names) if label_names else {}
        names = {c: sequential.get(c, f'class_{c}') for c in classes}
        ranking = confused_pairs(labels, predictions, classes)
        accuracy = float((labels == predictions).mean())

        print(f"=== {checkpoint_path}  (val accuracy {accuracy:.1%}, {features.shape[1]}-dim latent)")
        print("most confused pairs: " + ", ".join(
            f"{'/'.join(names[c] for c in sorted(p))} {rate:.1%}" for p, rate in ranking[:TOP_K]))
        for per_class in [10] * args.draws + [100]:
            try:
                result = check(features, labels, ranking, names, dataset, per_class, rng, model_id)
            except Exception as e:  # report and keep going
                print(f"  [{per_class}/class] ERROR {type(e).__name__}: {e}")
                continue
            print(f"  [{per_class}/class] prompt flags {len(result['prompt_flagged'])} pair(s), "
                  f"{result['prompt_flagged_in_top']} in the top {TOP_K} confused: {result['prompt_flagged']}")
            print(f"     LLM: {len(result['suggestions'])} suggestion(s), {result['suggested_in_top']} on a "
                  f"top-{TOP_K} confused pair, first one in top {TOP_K}: {result['first_in_top']}, "
                  f"'toward' on a confused pair: {result['toward_on_top_pair']}")
            for s in result['suggestions']:
                rank = f"#{s['confusion_rank']}" if s['confusion_rank'] > 0 else 'unranked'
                tighten = f" +tighten {','.join(s['tighten'])}" if s['tighten'] else ''
                print(f"       - {s['pair']} (confusion {rank}) {s['direction']} {s['scale']}{tighten}")
            if out:
                out.write(json.dumps({'checkpoint': checkpoint_path, 'accuracy': accuracy,
                                      'top_confused': [['/'.join(names[c] for c in sorted(p)), rate]
                                                       for p, rate in ranking[:10]],
                                      **result}) + '\n')
        print()
    if out:
        out.close()


if __name__ == '__main__':
    main()
