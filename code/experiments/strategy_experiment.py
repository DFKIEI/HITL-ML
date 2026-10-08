"""Do the LLM's suggestions make the classifier better? Headless training runs
that follow the app's protocol (main.py / ui/ui.py defaults: from scratch,
Adam lr 1e-4, batch 512, alpha 0.5, a pause every 5 epochs) for the
conditions that need no human:

- ce_only:      cross-entropy only (alpha 0) - the plain baseline;
- no_moves:     strategy 1 with nobody dragging - the interaction loss only
                holds the current 2D layout, so it separates "the extra loss
                term" from "what the LLM asks for";
- llm_2d:       strategy 3 - at every pause the LLM is asked and all its
                suggestions are applied to the 2D plot, as ui/ui_llm.py does;
- llm_high_dim: strategy 2 - the suggestions move the real latent space.

Strategies 1 and 4 with real drags need a person and are not covered.
Calls OpenRouter for the two llm_* conditions (one request per pause).

    python experiments/strategy_experiment.py --out experiments/results/strategies_CIFAR10_60ep.jsonl
    python experiments/strategy_experiment.py --conditions ce_only llm_2d --epochs 10
    python experiments/strategy_experiment.py --summarize experiments/results/strategies_CIFAR10_60ep.jsonl

With --out, runs already in that file are skipped, so an interrupted run can
simply be started again with the same command.
"""

import argparse
import json
import os
import random
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib

matplotlib.use('Agg')

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.optim as optim  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import load_dataset  # noqa: E402
from llm import openrouter  # noqa: E402
from llm.latent_state import get_class_names  # noqa: E402
from llm.suggestions import ask_for_suggestions  # noqa: E402
from model import get_model  # noqa: E402
from plots.plots import InteractivePlot  # noqa: E402
from training.training import train_model  # noqa: E402
from ui.ui_display import apply_llm_suggestion  # noqa: E402

CONDITIONS = {
    'ce_only': dict(strategy=1, alpha=0.0, llm=False),
    'no_moves': dict(strategy=1, alpha=0.5, llm=False),
    'llm_2d': dict(strategy=3, alpha=0.5, llm=True),
    'llm_high_dim': dict(strategy=2, alpha=0.5, llm=True),
}


class Value:
    """Stands in for the tk variables train_model reads."""

    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def headless_ui(plot):
    """The parts of the UI apply_llm_suggestion touches, rebuilt after every
    refresh like display_scatter_plot does (moved_points reset to the fresh
    features; plot.moved_2d_points is left alone, as in the app)."""
    features = np.asarray(plot.selected_features, dtype=float)
    labels = np.asarray(plot.selected_labels)
    unique = np.unique(labels)
    centers = np.array([features[labels == c].mean(axis=0) for c in unique])
    ax = MagicMock(**{'get_xlim.return_value': (features[:, 0].min(), features[:, 0].max()),
                      'get_ylim.return_value': (features[:, 1].min(), features[:, 1].max())})
    return SimpleNamespace(
        data={'centers': centers, 'labels': labels}, unique_labels=unique, ax=ax,
        moved_points=features.copy(), center_artists=[MagicMock() for _ in unique], plot=plot,
        point_tracker=MagicMock(), scatter=MagicMock(), scatter_fig=MagicMock(),
        incorrect_mask=np.asarray(plot.selected_predicted_labels) != labels)


def run(condition, seed, args, loaders, llm_model, log):
    config = CONDITIONS[condition]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    trainloader, valloader, testloader, num_classes, input_shape = loaders
    dev = device()
    model = get_model(f'CNN_{args.dataset}', input_shape, num_classes).to(dev)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    plot = InteractivePlot(model, valloader, 'scatter', args.dataset, 10)
    plot.prepare_data()
    names = get_class_names(plot)
    state = {'suggestions': None, 'requests': []}

    def on_pause():
        plot.prepare_data()  # what update_visualization does after a pause
        if not config['llm']:
            return
        try:
            summary, suggestions, _, raw = ask_for_suggestions(
                np.asarray(plot.latent_features, dtype=float), np.asarray(plot.selected_labels),
                names, args.dataset, model=llm_model)
        except Exception as e:  # a failed request leaves the old suggestions, as in the app
            state['requests'].append({'error': f'{type(e).__name__}: {e}'})
            return
        state['suggestions'] = list(suggestions)
        applied = 0
        if CONDITIONS[condition]['strategy'] == 3:
            ui = headless_ui(plot)
            applied = sum(bool(apply_llm_suggestion(ui, s)) for s in suggestions)
        state['requests'].append({
            'issue': summary.issue, 'applied': applied,
            'suggestions': [f"{s.class_i_name} {s.direction} {s.class_j_name} {s.scale}"
                            f"{' +ti' if s.tighten_i else ''}{' +tj' if s.tighten_j else ''}"
                            for s in suggestions]})

    epochs = []
    report_dir = tempfile.mkdtemp(prefix='strategy_experiment_')
    started = time.time()
    with open(os.devnull, 'w') as devnull:
        stdout, sys.stdout = sys.stdout, devnull  # train_model prints every batch
        try:
            train_model(model, optimizer, trainloader, valloader, testloader, dev, args.epochs, 10 ** 9,
                        Value(config['alpha']), report_dir, epoch_end_callback=on_pause,
                        pause_after_n_epochs=args.pause, plot=plot, logger=MagicMock(),
                        metrics_callback=epochs.append, strategy_var=Value(config['strategy']),
                        llm_suggestions_callback=lambda: state['suggestions'])
        finally:
            sys.stdout = stdout
    result = {'condition': condition, 'seed': seed, 'dataset': args.dataset, 'epochs': epochs,
              'llm_requests': state['requests'], 'minutes': (time.time() - started) / 60}
    final = epochs[-1] if epochs else {}
    log(f"{condition:13s} seed {seed}: val {final.get('val_accuracy', float('nan')):.2f}%  "
        f"test {final.get('test_accuracy', float('nan')):.2f}%  f1 {final.get('test_f1', float('nan')):.3f}  "
        f"({result['minutes']:.1f} min)")
    for request in state['requests']:
        log(f"    LLM: {request}")
    return result


def load_results(path):
    if not path or not os.path.exists(path):
        return []
    with open(path) as results_file:
        return [json.loads(line) for line in results_file if line.strip()]


def summarize(results, conditions=None):
    conditions = conditions or [c for c in CONDITIONS if any(r['condition'] == c for r in results)]
    print("final epoch, mean ± std over seeds:")
    print(f"   {'condition':13s} {'test acc':>15s} {'test f1':>15s} {'val acc':>15s}")
    for condition in conditions:
        finals = [r['epochs'][-1] for r in results if r['condition'] == condition and r['epochs']]
        if not finals:
            continue
        cells = []
        for key, fmt in (('test_accuracy', '{:6.2f} ± {:4.2f}'), ('test_f1', '{:6.3f} ± {:5.3f}'),
                         ('val_accuracy', '{:6.2f} ± {:4.2f}')):
            values = [f[key] for f in finals]
            cells.append(fmt.format(np.mean(values), np.std(values)))
        print(f"   {condition:13s} {cells[0]:>15s} {cells[1]:>15s} {cells[2]:>15s}  (n={len(finals)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--dataset', default='CIFAR10')
    parser.add_argument('--conditions', nargs='*', default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument('--seeds', nargs='*', type=int, default=[0, 1, 2])
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--pause', type=int, default=5)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--llm-model', default=None, dest='llm_model')
    parser.add_argument('--out', default=None, help='append one JSON line per run here; runs already in it '
                                                     'are skipped')
    parser.add_argument('--summarize', default=None, metavar='RESULTS',
                        help='only print the summary table of a results file')
    args = parser.parse_args()
    if args.summarize:
        summarize(load_results(args.summarize))
        return
    llm_model = args.llm_model or openrouter.get_default_model()
    if args.out:
        args.out = os.path.abspath(args.out)  # before the chdir below

    os.chdir(PROJECT_DIR)  # the loaders read 'dataset/...' relative to the project root
    print(f"{args.dataset}, {args.epochs} epochs, pause every {args.pause}, LLM {llm_model}, device {device()}",
          flush=True)
    loaders = load_dataset(args.dataset, args.batch)
    results = load_results(args.out)
    done = {(r['condition'], r['seed'], r['dataset'], len(r['epochs'])) for r in results}
    for seed in args.seeds:  # seed-major, so partial runs still compare conditions
        for condition in args.conditions:
            if (condition, seed, args.dataset, args.epochs) in done:
                print(f"{condition:13s} seed {seed}: already in {args.out}, skipped", flush=True)
                continue
            result = run(condition, seed, args, loaders, llm_model, lambda m: print(m, flush=True))
            results.append(result)
            if args.out:
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                with open(args.out, 'a') as out:
                    out.write(json.dumps(result) + '\n')

    print()
    summarize([r for r in results if r['dataset'] == args.dataset and len(r['epochs']) == args.epochs],
              args.conditions)


if __name__ == '__main__':
    main()
