"""Rebuild a user-study session from its event log.

Reads ``user_study_logs/<id>_<scenario>/session_events_id_<id>_scenario_<scenario>.jsonl``
(written by ``custom_logging.SessionRecorder``) and replays it:

- every latent space shown (``SPACE_SHOWN``, arrays in ``spaces/round_NNN.npz``),
- every modification of the 2D plot - center drags, point drags, class
  resets, applied LLM suggestions - replayed from its own parameters on top
  of the space it was made on, and checked against the 2D state logged
  after it (``state_after``); undo restores the logged state,
- the life of every LLM suggestion (shown, viewed, edited by fields / text /
  voice / plot, applied, removed) and how long it was on screen,
- time per space, per phase (setup / training / paused / ...) and per
  suggestion, and the training curve (``EPOCH_END``).

    python tools/reconstruct_session.py user_study_logs/P01_A1
    python tools/reconstruct_session.py user_study_logs/P01_A1 --export out/ --plot

``--export`` writes ``timeline.csv`` (one row per event), ``suggestions.csv``
and ``round_NNN_final.npz`` (2D points/centers when the round ended);
``--plot`` also writes a before/after PNG per round. Exit code 1 if any
replayed step does not match the log.
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

TOLERANCE = 1e-6


def find_log(path):
    if os.path.isfile(path):
        return path
    matches = sorted(glob.glob(os.path.join(path, 'session_events_*.jsonl')))
    if not matches:
        sys.exit(f"No session_events_*.jsonl in {path}")
    return matches[-1]


def load_events(log_file):
    with open(log_file, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


class Space:
    """One shown latent space and the 2D state the participant made of it."""

    def __init__(self, event, log_dir):
        self.round = event['round']
        self.event = event
        arrays = np.load(os.path.join(log_dir, event['space_file']))
        self.arrays = {k: arrays[k] for k in arrays.files}
        self.labels = self.arrays['labels']
        self.classes = np.unique(self.labels)
        self.points = self.arrays['features_2d'].astype(float).copy()
        self.centers = self.arrays['centers'].astype(float).copy()
        self.modifications = []

    def position(self, class_index):
        return int(np.where(self.classes == class_index)[0][0])

    def replay(self, event):
        """Apply one MODIFICATION; returns the max deviation from the logged
        state after it (None when the step is restored rather than replayed)."""
        kind = event['kind']
        deviation = None
        if kind == 'center_drag':
            c = event['class_index']
            self.points[self.labels == c] += np.asarray(event['delta'], dtype=float)
            self.centers[self.position(c)] = np.asarray(event['new_center'], dtype=float)
        elif kind == 'point_drag':
            self.points[event['point_index']] = np.asarray(event['new_position'], dtype=float)
        elif kind == 'class_reset':
            c = event['class_index']
            self.points[self.labels == c] = np.asarray(event['center'], dtype=float)
        elif kind == 'llm_apply':
            for c, vector in event['vectors'].items():
                c = int(c)
                self.points[self.labels == c] += np.asarray(vector, dtype=float)
                self.centers[self.position(c)] += np.asarray(vector, dtype=float)
            for c, factor in event['tighten_factors'].items():
                c = int(c)
                center = self.centers[self.position(c)]
                mask = self.labels == c
                self.points[mask] = center + (self.points[mask] - center) * factor
        after = event.get('state_after') or {}
        logged_points = np.asarray(after.get('points', self.points), dtype=float)
        logged_centers = np.asarray(after.get('centers', self.centers), dtype=float)
        if kind != 'undo':
            deviation = max(float(np.abs(self.points - logged_points).max(initial=0.0)),
                            float(np.abs(self.centers - logged_centers).max(initial=0.0)))
        # Continue from the logged state so one bad step does not spoil the rest.
        self.points, self.centers = logged_points.copy(), logged_centers.copy()
        self.modifications.append((event, deviation))
        return deviation


def reconstruct(events, log_dir):
    spaces = []
    suggestions = {}
    suggestion_events = defaultdict(Counter)
    time_spent = defaultdict(lambda: defaultdict(float))
    epochs = []
    mismatches = []
    llm_rounds = []

    for event in events:
        kind = event['type']
        if kind == 'SPACE_SHOWN':
            spaces.append(Space(event, log_dir))
        elif kind == 'MODIFICATION':
            if not spaces:
                mismatches.append((event['seq'], 'modification before any space was shown'))
                continue
            deviation = spaces[-1].replay(event)
            if deviation is not None and deviation > TOLERANCE:
                mismatches.append((event['seq'], f"{event['kind']} deviates by {deviation:.3g}"))
            uid = event.get('uid')
            if uid:
                suggestion_events[uid]['applied'] += 1
        elif kind == 'SUGGESTIONS_SHOWN':
            llm_rounds.append(event)
            for s in event['suggestions']:
                suggestions[s['uid']] = dict(s, shown_round=event['round'], model=event.get('model'))
                suggestion_events[s['uid']]['shown'] += 1
        elif kind == 'SUGGESTION_ADDED':
            suggestions[event['uid']] = dict(event['suggestion'], shown_round=event['round'], model='participant')
            suggestion_events[event['uid']]['added'] += 1
        elif kind == 'SUGGESTION_EDITED':
            suggestion_events[event['uid']][f"edited_{event['via']}"] += 1
            suggestions.setdefault(event['uid'], {}).update(final=event['suggestion'])
        elif kind == 'SUGGESTION_VIEWED':
            suggestion_events[event['uid']]['viewed'] += 1
        elif kind == 'SUGGESTION_REMOVED':
            suggestion_events[event['uid']]['removed'] += 1
        elif kind == 'TIME_SEGMENT':
            time_spent[event['kind']][event['key']] += event['duration_s']
        elif kind == 'EPOCH_END':
            epochs.append(event)
    return {
        'spaces': spaces, 'suggestions': suggestions, 'suggestion_events': suggestion_events,
        'time_spent': time_spent, 'epochs': epochs, 'mismatches': mismatches, 'llm_rounds': llm_rounds,
    }


def _fmt(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60:d}:{seconds % 60:02d}"


def report(events, result):
    first, last = events[0], events[-1]
    config = next((e for e in events if e['type'] == 'SESSION_CONFIG'), {})
    print(f"Participant {first.get('probant_id')}  scenario {first.get('scenario')}  "
          f"started {first['time']}  ({config.get('dataset')}, {config.get('model')})")
    print(f"{len(events)} events, last at {_fmt(last['elapsed_s'])}"
          + ("" if last['type'] == 'SESSION_END' else "  (no SESSION_END - the app did not close normally)"))

    phases = result['time_spent'].get('phase', {})
    if phases:
        print("Time per phase: " + ", ".join(f"{k} {_fmt(v)}" for k, v in phases.items()))

    print("\nRounds (one per latent space shown):")
    space_time = result['time_spent'].get('space', {})
    for space in result['spaces']:
        kinds = Counter(e['kind'] for e, _ in space.modifications)
        llm = [r for r in result['llm_rounds'] if r['round'] == space.round]
        print(f"  round {space.round:3d}  epoch {space.event.get('epoch', 0):3d}  "
              f"on screen {_fmt(space_time.get(str(space.round), 0.0))}  "
              f"modifications: {dict(kinds) or '-'}"
              + (f"  LLM answers: {len(llm)}" if llm else ""))

    if result['suggestions']:
        print("\nSuggestions:")
        on_screen = result['time_spent'].get('suggestion', {})
        for uid, s in result['suggestions'].items():
            counts = result['suggestion_events'][uid]
            final = s.get('final', s)
            print(f"  {uid:>5} r{s.get('shown_round', '?'):<3} {s.get('model') or '':<28} "
                  f"{final.get('class_i_name', '?')} {final.get('direction', '?')} {final.get('class_j_name', '?')} "
                  f"({final.get('scale', '?')})  viewed {_fmt(on_screen.get(uid, 0.0))}  "
                  f"{', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}")

    if result['epochs']:
        e = result['epochs'][-1]
        print(f"\nTraining: {len(result['epochs'])} epochs, last val acc {e.get('val_accuracy', 0):.2f}% "
              f"test acc {e.get('test_accuracy', 0):.2f}%")

    replayed = sum(1 for s in result['spaces'] for _, d in s.modifications if d is not None)
    if result['mismatches']:
        print(f"\nReplay: {len(result['mismatches'])} of {replayed} steps do NOT match the log:")
        for seq, message in result['mismatches'][:20]:
            print(f"  event {seq}: {message}")
    else:
        print(f"\nReplay: all {replayed} replayed modifications match the logged state.")


def export(events, result, out_dir, plot):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'timeline.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['seq', 'time', 'elapsed_s', 'round', 'type', 'detail'])
        for e in events:
            detail = {k: v for k, v in e.items()
                      if k not in ('seq', 'time', 'elapsed_s', 'round', 'type', 'state_after', 'state', 'raw')}
            writer.writerow([e['seq'], e['time'], e['elapsed_s'], e['round'], e['type'], json.dumps(detail)])

    with open(os.path.join(out_dir, 'suggestions.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['uid', 'round', 'model', 'class_i', 'class_j', 'direction', 'scale', 'expected_benefit',
                         'latent_separation_gain', 'plot_separation_gain', 'seconds_on_screen', 'events'])
        on_screen = result['time_spent'].get('suggestion', {})
        for uid, s in result['suggestions'].items():
            impact = s.get('impact') or {}
            writer.writerow([uid, s.get('shown_round'), s.get('model'), s.get('class_i_name'), s.get('class_j_name'),
                             s.get('direction'), s.get('scale'), s.get('expected_benefit'),
                             (impact.get('latent') or {}).get('separation_gain'),
                             (impact.get('plot') or {}).get('separation_gain'),
                             round(on_screen.get(uid, 0.0), 3), json.dumps(result['suggestion_events'][uid])])

    for space in result['spaces']:
        np.savez_compressed(os.path.join(out_dir, f'round_{space.round:03d}_final.npz'),
                            points=space.points, centers=space.centers, labels=space.labels)
        if plot and space.modifications:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
            for ax, points, title in ((axes[0], space.arrays['features_2d'], 'shown'),
                                      (axes[1], space.points, 'after the participant')):
                ax.scatter(points[:, 0], points[:, 1], c=space.labels, cmap='tab10', s=18)
                ax.set_title(f"round {space.round} - {title}")
            fig.savefig(os.path.join(out_dir, f'round_{space.round:03d}.png'), dpi=110, bbox_inches='tight')
            plt.close(fig)
    print(f"\nExported to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('path', help='session log directory or session_events_*.jsonl file')
    parser.add_argument('--export', default=None, help='write timeline/suggestion CSVs and per-round states here')
    parser.add_argument('--plot', action='store_true', help='with --export: also a PNG per round')
    args = parser.parse_args()

    log_file = find_log(args.path)
    events = load_events(log_file)
    if not events:
        sys.exit(f"{log_file} is empty")
    result = reconstruct(events, os.path.dirname(os.path.abspath(log_file)))
    report(events, result)
    if args.export:
        export(events, result, args.export, args.plot)
    sys.exit(1 if result['mismatches'] else 0)


if __name__ == '__main__':
    main()
