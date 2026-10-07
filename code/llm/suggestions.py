"""Prompting and parsing of the LLM's latent-space suggestions.

Ported from ``llm_integration/helpers_llm.py``: the model is shown only
categorical/semantic signals about the latent space (see
``latent_state.py``) and answers with freeform reasoning - a global
issue/strategy plus, per confused-or-close class pair, a movement intention
and why it is needed. There is nothing numeric to apply automatically; the
operator reads the suggestions and rearranges the plot themselves.
"""

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from llm import openrouter
from llm.latent_state import (
    build_global_summary_semantic,
    build_pair_summary_semantic,
    compute_centroids,
    compute_thresholds,
    get_class_names,
    pairwise_distances,
)

MAX_SUGGESTIONS = 5
MAX_PAIRS = 100

PROMPT_HEADER = """\
You advise the human operator of a human-in-the-loop training tool.

How the tool works: a neural network embeds every sample into a latent space.
The operator sees a 2D visualization of it as a scatter plot with one cluster
per class and can drag class clusters around to guide training - moving
confused classes apart, or tightening a diffuse one. The signals below come
from the model's actual latent representation, not that 2D visualization, and
you are not given exact coordinates, only how the classes relate to each other.

We provide only 5-level categorical latent signals (no numeric values):
Global semantic metrics:
- overall_overlap_level: very_low | low | medium | high | very_high
- overall_spread_level: very_compact | compact | medium | spread | very_spread
- spread_imbalance_level: very_low | low | medium | high | very_high
- outlier_burden_level: very_low | low | medium | high | very_high
- separation_health: very_poor | poor | medium | good | very_good
Pairwise semantic metrics:
- distance_relation: very_close | close | medium | far | very_far
- overlap_level: very_low | low | medium | high | very_high
  (absolute: under 10% / 10-20% / 20-35% / 35-50% / over 50% of the two
  classes' points sit inside each other's area)
- spread_i_level, spread_j_level: very_compact | compact | medium | spread | very_spread
- outlier_i_level, outlier_j_level: very_low | low | medium | high | very_high
"""

DIRECTIONS = ('toward_j', 'away_from_j', 'toward_empty_space')
SCALES = ('small', 'medium', 'large')

PROMPT_FOOTER = """\
Output MUST be strict JSON with exactly this shape and no extra keys:
{ "global": { "issue": "string", "strategy": "string" },
  "suggestions": [ { "class_i": int, "class_j": int, "suggestion": "string",
                     "direction": "toward_j" | "away_from_j" | "toward_empty_space",
                     "scale": "small" | "medium" | "large",
                     "tighten_i": true | false, "tighten_j": true | false } ] }
Write every text field for a non-technical reader: plain everyday words, short
sentences, and the class NAMES (never indices, metric names, level names such as
"very_high", or words like latent, centroid, embedding, vector or cluster
metrics). Think "the cat and dog groups overlap a lot, so push them apart".
The "global.issue" should summarize the main problem in one short sentence.
The "global.strategy" should give the overall plan in one short sentence.
The "suggestion" field must be one or two short sentences that say both:
1) what to do with this pair, naming both classes (e.g. "Move cat away from dog",
   "Pull the truck group tighter together"), and
2) why, in plain words (e.g. "they are easily confused and sit on top of each other").
Order suggestions from most to least important: the first one should be the
change that would help the classifier most.
The "direction" field states which way class_i should move: "toward_j" (towards
class_j), "away_from_j" (away from class_j), or "toward_empty_space" when
neither applies - e.g. class_i's problem is not localized to class_j, or it is
an outlier cluster that should simply move towards empty, unoccupied latent
space rather than towards or away from any one class.
The "scale" field states how large the movement should be: "small", "medium", or "large".
"tighten_i" / "tighten_j" are independent of direction/scale: set "tighten_i"
to true when class_i's own cluster is too spread out / diffuse and should be
condensed around its own center (and likewise "tighten_j" for class_j) -
overlap is often caused by spread, not just distance, and moving centers apart
alone will not fix that.
Only suggest a change for a real problem: a pair whose overlap_level is medium
or higher, or a class that is spread / very_spread AND overlaps another class.
distance_relation is relative to the other pairs, so the nearest pairs are
always called "close" even in a healthy space - a close pair with very_low or
low overlap is fine and needs no suggestion. If nothing has a real problem,
return an empty "suggestions" list and say in "global" that the layout looks
healthy. Fewer, correct suggestions are better than filling the list.
The text and the fields must describe the same move, because the operator may
edit the text and it is then read back into the fields: set "tighten_i" (or
"tighten_j") to true exactly when the "suggestion" text says, by name, to
tighten / pull together that class; match "scale" to the wording ("a little"
= small, "a lot" = large); and name class_i first, as the class that moves.
Use class_i and class_j as the integer class indices given in the data below.
Do NOT provide algorithmic or implementation instructions. Keep suggestions
focused on semantic movement in latent space. No text outside the JSON object.
"""


@dataclass
class GlobalSummary:
    issue: str = ''
    strategy: str = ''

    def is_empty(self):
        return not self.issue and not self.strategy

    def as_log_dict(self):
        return {'issue': self.issue, 'strategy': self.strategy}


@dataclass
class PairSuggestion:
    class_i: int
    class_j: int
    class_i_name: str
    class_j_name: str
    suggestion: str
    direction: str = 'toward_empty_space'
    scale: str = 'medium'
    tighten_i: bool = False
    tighten_j: bool = False
    source: str = 'llm'  # 'llm', or 'human' for one the operator added
    edited: bool = False  # the operator changed it after it arrived
    # Operator-set geometry from dragging the overlay arrow/circle on the
    # scatter plot, used instead of the categorical presets above (see
    # llm/movement.py): move_amount when scale == 'custom' (fraction of the
    # typical class-to-class distance), move_angle when direction ==
    # 'custom_angle' (degrees on the 2D plot, 0 = right, 90 = up), and
    # tighten_amount_i / tighten_amount_j for tighten_i / tighten_j
    # (fraction of the current size to shrink to). None = use the preset.
    move_amount: Optional[float] = None
    move_angle: Optional[float] = None
    tighten_amount_i: Optional[float] = None
    tighten_amount_j: Optional[float] = None

    def set_classes(self, class_i, class_j, class_names):
        self.class_i, self.class_j = int(class_i), int(class_j)
        self.class_i_name = class_names.get(self.class_i, f"class_{self.class_i}")
        self.class_j_name = class_names.get(self.class_j, f"class_{self.class_j}")

    def as_log_dict(self):
        return {
            'class_i': self.class_i,
            'class_j': self.class_j,
            'class_i_name': self.class_i_name,
            'class_j_name': self.class_j_name,
            'suggestion': self.suggestion,
            'direction': self.direction,
            'scale': self.scale,
            'tighten_i': self.tighten_i,
            'tighten_j': self.tighten_j,
            'source': self.source,
            'edited': self.edited,
            'move_amount': self.move_amount,
            'move_angle': self.move_angle,
            'tighten_amount_i': self.tighten_amount_i,
            'tighten_amount_j': self.tighten_amount_j,
        }


# Plain-language wording for the categorical fields, used by the suggestion
# cards (dropdown values) and by describe_movement. 'custom_angle' and
# 'custom' are operator-only (the LLM is never offered them, see DIRECTIONS
# and SCALES) and switch the card to its sliders.
DIRECTION_LABELS = {
    'away_from_j': 'away from',
    'toward_j': 'closer to',
    'toward_empty_space': 'into open space',
    'custom_angle': 'in a custom direction',
}
SCALE_LABELS = {'small': 'a little', 'medium': 'moderately', 'large': 'a lot', 'custom': 'custom amount'}

COMPASS_WORDS = ('right', 'up and right', 'up', 'up and left', 'left', 'down and left', 'down', 'down and right')


def compass_word(angle_degrees):
    """Nearest of 8 plain directions on the plot (0 = right, 90 = up)."""
    return COMPASS_WORDS[int(round((angle_degrees % 360) / 45.0)) % 8]


def describe_movement(suggestion):
    """One plain sentence for what a suggestion will do on the plot, e.g.
    "Move cat a lot away from dog. Also pull cat tighter together." """
    if suggestion.scale == 'custom' and suggestion.move_amount is not None:
        how_much = f"by a custom amount ({suggestion.move_amount:.0%})"
    else:
        how_much = SCALE_LABELS.get(suggestion.scale, suggestion.scale)
    if suggestion.direction == 'toward_j':
        text = f"Move {suggestion.class_i_name} {how_much} closer to {suggestion.class_j_name}."
    elif suggestion.direction == 'away_from_j':
        text = f"Move {suggestion.class_i_name} {how_much} away from {suggestion.class_j_name}."
    elif suggestion.direction == 'custom_angle' and suggestion.move_angle is not None:
        text = (f"Move {suggestion.class_i_name} {how_much} {compass_word(suggestion.move_angle)} "
                f"({suggestion.move_angle:.0f}°).")
    else:
        text = f"Move {suggestion.class_i_name} {how_much} into open space."

    if suggestion.scale == 'custom' and suggestion.move_amount == 0:
        text = f"Keep {suggestion.class_i_name} where it is."

    tighten_names, shrink = [], []
    for flag, name, amount in ((suggestion.tighten_i, suggestion.class_i_name, suggestion.tighten_amount_i),
                               (suggestion.tighten_j, suggestion.class_j_name, suggestion.tighten_amount_j)):
        if flag and amount is not None:
            shrink.append(f"{name} to {amount:.0%}")
        elif flag:
            tighten_names.append(name)
    if tighten_names:
        text += f" Also pull {' and '.join(tighten_names)} tighter together."
    if shrink:
        text += f" Shrink {' and '.join(shrink)} of its current size."
    return text


def request_suggestions(ui, model=None, user_goal=None):
    """Build the semantic state, ask the model, return
    (global_summary, suggestions, state, raw_content).

    The categorical signals come from the model's actual latent representation
    (``plot.latent_features``, pre-projection - the same features the classifier
    head sees), not the 2D scatter-plot positions. The 2D projection is a lossy
    visualization for the human; the LLM reasons about the real geometry, same
    as the offline pipeline in ``llm_integration/``."""
    return ask_for_suggestions(
        np.asarray(ui.plot.latent_features, dtype=float), np.asarray(ui.plot.selected_labels),
        get_class_names(ui.plot), getattr(ui.plot, 'dataset_name', 'the current'),
        model=model, user_goal=user_goal)


def ask_for_suggestions(points, labels, class_names, dataset_name, model=None, user_goal=None):
    """``request_suggestions`` without the UI: also used by
    ``tests/llm_eval.py`` to check the answers on known latent spaces."""
    state, class_indices = build_semantic_state(points, labels, class_names)
    messages = [
        {'role': 'system', 'content': 'Return only JSON that matches the requested schema.'},
        {'role': 'user', 'content': build_prompt(state, dataset_name, user_goal)},
    ]
    content, _ = openrouter.chat_completion(messages, model=model)

    global_summary, suggestions = parse_suggestions(content, class_names, set(class_indices))
    return global_summary, suggestions, state, content


def build_semantic_state(points, labels, class_names):
    """The categorical description of the latent space sent to the model,
    plus the class indices present."""
    class_indices = sorted(int(c) for c in np.unique(labels))
    centroids = compute_centroids(points, labels, class_indices)
    distances = pairwise_distances(centroids)
    thresholds = compute_thresholds(distances)

    pairs_summary = build_pair_summary_semantic(
        points, labels, centroids, distances, *thresholds,
        max_pairs=MAX_PAIRS, class_names=class_names,
    )
    global_metrics = build_global_summary_semantic(
        points, labels, centroids, distances, *thresholds,
    )
    return {'global_metrics': global_metrics, 'pairs': pairs_summary}, class_indices


def build_prompt(state, dataset_name, user_goal=None):
    parts = [
        PROMPT_HEADER,
        f'Return only the {MAX_SUGGESTIONS} most important suggestions (or fewer) to improve the '
        f'latent space of this {dataset_name} classifier.\n',
        PROMPT_FOOTER,
    ]
    if user_goal:
        parts.append(f"\nAdditional operator focus for this round: {user_goal}\n")
    parts.append(f"\nData (JSON):\n{json.dumps(state, indent=2)}\n")
    return ''.join(parts)


def parse_suggestions(content: str, class_names: Dict[int, str],
                       known_indices: Set[int]) -> Tuple[GlobalSummary, List[PairSuggestion]]:
    """Parse the model answer. Suggestions with unknown/duplicate classes are dropped."""
    payload = _extract_json(content)

    global_raw = payload.get('global') or {}
    if not isinstance(global_raw, dict):
        global_raw = {}
    global_summary = GlobalSummary(
        issue=str(global_raw.get('issue') or '').strip(),
        strategy=str(global_raw.get('strategy') or '').strip(),
    )

    raw_suggestions = payload.get('suggestions', payload.get('Suggestions', []))
    if isinstance(raw_suggestions, dict):
        raw_suggestions = [raw_suggestions]
    if not isinstance(raw_suggestions, list):
        raw_suggestions = []

    suggestions = []
    for raw in raw_suggestions[:MAX_SUGGESTIONS]:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get('suggestion') or '').strip()
        if not text:
            continue
        try:
            class_i = int(raw.get('class_i'))
            class_j = int(raw.get('class_j'))
        except (TypeError, ValueError):
            continue
        if class_i == class_j or class_i not in known_indices or class_j not in known_indices:
            continue

        direction = str(raw.get('direction') or '').strip()
        if direction not in DIRECTIONS:
            direction = 'toward_empty_space'
        scale = str(raw.get('scale') or '').strip()
        if scale not in SCALES:
            scale = 'medium'
        tighten_i = bool(raw.get('tighten_i'))
        tighten_j = bool(raw.get('tighten_j'))

        suggestions.append(PairSuggestion(
            class_i=class_i,
            class_j=class_j,
            class_i_name=class_names.get(class_i, f"class_{class_i}"),
            class_j_name=class_names.get(class_j, f"class_{class_j}"),
            suggestion=text,
            direction=direction,
            scale=scale,
            tighten_i=tighten_i,
            tighten_j=tighten_j,
        ))

    if not suggestions and global_summary.is_empty():
        raise ValueError("No usable content in the response.")
    return global_summary, suggestions


def _extract_json(content):
    text = content.strip()
    fenced = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find('{')
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find('{', start + 1)
    raise ValueError(f"Could not read JSON from the response: {content[:200]}")
