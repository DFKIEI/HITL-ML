"""Turn an LLM movement/tighten suggestion into 2D vectors and factors.

Shared by the three places that need to know "what would this suggestion do
to the class centroids/spreads": the high-dim strategies' loss target in
``training/training_utils.py`` (closing the loop - the model is nudged
towards the suggested layout), the scatter-plot overlay in
``ui/ui_display.py`` (showing the operator the same thing), and the "Apply"
button / auto-apply path in ``ui/ui_llm.py`` (applying it for real). All
three consume plain ``{class_index: np.ndarray([x, y])}`` centroids so this
module has no torch or matplotlib dependency.
"""

import dataclasses
from collections import defaultdict
from typing import Dict, Iterable, Optional

import numpy as np

# How far to move, as a fraction of the configuration's reference distance
# (see _configuration_scale) - deliberately NOT a fraction of the pair's own
# current gap, since that would make an already-overlapping pair (the case
# that most needs a push) get an even smaller one.
SCALE_FRACTIONS = {'small': 0.3, 'medium': 0.6, 'large': 1.0}
DEFAULT_SCALE_FRACTION = SCALE_FRACTIONS['medium']

# How much to shrink a flagged class's spread towards its own centroid.
TIGHTEN_FACTOR = 0.5

# Ranges for the operator's custom values (set by dragging the overlay
# arrows/circles on the scatter plot). A custom distance of 0 keeps the class
# where it is (e.g. tighten only).
CUSTOM_MOVE_RANGE = (0.0, 5.0)
CUSTOM_TIGHTEN_RANGE = (0.1, 0.95)

# Dragging an arrow within this many degrees of its suggestion's own
# direction keeps that direction (only the distance changes), so a small
# wobble doesn't turn "away from dog" into a fixed custom angle.
SNAP_DEGREES = 12.0

# For a repulsive (away_from_j) suggestion, class_j also gets pushed further
# away from class_i - not just class_i backing off - scaled down from
# class_i's own push so the named class still moves the most.
MUTUAL_REPULSION_FRACTION = 0.5


def configuration_scale(centroids: Dict[int, np.ndarray]) -> float:
    """A stable notion of "typical distance" across the whole set of class
    centroids, used as the movement reference instead of one pair's own
    current gap. If two classes are already sitting on top of each other,
    scaling the "move apart" push by their own (tiny) gap would produce a
    tiny push - exactly backwards from what's needed."""
    values = list(centroids.values())
    if len(values) < 2:
        return 1.0
    mean_center = np.mean(values, axis=0)
    scale = float(np.mean([np.linalg.norm(c - mean_center) for c in values]))
    return scale if scale > 1e-8 else 1.0


def _unit_away_from_crowd(center_i: np.ndarray, centroids: Dict[int, np.ndarray]) -> np.ndarray:
    """Fallback push direction: away from the mean of all centroids, i.e.
    towards the least crowded region. Used for 'toward_empty_space' and as a
    fallback when a pair's centroids coincide (no i->j axis to repel along)."""
    mean_center = np.mean(list(centroids.values()), axis=0)
    away = center_i - mean_center
    norm = float(np.linalg.norm(away))
    return away / norm if norm > 1e-8 else np.array([1.0, 0.0])


def move_fraction(suggestion) -> float:
    """How far to move, as a fraction of the configuration scale: the
    operator's custom amount when scale is 'custom', else the preset."""
    amount = getattr(suggestion, 'move_amount', None)
    if suggestion.scale == 'custom' and amount is not None:
        return float(np.clip(amount, *CUSTOM_MOVE_RANGE))
    return SCALE_FRACTIONS.get(suggestion.scale, DEFAULT_SCALE_FRACTION)


def tighten_factor(suggestion, class_index) -> float:
    """Fraction of its current spread ``class_index`` (the suggestion's
    class_i or class_j) shrinks to when the suggestion tightens it."""
    field = 'tighten_amount_i' if class_index == suggestion.class_i else 'tighten_amount_j'
    amount = getattr(suggestion, field, None)
    if amount is not None:
        return float(np.clip(amount, *CUSTOM_TIGHTEN_RANGE))
    return TIGHTEN_FACTOR


def _unit_from_angle(angle_degrees, center_i, centroids):
    """Unit vector for an operator-chosen angle on the 2D plot. An angle
    only means something in 2D, so any other space falls back to moving
    into open space."""
    if angle_degrees is None or np.shape(center_i) != (2,):
        return _unit_away_from_crowd(center_i, centroids)
    radians = np.deg2rad(angle_degrees)
    return np.array([np.cos(radians), np.sin(radians)])


def suggestion_vectors(suggestion, centroids: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """Displacement vectors for every class this suggestion moves, keyed by
    class index. ``class_i`` gets a vector whenever the suggestion can be
    computed at all; a repulsive (``away_from_j``) suggestion also pushes
    ``class_j`` further away - real mutual separation, not just one side
    backing off - scaled down by ``MUTUAL_REPULSION_FRACTION``. Empty if it
    can't be computed (e.g. a named class has no points on screen right now)."""
    center_i = centroids.get(suggestion.class_i)
    if center_i is None:
        return {}
    fraction = move_fraction(suggestion)
    reference = configuration_scale(centroids)
    direction = suggestion.direction

    if direction in ('toward_j', 'away_from_j'):
        center_j = centroids.get(suggestion.class_j)
        if center_j is None:
            return {}
        delta = center_j - center_i
        dist = float(np.linalg.norm(delta))

        if direction == 'away_from_j':
            unit = delta / dist if dist > 1e-8 else _unit_away_from_crowd(center_i, centroids)
            return {
                suggestion.class_i: -unit * fraction * reference,
                suggestion.class_j: unit * fraction * MUTUAL_REPULSION_FRACTION * reference,
            }

        # toward_j (attract): never overshoot past class_j's own centroid.
        if dist < 1e-8:
            return {}  # already coincide - nothing to attract towards
        unit = delta / dist
        magnitude = min(fraction * reference, 0.9 * dist)
        return {suggestion.class_i: unit * magnitude}

    if direction == 'custom_angle':
        unit = _unit_from_angle(getattr(suggestion, 'move_angle', None), center_i, centroids)
        return {suggestion.class_i: unit * fraction * reference}

    # "toward_empty_space" (or anything unrecognized).
    return {suggestion.class_i: _unit_away_from_crowd(center_i, centroids) * fraction * reference}


def compute_class_movements(suggestions: Iterable, centroids: Dict[int, np.ndarray]
                             ) -> Dict[int, np.ndarray]:
    """Average, per class, the displacement vectors of every suggestion that
    moves it - as ``class_i`` directly, or as the ``class_j`` of a repulsive
    suggestion. Classes not mentioned are left out of the result (i.e. no
    suggested movement)."""
    contributions = defaultdict(list)
    for suggestion in suggestions or []:
        for class_index, vector in suggestion_vectors(suggestion, centroids).items():
            contributions[class_index].append(vector)
    return {class_index: np.mean(vectors, axis=0) for class_index, vectors in contributions.items()}


def compute_class_tighten_factors(suggestions: Iterable) -> Dict[int, float]:
    """Which classes should have their spread condensed around their own
    centroid, per the LLM's ``tighten_i``/``tighten_j`` flags - the part of a
    suggestion that a pure center-to-center movement can never express. A
    class tightened by several suggestions takes the tightest factor."""
    factors = {}
    for suggestion in suggestions or []:
        for flag, class_index in (('tighten_i', suggestion.class_i), ('tighten_j', suggestion.class_j)):
            if getattr(suggestion, flag, False):
                factor = tighten_factor(suggestion, class_index)
                factors[class_index] = min(factor, factors.get(class_index, factor))
    return factors


def fit_suggestion_to_vector(suggestion, vector, centroids: Dict[int, np.ndarray],
                             base_direction: Optional[str] = None) -> None:
    """Change ``suggestion`` (in place) so it moves class_i by ``vector`` on
    the 2D plot - what dragging its arrow tip means. Within SNAP_DEGREES of
    ``base_direction`` (the suggestion's own direction when the drag started)
    that direction is kept and only the distance becomes custom; otherwise
    it becomes a custom angle."""
    reference = configuration_scale(centroids)
    vector = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(vector))
    suggestion.scale = 'custom'
    if length < 1e-8:
        suggestion.move_amount = 0.0
        return

    if base_direction is not None and base_direction != 'custom_angle':
        preset = dataclasses.replace(suggestion, direction=base_direction, scale='medium')
        preset_vector = suggestion_vectors(preset, centroids).get(suggestion.class_i)
        if preset_vector is not None and np.any(preset_vector):
            unit = preset_vector / np.linalg.norm(preset_vector)
            cosine = float(np.dot(vector, unit)) / length
            if cosine >= np.cos(np.deg2rad(SNAP_DEGREES)):
                suggestion.direction = base_direction
                suggestion.move_amount = float(np.clip(length * cosine / reference, *CUSTOM_MOVE_RANGE))
                return

    suggestion.direction = 'custom_angle'
    suggestion.move_angle = float(np.degrees(np.arctan2(vector[1], vector[0])) % 360)
    suggestion.move_amount = float(np.clip(length / reference, *CUSTOM_MOVE_RANGE))
