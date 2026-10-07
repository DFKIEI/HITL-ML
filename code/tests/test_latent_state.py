"""The categorical signals sent to the LLM (llm/latent_state.py)."""

import numpy as np

from llm.suggestions import build_semantic_state

NAMES = {i: f"c{i}" for i in range(6)}


def space(overlapping_pair=True, seed=0, dim=16, per_class=40):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1, (6, dim))
    centers = 10 * centers / np.linalg.norm(centers, axis=1, keepdims=True)
    if overlapping_pair:
        centers[1] = centers[0]
    points = np.concatenate([c + rng.normal(0, 0.3, (per_class, dim)) for c in centers])
    return points, np.repeat(np.arange(6), per_class)


def overlap_levels(points, labels):
    state, _ = build_semantic_state(points, labels, NAMES)
    return {(p['class_i'], p['class_j']): p['overlap_level'] for p in state['pairs']}


def test_only_the_overlapping_pair_is_reported_high():
    # Most pairs share no points; they must be very_low, not the top level.
    levels = overlap_levels(*space())
    assert levels.pop((0, 1)) == 'very_high'
    assert set(levels.values()) == {'very_low'}


def test_no_overlap_anywhere_is_very_low():
    assert set(overlap_levels(*space(overlapping_pair=False)).values()) == {'very_low'}


def test_overlap_levels_are_absolute():
    # A heavy and a mild overlap must not share a level just because they
    # are the only overlapping pairs.
    rng = np.random.default_rng(0)
    dim, per_class = 32, 60
    centers = rng.normal(0, 1, (6, dim))
    centers = 10 * centers / np.linalg.norm(centers, axis=1, keepdims=True)
    centers[1] = centers[0] + 0.2 * rng.normal(0, 1, dim) / np.sqrt(dim)
    centers[3] = centers[2] + 0.08 * (centers[3] - centers[2])
    points = np.concatenate([c + rng.normal(0, 2.0 / np.sqrt(dim), (per_class, dim)) for c in centers])
    levels = overlap_levels(points, np.repeat(np.arange(6), per_class))
    assert levels[(0, 1)] == 'very_high'
    assert levels[(2, 3)] in ('medium', 'high')


def class_levels(points, labels):
    state, _ = build_semantic_state(points, labels, NAMES)
    levels = {}
    for p in state['pairs']:
        levels[p['class_i']] = (p['spread_i_level'], p['outlier_i_level'])
        levels[p['class_j']] = (p['spread_j_level'], p['outlier_j_level'])
    return state['global_metrics'], levels


def test_equal_spreads_are_all_medium():
    # Quintile levels used to call two of six identical classes very_spread,
    # and the model then "fixed" a healthy space.
    _, levels = class_levels(*space(overlapping_pair=False))
    assert {spread for spread, _ in levels.values()} == {'medium'}


def test_diffuse_class_is_spread():
    points, labels = space(overlapping_pair=False)
    mask = labels == 2
    points[mask] = points[mask].mean(axis=0) + 4 * (points[mask] - points[mask].mean(axis=0))
    _, levels = class_levels(points, labels)
    assert levels[2][0] == 'very_spread'
    assert all(levels[c][0] == 'medium' for c in levels if c != 2)


def test_global_levels_follow_the_worst_pair():
    overlapping, _ = class_levels(*space())
    healthy, _ = class_levels(*space(overlapping_pair=False))
    assert overlapping['separation_health'] == 'very_poor'
    assert overlapping['overall_overlap_level'] != 'very_low'
    assert healthy['separation_health'] == 'very_good'
    assert healthy['overall_overlap_level'] == 'very_low'
