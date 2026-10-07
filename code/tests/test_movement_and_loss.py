"""Suggestion -> movement on the 2D plot -> interaction loss target.

Covers the chain the operator relies on in strategy 4: the text is read into
fields, Apply moves the 2D points (ui_display.apply_llm_suggestion), the
training loop builds its target from the moved points
(training_utils.compute_ideal_structure) and relative_distance_loss pulls
the real features that way. Strategy 2's high-dim target is checked too."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from llm.movement import (CUSTOM_MOVE_RANGE, SCALE_FRACTIONS, TIGHTEN_FACTOR, compute_class_tighten_factors,
                          suggestion_vectors)
from llm.suggestions import PairSuggestion, describe_movement
from llm.text_to_movement import interpret_text
from training.losses import relative_distance_loss
from training.training_utils import compute_ideal_structure, compute_llm_ideal_structure
from ui.ui_display import apply_llm_suggestion

NAMES = {0: 'cat', 1: 'dog', 2: 'ship'}
PER_CLASS = 50


def make_points(dim=2, seed=0):
    """Cat and dog overlapping near the origin, ship far away. Sorted by
    class with equal counts, the layout compute_ideal_structure expects."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((3, dim))
    centers[1, 0] = 0.5
    centers[2, 0], centers[2, 1 % dim] = 6.0, 4.0
    spreads = (1.0, 0.6, 0.4)
    points = np.concatenate([centers[c] + rng.normal(0, spreads[c], (PER_CLASS, dim)) for c in range(3)])
    labels = np.repeat(np.arange(3), PER_CLASS)
    return points, labels


def centroids_of(points, labels):
    return {c: points[labels == c].mean(axis=0) for c in np.unique(labels)}


def suggestion(text='', **fields):
    base = dict(class_i=0, class_j=1, class_i_name='cat', class_j_name='dog', suggestion=text)
    base.update(fields)
    return PairSuggestion(**base)


def from_text(text):
    understood = interpret_text(text, NAMES)
    s = suggestion(text)
    s.set_classes(understood.get('class_i', 0), understood.get('class_j', 1), NAMES)
    for field in ('direction', 'scale', 'tighten_i', 'tighten_j'):
        if field in understood:
            setattr(s, field, understood[field])
    return s


def fake_ui(points, labels):
    """Just enough of the UI object for apply_llm_suggestion."""
    unique = np.unique(labels)
    return SimpleNamespace(
        data={'centers': np.array([points[labels == c].mean(axis=0) for c in unique]), 'labels': labels},
        unique_labels=unique,
        ax=MagicMock(),
        moved_points=points.copy(),
        center_artists=[MagicMock() for _ in unique],
        plot=MagicMock(),
        point_tracker=MagicMock(),
        scatter=MagicMock(),
        scatter_fig=MagicMock(),
    )


def spread(points, labels, c):
    p = points[labels == c]
    return float(np.linalg.norm(p - p.mean(axis=0), axis=1).mean())


def gap(points, labels, i, j):
    centers = centroids_of(points, labels)
    return float(np.linalg.norm(centers[i] - centers[j]))


# ------------------------------------------------------------ movement vectors
def test_away_moves_both_classes_apart():
    points, labels = make_points()
    centers = centroids_of(points, labels)
    vectors = suggestion_vectors(suggestion(direction='away_from_j'), centers)
    before = np.linalg.norm(centers[0] - centers[1])
    after = np.linalg.norm(centers[0] + vectors[0] - centers[1] - vectors[1])
    assert after > before + 0.5


def test_toward_never_overshoots():
    points, labels = make_points()
    centers = centroids_of(points, labels)
    s = suggestion(class_j=2, direction='toward_j', scale='large')
    moved = centers[0] + suggestion_vectors(s, centers)[0]
    assert np.linalg.norm(moved - centers[2]) < np.linalg.norm(centers[0] - centers[2])
    assert np.dot(moved - centers[2], centers[0] - centers[2]) > 0  # still on cat's side of ship


def test_scales_are_ordered():
    points, labels = make_points()
    centers = centroids_of(points, labels)
    lengths = [np.linalg.norm(suggestion_vectors(suggestion(direction='away_from_j', scale=s), centers)[0])
               for s in ('small', 'medium', 'large')]
    assert lengths[0] < lengths[1] < lengths[2]


def test_custom_amount_scales_the_arrow():
    points, labels = make_points()
    centers = centroids_of(points, labels)
    medium = suggestion_vectors(suggestion(direction='away_from_j', scale='medium'), centers)[0]
    custom = suggestion_vectors(suggestion(direction='away_from_j', scale='custom',
                                           move_amount=2 * SCALE_FRACTIONS['medium']), centers)[0]
    np.testing.assert_allclose(custom, 2 * medium)


def test_custom_amount_is_clamped():
    points, labels = make_points()
    centers = centroids_of(points, labels)
    huge = suggestion_vectors(suggestion(direction='away_from_j', scale='custom', move_amount=50), centers)[0]
    top = suggestion_vectors(suggestion(direction='away_from_j', scale='custom',
                                        move_amount=CUSTOM_MOVE_RANGE[1]), centers)[0]
    np.testing.assert_allclose(huge, top)


def test_custom_amount_ignored_for_presets():
    points, labels = make_points()
    centers = centroids_of(points, labels)
    a = suggestion_vectors(suggestion(direction='away_from_j', scale='small', move_amount=1.4), centers)[0]
    b = suggestion_vectors(suggestion(direction='away_from_j', scale='small'), centers)[0]
    np.testing.assert_allclose(a, b)


@pytest.mark.parametrize('angle, expected', [(0, (1, 0)), (90, (0, 1)), (180, (-1, 0)), (270, (0, -1))])
def test_custom_angle_points_that_way(angle, expected):
    points, labels = make_points()
    centers = centroids_of(points, labels)
    s = suggestion(direction='custom_angle', move_angle=angle)
    vectors = suggestion_vectors(s, centers)
    assert set(vectors) == {0}  # only the named class moves
    unit = vectors[0] / np.linalg.norm(vectors[0])
    np.testing.assert_allclose(unit, expected, atol=1e-9)


def test_custom_angle_in_high_dim_falls_back_to_open_space():
    points, labels = make_points(dim=16)
    centers = centroids_of(points, labels)
    angled = suggestion_vectors(suggestion(direction='custom_angle', move_angle=45), centers)[0]
    empty = suggestion_vectors(suggestion(direction='toward_empty_space'), centers)[0]
    np.testing.assert_allclose(angled, empty)


def test_tighten_factors_custom_and_tightest_wins():
    a = suggestion(tighten_i=True)
    b = suggestion(class_i=1, class_j=0, tighten_j=True, tighten_amount_j=0.3)
    c = suggestion(class_i=1, class_j=2, tighten_i=True, tighten_amount_i=0.8)
    assert compute_class_tighten_factors([a]) == {0: TIGHTEN_FACTOR}
    assert compute_class_tighten_factors([a, b, c]) == {0: 0.3, 1: 0.8}


def test_describe_custom():
    s = suggestion(direction='custom_angle', move_angle=90, scale='custom', move_amount=0.8,
                   tighten_i=True, tighten_amount_i=0.3)
    text = describe_movement(s)
    assert 'up' in text and '80%' in text and '30%' in text


# ----------------------------------------------- apply on the 2D scatter plot
@pytest.mark.parametrize('text', [
    "Move cat a lot away from dog, they overlap.",
    "Push cat and dog apart.",
])
def test_apply_away_text_separates_on_plot(text):
    points, labels = make_points()
    ui = fake_ui(points, labels)
    assert apply_llm_suggestion(ui, from_text(text))
    assert gap(ui.moved_points, labels, 0, 1) > gap(points, labels, 0, 1) + 0.5
    # data['centers'] (what the next ideal_structure reads) follows the points
    np.testing.assert_allclose(ui.data['centers'][0], ui.moved_points[labels == 0].mean(axis=0))
    ui.plot.update_latent_space.assert_called_once()


def test_apply_toward_text_brings_closer():
    points, labels = make_points()
    ui = fake_ui(points, labels)
    apply_llm_suggestion(ui, from_text("Move ship closer to dog."))
    assert gap(ui.moved_points, labels, 2, 1) < gap(points, labels, 2, 1)


def test_apply_tighten_text_shrinks_spread():
    points, labels = make_points()
    ui = fake_ui(points, labels)
    apply_llm_suggestion(ui, from_text("Move cat away from dog and pull them tighter."))
    for c in (0, 1):
        assert spread(ui.moved_points, labels, c) == pytest.approx(TIGHTEN_FACTOR * spread(points, labels, c))
    assert spread(ui.moved_points, labels, 2) == pytest.approx(spread(points, labels, 2))


def test_apply_custom_circle_size():
    points, labels = make_points()
    ui = fake_ui(points, labels)
    apply_llm_suggestion(ui, suggestion(direction='away_from_j', tighten_i=True, tighten_amount_i=0.25,
                                        tighten_j=True))
    assert spread(ui.moved_points, labels, 0) == pytest.approx(0.25 * spread(points, labels, 0))
    assert spread(ui.moved_points, labels, 1) == pytest.approx(TIGHTEN_FACTOR * spread(points, labels, 1))


def test_apply_custom_angle_moves_up():
    points, labels = make_points()
    ui = fake_ui(points, labels)
    before = ui.data['centers'][0].copy()
    apply_llm_suggestion(ui, suggestion(direction='custom_angle', move_angle=90, scale='custom', move_amount=1.0))
    delta = ui.data['centers'][0] - before
    assert delta[1] > 0 and abs(delta[0]) < 1e-9


# ------------------------------------------------- the interaction loss target
def gradient_steps(points, labels, ideal_structure, steps=1000, lr=5.0):
    """Optimise the features against the interaction loss alone - a proxy
    for what alpha * relative_distance_loss does to the model's output."""
    features = torch.tensor(points, dtype=torch.float32, requires_grad=True)
    label_t = torch.tensor(labels)
    optimizer = torch.optim.SGD([features], lr=lr)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = relative_distance_loss(features, label_t, ideal_structure)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return features.detach().numpy(), losses


@pytest.mark.parametrize('text, check', [
    ("Move cat a lot away from dog.", lambda before, after: gap(after, LABELS, 0, 1) > gap(before, LABELS, 0, 1) + 0.5),
    ("Move ship closer to cat.", lambda before, after: gap(after, LABELS, 2, 0) < gap(before, LABELS, 2, 0) - 0.5),
    ("Pull cat tighter together.", lambda before, after: spread(after, LABELS, 0) < 0.8 * spread(before, LABELS, 0)),
])
def test_2d_loss_pulls_towards_applied_suggestion(text, check):
    """Strategy 4: text -> Apply -> ideal_structure from the moved 2D
    points -> optimising the loss makes the real points do the same."""
    points, labels = make_points()
    ui = fake_ui(points, labels)
    assert apply_llm_suggestion(ui, from_text(text))

    ideal = compute_ideal_structure(torch.tensor(ui.moved_points, dtype=torch.float32), PER_CLASS, 3)
    label_t = torch.tensor(labels)
    loss_before = relative_distance_loss(torch.tensor(points, dtype=torch.float32), label_t, ideal).item()
    loss_target = relative_distance_loss(torch.tensor(ui.moved_points, dtype=torch.float32), label_t, ideal).item()
    assert loss_target < loss_before

    trained, losses = gradient_steps(points, labels, ideal)
    assert losses[-1] < losses[0]
    assert check(points, trained)


def test_custom_zero_distance_only_tightens():
    points, labels = make_points()
    ui = fake_ui(points, labels)
    before = ui.data['centers'][0].copy()
    s = suggestion(direction='away_from_j', scale='custom', move_amount=0.0, tighten_i=True)
    assert describe_movement(s).startswith("Keep cat where it is.")
    apply_llm_suggestion(ui, s)
    np.testing.assert_allclose(ui.data['centers'][0], before, atol=1e-9)
    np.testing.assert_allclose(ui.data['centers'][1], points[labels == 1].mean(axis=0), atol=1e-9)
    assert spread(ui.moved_points, labels, 0) < spread(points, labels, 0)


@pytest.mark.xfail(strict=True, reason=(
    "relative_distance_loss's separation hinge keeps every pair at least "
    "spread_i + spread_j apart, so a 'closer' suggestion that goes inside "
    "that distance is shown on the plot but only partly trained towards."))
def test_2d_loss_honours_large_toward_suggestion():
    rng = np.random.default_rng(1)
    centers = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 8.0]])
    points = np.concatenate([centers[c] + rng.normal(0, 0.8, (PER_CLASS, 2)) for c in range(3)])
    ui = fake_ui(points, LABELS)
    apply_llm_suggestion(ui, suggestion(direction='toward_j', scale='large'))
    ideal = compute_ideal_structure(torch.tensor(ui.moved_points, dtype=torch.float32), PER_CLASS, 3)
    trained, _ = gradient_steps(points, LABELS, ideal, steps=2000)
    assert gap(trained, LABELS, 0, 1) == pytest.approx(gap(ui.moved_points, LABELS, 0, 1), abs=0.2)


def test_high_dim_target_follows_suggestion():
    """Strategy 2: the target is built in the full latent space."""
    points, labels = make_points(dim=32)
    s = suggestion(direction='away_from_j', scale='large', tighten_j=True)
    ideal = compute_llm_ideal_structure(torch.tensor(points, dtype=torch.float32), PER_CLASS, 3, [s])
    current = compute_ideal_structure(torch.tensor(points, dtype=torch.float32), PER_CLASS, 3)

    def center_gap(structure):
        return float(torch.norm(structure[0]['center'] - structure[1]['center']))

    assert center_gap(ideal) > center_gap(current)
    assert float(ideal[1]['spread']) == pytest.approx(TIGHTEN_FACTOR * float(current[1]['spread']))
    torch.testing.assert_close(ideal[2]['center'], current[2]['center'])  # ship untouched

    trained, losses = gradient_steps(points, labels, ideal)
    assert losses[-1] < losses[0]
    assert gap(trained, labels, 0, 1) > gap(points, labels, 0, 1)


LABELS = np.repeat(np.arange(3), PER_CLASS)
