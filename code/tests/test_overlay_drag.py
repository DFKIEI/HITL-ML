"""Dragging the suggestion arrows/circles on the scatter plot (strategy 4):
the dragged tip/edge follows the pointer, and the suggestion is reshaped so
Apply and the loss use exactly what is drawn."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from llm.movement import SNAP_DEGREES, compute_class_movements, fit_suggestion_to_vector, suggestion_vectors
from llm.suggestions import PairSuggestion
from ui.ui_display import (apply_llm_suggestion, llm_overlay_hover, llm_overlay_motion, llm_overlay_press,
                           llm_overlay_release, refresh_llm_overlay)
from test_movement_and_loss import make_points, spread


def suggestion(**fields):
    base = dict(class_i=0, class_j=1, class_i_name='cat', class_j_name='dog', suggestion='',
                direction='away_from_j')
    base.update(fields)
    return PairSuggestion(**base)


def plot_ui(suggestions, strategy=4):
    points, labels = make_points()
    unique = np.unique(labels)
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1])
    ax.set_xlim(-6, 14)
    ax.set_ylim(-6, 10)
    ui = SimpleNamespace(
        ax=ax, scatter_fig=fig, scatter=MagicMock(), incorrect_scatter=MagicMock(),
        data={'centers': np.array([points[labels == c].mean(axis=0) for c in unique]), 'labels': labels,
              'features': points.copy()},
        unique_labels=unique, moved_points=points.copy(), original_points=points.copy(),
        class_colors={0: 'red', 1: 'blue', 2: 'green'}, center_artists=[MagicMock() for _ in unique],
        point_colors=np.ones((len(labels), 4)), incorrect_mask=np.zeros(len(labels), bool),
        incorrect_colors=np.ones((0, 4)), plot=MagicMock(), point_tracker=MagicMock(),
        llm_tracker=MagicMock(), llm_panel=MagicMock(), strategy_var=SimpleNamespace(get=lambda: strategy),
        latest_llm_suggestions=list(suggestions), applied_llm_suggestion_ids=set(),
        highlighted_classes=None, llm_drag=None,
    )
    refresh_llm_overlay(ui)
    return ui, points, labels


def event(ui, xy):
    x, y = ui.ax.transData.transform(xy)
    return SimpleNamespace(inaxes=ui.ax, x=x, y=y, xdata=float(xy[0]), ydata=float(xy[1]))


def drag(ui, start, end, steps=5):
    assert llm_overlay_press(ui, event(ui, start))
    for t in np.linspace(0, 1, steps + 1)[1:]:
        llm_overlay_motion(ui, event(ui, np.asarray(start) + t * (np.asarray(end) - np.asarray(start))))
    assert llm_overlay_release(ui, event(ui, end))


def handle(ui, kind, class_index):
    return next(h for h in ui.llm_handles if h['kind'] == kind and h['class'] == class_index)


def drawn_tip(ui, class_index):
    centroids = {int(l): ui.data['centers'][i] for i, l in enumerate(ui.unique_labels)}
    return centroids[class_index] + compute_class_movements(ui.latest_llm_suggestions, centroids)[class_index]


def test_handles_only_when_editable():
    s = suggestion(tighten_i=True)
    assert {(h['kind'], h['class']) for h in plot_ui([s])[0].llm_handles} == {('arrow', 0), ('arrow', 1), ('circle', 0)}
    assert plot_ui([s], strategy=3)[0].llm_handles == []


def test_class_j_pushed_away_has_independent_arrow_handle():
    ui, *_ = plot_ui([suggestion()])
    assert [h['class'] for h in ui.llm_handles if h['kind'] == 'arrow'] == [0, 1]


def test_press_elsewhere_is_not_grabbed():
    ui, *_ = plot_ui([suggestion()])
    assert not llm_overlay_press(ui, event(ui, (12.0, -5.0)))


def test_drag_arrow_off_axis_sets_custom_direction_and_follows_pointer():
    s = suggestion()
    ui, *_ = plot_ui([s])
    start = handle(ui, 'arrow', 0)['end']
    target = ui.data['centers'][0] + np.array([0.0, 3.0])  # straight up
    drag(ui, start, target)
    assert s.direction == 'custom_angle' and s.scale == 'custom' and s.edited
    assert s.move_angle == pytest.approx(90.0, abs=1e-6)
    np.testing.assert_allclose(drawn_tip(ui, 0), target, atol=1e-6)
    ui.llm_tracker.log_edited_on_plot.assert_called_with(s)
    ui.llm_panel.refresh_cards.assert_called_once()


def test_drag_arrow_along_its_direction_keeps_direction():
    s = suggestion()
    ui, *_ = plot_ui([s])
    h = handle(ui, 'arrow', 0)
    longer = h['start'] + 2.0 * (h['end'] - h['start'])
    drag(ui, h['end'], longer)
    assert s.direction == 'away_from_j' and s.scale == 'custom'
    np.testing.assert_allclose(drawn_tip(ui, 0), longer, atol=1e-6)
    # still a mutual push: dog keeps moving too
    assert 1 in compute_class_movements([s], {int(l): ui.data['centers'][i] for i, l in enumerate(ui.unique_labels)})


def test_drag_arrow_to_its_tail_means_stay():
    s = suggestion(tighten_i=True)
    ui, *_ = plot_ui([s])
    h = handle(ui, 'arrow', 0)
    drag(ui, h['end'], h['start'])
    assert s.move_amount == pytest.approx(0.0, abs=1e-9)


def test_drag_arrow_with_two_suggestions_moves_the_mean_to_pointer():
    first = suggestion()
    second = suggestion(class_j=2, direction='toward_j')
    ui, *_ = plot_ui([first, second])
    target = ui.data['centers'][0] + np.array([0.0, -3.0])  # well off the away-from-dog axis
    drag(ui, handle(ui, 'arrow', 0)['end'], target)
    assert first.direction == 'custom_angle'
    np.testing.assert_allclose(drawn_tip(ui, 0), target, atol=1e-6)
    assert second.scale != 'custom'  # only the grabbed suggestion changes


def test_drag_near_axis_snaps_onto_it():
    s = suggestion()
    ui, *_ = plot_ui([s])
    h = handle(ui, 'arrow', 0)
    axis = (h['end'] - h['start']) / np.linalg.norm(h['end'] - h['start'])
    normal = np.array([-axis[1], axis[0]])
    near = h['start'] + 3.0 * axis + 0.2 * normal  # ~4 degrees off
    drag(ui, h['end'], near)
    assert s.direction == 'away_from_j'
    np.testing.assert_allclose(drawn_tip(ui, 0), h['start'] + 3.0 * axis, atol=1e-6)


def test_drag_circle_sets_that_class_only():
    s = suggestion(tighten_i=True, tighten_j=True)
    ui, points, labels = plot_ui([s])
    h = handle(ui, 'circle', 0)
    edge = h['center'] + np.array([h['radius'], 0.0])
    drag(ui, edge, h['center'] + np.array([0.3 * h['mean_radius'], 0.0]))
    assert s.tighten_amount_i == pytest.approx(0.3)
    assert s.tighten_amount_j is None
    assert handle(ui, 'circle', 0)['radius'] == pytest.approx(0.3 * h['mean_radius'])

    apply_llm_suggestion(ui, s)
    assert spread(ui.moved_points, labels, 0) == pytest.approx(0.3 * spread(points, labels, 0))


def test_circle_grabbed_anywhere_on_edge():
    ui, *_ = plot_ui([suggestion(tighten_i=True)])
    h = handle(ui, 'circle', 0)
    assert llm_overlay_press(ui, event(ui, h['center'] + h['radius'] * np.array([0.0, -1.0])))
    assert ui.llm_drag['kind'] == 'circle'


def test_hover_sets_hand_cursor():
    ui, *_ = plot_ui([suggestion()])
    ui.scatter_fig.canvas.set_cursor = MagicMock()
    llm_overlay_hover(ui, event(ui, handle(ui, 'arrow', 0)['end']))
    assert ui.llm_cursor_on_handle


@pytest.mark.parametrize('degrees, snapped', [(SNAP_DEGREES - 2, True), (SNAP_DEGREES + 5, False)])
def test_snap_threshold(degrees, snapped):
    points, labels = make_points()
    centroids = {c: points[labels == c].mean(axis=0) for c in range(3)}
    s = suggestion()
    base = suggestion_vectors(s, centroids)[0]
    angle = np.deg2rad(degrees)
    rotated = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) @ base
    fit_suggestion_to_vector(s, rotated, centroids, 'away_from_j')
    assert (s.direction == 'away_from_j') == snapped


def test_suggestion_view_hides_other_clusters_and_restores_everything():
    from ui.ui_display import highlight_classes
    s = suggestion()
    ui, points, labels = plot_ui([s])
    ui.llm_focus_suggestion = s
    ui.llm_only_suggestion_clusters = True
    highlight_classes(ui, None)
    colors = ui.scatter.set_facecolors.call_args.args[0]
    assert np.all(colors[labels == 2, 3] == 0)
    assert np.all(colors[labels != 2, 3] > 0)
    ui.center_artists[2].set_visible.assert_called_with(False)
    # A hover exit or redraw must not restore unrelated clusters.
    refresh_llm_overlay(ui)
    colors = ui.scatter.set_facecolors.call_args.args[0]
    assert np.all(colors[labels == 2, 3] == 0)
    ui.llm_only_suggestion_clusters = False
    highlight_classes(ui, None)
    np.testing.assert_array_equal(ui.scatter.set_facecolors.call_args.args[0], ui.point_colors)
    ui.center_artists[2].set_visible.assert_called_with(True)
    np.testing.assert_array_equal(ui.moved_points, points)
    plt.close(ui.scatter_fig)


def test_suggestion_view_tracks_current_pair_while_editing_and_navigation():
    from ui.ui_display import highlight_classes
    s = suggestion()
    ui, _, labels = plot_ui([s])
    ui.llm_only_suggestion_clusters = True
    ui.llm_focus_suggestion = s
    s.class_j = 2  # dropdown/text edit changes the current pair in place
    refresh_llm_overlay(ui)
    colors = ui.scatter.set_facecolors.call_args.args[0]
    assert np.all(colors[labels == 1, 3] == 0)
    assert np.all(colors[labels == 2, 3] > 0)
    ui.llm_focus_suggestion = suggestion(class_i=1, class_j=2)
    highlight_classes(ui, None)
    colors = ui.scatter.set_facecolors.call_args.args[0]
    assert np.all(colors[labels == 0, 3] == 0)
    ui.llm_focus_suggestion = None
    highlight_classes(ui, None)
    assert np.all(ui.scatter.set_facecolors.call_args.args[0][:, 3] > 0)
    plt.close(ui.scatter_fig)


def test_edit_highlight_keeps_both_movement_arrows_visible():
    from ui.ui_display import highlight_classes
    ui, *_ = plot_ui([suggestion()])
    highlight_classes(ui, {0})
    refresh_llm_overlay(ui)
    arrows = [a for a, c in ui.llm_overlay_classes if c == 1]
    assert arrows and all(a.get_alpha() == .9 and a.get_visible() for a in arrows)
    plt.close(ui.scatter_fig)


def test_preview_arrow_endpoints_remain_in_view():
    s = suggestion(scale='custom', move_amount=4.0)
    ui, *_ = plot_ui([s])
    ui.ax.set_xlim(-.1, .1)
    ui.ax.set_ylim(-.1, .1)
    refresh_llm_overlay(ui)
    for artist in ui.llm_overlay_artists:
        if hasattr(artist, 'arrow_patch'):
            x,y = artist.xy
            assert ui.ax.get_xlim()[0] <= x <= ui.ax.get_xlim()[1]
            assert ui.ax.get_ylim()[0] <= y <= ui.ax.get_ylim()[1]
            assert artist.get_annotation_clip() is False
    plt.close(ui.scatter_fig)


@pytest.mark.parametrize('edited_class,other_class', [(0, 1), (1, 0)])
def test_each_arrow_moves_independently_and_apply_matches_preview(edited_class, other_class):
    from llm.impact import apply_to_points
    s = suggestion()
    ui, points, labels = plot_ui([s])
    centroids = {int(c): ui.data['centers'][i].copy() for i,c in enumerate(ui.unique_labels)}
    before_other = suggestion_vectors(s, centroids)[other_class].copy()
    original = handle(ui, 'arrow', edited_class)['end'].copy()
    target = centroids[edited_class] + np.array([1.0, 2.0])
    drag(ui, original, target)
    after = suggestion_vectors(s, centroids)
    np.testing.assert_allclose(after[other_class], before_other, atol=1e-8)
    np.testing.assert_allclose(centroids[edited_class] + after[edited_class], target, atol=1e-8)
    predicted = apply_to_points(points, labels, [s])
    assert apply_llm_suggestion(ui, s)
    np.testing.assert_allclose(ui.moved_points, predicted, atol=1e-8)
    plt.close(ui.scatter_fig)


def test_editing_primary_then_secondary_preserves_primary_move():
    s = suggestion()
    ui, *_ = plot_ui([s])
    drag(ui, handle(ui, 'arrow', 0)['end'], ui.data['centers'][0] + [0., 2.])
    before = handle(ui, 'arrow', 0)['end'].copy()
    drag(ui, handle(ui, 'arrow', 1)['end'], ui.data['centers'][1] + [2., 0.])
    np.testing.assert_allclose(handle(ui, 'arrow', 0)['end'], before)
    # Serializing and reconstructing a suggestion preserves both independent moves.
    import json
    restored = PairSuggestion(**json.loads(json.dumps(s.as_log_dict())))
    centroids = {int(c):ui.data['centers'][i] for i,c in enumerate(ui.unique_labels)}
    for c,vector in suggestion_vectors(s,centroids).items():
        np.testing.assert_allclose(suggestion_vectors(restored,centroids)[c], vector)
    plt.close(ui.scatter_fig)


def test_changing_class_pair_clears_old_movement_overrides():
    s = suggestion(move_overrides={0:[1.,0.],1:[-1.,0.]})
    s.set_classes(0,2,{0:'cat',1:'dog',2:'bird'})
    assert s.move_overrides == {}


def test_zero_movement_keeps_handle_for_editing_again():
    s = suggestion()
    ui, *_ = plot_ui([s])
    start = handle(ui, 'arrow', 0)['start'].copy()
    drag(ui, handle(ui, 'arrow', 0)['end'], start)
    np.testing.assert_allclose(handle(ui, 'arrow', 0)['end'], start)
    drag(ui, start, start + [0., 1.])
    np.testing.assert_allclose(handle(ui, 'arrow', 0)['end'], start + [0., 1.])
    plt.close(ui.scatter_fig)


@pytest.mark.parametrize('direction', ['toward_j', 'toward_empty_space', 'custom_angle'])
def test_stationary_reference_class_can_be_moved_independently(direction):
    s = suggestion(direction=direction, move_angle=90.)
    ui, *_ = plot_ui([s])
    first_tip = handle(ui, 'arrow', 0)['end'].copy()
    second = handle(ui, 'arrow', 1)
    np.testing.assert_allclose(second['end'], ui.data['centers'][1])
    assert any('drag to move' in a.get_text() for a in ui.llm_overlay_artists if hasattr(a, 'get_text'))
    target = second['start'] + [1., 1.]
    drag(ui, second['end'], target)
    np.testing.assert_allclose(handle(ui, 'arrow', 1)['end'], target)
    np.testing.assert_allclose(handle(ui, 'arrow', 0)['end'], first_tip)
    plt.close(ui.scatter_fig)
