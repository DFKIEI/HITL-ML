"""Session log -> reconstruction, impact metrics, and the one-at-a-time
overlay.

The session log has to be complete enough to rebuild what the participant
did: every modification is replayed from its own parameters by
tools/reconstruct_session.py and must land exactly on the logged state."""

import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))

from custom_logging import SessionRecorder  # noqa: E402
from llm.impact import apply_to_points, describe_impact, measured_benefit_level, suggestion_impact  # noqa: E402
from llm.suggestions import PairSuggestion, parse_suggestions  # noqa: E402
from ui.ui_display import _log_space_shown, _shown_suggestions, apply_llm_suggestion, log_modification  # noqa: E402
from test_movement_and_loss import NAMES, fake_ui, make_points  # noqa: E402
import reconstruct_session  # noqa: E402


def suggestion(**fields):
    base = dict(class_i=0, class_j=1, class_i_name='cat', class_j_name='dog', suggestion='',
                direction='away_from_j')
    base.update(fields)
    return PairSuggestion(**base)


def session_ui(tmp_path, dim_latent=16):
    points, labels = make_points()
    ui = fake_ui(points, labels)
    ui.data.update(features=points.copy(), predicted_labels=labels.copy(), dataset_name='test')
    ui.plot = SimpleNamespace(latent_features=np.random.default_rng(1).normal(size=(len(labels), dim_latent)),
                              selected_labels=labels, samples_to_track=list(range(len(labels))),
                              update_center=lambda *a: None, update_latent_space=lambda *a: None)
    ui.strategy_var = MagicMock(**{'get.return_value': 4})
    ui.events = SessionRecorder('p1', 'A1', str(tmp_path))
    return ui


# ------------------------------------------------------------------ recorder
def test_recorder_writes_timestamped_events_and_time_segments(tmp_path):
    events = SessionRecorder('p1', 'A1', str(tmp_path))
    events.enter('phase', 'setup')
    events.log('SOMETHING', value=np.float32(1.5), array=np.arange(3))
    events.enter('phase', 'training')
    events.close()
    events.log('AFTER_CLOSE')  # ignored, never raises

    lines = [json.loads(line) for line in open(events.path)]
    types = [e['type'] for e in lines]
    assert types[0] == 'SESSION_START' and types[-1] == 'SESSION_END'
    assert 'AFTER_CLOSE' not in types
    assert [e['seq'] for e in lines] == list(range(1, len(lines) + 1))
    something = lines[types.index('SOMETHING')]
    assert something['value'] == 1.5 and something['array'] == [0, 1, 2]
    assert len(something['time']) == len('2026-01-01T00:00:00.000')
    segments = [e for e in lines if e['type'] == 'TIME_SEGMENT']
    assert [s['key'] for s in segments] == ['setup', 'training']
    assert set(lines[-1]['time_spent']['phase']) == {'setup', 'training'}


# ------------------------------------------------------------ reconstruction
def test_session_replays_exactly(tmp_path):
    ui = session_ui(tmp_path)
    _log_space_shown(ui, ui.data)

    # A center drag, the way the scatter plot's handlers do it.
    old_center = ui.data['centers'][2].copy()
    delta = np.array([1.5, -0.5])
    ui.moved_points[ui.data['labels'] == 2] += delta
    ui.data['centers'][2] = old_center + delta
    log_modification(ui, 'center_drag', class_index=2, old_center=old_center,
                     new_center=ui.data['centers'][2], delta=delta)

    # A single point drag.
    old = ui.moved_points[3].copy()
    ui.moved_points[3] = [9.0, 9.0]
    log_modification(ui, 'point_drag', point_index=3, class_index=0, old_position=old,
                     new_position=ui.moved_points[3])

    # An applied LLM suggestion with tightening on both classes.
    assert apply_llm_suggestion(ui, suggestion(tighten_i=True, tighten_j=True, tighten_amount_j=0.3),
                                source='participant')
    ui.events.close()

    events = reconstruct_session.load_events(ui.events.path)
    result = reconstruct_session.reconstruct(events, str(tmp_path))
    assert result['mismatches'] == []
    space = result['spaces'][0]
    assert [e['kind'] for e, _ in space.modifications] == ['center_drag', 'point_drag', 'llm_apply']
    np.testing.assert_allclose(space.points, ui.moved_points)
    np.testing.assert_allclose(space.centers, ui.data['centers'])


def test_reconstruction_flags_a_step_that_does_not_match(tmp_path):
    ui = session_ui(tmp_path)
    _log_space_shown(ui, ui.data)
    ui.moved_points[0] = [5.0, 5.0]  # changed, but logged as if moved elsewhere
    log_modification(ui, 'point_drag', point_index=0, class_index=0, old_position=[0, 0], new_position=[4.0, 4.0])
    ui.events.close()
    result = reconstruct_session.reconstruct(reconstruct_session.load_events(ui.events.path), str(tmp_path))
    assert len(result['mismatches']) == 1


# ------------------------------------------------------------------- impact
def test_apply_to_points_matches_the_plot_apply():
    points, labels = make_points()
    s = suggestion(tighten_i=True)
    ui = fake_ui(points, labels)
    apply_llm_suggestion(ui, s)
    np.testing.assert_allclose(apply_to_points(points, labels, [s]), ui.moved_points)


@pytest.mark.parametrize('dim', [2, 64])
def test_separating_an_overlapping_pair_is_measured_as_a_gain(dim):
    points, labels = make_points(dim=dim)
    push = suggestion(scale='large', tighten_i=True)
    pull = suggestion(direction='toward_j', scale='large')

    gain = suggestion_impact(points, labels, push)
    loss = suggestion_impact(points, labels, pull)
    assert gain['dim'] == dim
    assert gain['separation_gain'] > 0.05 > loss['separation_gain']
    assert gain['pair_overlap_after'] < gain['pair_overlap_before']
    assert gain['fisher_after'] > gain['fisher_before']
    assert gain['loss_target'] > 0
    assert 0 <= gain['knn_preserved'] <= 1
    assert measured_benefit_level(gain) == 'high'
    assert 'LLM expects low' in describe_impact(gain, 'low')


def test_no_op_suggestion_changes_nothing():
    points, labels = make_points()
    keep = suggestion(scale='custom', move_amount=0.0)
    impact = suggestion_impact(points, labels, keep)
    assert impact['shift_ratio'] == pytest.approx(0.0, abs=1e-12)
    assert impact['knn_preserved'] == 1.0
    assert impact['subspace_rotation_deg'] == pytest.approx(0.0, abs=1e-3)
    # Not 0: the loss's separation term still pushes the overlapping pair.
    moved = suggestion_impact(points, labels, suggestion(scale='large', tighten_i=True))
    assert 0 <= impact['loss_target'] < moved['loss_target']


def test_expected_benefit_is_parsed_and_logged():
    content = json.dumps({'global': {'issue': 'x', 'strategy': 'y'}, 'suggestions': [
        {'class_i': 0, 'class_j': 1, 'suggestion': 'Move cat away from dog', 'direction': 'away_from_j',
         'scale': 'large', 'expected_benefit': 'HIGH'},
        {'class_i': 1, 'class_j': 2, 'suggestion': 'Move dog away from ship', 'expected_benefit': 'huge'}]})
    _, suggestions = parse_suggestions(content, NAMES, {0, 1, 2})
    assert [s.expected_benefit for s in suggestions] == ['high', None]
    assert suggestions[0].uid != suggestions[1].uid
    assert suggestions[0].as_log_dict()['expected_benefit'] == 'high'


# --------------------------------------------------------- one at a time
def test_overlay_only_draws_the_suggestion_on_screen():
    first, second = suggestion(), suggestion(class_i=2, class_j=0)
    ui = SimpleNamespace(latest_llm_suggestions=[first, second], applied_llm_suggestion_ids=set(),
                         llm_focus_suggestion=second)
    assert _shown_suggestions(ui) == [second]
    ui.applied_llm_suggestion_ids = {id(second)}
    assert _shown_suggestions(ui) == []
    ui.llm_focus_suggestion = None
    assert _shown_suggestions(ui) == [first]
