"""Plain-language suggestion text -> the categorical fields that get applied
(llm/text_to_movement.py). The phrasings follow what the LLM is prompted to
write (llm/suggestions.py): an instruction naming both classes, then why."""

import pytest

from llm.text_to_movement import interpret_text

CIFAR = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
         5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
PAMAP = {0: 'lying', 1: 'sitting', 2: 'standing', 3: 'walking', 4: 'running', 5: 'cycling',
         6: 'nordic_walking', 7: 'ascending_stairs', 8: 'descending_stairs',
         9: 'vacuum_cleaning', 10: 'ironing', 11: 'rope_jumping'}

CASES = [
    ("Move cat away from dog, they are easily confused and sit on top of each other.", CIFAR,
     dict(class_i=3, class_j=5, direction='away_from_j')),
    ("Move cat a lot away from dog and pull the dog group tighter.", CIFAR,
     dict(class_i=3, class_j=5, direction='away_from_j', scale='large', tighten_i=False, tighten_j=True)),
    ("Push automobile and truck apart; they look very similar to the model.", CIFAR,
     dict(class_i=1, class_j=9, direction='away_from_j')),
    ("Separate deer from horse slightly because they overlap a little.", CIFAR,
     dict(class_i=4, class_j=7, direction='away_from_j', scale='small')),
    ("Pull the bird group tighter together, it is very spread out and mixes with airplane.", CIFAR,
     dict(class_i=2, tighten_i=True)),
    ("Move frog into open space because it is far from everything but has many stray points.", CIFAR,
     dict(class_i=6, direction='toward_empty_space')),
    ("Move ship a little closer to airplane since they share a blue background.", CIFAR,
     dict(class_i=8, class_j=0, direction='toward_j', scale='small')),
    ("Move dog away from cat. They are close together and get mixed up.", CIFAR,
     dict(class_i=5, class_j=3, direction='away_from_j')),
    ("Keep cat and dog apart and make both groups tighter.", CIFAR,
     dict(class_i=3, class_j=5, direction='away_from_j', tighten_i=True, tighten_j=True)),
    ("Move cat away from dog and pull them tighter.", CIFAR,
     dict(class_i=3, class_j=5, direction='away_from_j', tighten_i=True, tighten_j=True)),
    ("Tighten cat and dog.", CIFAR, dict(class_i=3, class_j=5, tighten_i=True, tighten_j=True)),
    ("Move the cats away from the dogs.", CIFAR, dict(class_i=3, class_j=5, direction='away_from_j')),
    ("Move truck a lot farther from automobile because they are very close and overlap heavily.", CIFAR,
     dict(class_i=9, class_j=1, direction='away_from_j', scale='large')),
    ("Bring deer closer to horse, they are both four-legged animals in similar scenes.", CIFAR,
     dict(class_i=4, class_j=7, direction='toward_j')),
    ("Move horse toward empty space, it is crowded by deer and dog.", CIFAR,
     dict(class_i=7, direction='toward_empty_space')),
    ("Move horse towards the open space.", CIFAR, dict(class_i=7, direction='toward_empty_space')),
    ("Move bird far from airplane.", CIFAR,
     dict(class_i=2, class_j=0, direction='away_from_j', scale='large')),
    ("Because cat and dog overlap, move cat away from dog.", CIFAR,
     dict(class_i=3, class_j=5, direction='away_from_j')),
    ("Move nordic walking away from walking, they are nearly identical movements.", PAMAP,
     dict(class_i=6, class_j=3, direction='away_from_j')),
    ("Move ascending stairs moderately away from descending stairs.", PAMAP,
     dict(class_i=7, class_j=8, direction='away_from_j', scale='medium')),
    ("Sitting and standing overlap a lot; move sitting far away from standing.", PAMAP,
     dict(class_i=1, class_j=2, direction='away_from_j', scale='large')),
    ("Ironing is too spread out - condense ironing and move it away from vacuum cleaning.", PAMAP,
     dict(class_i=10, class_j=9, direction='away_from_j', tighten_i=True)),
    ("Move cycling slightly toward running because the gap between them is too big.", PAMAP,
     dict(class_i=5, class_j=4, direction='toward_j', scale='small')),
    ("Rope jumping has many outliers; move it to an empty area away from running.", PAMAP,
     dict(class_i=11, direction='toward_empty_space')),
]


@pytest.mark.parametrize('text, names, expected', CASES, ids=[c[0][:40] for c in CASES])
def test_interpret_text(text, names, expected):
    understood = interpret_text(text, names)
    assert {k: understood.get(k) for k in expected} == expected


@pytest.mark.parametrize('text', [
    "Lying is far from everything else, so tighten lying a bit.",
    "Move cat away from dog because they are far from the other animals.",
])
def test_descriptive_far_is_not_a_scale(text):
    assert interpret_text(text, {**PAMAP, **{20: 'cat', 21: 'dog'}}).get('scale') != 'large'


@pytest.mark.parametrize('text', [
    "Move deer away from horse and pull the deer group tighter; the horse group is already tight.",
    "Move deer away from horse a medium amount. Both groups are already fairly tight.",
])
def test_described_tightness_is_not_an_instruction(text):
    understood = interpret_text(text, CIFAR)
    assert not understood.get('tighten_j')
    assert understood.get('tighten_i', False) == ('pull the deer group tighter' in text)


def test_unknown_text_changes_nothing():
    assert interpret_text("Looks fine to me.", CIFAR) == {}
