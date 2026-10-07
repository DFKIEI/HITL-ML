"""Read a plain-language suggestion ("Move cat a lot away from dog and pull
the dog group tighter") into the categorical fields the movement/loss code
works with (class_i, class_j, direction, scale, tighten_i, tighten_j).

Used when the operator edits or writes a suggestion's text in strategy 4, so
that what they typed is what gets applied to the plot - and therefore what
the interaction loss is pulled towards (see ``llm/movement.py`` and
``training/training_utils.py``). Rule-based and instant on purpose: no API
call per keystroke, and the dropdowns on the card show exactly what was
understood so the operator can correct it.
"""

import re
from typing import Dict, List, Tuple

TIGHTEN_PATTERNS = (
    r'tight\w*', r'compact\w*', r'condens\w*', r'shrink\w*', r'dens\w*',
    r'pull\s+(?:\w+\s+){0,4}together', r'bring\s+(?:\w+\s+){0,4}together',
    r'group\s+together', r'gather\w*', r'less\s+spread', r'less\s+scattered',
    r'concentrat\w*',
)
AWAY_PATTERNS = (
    r'away', r'apart', r'(?<!is\s)(?<!are\s)far\s+from', r'separat\w*', r'further', r'farther', r'distance\w*',
    r'push\w*', r'repel\w*', r'split\w*',
)
TOWARD_PATTERNS = (
    r'closer', r'towards?', r'nearer', r'near', r'merge\w*', r'join\w*',
    r'approach\w*', r'next\s+to', r'attract\w*',
)
EMPTY_PATTERNS = (
    r'empty', r'open\s+space', r'free\s+space', r'unoccupied', r'elsewhere',
    r'isolat\w*', r'on\s+its\s+own', r'by\s+itself',
)
SMALL_PATTERNS = (r'slight\w*', r'a\s+little', r'a\s+bit', r'small', r'gentl\w*', r'somewhat', r'minor')
LARGE_PATTERNS = (
    r'a\s+lot', r'(?<!is\s)(?<!are\s)far', r'large', r'strong\w*', r'significant\w*', r'much',
    r'big', r'major', r'well\s+away', r'clearly', r'completely',
)
MEDIUM_PATTERNS = (r'moderat\w*', r'medium', r'some\s+distance')
# The reason part of a suggestion ("... because they are far from
# everything") describes the current layout, not the move; it is blanked out
# before looking for direction/scale/tighten words. Ends at the next comma or
# sentence break so "Because X, move cat away from dog" keeps its instruction.
REASON_PATTERN = r'\b(?:because|since|given\s+that|due\s+to|as\s+(?:they|it))\b[^.,;!?\n]*'
# "the horse group is already tight" describes, it does not ask to tighten.
DESCRIBED_TIGHT_PATTERN = (r'\b(?:is|are|already|stays?|remains?)\s+'
                           r'(?:(?:already|fairly|very|quite|pretty|nicely|still)\s+)*'
                           r'(?:tight|compact|dense|condensed|concentrated)\w*')
# "pull them tighter" / "make both groups tighter" - both classes of the pair.
BOTH_PATTERN = r'\b(?:both|them|each|all|the\s+pair)\b'
# "toward(s) (the) empty space" is the empty-space direction, not toward_j.
TOWARD_EMPTY_GAP = r'^\s*(?:\w+\s+){0,2}$'


def _find_any(patterns, text):
    """Start of the first match of any pattern (whole words), or None."""
    best = None
    for pattern in patterns:
        match = re.search(r'\b' + pattern + r'\b', text)
        if match and (best is None or match.start() < best):
            best = match.start()
    return best


def _class_mentions(text: str, class_names: Dict[int, str]) -> List[Tuple[int, int]]:
    """(position, class_index) for every class named in ``text``, in reading
    order. Longest names are matched first and masked out, so e.g.
    "nordic walking" is not also counted as "walking"."""
    candidates = []
    for index, name in class_names.items():
        base = str(name).lower().replace('_', ' ').strip()
        if base:
            candidates.append((base, int(index)))
    candidates.sort(key=lambda item: len(item[0]), reverse=True)

    masked = text
    found = []
    for base, index in candidates:
        words = r'[\s_-]+'.join(re.escape(word) for word in base.split())
        pattern = r'\b' + words + r'(?:e?s)?\b'  # "cat" also matches "cats"
        for match in re.finditer(pattern, masked):
            found.append((match.start(), index))
            masked = masked[:match.start()] + ' ' * (match.end() - match.start()) + masked[match.end():]
    found.sort()
    return found


def interpret_text(text: str, class_names: Dict[int, str]) -> dict:
    """Fields understood from ``text``; keys that could not be worked out are
    left out, so the caller keeps its current value for them.

    The first class named is the one that moves (class_i), the second the one
    it moves relative to (class_j). Tightening applies to the classes named
    in the same sentence/clause as the tighten wording (class_i if none).
    Reason clauses (``REASON_PATTERN``) are ignored for everything but
    finding the classes."""
    lowered = (text or '').lower()
    result = {}

    mentions = _class_mentions(lowered, class_names)
    for pattern in (REASON_PATTERN, DESCRIBED_TIGHT_PATTERN):
        lowered = re.sub(pattern, lambda m: ' ' * len(m.group(0)), lowered)
    ordered = []
    for _, index in mentions:
        if index not in ordered:
            ordered.append(index)
    if ordered:
        result['class_i'] = ordered[0]
    if len(ordered) > 1:
        result['class_j'] = ordered[1]

    tighten = set()
    direction = None
    offset = 0
    clause_breaks = (r'[.;!?\n]|,|\b(?:and then|then|also|while|but)\b'
                     r'|\band\b(?=\s+(?:pull|tighten|make|bring|condense|compact|keep|shrink|gather|move|push))')
    for clause in re.split(clause_breaks, lowered):
        clause_start = lowered.find(clause, offset)
        offset = max(offset, clause_start)
        if not clause.strip():
            continue
        clause_mentions = _class_mentions(clause, class_names)

        tighten_at = _find_any(TIGHTEN_PATTERNS, clause)
        if tighten_at is not None:
            has_direction = any(_find_any(patterns, re.sub(
                r'\b(?:' + '|'.join(TIGHTEN_PATTERNS) + r')\b', ' ', clause)) is not None
                for patterns in (AWAY_PATTERNS, TOWARD_PATTERNS, EMPTY_PATTERNS))
            if not clause_mentions:
                targets = ordered[:2] if re.search(BOTH_PATTERN, clause) else ordered[:1]
            elif re.search(r'\b(?:both|each|all)\b', clause) or not has_direction:
                # "pull cat and dog tighter" - every class named here
                targets = [index for _, index in clause_mentions]
            else:
                # "move cat away from dog, tightening dog" - the class named
                # nearest to the tighten wording
                targets = [min(clause_mentions, key=lambda m: abs(m[0] - tighten_at))[1]]
            tighten.update(targets)
            # Words like "together"/"closer" inside a tighten phrase are not
            # a direction; only look for one outside it.
            clause = re.sub(r'\b(?:' + '|'.join(TIGHTEN_PATTERNS) + r')\b', ' ', clause)

        if direction is None:
            hits = {
                'toward_empty_space': _find_any(EMPTY_PATTERNS, clause),
                'away_from_j': _find_any(AWAY_PATTERNS, clause),
                'toward_j': _find_any(TOWARD_PATTERNS, clause),
            }
            hits = {key: pos for key, pos in hits.items() if pos is not None}
            if hits:
                direction = min(hits, key=hits.get)
                if (direction == 'toward_j' and 'toward_empty_space' in hits and re.match(
                        TOWARD_EMPTY_GAP, re.sub(r'^\S+', '', clause[hits['toward_j']:hits['toward_empty_space']]))):
                    direction = 'toward_empty_space'

    if direction is not None:
        if direction == 'toward_j' and len(ordered) < 2:
            direction = 'toward_empty_space' if _find_any(EMPTY_PATTERNS, lowered) is not None else None
        if direction is not None:
            result['direction'] = direction

    small = _find_any(SMALL_PATTERNS, lowered)
    large = _find_any(LARGE_PATTERNS, lowered)
    medium = _find_any(MEDIUM_PATTERNS, lowered)
    scales = {key: pos for key, pos in (('small', small), ('large', large), ('medium', medium))
              if pos is not None}
    if scales:
        result['scale'] = min(scales, key=scales.get)

    class_i = result.get('class_i')
    class_j = result.get('class_j')
    if tighten:
        result['tighten_i'] = class_i in tighten
        result['tighten_j'] = class_j in tighten if class_j is not None else False
    return result
