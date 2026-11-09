"""LLM interpretation of edited instructions; validates output before any movement."""
import json
import math
from llm import openrouter


def interpret_instruction(text, class_names, current, model=None):
    schema = {'class_i': current['class_i'], 'class_j': current['class_j'],
              'direction': current.get('direction', 'away_from_j'),
              'scale': current.get('scale', 'medium'),
              'tighten_i': current.get('tighten_i', False), 'tighten_j': current.get('tighten_j', False)}
    for key in ('move_amount', 'move_angle'):
        if current.get(key) is not None:
            schema[key] = current[key]
    messages = [
        {'role': 'system', 'content': 'Interpret the operator instruction as a class-cluster edit. '
         'Return only one JSON object with class_i and class_j (distinct known integer class IDs), '
         'direction (away_from_j, toward_j, toward_empty_space or custom_angle), '
         'scale (small, medium, large or custom), tighten_i and tighten_j (booleans). '
         'For custom angle include move_angle in degrees, 0 right, 90 up. '
         'For custom scale include move_amount, 0 to 5 times the layout scale. '
         'Preserve existing fields not changed by the instruction. Do not invent class IDs.'},
        {'role': 'user', 'content': json.dumps({'instruction': text, 'classes': class_names,
                                               'current': schema}, ensure_ascii=False)}]
    raw, _ = openrouter.chat_completion(messages, model=model, temperature=0., max_tokens=1500, timeout=60)
    clean = raw.strip()
    if clean.startswith('```'):
        clean = clean.split('\n', 1)[1].rsplit('```', 1)[0]
    result = json.loads(clean)
    if not isinstance(result, dict):
        raise ValueError('Interpretation must be an object.')
    for key in ('class_i', 'class_j'):
        if type(result.get(key)) is not int or result[key] not in class_names:
            raise ValueError('Interpretation names an unknown class.')
    if result['class_i'] == result['class_j']:
        raise ValueError('Choose two different classes.')
    if result.get('direction') not in ('away_from_j', 'toward_j', 'toward_empty_space', 'custom_angle'):
        raise ValueError('Invalid direction.')
    if result.get('scale') not in ('small', 'medium', 'large', 'custom'):
        raise ValueError('Invalid movement size.')
    for key in ('tighten_i', 'tighten_j'):
        if type(result.get(key)) is not bool:
            raise ValueError('Invalid tightening flag.')
    for key, low, high, required in [('move_amount', 0, 5, result['scale']=='custom'),
                                     ('move_angle', 0, 360, result['direction']=='custom_angle')]:
        value = result.get(key)
        if (required or value is not None) and (type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high):
            raise ValueError(f'Invalid {key}.')
    allowed = set(schema) | {'move_amount', 'move_angle'}
    return {k:v for k,v in result.items() if k in allowed}, raw
