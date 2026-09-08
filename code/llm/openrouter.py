"""Minimal OpenRouter client.

Uses only the standard library (urllib) so the tool keeps its current
dependency list. OpenRouter exposes an OpenAI-compatible chat-completions
endpoint, so any model id listed on https://openrouter.ai/models works here.
"""

import json
import os
import urllib.error
import urllib.request

API_URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_ENV_VAR = "OPENROUTER_API_KEY"
MODEL_ENV_VAR = "OPENROUTER_MODEL"

# Any OpenRouter model id can be used; this is only the default shown in the UI.
DEFAULT_MODEL = "anthropic/claude-opus-5"

# Small, cheap models that are effective enough for the latent-space
# suggestions and are offered in the UI dropdown. Any other OpenRouter model
# id (see https://openrouter.ai/models) can still be typed into the same
# field - this list is only a curated shortlist, not a restriction.
RECOMMENDED_MODELS = [
    DEFAULT_MODEL,
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "google/gemini-2.5-flash",
]

# Put the key in this file (KEY=VALUE per line) instead of exporting it every
# run. It is searched next to this module, in code/ and in the project root,
# and it is git-ignored - never commit a filled in copy.
CONFIG_FILENAME = "llm_config.txt"


# True once a key was pasted into the suggestion window during this session
_key_set_in_session = False


class OpenRouterError(RuntimeError):
    """Raised when the OpenRouter request fails or returns an unusable answer."""


def config_paths():
    """The places llm_config.txt is looked for, in that order."""
    llm_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.dirname(llm_dir)
    project_dir = os.path.dirname(code_dir)
    return [os.path.join(directory, CONFIG_FILENAME)
            for directory in (project_dir, code_dir, llm_dir)]


def read_config():
    """Read the first llm_config.txt that exists. Returns {} if there is none.

    The file is read on every lookup, so it can be edited while the tool runs.
    """
    for path in config_paths():
        if not os.path.isfile(path):
            continue
        settings = {'_path': path}
        try:
            with open(path, 'r', encoding='utf-8') as config_file:
                for line in config_file:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    name, _, value = line.partition('=')
                    settings[name.strip().upper()] = value.strip().strip('"').strip("'")
        except OSError as e:
            print(f"Could not read {path}: {e}")
            continue
        return settings
    return {}


def get_api_key():
    """Key from the environment, else from llm_config.txt."""
    key = (os.environ.get(KEY_ENV_VAR) or "").strip()
    if key:
        return key
    return (read_config().get(KEY_ENV_VAR) or "").strip()


def get_key_source():
    """Where the key came from - shown in the suggestion window."""
    if (os.environ.get(KEY_ENV_VAR) or "").strip():
        if _key_set_in_session:
            return "the key pasted in this session"
        return f"the {KEY_ENV_VAR} environment variable"
    config = read_config()
    if (config.get(KEY_ENV_VAR) or "").strip():
        return os.path.relpath(config['_path'])
    return None


def set_api_key(key):
    """Store the key for this process only (never written to disk)."""
    global _key_set_in_session
    _key_set_in_session = True
    os.environ[KEY_ENV_VAR] = (key or "").strip()


def get_default_model():
    model = (os.environ.get(MODEL_ENV_VAR) or "").strip()
    if model:
        return model
    return (read_config().get(MODEL_ENV_VAR) or DEFAULT_MODEL).strip()


def chat_completion(messages, model=None, temperature=0.2, max_tokens=4000,
                    json_mode=True, timeout=120):
    """Send a chat completion request and return (content, raw_response_dict)."""
    api_key = get_api_key()
    if not api_key:
        raise OpenRouterError(
            f"No OpenRouter API key found. Put it in {CONFIG_FILENAME} "
            f"(as {KEY_ENV_VAR}=sk-or-...), set the {KEY_ENV_VAR} environment "
            "variable, or paste the key in the suggestions window."
        )

    model = (model or get_default_model()).strip()
    payload = {
        'model': model,
        'messages': messages,
        'temperature': temperature,
        'max_tokens': max_tokens,
    }
    if json_mode:
        payload['response_format'] = {'type': 'json_object'}

    try:
        return _post(payload, api_key, timeout)
    except OpenRouterError as e:
        # Not every model on OpenRouter accepts response_format; retry plain.
        if json_mode and 'response_format' in str(e):
            payload.pop('response_format', None)
            return _post(payload, api_key, timeout)
        raise


def _post(payload, api_key, timeout):
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            # Optional OpenRouter attribution headers.
            'HTTP-Referer': 'https://github.com/DFKIEI/HITL-ML',
            'X-Title': 'HITL-ML',
        },
        method='POST',
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        detail = e.read().decode('utf-8', errors='replace')[:500]
        raise OpenRouterError(f"OpenRouter returned HTTP {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise OpenRouterError(f"Could not reach OpenRouter: {e.reason}")
    except json.JSONDecodeError as e:
        raise OpenRouterError(f"OpenRouter returned invalid JSON: {e}")

    if 'error' in body and not body.get('choices'):
        raise OpenRouterError(f"OpenRouter error: {body['error']}")

    try:
        choice = body['choices'][0]
        content = choice['message']['content']
    except (KeyError, IndexError):
        raise OpenRouterError(f"Unexpected OpenRouter response: {str(body)[:500]}")

    if not content:
        raise OpenRouterError("OpenRouter returned an empty message.")

    print(f"[LLM DEBUG] model={payload.get('model')} "
          f"finish_reason={choice.get('finish_reason')} response={content}")

    if choice.get('finish_reason') == 'length':
        raise OpenRouterError(
            f"OpenRouter response was cut off by the max_tokens limit "
            f"({payload.get('max_tokens')}). Raise max_tokens and try again."
        )

    return content, body
