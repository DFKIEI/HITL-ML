"""Voice dictation for suggestion edits: record from the microphone, then
transcribe through OpenRouter with an audio-capable model (same key as the
suggestions, no extra service).

Recording needs the optional ``sounddevice`` package (``pip install
sounddevice``); without it ``available()`` is False and the UI hides the
dictate buttons' function behind a hint. On macOS the terminal / IDE that
runs the tool needs microphone permission (System Settings > Privacy &
Security > Microphone).

The transcription model is ``OPENROUTER_TRANSCRIBE_MODEL`` from the
environment or ``llm_config.txt`` (default ``DEFAULT_TRANSCRIBE_MODEL``).
"""

import base64
import io
import os
import threading
import wave

import numpy as np

from llm import openrouter

SAMPLE_RATE = 16000
MAX_SECONDS = 120
TRANSCRIBE_ENV_VAR = "OPENROUTER_TRANSCRIBE_MODEL"
DEFAULT_TRANSCRIBE_MODEL = "google/gemini-2.5-flash"

TRANSCRIBE_PROMPT = (
    "Transcribe this short voice note exactly, in the language spoken. It is an "
    "instruction for rearranging class clusters on a plot (e.g. \"move cat away from "
    "dog a lot\"). Return only the transcript, no quotes or comments."
)

try:
    import sounddevice as _sd
except Exception:  # missing package or no PortAudio
    _sd = None


def available():
    return _sd is not None


def unavailable_reason():
    if _sd is None:
        return "Dictation needs the sounddevice package: pip install sounddevice"
    return None


def get_transcribe_model():
    model = (os.environ.get(TRANSCRIBE_ENV_VAR) or "").strip()
    if model:
        return model
    return (openrouter.read_config().get(TRANSCRIBE_ENV_VAR) or DEFAULT_TRANSCRIBE_MODEL).strip()


class Recorder:
    """Collects 16 kHz mono audio between ``start()`` and ``stop()``."""

    def __init__(self):
        self._frames = []
        self._stream = None
        self._lock = threading.Lock()

    @property
    def recording(self):
        return self._stream is not None

    def start(self):
        if _sd is None:
            raise RuntimeError(unavailable_reason())
        self._frames = []

        def callback(indata, _frames, _time, _status):
            with self._lock:
                if sum(len(f) for f in self._frames) < SAMPLE_RATE * MAX_SECONDS:
                    self._frames.append(indata.copy())

        self._stream = _sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', callback=callback)
        self._stream.start()

    def stop(self):
        """Stop and return (wav_bytes, seconds)."""
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
        with self._lock:
            audio = np.concatenate(self._frames) if self._frames else np.zeros((0, 1), dtype=np.int16)
        return to_wav(audio), len(audio) / SAMPLE_RATE


def to_wav(samples):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
    return buffer.getvalue()


def transcribe(wav_bytes, model=None):
    """Transcript of ``wav_bytes`` (a WAV file) via OpenRouter. Returns
    (text, model)."""
    model = model or get_transcribe_model()
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': TRANSCRIBE_PROMPT},
            {'type': 'input_audio',
             'input_audio': {'data': base64.b64encode(wav_bytes).decode('ascii'), 'format': 'wav'}},
        ],
    }]
    content, _ = openrouter.chat_completion(messages, model=model, temperature=0.0, max_tokens=500,
                                            json_mode=False, timeout=60)
    return content.strip().strip('"').strip(), model
