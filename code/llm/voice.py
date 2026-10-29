"""Local voice dictation for suggestion edits.

Install requirements-voice.txt and run `python -m llm.voice --download` from
code/ once. Recording and transcription then run offline; audio is never
sent to OpenRouter. HITL_WHISPER_MODEL selects a model name or local path.
"""

import argparse
import importlib.util
from functools import lru_cache
from pathlib import Path
import io
import os
import threading
import wave

import numpy as np

SAMPLE_RATE = 16000
MAX_SECONDS = 120
DEFAULT_TRANSCRIBE_MODEL = "base"
MODEL_CACHE = Path(__file__).resolve().parents[2] / '.voice_models'
_transcription_lock = threading.Lock()

try:
    import sounddevice as _sd
except Exception:  # missing package or no PortAudio
    _sd = None


def available():
    return unavailable_reason() is None


def unavailable_reason():
    if _sd is None:
        return "Local dictation needs sounddevice: python -m pip install -r requirements-voice.txt"
    if importlib.util.find_spec('faster_whisper') is None:
        return "Local dictation needs faster-whisper: python -m pip install -r requirements-voice.txt"
    return None


def get_transcribe_model():
    return os.environ.get('HITL_WHISPER_MODEL', '').strip() or DEFAULT_TRANSCRIBE_MODEL


@lru_cache(maxsize=1)
def _load_model(model_name, local_files_only=True):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Install local dictation: python -m pip install -r requirements-voice.txt") from exc
    try:
        return WhisperModel(model_name, device='cpu', compute_type='int8',
                            download_root=str(MODEL_CACHE), local_files_only=local_files_only)
    except Exception as exc:
        if local_files_only:
            raise RuntimeError("Local speech model could not load. From code/, run "
                               "python -m llm.voice --download once while online, then retry. "
                               f"Details: {exc}") from exc
        raise


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
    """Transcribe WAV bytes on CPU, without network access. Return (text, model)."""
    model_name = model or get_transcribe_model()
    with _transcription_lock:
        recognizer = _load_model(model_name)
        segments, _ = recognizer.transcribe(io.BytesIO(wav_bytes), beam_size=5,
                                            vad_filter=True, condition_on_previous_text=False,
                                            initial_prompt="Class cluster editing: move, tighten, loosen, "
                                                           "spread, separate, pull together, push apart.")
        transcript = ' '.join(segment.text.strip() for segment in segments).strip()
    return transcript, f'local-whisper:{model_name}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='download speech model for offline use')
    parser.add_argument('--model', default=None, help='model name or local model directory')
    parser.add_argument('--audio', type=Path, help='transcribe an existing WAV file locally')
    args = parser.parse_args()
    model = args.model or get_transcribe_model()
    if args.download:
        _load_model(model, local_files_only=False)
        print(f'Local speech model ready: {model}; cache: {MODEL_CACHE}')
    if args.audio:
        text, used_model = transcribe(args.audio.read_bytes(), model=model)
        print(f'{used_model}: {text}')
    if not args.download and not args.audio:
        parser.print_help()


if __name__ == '__main__':
    main()
