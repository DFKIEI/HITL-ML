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
import subprocess
import sys
import tempfile
import time

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
    """Microphone in a child process, isolating native audio-driver shutdown.

    The child writes WAV data before closing PortAudio. If native shutdown
    deadlocks, the parent can terminate that child and still transcribe the WAV.
    """

    def __init__(self):
        self._process = None
        self._directory = None

    @property
    def recording(self):
        return self._process is not None

    def start(self):
        if self.recording:
            raise RuntimeError("A recording is already running.")
        self._directory = tempfile.TemporaryDirectory(prefix='hitl-mic-')
        directory = Path(self._directory.name)
        try:
            self._process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), '--record-worker', str(directory)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            deadline = time.monotonic() + 10
            while not (directory / 'ready').exists():
                error = directory / 'error.txt'
                if error.exists():
                    raise RuntimeError(error.read_text())
                if self._process.poll() is not None:
                    raise RuntimeError("Microphone process exited before recording started.")
                if time.monotonic() > deadline:
                    raise RuntimeError("Microphone did not start within 10 seconds.")
                time.sleep(.03)
        except Exception:
            self._cleanup()
            raise

    def _cleanup(self):
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            if process.stdin:
                process.stdin.close()
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None

    def stop(self):
        if self._process is None:
            return to_wav(np.zeros((0, 1), dtype=np.int16)), 0.0
        process = self._process
        directory = Path(self._directory.name)
        try:
            try:
                process.stdin.write(b'STOP\n')
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass  # an auto-stopped recording may already be on disk
            deadline = time.monotonic() + 5
            while not (directory / 'audio.wav').exists():
                error = directory / 'error.txt'
                if error.exists():
                    raise RuntimeError(error.read_text())
                if process.poll() is not None:
                    raise RuntimeError("Microphone exited without saving the recording.")
                if time.monotonic() > deadline:
                    raise RuntimeError("Microphone did not return audio within 5 seconds. Please retry.")
                time.sleep(.02)
            audio = (directory / 'audio.wav').read_bytes()
            with wave.open(io.BytesIO(audio)) as wav:
                seconds = wav.getnframes() / wav.getframerate()
            try:
                process.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                pass  # _cleanup kills a driver stuck in stop/close; audio is already safe
            return audio, seconds
        finally:
            self._cleanup()


def _record_worker(directory):
    """Only the child process touches the microphone and native stream."""
    directory = Path(directory)
    frames, lock = [], threading.Lock()
    state = {'samples': 0, 'accepting': True}
    stream = None
    try:
        if _sd is None:
            raise RuntimeError("Microphone recording requires sounddevice.")

        def callback(indata, _frames, _time, _status):
            with lock:
                if not state['accepting']:
                    return
                remaining = SAMPLE_RATE * MAX_SECONDS - state['samples']
                if remaining > 0:
                    block = indata[:remaining].copy()
                    frames.append(block)
                    state['samples'] += len(block)

        stream = _sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                 dtype='int16', callback=callback)
        stream.start()
        (directory / 'ready').touch()
        sys.stdin.buffer.readline()  # STOP or parent exit (EOF)
        with lock:
            state['accepting'] = False
            audio = np.concatenate(frames) if frames else np.zeros((0, 1), dtype=np.int16)
        temporary = directory / 'audio.tmp'
        temporary.write_bytes(to_wav(audio))
        temporary.replace(directory / 'audio.wav')  # publish complete audio before native shutdown
    except Exception as exc:
        (directory / 'error.txt').write_text(f'{type(exc).__name__}: {exc}')
    finally:
        if stream is not None:
            try:
                stream.abort()
            finally:
                stream.close()


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
    parser.add_argument('--record-worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--download', action='store_true', help='download speech model for offline use')
    parser.add_argument('--model', default=None, help='model name or local model directory')
    parser.add_argument('--audio', type=Path, help='transcribe an existing WAV file locally')
    args = parser.parse_args()
    if args.record_worker:
        _record_worker(args.record_worker)
        return
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
