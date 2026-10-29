"""Local dictation: decoding stays offline and recording needs no API key."""
import io
import sys
import wave
from types import SimpleNamespace, ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest

from llm import voice
from ui.ui_llm import LLMSuggestionsPanel


def test_wav_encoding_is_mono_16khz_pcm():
    audio = np.array([0, 100, -100], dtype=np.int16)
    with wave.open(io.BytesIO(voice.to_wav(audio))) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        assert wav.readframes(3) == audio.tobytes()


def test_model_loading_is_offline_cpu_by_default(monkeypatch):
    constructor = MagicMock()
    module = ModuleType('faster_whisper')
    module.WhisperModel = constructor
    monkeypatch.setitem(sys.modules, 'faster_whisper', module)
    voice._load_model.cache_clear()
    try:
        voice._load_model('base')
        assert constructor.call_args.kwargs['local_files_only'] is True
        assert constructor.call_args.kwargs['device'] == 'cpu'
        assert constructor.call_args.kwargs['compute_type'] == 'int8'
    finally:
        voice._load_model.cache_clear()


def test_transcription_consumes_segments_and_logs_local_model(monkeypatch):
    recognizer = MagicMock()
    recognizer.transcribe.return_value = (iter([SimpleNamespace(text=' Move cat '), SimpleNamespace(text='away from dog. ')]), None)
    loader = MagicMock(return_value=recognizer)
    monkeypatch.setattr(voice, '_load_model', loader)
    wav = voice.to_wav(np.zeros(1600, dtype=np.int16))
    assert voice.transcribe(wav, model='base') == ('Move cat away from dog.', 'local-whisper:base')
    loader.assert_called_once_with('base')
    assert recognizer.transcribe.call_args.kwargs['vad_filter'] is True


def test_missing_model_has_setup_message_without_cloud_fallback(monkeypatch):
    module = ModuleType('faster_whisper')
    module.WhisperModel = MagicMock(side_effect=FileNotFoundError('not cached'))
    monkeypatch.setitem(sys.modules, 'faster_whisper', module)
    voice._load_model.cache_clear()
    try:
        with pytest.raises(RuntimeError, match='python -m llm.voice --download'):
            voice._load_model('base')
        assert module.WhisperModel.call_count == 1
    finally:
        voice._load_model.cache_clear()


def test_recording_does_not_require_api_key(monkeypatch):
    monkeypatch.setattr(voice, 'available', lambda: True)
    panel = SimpleNamespace(recorder=SimpleNamespace(recording=False, start=MagicMock()), status_var=MagicMock())
    button, suggestion = MagicMock(), object()
    LLMSuggestionsPanel._toggle_dictation(panel, button, suggestion)
    panel.recorder.start.assert_called_once()
    assert panel.dictation_target is suggestion
    button.configure.assert_called_once_with(text='Stop')


def test_stop_keeps_ui_free_while_microphone_shutdown_blocks(monkeypatch):
    import queue
    import threading
    from functools import partial
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    ui_thread = threading.get_ident()
    wav = voice.to_wav(np.zeros(16000, dtype=np.int16))

    def stop():
        assert threading.get_ident() != ui_thread
        entered.set()
        assert release.wait(3)
        return wav, 1.0

    panel = SimpleNamespace(recorder=SimpleNamespace(recording=True, stop=stop),
                            dictation_target=object(), dictation_busy=False,
                            status_var=MagicMock(), result_queue=queue.Queue())
    def worker(target):
        try:
            LLMSuggestionsPanel._stop_and_transcribe_worker(panel, target)
        finally:
            finished.set()
    panel._stop_and_transcribe_worker = worker
    monkeypatch.setattr(voice, 'transcribe', lambda _: ('Move cat', 'local-whisper:base'))
    button = MagicMock()
    try:
        LLMSuggestionsPanel._toggle_dictation(panel, button, panel.dictation_target)
        assert entered.wait(1)
        assert not finished.is_set()  # callback returned despite blocked audio shutdown
        assert panel.dictation_busy is True
        button.configure.assert_called_with(text='Working…', state='disabled')
        # Extra clicks cannot launch another recording/transcription job.
        LLMSuggestionsPanel._toggle_dictation(panel, button, panel.dictation_target)
    finally:
        release.set()
        assert finished.wait(3)
    assert panel.result_queue.get_nowait()[0] == 'voice_status'
    status, payload = panel.result_queue.get_nowait()
    assert status == 'voice' and payload[3] == 'Move cat'


def test_stop_failure_is_reported_and_button_can_be_used_again():
    import queue
    panel = SimpleNamespace(recorder=SimpleNamespace(stop=MagicMock(side_effect=RuntimeError('device gone'))),
                            result_queue=queue.Queue(), dictation_busy=True,
                            dictation_button=MagicMock(), status_var=MagicMock(),
                            ui=SimpleNamespace(events=None), _log=MagicMock())
    LLMSuggestionsPanel._stop_and_transcribe_worker(panel, object())
    _, payload = panel.result_queue.get_nowait()
    assert 'device gone' in payload[5]
    LLMSuggestionsPanel._dictation_done(panel, *payload)
    assert panel.dictation_busy is False
    panel.dictation_button.configure.assert_called_with(text='Dictate', state='normal')
    assert 'device gone' in panel.status_var.set.call_args.args[0]


def test_empty_recording_releases_busy_state(monkeypatch):
    import queue
    recognizer = MagicMock()
    monkeypatch.setattr(voice, 'transcribe', recognizer)
    panel = SimpleNamespace(recorder=SimpleNamespace(stop=lambda: (b'wav', .1)),
                            result_queue=queue.Queue(), dictation_busy=True,
                            dictation_button=MagicMock(), status_var=MagicMock(),
                            ui=SimpleNamespace(events=None), _log=MagicMock())
    LLMSuggestionsPanel._stop_and_transcribe_worker(panel, object())
    _, payload = panel.result_queue.get_nowait()
    LLMSuggestionsPanel._dictation_done(panel, *payload)
    recognizer.assert_not_called()
    assert panel.dictation_busy is False
    panel.status_var.set.assert_called_with('No speech detected. Try again.')
