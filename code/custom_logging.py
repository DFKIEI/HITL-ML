import json
import logging
import numpy as np
from typing import Optional
import os


class PointTracker:
    def __init__(self, probant_id: str = 'test', scenario: str = 'A1', log_path: str = ''):
        # Configure logging
        self.logger = logging.getLogger('PointTracker')
        self.logger.setLevel(logging.INFO)
        log_file = f'{log_path}{os.sep}point_movements_id_{probant_id}_scenario_{scenario}.log'
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)

        # Add handler if not already added
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
        self.init_logging_header(probant_id, scenario)

    def log_center_movement(self,
                            class_label: int,
                            old_center: np.ndarray,
                            new_center: np.ndarray):
        """
        Log movement of a class center

        Args:
            class_label (int): The label of the class
            old_center (np.ndarray): Original center coordinates
            new_center (np.ndarray): New center coordinates
        """
        self.logger.info(
            f"CM, {class_label},{old_center},{new_center}"
        )

    def init_logging_header(self,
                            id: str,
                            scenario: str):
        """
        Intialize the header of the log file

        Args:
            id (str): Unique Identifier for participant
            scenario (str): Nome of teh performed scenario
            new_center (np.ndarray): New center coordinates
        """
        self.logger.info(
            '##############      '
            f"Participant id : -  {id}"
            f" ||  Performed scenario to {scenario}"
            '      ##############'
        )

    def log_individual_point_movement(self,
                                      point_index: int,
                                      old_position: np.ndarray,
                                      new_position: np.ndarray,
                                      class_label: Optional[int] = None):
        """
        Log movement of an individual point

        Args:
            point_index (int): Index of the point
            old_position (np.ndarray): Original point coordinates
            new_position (np.ndarray): New point coordinates
            class_label (Optional[int]): Label of the point's class
        """
        self.logger.info(
            f"PM, {point_index},{old_position},{new_position}"
        )

    def log_class_points_reset(self,
                               class_label: int,
                               center: np.ndarray,
                               num_points: int):
        """
        Log resetting of all points in a class to its center

        Args:
            class_label (int): The label of the class
            center (np.ndarray): The center coordinates
            num_points (int): Number of points reset
        """
        self.logger.info(
            f"CR, {class_label}, {center}, _"
        )

    def log_llm_center_movement(self,
                                class_label: int,
                                old_center: np.ndarray,
                                new_center: np.ndarray):
        """
        Log a class center movement that came from an applied LLM suggestion

        Args:
            class_label (int): The label of the class
            old_center (np.ndarray): Original center coordinates
            new_center (np.ndarray): New center coordinates
        """
        self.logger.info(
            f"LLM_CM, {class_label},{old_center},{new_center}"
        )

    def log_llm_class_scaling(self,
                              class_label: int,
                              center: np.ndarray,
                              factor: float,
                              num_points: int):
        """
        Log the rescaling of a class spread that came from an applied LLM suggestion

        Args:
            class_label (int): The label of the class
            center (np.ndarray): The center the points were scaled around
            factor (float): Scaling factor applied to the distances to the center
            num_points (int): Number of points affected
        """
        self.logger.info(
            f"LLM_CS, {class_label}, {center}, {factor}, {num_points}"
        )

    def undo_last_step(self):
        """
        Log undoing of last step

        """
        self.logger.info(
            f"----------- Undo Last step ----------- "
        )


class AllDataPointsTracker:
    def __init__(self, probant_id: str = 'test', scenario: str = 'A1', log_path: str = ''):
        # Configure logging
        self.iter_counter = 0
        self.logger = logging.getLogger('AllDataPointsTracker')
        self.logger.setLevel(logging.INFO)
        log_file = f'{log_path}{os.sep}all_datapoints_id_{probant_id}_scenario_{scenario}.log'
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)

        # Add handler if not already added
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)

    def log_datapoints_state(self, data: dict, moved_points):
        """
        Log movement of all datapoints
        """
        self.logger.info(
            f"{self.iter_counter}, {data},{moved_points}"
        )
        self.iter_counter += 1


class ModelTracker:
    def __init__(self, probant_id: str = 'test', scenario: str = 'A1', log_path: str = ''):
        # Configure logging
        self.iter_counter = 0
        self.logger = logging.getLogger('ModelTracker')
        self.logger.setLevel(logging.INFO)
        log_file = f'{log_path}{os.sep}model_scores_id_{probant_id}_scenario_{scenario}.log'
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)

        # Add handler if not already added
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)

    def log_model_results(self, message: str = ''):
        """
        Log movement of all datapoints
        """
        self.logger.info(
            message
        )
        self.iter_counter += 1


class LLMTracker:
    def __init__(self, probant_id: str = 'test', scenario: str = 'A1', log_path: str = ''):
        # Configure logging
        self.iter_counter = 0
        self.logger = logging.getLogger('LLMTracker')
        self.logger.setLevel(logging.INFO)
        log_file = f'{log_path}{os.sep}llm_suggestions_id_{probant_id}_scenario_{scenario}.log'
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)

        # Add handler if not already added
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)

    def log_request(self, model: str, goal, state: dict):
        """Log the latent space state that was sent to the LLM"""
        self.iter_counter += 1
        self.logger.info(
            f"REQUEST, {self.iter_counter}, {model}, goal={goal}, {json.dumps(state)}"
        )

    def log_response(self, model: str, raw_response: str):
        """Log the raw answer of the LLM"""
        self.logger.info(
            f"RESPONSE, {self.iter_counter}, {model}, {json.dumps(raw_response)}"
        )

    def log_suggestions(self, global_summary, suggestions):
        """Log the global assessment and per-pair suggestions shown to the participant"""
        payload = {
            'global': global_summary.as_log_dict(),
            'suggestions': [s.as_log_dict() for s in suggestions],
        }
        self.logger.info(
            f"SHOWN, {self.iter_counter}, {json.dumps(payload)}"
        )

    def log_dismissed(self, suggestion):
        """Log a suggestion the participant rejected"""
        self.logger.info(
            f"DISMISSED, {self.iter_counter}, {json.dumps(suggestion.as_log_dict())}"
        )

    def log_applied(self, suggestion):
        """Log a suggestion the participant applied to the 2D plot directly"""
        self.logger.info(
            f"APPLIED, {self.iter_counter}, {json.dumps(suggestion.as_log_dict())}"
        )

    def log_edited_on_plot(self, suggestion):
        """Log a suggestion the participant reshaped by dragging its arrow or circle"""
        self.logger.info(
            f"EDITED_ON_PLOT, {self.iter_counter}, {json.dumps(suggestion.as_log_dict())}"
        )

    def log_error(self, message: str):
        """Log a failed request or a failed execution"""
        self.logger.info(
            f"ERROR, {self.iter_counter}, {message}"
        )


def _to_jsonable(value):
    """numpy/torch values -> plain JSON types, so every event can be dumped."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, 'detach'):  # torch tensor
        return value.detach().cpu().tolist()
    return value


class SessionRecorder:
    """One structured, timestamped event stream per participant/scenario -
    everything needed to reconstruct the session afterwards (see
    ``tools/reconstruct_session.py``): settings, training phases, every shown
    latent space (saved as .npz next to the log), every modification with the
    2D state after it, every LLM request/answer/suggestion and what the
    participant did with it, and how long they spent on each space,
    phase and suggestion.

    Each line of ``session_events_id_<id>_scenario_<scenario>.jsonl`` is
    ``{"seq", "time" (ISO, ms), "elapsed_s" (since session start), "type", ...}``.
    Thread-safe: the training thread logs epochs while the UI logs drags.
    """

    def __init__(self, probant_id: str = 'test', scenario: str = 'A1', log_path: str = ''):
        import threading
        import time
        self._time = time
        self._lock = threading.Lock()
        self.log_path = log_path
        self.path = os.path.join(log_path, f'session_events_id_{probant_id}_scenario_{scenario}.jsonl')
        self.spaces_dir = os.path.join(log_path, 'spaces')
        self.audio_dir = os.path.join(log_path, 'audio')
        self._file = open(self.path, 'a', encoding='utf-8')
        self._start = time.monotonic()
        self.seq = 0
        self.round = 0  # index of the latent space currently shown
        # kind -> (key, start elapsed_s); kind -> key -> total seconds
        self._open_segments = {}
        self.totals = {}
        self.closed = False
        self.log('SESSION_START', probant_id=probant_id, scenario=scenario, log_path=log_path,
                 wall_clock=self._time.strftime('%Y-%m-%d %H:%M:%S %z'))

    def elapsed(self):
        return self._time.monotonic() - self._start

    def log(self, event_type: str, **payload):
        with self._lock:
            if self.closed:
                return None
            self.seq += 1
            now = self._time.time()
            record = {
                'seq': self.seq,
                'time': self._time.strftime('%Y-%m-%dT%H:%M:%S', self._time.localtime(now))
                + f'.{int(now * 1000) % 1000:03d}',
                'elapsed_s': round(self.elapsed(), 3),
                'round': self.round,
                'type': event_type,
            }
            record.update(_to_jsonable(payload))
            self._file.write(json.dumps(record) + '\n')
            self._file.flush()
            return self.seq

    # ------------------------------------------------------------- snapshots
    def save_space(self, **arrays):
        """Start a new round: store the arrays of the latent space just shown
        (2D points, high-dim features, labels, ...) and return the file path,
        relative to the log directory."""
        os.makedirs(self.spaces_dir, exist_ok=True)
        with self._lock:
            self.round += 1
            name = f'round_{self.round:03d}.npz'
        np.savez_compressed(os.path.join(self.spaces_dir, name),
                            **{k: np.asarray(v) for k, v in arrays.items() if v is not None})
        return os.path.join('spaces', name)

    def audio_path(self):
        os.makedirs(self.audio_dir, exist_ok=True)
        return os.path.join(self.audio_dir, f'dictation_{self.seq + 1:05d}.wav')

    # ----------------------------------------------------------- time spent
    def enter(self, kind: str, key):
        """Switch the open time segment of ``kind`` (e.g. 'space', 'phase',
        'suggestion') to ``key``; the previous one is closed and logged as
        TIME_SEGMENT. ``key=None`` just closes it."""
        key = None if key is None else str(key)
        current = self._open_segments.get(kind)
        if current is not None and current[0] == key:
            return
        now = self.elapsed()
        if current is not None:
            self._close_segment(kind, current, now)
        if key is None:
            self._open_segments.pop(kind, None)
        else:
            self._open_segments[kind] = (key, now)

    def _close_segment(self, kind, segment, now):
        key, start = segment
        duration = now - start
        self.totals.setdefault(kind, {})
        self.totals[kind][key] = self.totals[kind].get(key, 0.0) + duration
        self.log('TIME_SEGMENT', kind=kind, key=key, start_s=round(start, 3), end_s=round(now, 3),
                 duration_s=round(duration, 3))

    def time_in(self, kind: str, key=None):
        """Seconds spent so far in ``kind``/``key`` (the open segment counts),
        or in the open segment of ``kind`` when ``key`` is None."""
        current = self._open_segments.get(kind)
        if key is None:
            return self.elapsed() - current[1] if current else 0.0
        total = self.totals.get(kind, {}).get(str(key), 0.0)
        if current is not None and current[0] == str(key):
            total += self.elapsed() - current[1]
        return total

    def close(self):
        if self.closed:
            return
        now = self.elapsed()
        for kind, segment in list(self._open_segments.items()):
            self._close_segment(kind, segment, now)
        self._open_segments = {}
        totals = {kind: {k: round(v, 3) for k, v in keys.items()} for kind, keys in self.totals.items()}
        self.log('SESSION_END', duration_s=round(now, 3), time_spent=totals)
        with self._lock:
            self.closed = True
            self._file.close()
