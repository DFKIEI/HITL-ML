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

    def log_error(self, message: str):
        """Log a failed request or a failed execution"""
        self.logger.info(
            f"ERROR, {self.iter_counter}, {message}"
        )
