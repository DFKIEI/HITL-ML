import logging
import numpy as np
from typing import Optional


class PointTracker:
    def __init__(self, probant_id: str = 'test', scenario: str = 'A1'):
        # Configure logging
        self.logger = logging.getLogger('PointTracker')
        self.logger.setLevel(logging.INFO)
        log_file = f'point_movements_id_{probant_id}_scenario_{scenario}.log'
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

    def undo_last_step(self):
        """
        Log undoing of last step

        """
        self.logger.info(
            f"----------- Undo Last step ----------- "
        )
