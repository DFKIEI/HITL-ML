import logging
import numpy as np
from typing import Optional


class PointTracker:
    def __init__(self, log_file: str = 'point_movements.log'):
        # Configure logging
        self.logger = logging.getLogger('PointTracker')
        self.logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)

        # Add handler if not already added
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)

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
            f"Center movement - Class {class_label}: "
            f"From {old_center} to {new_center}"
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
        message = (
            f"Point movement - Index {point_index}: "
            f"From {old_position} to {new_position}"
        )
        if class_label is not None:
            message += f" (Class {class_label})"

        self.logger.info(message)

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
            f"Class points reset - Class {class_label}: "
            f"All {num_points} points moved to center {center}"
        )

    def undo_last_step(self):
        """
        Log undoing of last step

        """
        self.logger.info(
            f"----------- Undo Last step ----------- "
        )
