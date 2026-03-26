from typing import Tuple
import numpy as np

"""
    This class defines a simulation of a pSCT telescope object.
    A pSCT telescope represents the current state of the telescope.
"""
class pSCT:
    """
        Default pSCT constructor.
    """
    def __init__(self):
        pass

    """
        Returns the image currently seen by the pSCT camera of an on-axis star.
        If normalized is True, the image returned is in the range of [-1, 1],
        Otherwise, the image returned is in the range of [0, 255].
        The returned value is a numpy array with shape (img_size, img_size).
    """
    def get_image(self, normalized: bool = True):
        pass

    """
        Retrieves the true focal plane coordinates of each centroid for 
        this telescope.
    """
    def get_true_centroids(self):
        pass

    """
        Rotate the panel specified by 'panel_id' by 'rotation' amount.
        panel_id: integer representation of the panel's id. ex: 1121
        rotation: the amount to rotate the panel by. must have shape (1, 2).

        Throws: ValueError: shape mismatch if rotation is not (1, 2)
    """
    def rotate_panel(self, panel_id: int, rotation: np.ndarray):
        pass

    """
        Randomly misaligns every panel in the telescope
    """
    def set_random_rotations(self):
        pass