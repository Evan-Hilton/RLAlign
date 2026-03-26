import random
import numpy as np
import gymnasium as gym
from scipy.ndimage import gaussian_filter, maximum_filter, label, center_of_mass
import yaml
from gymnasium import spaces
from pSCT import pSCT

""" 
    This class represents a simplified simulation of the pSCT telescope
    for use in RL development.
"""
class pSCT_environment(gym.Env):

    def __init__(self,
                 n_panels = 2):
        
        # bookkeeping
        self.step_count = 0
        
        # panels
        self.P1s = [1111, 1112, 1113, 1114, 1211, 1212, 1213, 1214, 1311, 1312, 1313, 1314, 1411, 1412, 1413, 1414],
        self.n_panels = n_panels

        # the pSCT telescope
        self.telescope = pSCT()

        # Observation: single-channel image, unnormalized.
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, self.img_size, self.img_size),  # CHW for SB3 CNN
            dtype=np.int8,
        )

        # Action: (panel choice, rx, ry)
        # rx and ry are discretized to 25 unique values.
        self.action_space = spaces.MultiDiscrete([self.n_panels, 25, 25], dtype=np.int8)
    
    # =================================== API ===================================
    
    """
        -Run one timestep of the environment's dynamics using the agent actions.
        For each panel in the simulation, the agent chooses to update its position
        by providing a rotation rx, ry. 
        -Takes each rotation and updates the location of the panel's corresponding 
        image and renders it.
        -Calculates and returns the reward corresponding with the provided action
        
        params:
        action (numpy.ndarray with shape (1, 2)):        panel rotations. the nth panel is rotated rx, ry = action[n]

        returns:
        observation (numpy.ndarray with shape (img_size, img_size)):    the new environment state
        reward (Float):                                                 how beneficial the action was
        terminated (bool):                                              whether the agent reaches the terminal state
        truncated (bool):                                               whether the agent reaches a state that should cause the simulation to stop early
        info (dict):                                                    contains debugging information
    """
    def step(self, action):

        self.telescope.rotate_panel(self.P1s[action[0]], )
        return observation.astype(np.float32)[None, :, :], reward, terminated, truncated, info

    """
        Resets the environment to an initial internal state, returning an initial observation and info.
        Places each panel's image to a new random location.

        params:
        seed (None):                Satisfies the API but we introduce our own PRNG (numpy random number generator) so keep as None
        options (optional dict):    optional debug information to include about how the environment is reset. RNG seed for example.

        returns:
        observation (numpy.ndarray with shape (n_panels, 2)):   the new environment state
        info (dict):                                            contains debugging information. Should be the same information as returned from step()
    """
    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        self.telescope.set_random_rotations()

        return observation.astype(np.float32)[None, :, :], info
    
    # ============================== Helper Functions ==============================