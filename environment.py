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
        
        self.n_panels = n_panels

        # the pSCT telescope
        self.telescope = pSCT()

        self.step_count = 0

        # action / observation
        # Observation: single-channel image
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1, self.img_size, self.img_size),  # CHW for SB3 CNN
            dtype=np.float32,
        )

        # Action: (panel choice, rx, ry)
        # rx and ry are discretized to 25 unique values.
        self.action_space = spaces.MultiDiscrete([17, 25, 25], dtype=np.int8)
    # =================================== API ===================================
    """
    Please note that render() and close() are not declared. They are optionally a part of the API.

    render() is not defined because the observation returned by step() and reset()
    are inherently visual (an image). Any rendering or visualization that a user
    might want should be done in a seperate class. (It is done in this 
    project in visualize.py and visualizeDebug.py)
    
    close() is not defined because there is nothing to close. close() would need
    to be defined if we were doing any threading, using the internet, or using
    some other data gathering that requires having a connection.
    """
    
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