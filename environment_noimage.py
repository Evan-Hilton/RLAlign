import random
import numpy as np
import gymnasium as gym
import yaml
from gymnasium import spaces
from pSCT import pSCT
from image_analyzer import image_analyzer

""" 
    This class represents how the agent interacts with the pSCT.
    This class should be used as the environment class for training 
    an agent to align the pSCT mirrors.

    This class uses a simplified version of the pSCT that does not
    have mirrors but insead interacts directly with the true position
    of the centroids. This is done mainly to prove that the method will
    work for the image version of the pSCT and to drastically increase
    training efficiency, since there is a large bottleneck when creating
    images.
"""
class pSCT_environment(gym.Env):

    def __init__(self,
                 n_panels = 2,
                 memory_time = 1 # how many steps backward in time the agent can see
                 ):
        
        # bookkeeping
        self.step_count = 0
        self.max_steps = 512 # the maximum amount of time the agent is allowed to move for
        self.prev_cost = 0
        
        # panels
        self.P1s = [1111, 1112, 1113, 1114, 1211, 1212, 1213, 1214, 1311, 1312, 1313, 1314, 1411, 1412, 1413, 1414]
        self.n_panels = n_panels
        # discretize the action rotations into self.action_quant amount of discrete values
        # note that action_quant should be odd so that (action_quant - 1) / 2 maps to rotation = 0 (allow the agent to not move a panel)
        self.action_quant: int = 25 # if this is 25, then the agent can choose between 25 values to move the panels by. 0 and 25 represent maximum motion

        # the pSCT telescope
        self.telescope = pSCT(n_panels=self.n_panels)

        # image information
        self.memory_time = memory_time # m frames of memory in the observation
        self.memory = None # np array with shape: (2 * self.n_panels, self.memory_time)

        # Observation: each true centroid location given by (x1, y1, x2, y2, ..., xn, yn). this is stacked
        # for each time step in the past that the agent has access to (see memory_time). this vector is then
        # flattened to provide a single array to pass as the observation.
        #
        # -1 is the far left of the screen, 1 is the far right of the screen
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.n_panels * 2 * self.memory_time),
            dtype=np.float32,
        )

        # Action: (panel choice, rx, ry)
        # panel choice is a number between 0 and n_panels - 1. 
        # rx and ry are discretized to 25 unique values.
        self.action_space = spaces.MultiDiscrete([self.n_panels, self.action_quant, self.action_quant], dtype=np.uint8)
    
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
        # normalize actions given by the network. map [0, action_quant] -> [-1, 1]
        rotation_x = action[1] - ((self.action_quant - 1) / 2)          # action values surround zero
        rotation_x = rotation_x * 1.0 / ((self.action_quant - 1) / 2)   # action scaled between -1 and 1
        rotation_y = action[2] - ((self.action_quant - 1) / 2)          # action values surround zero
        rotation_y = rotation_y * 1.0 / ((self.action_quant - 1) / 2)   # action scaled between -1 and 1

        # rotate the panel
        self.telescope.rotate_panel(self.P1s[action[0]], rotation_x, rotation_y)

        # update memory - give the new observation to the memory
        single_step_obs = self.telescope.get_normalized_centroid_fp_coords_to_screen()
        self.increment_memory(single_step_obs)

        # calculate reward and reward shaping
        cost = self.cost_from_detected_centroids(self.telescope.true_centroids)
        reward = -cost
        improve = self.prev_cost - cost
        reward += 0.5 * improve

        terminated = False
        if self.telescope.all_centroids_at_center():
            reward += 10
            terminated = True
        if self.telescope.any_centroid_outside_image():
            reward -= 35 # truncation penalty should be 5x-20x worse than average reward (currently at ~-0.4)
            terminated = True
        reward -= 0.1 # time penalty. incentivices fast solutions
        self.prev_cost = cost

        # bookkeeping
        truncated = self.step_count >= self.max_steps
        self.step_count += 1
        info = {}

        return self.memory.flatten(), reward, terminated, truncated, info

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
        # bookkeeping
        self.step_count = 0
        self.telescope.set_random_rotations()

        # set up the observation
        img = self.telescope.get_image(self.P1s[:self.n_panels])
        self.memory = np.zeros((self.memory_time, self.telescope.img_size, self.telescope.img_size), dtype=np.uint8)
        self.memory[:] = img

        # set up reward shaping
        detected_centroids = image_analyzer.get_centroid_locations(self.memory[0])
        self.prev_cost = self.cost_from_detected_centroids(detected_centroids)

        return self.memory, {}
    
    # ============================== Helper Functions ==============================

    def cost_from_detected_centroids(self, detected_fp_coords):
        d = detected_fp_coords - self.telescope.center[None, :]
        mean_r2 = float(np.mean(np.sqrt(np.sum(d**2, axis=1))))

        cost = mean_r2
        return cost

    def increment_memory(self, img):
        self.memory[1:] = self.memory[:-1] # shift all frames forward (ignoring first fram and overriding last frame)
        self.memory[0] = img