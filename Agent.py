from stable_baselines3 import PPO
from environment import pSCT_environment
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

"""
An Agent contains five things:
    - a panel ID which represents the ID of the panel that the agent controls
    - the reward gained at each time step in the current eval episode
    - the trained RL model
    - the environment the model was trained in
    - the environment's current state (what the observation is)
An Agent is used for evaluation purposes only (NO TRAINING!)
"""
class Agent():
    def __init__(self, path_to_model: str, path_to_env: str, id: int):
        # the following five instance variables are API
        self.ID = id

        self.rewards = []

        self.model = PPO.load(path_to_model)

        self.env = make_vec_env(pSCT_environment, n_envs=1)
        self.env = VecNormalize.load(path_to_env, self.env)
        self.env.training = False
        self.env.norm_reward = False

        self.obs = self.env.reset()

        # the following instance variables are internal tracking variables
        self.playing = True
    
    def step(self):
        if(self.playing):
            action, _ = self.model.predict(self.obs, deterministic=True)
            self.obs, reward, terminated, truncated, _ = self.env.step(action)
            self.rewards.append(reward)
            if (terminated or truncated):
                self.playing = False
        return self.obs

    def reset(self):
        self.obs = self.env.reset()
        self.rewards = []
        self.playing = True