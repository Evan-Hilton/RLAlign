# this script is simply set up to continue training an already trained model. It loads
# the model and then continues training

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from environment_noimage import pSCT_environment
from stable_baselines3 import PPO

# load this model and env
version = "v7.3.6"
time_steps = 1_000_000

path_to_saved_model = "models/" + version
path_to_saved_env   = "envs/" + version

# save to this model and env
path_to_new_model   = "models/" + version
path_to_new_env     = "envs/" + version

if __name__ == '__main__':

    env = make_vec_env(
        pSCT_environment,
        n_envs=8,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"n_panels":6}
    )
    #env = VecNormalize.load(path_to_saved_env, env)
    env.training = True
    #env.norm_reward = True   # match what was trained with

    env.reset()

    model = PPO.load(
        path_to_saved_model,
        env=env,   # VERY IMPORTANT
        device="cpu"
    )

    model.learn(total_timesteps=time_steps, reset_num_timesteps=False)
    model.save(path_to_new_model)
    #env.save(path_to_new_env)