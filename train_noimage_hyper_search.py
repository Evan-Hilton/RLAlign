import torch.nn as nn
import warnings
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env
from environment_noimage import pSCT_environment


# Check that the model parameters are defined correctly in accordance with Stable_Baselines
env = pSCT_environment()

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=".*observation.*dtype.*np.uint8.*"
    )
    warnings.filterwarnings(
        "ignore",
        message=".*observation space.*not in \\[0, 255\\].*"
    )
    check_env(env, warn=True)

def train_model(lr=1e-4, bs=64, n_epo=10, e_coef=0.001, path="default", model_name="v7.3.1", n_panl=1):
    env = make_vec_env(
        pSCT_environment,
        n_envs=8,
        vec_env_cls=SubprocVecEnv, # recommended in the documentation for speeding up training
        env_kwargs={"n_panels": n_panl}
    )
    #env = VecNormalize(env, norm_reward=True, norm_obs=False) # normalize the reward so that gradient updates aren't clipped too much
    # ultimately, env wraps VecNormalize, which wraps SupprocVecEnv, which wraps MirrorEnvImageDetect

    model = PPO(
        policy="MlpPolicy",
        env=env,
        device="cpu",
        policy_kwargs = dict(
            net_arch=dict(
                pi=[256, 256],         # policy MLP
                vf=[256, 256]          # value MLP
            ),
            activation_fn=nn.ReLU,
        ),
        learning_rate=lr,
        n_steps=1024,
        batch_size=bs,
        n_epochs=n_epo,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=e_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        normalize_advantage=True,
        tensorboard_log="./ppo_logs/v8/experiment4/" + path + "/",
    )

    version = model_name

    model.learn(total_timesteps=1_000_000)
    model.save("models/" + version)
    #env.save("envs/" + version)
    env.close()

# Train
if __name__ == "__main__":
    # 7.2
    # model_num = 1
    # model_num += 1
    # for lr in [3e-4]:
    #     train_model(lr=lr, path="learning_rate_exp", model_name="v7.2." + str(model_num))
    #     model_num += 1
    # for batch_size in [32, 128]:
    #     train_model(bs=batch_size, path="batch_size_exp", model_name="v7.2." + str(model_num))
    #     model_num += 1
    # for n_epochs in [5, 15]:
    #     train_model(n_epo=n_epochs, path="n_epochs_exp", model_name="v7.2." + str(model_num))
    #     model_num += 1
    # for ent_coef in [0.001, 0.02, 0.05]:
    #     train_model(e_coef=ent_coef, path="ent_coef_exp", model_name="v7.2." + str(model_num))
    #     model_num += 1
    for i in [6, 7, 8, 9, 10]:
        train_model(path="n_panels_exp", model_name="v8.4." + str(i), n_panl=i)