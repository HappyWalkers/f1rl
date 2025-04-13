import gym
import torch
import time
import numpy as np
import random
import torch.distributions as D
import yaml
import json
import os
from argparse import Namespace
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.models.torch import Model
from absl import app
from absl import flags
from absl import logging
from gap_follow_agent import Gap_follower
import sk
import stablebaseline3
from rl_env import F110GymWrapper
from stable_baselines3.common.utils import set_random_seed
import stablebaseline3.rl
from utils.Track import Track
from utils import utils
from matplotlib import pyplot as plt
from PIL import Image

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 42, "Random seed for reproducibility")
flags.DEFINE_integer("map_index", 63, "Index of the map to use")
flags.DEFINE_integer("num_agents", 1, "Number of agents")
flags.DEFINE_string("logging_level", "INFO", "Logging level")
flags.DEFINE_boolean("eval_only", False, "Run only evaluation (no training)")
flags.DEFINE_string("model_path", "./logs/best_model/best_model.zip", "Path to the model to evaluate")
flags.DEFINE_string("algorithm", "SAC", "Algorithm used (SAC, PPO, DDPG, TD3, WALL_FOLLOW, PURE_PURSUIT)")
flags.DEFINE_integer("num_eval_episodes", 5, "Number of episodes to evaluate")
flags.DEFINE_boolean("use_imitation_learning", True, "Whether to use imitation learning before RL training")
flags.DEFINE_enum("imitation_policy", "PURE_PURSUIT", ["WALL_FOLLOW", "PURE_PURSUIT"],
                  "Policy to use for imitation learning.")
flags.DEFINE_integer("num_envs", 4, "Number of parallel environments for training")
flags.DEFINE_boolean("use_domain_randomization", False, "Apply domain randomization during training")


os.environ['F110GYM_PLOT_SCALE'] = str(60.)

    
def test_env(env):
    gap_follower = Gap_follower()

    for ep_i in range(20):
        obs = env.reset()
        done = False
        i = 0
        min_obs = []
        while not (done):
            i += 1
            env.render()
            steer = 0
            speed = 1
            ##### use policy ######
            # breakpoint()
            action, metric = gap_follower.planning(obs)
            obs, step_reward, done, info = env.step(action)
        print('finish one episode')


def main(argv):
    # set logging level
    logging.set_verbosity(getattr(logging, FLAGS.logging_level))
    
    # Set seeds for reproducibility
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    set_random_seed(FLAGS.seed)

    # Load track
    class Config(utils.ConfigYAML):
        sim_time_step = 0.1
        map_dir = './f1tenth_racetracks/'
        use_blank_map = True
        map_ext = '.png'
        map_scale = 1
        map_ind = FLAGS.map_index
    config = Config()
    map_info = np.genfromtxt(config.map_dir + 'map_info.txt', delimiter='|', dtype='str')
    track, config = Track.load_map(config.map_dir, map_info, config.map_ind, config, scale=config.map_scale, downsample_step=1)

    # Create the environment wrapper
    # Base environment arguments (without domain randomization)
    base_env_kwargs = {
        'waypoints': track.waypoints,
        'map_path': config.map_dir + map_info[config.map_ind][1].split('.')[0],
        'num_agents': FLAGS.num_agents,
        'track': track,
        # seed will be handled by make_vec_env/make_env
        # Other DR parameters will be added in rl.py if enabled
    }

    # Branch based on mode (train or evaluate)
    # Create a vectorized environment
    # We use the same env creation function as training for consistency
    vec_env = stablebaseline3.rl.create_vec_env(
        env_kwargs=base_env_kwargs,
        seed=FLAGS.seed,
        num_envs=FLAGS.num_envs,
        use_domain_randomization=FLAGS.use_domain_randomization
    )
    if FLAGS.eval_only:
        stablebaseline3.rl.evaluate(
            eval_env=vec_env,
            model_path=FLAGS.model_path,
            algorithm=FLAGS.algorithm,
            num_episodes=FLAGS.num_eval_episodes,
        )
    else:
        # Train with vectorized environments
        model = stablebaseline3.rl.train(
            env=vec_env,
            seed=FLAGS.seed,
            num_envs=FLAGS.num_envs,
            use_domain_randomization=FLAGS.use_domain_randomization,
            use_imitation_learning=FLAGS.use_imitation_learning,
            imitation_policy_type=FLAGS.imitation_policy,
            algorithm=FLAGS.algorithm
        )


if __name__ == "__main__":
    app.run(main)
