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
import sk
import stablebaseline3
from rl_env import F110GymWrapper
from stable_baselines3.common.utils import set_random_seed
import stablebaseline3.rl
from utils.Track import Track
from utils import utils
from matplotlib import pyplot as plt
from PIL import Image
import wandb

FLAGS = flags.FLAGS

flags.DEFINE_boolean("racing_mode", False, "Enable racing mode with two cars")
flags.DEFINE_integer("num_agents", 1, "Number of agents")
flags.DEFINE_boolean("use_il", True, "Whether to use imitation learning before RL training")
flags.DEFINE_enum("il_policy", "PURE_PURSUIT", ["WALL_FOLLOW", "PURE_PURSUIT", "LATTICE"], "Policy to use for imitation learning.")
flags.DEFINE_integer("num_envs", 24, "Number of parallel environments for training")
flags.DEFINE_boolean("use_dr", True, "Apply domain randomization during training")
flags.DEFINE_integer("num_param_cmbs", 24, "Number of parameter combinations to use for domain randomization")
flags.DEFINE_boolean("include_params_in_obs", False, "Include environment parameters in observations for contextual RL")
flags.DEFINE_enum("lidar_scan_in_obs_mode", "DOWNSAMPLED", ["NONE", "FULL", "DOWNSAMPLED"], "Lidar scan mode in observations: NONE (no lidar), FULL (1080 points), DOWNSAMPLED (108 points - 1 in 10)")

flags.DEFINE_boolean("eval", False, "Run only evaluation (no training)")
flags.DEFINE_integer("num_eval_episodes", 1, "Number of episodes to evaluate")
flags.DEFINE_boolean("render_in_eval", True, "Render in evaluation")
flags.DEFINE_boolean("plot_in_eval", True, "Plot in evaluation")
flags.DEFINE_integer("seed", 42, "Random seed for reproducibility")
flags.DEFINE_integer("map_index", 63, "Index of the map to use")
flags.DEFINE_string("logging_level", "INFO", "Logging level")
flags.DEFINE_string("model_path", "./logs/best_model/best_model.zip", "Path to the model to evaluate")
flags.DEFINE_string("vecnorm_path", "./logs/best_model/vec_normalize.pkl", "Path to the VecNormalize statistics file. If None, will try to infer from model_path.")
flags.DEFINE_enum("algorithm", "RECURRENT_PPO", ["SAC", "PPO", "RECURRENT_PPO", "DDPG", "TD3", "WALL_FOLLOW", "PURE_PURSUIT", "LATTICE"], "Algorithm used")
flags.DEFINE_enum("feature_extractor", "RESNET", ["MLP", "RESNET", "FILM", "TRANSFORMER", "MOE"], "Feature extractor architecture to use")

# WandB flags
flags.DEFINE_boolean("use_wandb", True, "Whether to use Weights & Biases for experiment tracking")
flags.DEFINE_string("wandb_run_name", None, "WandB run name. Required when use_wandb=True.")
flags.DEFINE_string("wandb_notes", "", "Notes for the WandB run")
flags.DEFINE_enum("wandb_mode", "online", ["online", "offline", "disabled"], "WandB mode")

os.environ['F110GYM_PLOT_SCALE'] = str(60.)


def setup_wandb_config():
    """Setup WandB configuration from FLAGS"""
    config = {
        # Training hyperparameters
        "algorithm": FLAGS.algorithm,
        "feature_extractor": FLAGS.feature_extractor,
        "num_envs": FLAGS.num_envs,
        "num_param_cmbs": FLAGS.num_param_cmbs,
        "seed": FLAGS.seed,
        "map_index": FLAGS.map_index,
        
        # Environment settings
        "racing_mode": FLAGS.racing_mode,
        "num_agents": FLAGS.num_agents,
        "use_dr": FLAGS.use_dr,
        "include_params_in_obs": FLAGS.include_params_in_obs,
        "lidar_scan_in_obs_mode": FLAGS.lidar_scan_in_obs_mode,
        
        # Imitation learning
        "use_il": FLAGS.use_il,
        "il_policy": FLAGS.il_policy,
        
        # Evaluation
        "num_eval_episodes": FLAGS.num_eval_episodes,
        
        # System
        "logging_level": FLAGS.logging_level,
    }
    return config


def initialize_wandb():
    """Initialize WandB with configuration"""
    if not FLAGS.use_wandb or FLAGS.wandb_mode == "disabled":
        return None
    
    # Require run name when wandb is enabled
    if FLAGS.wandb_run_name is None:
        raise ValueError("wandb_run_name is required when use_wandb=True. Please specify --wandb_run_name.")
    
    # Initialize wandb
    run = wandb.init(
        project="f1tenth-rl",
        entity=None,  # Use default entity
        name=FLAGS.wandb_run_name,
        config=setup_wandb_config(),
        notes=FLAGS.wandb_notes,
        mode=FLAGS.wandb_mode,
        sync_tensorboard=True,  # Sync tensorboard logs if available
    )
    
    logging.info(f"Initialized WandB run: {run.name} (ID: {run.id})")
    logging.info(f"WandB URL: {run.url}")
    
    return run

def wandb_run_main(argv):
    # Initialize WandB
    wandb_run = None
    if FLAGS.use_wandb and not FLAGS.eval:
        wandb_run = initialize_wandb()
    
    try:
        main(argv)
    finally:
        # Clean up WandB
        if wandb_run:
            wandb.finish()


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
    # map_name = map_info[config.map_ind][1].split('/')[0]
    base_env_kwargs = {
        'waypoints': track.waypoints,
        'map_path': config.map_dir + map_info[config.map_ind][1].split('.')[0],
        # 'map_path': os.path.join(config.map_dir, map_name, map_name + "_map"),
        'num_agents': FLAGS.num_agents,
        'track': track,
        'include_params_in_obs': FLAGS.include_params_in_obs,
        'racing_mode': FLAGS.racing_mode,
        'lidar_scan_in_obs_mode': FLAGS.lidar_scan_in_obs_mode
        # seed will be handled by make_vec_env/make_env
        # Other DR parameters will be added in rl.py if enabled
    }

    # Branch based on mode (train or evaluate)
    # Create a vectorized environment
    # We use the same env creation function as training for consistency
    vec_env = stablebaseline3.rl.create_vec_env(
        env_kwargs=base_env_kwargs,
        seed=FLAGS.seed
    )
    if FLAGS.eval:
        stablebaseline3.rl.evaluate(eval_env=vec_env)
    else:
        # Train with vectorized environments
        stablebaseline3.rl.train(env=vec_env, seed=FLAGS.seed)


if __name__ == "__main__":
    app.run(wandb_run_main)
