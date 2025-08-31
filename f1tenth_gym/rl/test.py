import math
import os
from absl.testing import flagsaver
from absl import logging as logger

# Import flags and entrypoint (package-relative)
from .main import main, FLAGS
from .stablebaseline3.rl import (
    ALGO_RECURRENT_PPO,
    ALGO_PURE_PURSUIT,
    ALGO_WALL_FOLLOW,
    ALGO_LATTICE,
)
from .stablebaseline3 import rl as sbrl
from stable_baselines3.common.vec_env import VecNormalize


def _parse_flags_once():
    # Ensure ABSEIL flags are parsed exactly once in the test session
    if not FLAGS.is_parsed():
        FLAGS(["pytest"])  # minimal argv to satisfy absl


def _basic_assertions(metrics: dict):
    assert isinstance(metrics, dict)
    
    assert "mean_reward" in metrics
    assert math.isfinite(float(metrics["mean_reward"]))

    assert "mean_episode_length" in metrics
    assert float(metrics["mean_episode_length"]) > 0


def test_main_eval_pure_pursuit_metrics():
    _parse_flags_once()
    with flagsaver.flagsaver(
        eval=True,
        use_wandb=False,
        algorithm=ALGO_PURE_PURSUIT,
        num_envs=1,
        num_eval_episodes=3,
        seed=123,
        map_index=63,
        lidar_scan_in_obs_mode="DOWNSAMPLED",
        include_params_in_obs=False,
        use_dr=False,
        use_il=False,
        render_in_eval=False,
        plot_in_eval=False,
        logging_level="ERROR",
    ):
        metrics = main(None)

    _basic_assertions(metrics)
    assert metrics["mean_reward"] > -500 # TODO: could be better
    assert metrics["mean_episode_length"] > 100


def test_main_eval_wall_follow_metrics():
    _parse_flags_once()
    # WallFollow expects lidar; use downsampled to keep obs small
    with flagsaver.flagsaver(
        eval=True,
        use_wandb=False,
        algorithm=ALGO_WALL_FOLLOW,
        num_envs=1,
        num_eval_episodes=3,
        seed=123,
        map_index=63,
        lidar_scan_in_obs_mode="DOWNSAMPLED",
        include_params_in_obs=False,
        use_dr=False,
        use_il=False,
        render_in_eval=False,
        plot_in_eval=False,
        logging_level="ERROR",
    ):
        metrics = main(None)

    _basic_assertions(metrics)
    assert metrics["mean_reward"] > 300.0
    assert metrics["mean_episode_length"] > 500


def test_main_eval_lattice_metrics():
    _parse_flags_once()
    # Lattice planner expects lidar for obstacle detection; use downsampled to keep obs small
    with flagsaver.flagsaver(
        eval=True,
        use_wandb=False,
        algorithm=ALGO_LATTICE,
        num_envs=1,
        num_eval_episodes=3,
        seed=123,
        map_index=63,
        lidar_scan_in_obs_mode="DOWNSAMPLED",
        include_params_in_obs=False,
        use_dr=False,
        use_il=False,
        render_in_eval=False,
        plot_in_eval=False,
        logging_level="ERROR",
    ):
        metrics = main(None)

    _basic_assertions(metrics)
    assert metrics["mean_reward"] > -500 # TODO: could be better
    assert metrics["mean_episode_length"] > 100



def test_pure_rl_training_improves_reward(tmp_path):
    _parse_flags_once()

    seed = 42
    map_index = 63

    # Temporary paths for models and stats
    temp_dir = tmp_path / "rl_train_test"
    temp_dir.mkdir(parents=True, exist_ok=True)
    baseline_model_path = os.path.join(str(temp_dir), "baseline_model")
    baseline_vecnorm_path = os.path.join(str(temp_dir), "baseline_vecnormalize.pkl")
    trained_model_path = os.path.join(str(temp_dir), "trained_model")
    trained_vecnorm_path = os.path.join(str(temp_dir), "trained_vecnormalize.pkl")

    # ------------------------------------------------------------------
    # Baseline evaluation (before training)
    # Train very briefly to get a minimally trained baseline
    with flagsaver.flagsaver(
        # Provided by user:
        use_il=False,
        use_dr=False,
        use_wandb=False,
        # All required flags:
        algorithm=ALGO_RECURRENT_PPO,
        feature_extractor="RESNET",
        include_params_in_obs=False,
        lidar_scan_in_obs_mode="DOWNSAMPLED",  # default in main.py
        num_envs=1,
        num_param_cmbs=1,
        total_timesteps=1,
        seed=seed,
        map_index=map_index,
        racing_mode=False,
        num_agents=1,
        eval=False,
        render_in_eval=False,
        plot_in_eval=False,             # default is True; keep False for test speed
        logging_level="INFO",         # default is INFO; quieter for tests
        wandb_mode="disabled",
        model_path=str(baseline_model_path),
        vecnorm_path=str(baseline_vecnorm_path),
    ):
        _ = main(None)

    with flagsaver.flagsaver(
        eval=True,
        use_wandb=False,
        algorithm=ALGO_RECURRENT_PPO,
        num_envs=1,
        num_eval_episodes=1,
        seed=seed,
        map_index=map_index,
        lidar_scan_in_obs_mode="DOWNSAMPLED",
        include_params_in_obs=False,
        use_dr=False,
        use_il=False,
        racing_mode=False,
        num_agents=1,
        render_in_eval=False,
        plot_in_eval=False,             # default True; keep False in tests
        logging_level="INFO",         # default INFO; quieter
        model_path=str(baseline_model_path),
        vecnorm_path=str(baseline_vecnorm_path),
    ):
        metrics_before = main(None)
    logger.info(f"Metrics before training: {metrics_before}")
    _basic_assertions(metrics_before)

    # ------------------------------------------------------------------
    # Train briefly and rely on callback saving best model + VecNormalize stats
    with flagsaver.flagsaver(
        # Provided by user:
        use_il=False,
        use_dr=False,
        use_wandb=False,
        # All required flags:
        algorithm=ALGO_RECURRENT_PPO,
        feature_extractor="RESNET",
        include_params_in_obs=False,
        lidar_scan_in_obs_mode="DOWNSAMPLED",
        num_envs=1,
        num_param_cmbs=1,
        total_timesteps=4000,
        seed=seed,
        map_index=map_index,
        racing_mode=False,
        num_agents=1,
        eval=False,
        render_in_eval=False,
        plot_in_eval=False,             # default True; keep False in tests
        logging_level="INFO",         # default INFO; quieter
        wandb_mode="disabled",
        model_path=str(trained_model_path),
        vecnorm_path=str(trained_vecnorm_path),
    ):
        _ = main(None)

    # ------------------------------------------------------------------
    # Post-training evaluation using the saved model and VecNormalize stats
    with flagsaver.flagsaver(
        eval=True,
        use_wandb=False,
        algorithm=ALGO_RECURRENT_PPO,
        num_envs=1,
        num_eval_episodes=1,
        seed=seed,
        map_index=map_index,
        lidar_scan_in_obs_mode="DOWNSAMPLED",
        include_params_in_obs=False,
        use_dr=False,
        use_il=False,
        racing_mode=False,
        num_agents=1,
        render_in_eval=False,
        plot_in_eval=False,             # default True; keep False in tests
        logging_level="INFO",         # default INFO; quieter
        model_path=str(trained_model_path),
        vecnorm_path=str(trained_vecnorm_path),
    ):
        metrics_after = main(None)
    logger.info(f"Metrics after training: {metrics_after}")
    _basic_assertions(metrics_after)

    # Expect improvement after training
    assert float(metrics_after["mean_reward"]) > float(metrics_before["mean_reward"]) 
