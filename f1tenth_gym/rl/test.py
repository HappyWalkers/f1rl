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
    ALGO_MPC,
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


def _run_training_and_evaluation_test(
    tmp_path,
    test_name: str,
    train_config: dict,
    eval_config: dict,
    expected_min_reward: float,
    log_message_suffix: str
):
    """
    Helper function to run training and evaluation tests with different configurations.
    
    Args:
        tmp_path: pytest tmp_path fixture
        test_name: Name for the test directory
        train_config: Dictionary of training configuration flags
        eval_config: Dictionary of evaluation configuration flags  
        expected_min_reward: Minimum expected reward after training
        log_message_suffix: Suffix for log message (e.g., "training", "IL + RL training")
    """
    _parse_flags_once()

    seed = 42
    map_index = 63

    # Temporary paths for models and stats
    temp_dir = tmp_path / test_name
    temp_dir.mkdir(parents=True, exist_ok=True)
    trained_model_path = os.path.join(str(temp_dir), "trained_model")
    trained_vecnorm_path = os.path.join(str(temp_dir), "trained_vecnormalize.pkl")

    # Base configuration that's common to all tests
    base_config = {
        "use_wandb": False,
        "algorithm": ALGO_RECURRENT_PPO,
        "feature_extractor": "RESNET",
        "lidar_scan_in_obs_mode": "DOWNSAMPLED",
        "seed": seed,
        "map_index": map_index,
        "racing_mode": False,
        "num_agents": 1,
        "eval": False,
        "render_in_eval": False,
        "plot_in_eval": False,
        "logging_level": "INFO",
        "wandb_mode": "disabled",
        "model_path": str(trained_model_path),
        "vecnorm_path": str(trained_vecnorm_path),
        "total_timesteps": int(1e6),
    }

    # ------------------------------------------------------------------
    # Training phase
    training_flags = {**base_config, **train_config}
    
    with flagsaver.flagsaver(**training_flags):
        _ = main(None)

    # ------------------------------------------------------------------
    # Evaluation phase
    eval_base_config = {
        "eval": True,
        "use_wandb": False,
        "algorithm": ALGO_RECURRENT_PPO,
        "num_eval_episodes": 10,
        "seed": seed,
        "map_index": map_index,
        "lidar_scan_in_obs_mode": "DOWNSAMPLED",
        "racing_mode": False,
        "num_agents": 1,
        "render_in_eval": False,
        "plot_in_eval": False,
        "logging_level": "INFO",
        "model_path": str(trained_model_path),
        "vecnorm_path": str(trained_vecnorm_path),
    }
    
    evaluation_flags = {**eval_base_config, **eval_config}
    
    with flagsaver.flagsaver(**evaluation_flags):
        metrics_after = main(None)
    
    logger.info(f"Metrics after {log_message_suffix}: {metrics_after}")
    _basic_assertions(metrics_after)

    # Check performance expectation
    assert float(metrics_after["mean_reward"]) > expected_min_reward

    return metrics_after


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


def test_main_eval_mpc_metrics():
    _parse_flags_once()
    # MPC controller uses track geometry and kinematics; use downsampled lidar to keep obs small
    with flagsaver.flagsaver(
        eval=True,
        use_wandb=False,
        algorithm=ALGO_MPC,
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
    assert metrics["mean_reward"] > 0  # MPC should perform reasonably well
    assert metrics["mean_episode_length"] > 200



def test_pure_rl_training(tmp_path):
    train_config = {
        "use_il": False,
        "use_dr": False,
        "include_params_in_obs": False,
    }
    
    eval_config = {
        "num_envs": 1,
        "use_dr": False,
        "include_params_in_obs": False,
    }
    
    _run_training_and_evaluation_test(
        tmp_path=tmp_path,
        test_name="rl_train_test",
        train_config=train_config,
        eval_config=eval_config,
        expected_min_reward=0,
        log_message_suffix="RL training"
    )


def test_il_rl_training(tmp_path):
    train_config = {
        "use_il": True,
        "il_policy": "WALL_FOLLOW",
        "il_num_transitions": 1e6,
        "use_dr": False,
        "include_params_in_obs": False,
    }
    
    eval_config = {
        "num_envs": 1,
        "use_dr": False,
        "include_params_in_obs": False,
    }
    
    _run_training_and_evaluation_test(
        tmp_path=tmp_path,
        test_name="il_rl_train_test",
        train_config=train_config,
        eval_config=eval_config,
        expected_min_reward=2500,
        log_message_suffix="IL + RL training"
    )


def test_il_rl_dr_training(tmp_path):
    train_config = {
        "use_il": True,
        "il_policy": "WALL_FOLLOW",
        "il_num_transitions": int(1e6),
        "use_dr": True,
        "include_params_in_obs": True,  # Enable contextual RL with DR
    }
    
    eval_config = {
        "use_dr": True,  # Keep DR enabled for eval
        "include_params_in_obs": True,  # Match training setup
    }
    
    _run_training_and_evaluation_test(
        tmp_path=tmp_path,
        test_name="il_rl_dr_train_test",
        train_config=train_config,
        eval_config=eval_config,
        expected_min_reward=0,
        log_message_suffix="IL + RL + DR training"
    )

