import math
from absl.testing import flagsaver

# Import flags and entrypoint (package-relative)
from .main import main, FLAGS
from .stablebaseline3.rl import (
    ALGO_PURE_PURSUIT,
    ALGO_WALL_FOLLOW,
    ALGO_LATTICE,
)


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
    assert metrics["mean_reward"] > -500
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
    assert metrics["mean_reward"] > -500
    assert metrics["mean_episode_length"] > 100


