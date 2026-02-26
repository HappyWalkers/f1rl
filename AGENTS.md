# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is an F1Tenth Reinforcement Learning project for autonomous racing. The main code lives in `f1tenth_gym/rl/`. It uses Python 3.10, PyTorch, Stable Baselines3, and a custom F1Tenth gym simulator.

### Python Environment

- A Python 3.10 virtualenv is at `/workspace/.venv`. Activate it with `source /workspace/.venv/bin/activate`.
- The `f1tenth_gym` package is installed in editable mode (`pip install -e .` from `f1tenth_gym/`).

### Key Gotchas

- **gym version**: The codebase requires `gym==0.19.0`. Modern pip/setuptools cannot build this version from source due to an invalid version specifier (`opencv-python>=3.`) in its `extras_require`. The workaround is to download the source, fix the specifier to `opencv-python>=3.0`, and install with `--no-deps`. The update script handles this automatically.
- **jax/jaxlib**: The pinned versions in `requirements.txt` (0.4.13) are unavailable on this platform. `jax==0.4.30` / `jaxlib==0.4.30` work as replacements. JAX is only used in utility files (`utils/Track.py`, `utils/jax_utils.py`, `utils/trainer_jax.py`), not in the core RL pipeline.
- **numpy must stay at 1.24.4**: Many packages (numba, matplotlib, scikit-learn) require numpy<2. Installing newer JAX may upgrade numpy; always restore `numpy==1.24.4` afterwards.
- **No GPU in Cloud VM**: PyTorch runs on CPU. Training is slow but functional. Expert policy evaluations run quickly (~seconds).

### Running Tests

From `f1tenth_gym/`:
```bash
source /workspace/.venv/bin/activate
python -m pytest rl/test.py -v -k "not mpc"  # MPC test has a known reward assertion failure
```

The eval tests (`test_main_eval_wall_follow_metrics`, `test_main_eval_pure_pursuit_metrics`, `test_main_eval_lattice_metrics`) run in ~seconds each. The training tests (`test_pure_rl_training`, `test_il_rl_training`, `test_il_rl_dr_training`) take minutes+ on CPU.

### Running the Application

See `readme.md` for training/evaluation CLI commands. Quick examples:

```bash
cd /workspace/f1tenth_gym
source /workspace/.venv/bin/activate

# Expert policy evaluation (fast, no trained model needed)
python -m rl.main --eval=true --algorithm=wall_follow --num_envs=1 --use_wandb=false --render_in_eval=false

# Short RL training (CPU, no IL for speed)
python -m rl.main --use_wandb=false --num_envs=2 --total_timesteps=2048 --use_il=false --use_dr=false --include_params_in_obs=false
```

### Lint

No dedicated linter config (pylint/ruff/flake8) is set up in the repo. Use standard Python linting tools if needed.
