# F1Tenth Reinforcement Learning — Contribution Guide

This document captures the detailed standards, conventions, and expectations for this repository.

## Scope and Goals

- Build robust, high-performance autonomous racing agents suitable for real-world F1Tenth vehicles.
- Emphasize sim-to-real transfer via domain randomization and robust training.
- Maintain clarity, reproducibility, and operational reliability.

## Technology Stack

- Python 3.10+
- Stable Baselines3 (SB3) with custom extensions
- PyTorch 2.0+
- Gymnasium, F1Tenth Gym (F110)
- Simulator: F1Tenth racing simulator with dynamic vehicle models

## Non‑Negotiable Standards

- Type hints everywhere (functions, classes, public APIs)

## Environment and Project Mandates

- Always extend `F110GymWrapper` for custom envs
- Support both single‑agent and multi‑agent (racing) modes
- Use vectorized environments for training (e.g., `create_vec_env`)
- Prefer recurrent policies (LSTM) to capture temporal dependencies
- Apply domain randomization during training for sim‑to‑real

## RL Training Pipeline

- Algorithms: PPO, SAC, and recurrent PPO (preferred for temporal tasks)
- Vectorized envs: maximize parallelism while staying memory‑safe

## Expert Policies and Planners

Provide baselines and/or hybrid controllers:

- Pure Pursuit: waypoint following with lookahead distance
- Wall Following: LiDAR‑based reactive control
- Lattice Planner: sampling‑based trajectories
- Gap Following: reactive obstacle avoidance for racing

These are useful for imitation learning and for safety fallbacks.

## Sim‑to‑Real Transfer Guidelines

- Parameter randomization: mass, inertia, tire friction, drag, actuator delays
- Sensor modeling: LiDAR and state estimation noise; dropouts; bias
- Randomize track friction patches and lighting if image sensors are used
- Train with ranges broad enough to cover expected real‑world variation

## Configuration Management

- Log all hyperparameters, seeds, and environment parameters
- Keep default configurations conservative and reproducible

## Data, Observation Processing, and Feature Extractors

- Keep observation pre‑processing explicit and documented
- For LiDAR: support full, downsampled, and none (for ablations)

## Performance and Memory

- Batch operations in vectorized envs; avoid per‑step Python loops
- Profile and remove bottlenecks; prefer tensor ops over Python control flow

## Training Stability

- Normalize observations and rewards where beneficial

## Reproducibility

- Set all random seeds (Python, NumPy, PyTorch, envs)
- Use deterministic algorithms where possible
- Log versions of all dependencies
- Save configs and code snapshot alongside checkpoints

## Experiment Management

- Save model checkpoints regularly with metadata (step, env params, seed)
- Log training curves/metrics; support TensorBoard
- Implement evaluation protocols (fixed seeds + randomized sweeps)

## File Organization

```
f1tenth_gym/rl/
├── main.py                  # Main training/evaluation script
├── rl_env.py               # Environment wrapper
├── stablebaseline3/        # Custom SB3 extensions
│   ├── rl.py              # Training pipeline
│   ├── feature_extractor.py  # Neural network architectures
│   └── rl_node.py         # ROS integration (if applicable)
├── utils/
│   ├── Track.py           # Track handling
│   ├── utils.py           # General utilities
│   └── torch_utils.py     # PyTorch utilities
├── pure_pursuit.py         # Pure pursuit controller
├── wall_follow.py          # Wall following controller
├── lattice_planner.py      # Lattice planner
└── gap_follow_agent.py     # Gap following agent
```

## Code Review Checklist

- [ ] All functions have proper type annotations
- [ ] Logging for important events and errors
- [ ] Performance considerations addressed (batching, vectorization)
- [ ] Memory usage optimized
- [ ] Reproducibility ensured (seeds, deterministic ops)
- [ ] Configuration and hyperparameters logged
- [ ] Code follows project conventions and directory layout

## Security and Deployment

- Validate trained models before deployment; sanity tests on real hardware
- Monitor system performance during deployment; log runtime anomalies

---

If a choice is ambiguous, prioritize reliability and readability. When in doubt, add a small test or example and document assumptions here. 