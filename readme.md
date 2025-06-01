# F1RL

F1Tenth Reinforcement Learning project focused on autonomous racing with sim-to-real transfer capabilities.

## Demo

https://github.com/user-attachments/assets/e45a3859-e614-4d35-a0af-d0be6296719d

https://github.com/user-attachments/assets/ee10fbbc-1e6e-435b-ae7f-ddb40ccaf1e7

https://github.com/user-attachments/assets/e7c1aa41-8e63-48bb-9948-68f11f25d3ef

## Directory Structure

```
├── f1tenth_gym/              # Modified F1Tenth gym environment (from @https://github.com/f1tenth/f1tenth_gym)
│   ├── rl/                   # Main RL implementation directory
│   │   ├── main.py          # Main training/evaluation script with command-line interface
│   │   ├── rl_env.py        # F110GymWrapper with domain randomization support
│   │   ├── stablebaseline3/  # RL code using stable baselines 3
│   │   │   ├── rl.py        # Training pipeline with imitation learning integration
│   │   │   ├── feature_extractor.py  # Neural network architectures (ResNet, MLP, FILM, etc.)
│   │   │   └── rl_node.py   # ROS node for real robot deployment
│   │   ├── pure_pursuit.py  # Expert policy for imitation learning: Pure pursuit controller
│   │   ├── wall_follow.py   # Expert policy for imitation learning: Wall following controller
│   │   ├── lattice_planner.py  # Expert policy for imitation learning: Lattice-based trajectory planner
│   │   └── gap_follow_agent.py  # Expert policy for imitation learning: Gap following reactive controller
│   ├── f1tenth_racetracks/  # Track maps (from @https://github.com/f1tenth/f1tenth_racetracks)
│   └── gym/                 # Core F1Tenth gym implementation
└── papers/                  # Related research papers
```

## Quick Start

### Docker Setup

Pull the pre-built Docker image:

```bash
docker pull wudidewo/f1tenth-rl
```

Run the container:

```bash
docker run -it --rm wudidewo/f1tenth-rl
```

### Training
Note that the image contains the code for this project. However, clone from this repo if you want the most up-to-date code.

Basic training with default settings:

```bash
cd /f1tenth_gym
python3 rl/main.py --use_wandb=false --num_envs=2
```

A GPU will help accelerate the training process. Improve the num_envs if your computer has multiple cores.

### Evaluation

After the training, the model should be saved in the path like "f1tenth_gym/logs/RECURRENT_PPO_RESNET_envs24_params24_dr1_il1_crl0_racing0_PURE_PURSUIT_seed42_20250531_145431/best_model/best_model.zip". We need to copy the model and normalization file into the evaluation directory, "f1tenth_gym/logs/best_model".

Evaluate a trained model:

```bash
python3 rl/main.py --eval=true --num_envs=2 --model_path=./logs/best_model/best_model.zip --vecnorm_path=./logs/best_model/vec_normalize.pkl
```

### Check more flags
```bash
python3 rl/main.py --help
```

### Deployment

For real robot deployment in ROS environments:

```bash
# Run the ROS node (requires ROS setup, lidar messages, and odom messages)
python stablebaseline3/rl_node.py --model_path=./logs/best_model/best_model.zip --vecnorm_path=./logs/best_model/vec_normalize.pkl
```

The `rl_node.py` provides ROS integration for deploying trained policies on physical F1Tenth vehicles.


## Key Features
This project use recurrent policy for reinforcement learning, domain randomization for mitigating the sim-to-real gap, imitation learning for the model bootstrap, contextual reinforcement learning for improving the performance. More details can be seen in a slightly outdated report: https://drive.google.com/file/d/1Q0JvorN-uOZdv618uBwWlkrfrRAe2O4r/view?usp=sharing

## Project Status

This is an active project. Current progress and ideas are tracked in:

- **Slides**: https://docs.google.com/presentation/d/1MN3k4OcoNS2_rCE4_UrtA76lchaS4XlhmnMyM7rifgE/edit?usp=sharing
- **Todo List**: https://docs.google.com/document/d/1KLnU6vMZAfi7sqKzXk04mfPIC0vFzXK4BlMCgbjqpX0/edit?usp=sharing