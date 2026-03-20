# Cabinet Door Opening Robot - CS 188 Starter Project

### Introduction

The project aims toward implementing a diffusion policy to handle the OpenCabient task in RoboCasa. Given the template starter project for CS188, we modify the the overall diffusion policy from the ([suggested implemention](https://github.com/robocasa-benchmark/diffusion_policy)) to achieve the following goals: (1) to train and evaluate diffusion policy architectures from the official Diffusion Policy repository on a pre-collected OpenCabinet demonstration dataset; (2) to investigate whether augmenting the robot’s proprioceptive state with explicit task-relevant features improves policy performance; (3) to examine sources of failure in the evaluation pipeline and develop principled corrections; and (4) to characterize the trade-offs between model capacity, training efficiency, and task success rate under realistic compute constraints.

With this basic starter code, we call our ([modified diffusion policy](https://github.com/joycechen721/diffusion_policy)) for training the model given a set of demonstrations and added our own scripts to evaluate the training checkpoints.

## Overview

In this project you will build a robot that learns to open kitchen cabinet doors
using **RoboCasa365**, a large-scale simulation benchmark for everyday robot
tasks. You will progress from understanding the simulation environment, to
collecting demonstrations, to training a neural-network policy that controls the
robot autonomously.

### What you will learn

1. How robotic manipulation environments are structured (MuJoCo + robosuite + RoboCasa)
2. How the `OpenCabinet` task works -- sensors, actions, success criteria
3. How to collect and use demonstration datasets (human + MimicGen)
4. How to train a behavior-cloning policy from demonstrations
5. How to evaluate your trained policy in simulation

### The robot

We use the **PandaOmron** mobile manipulator -- a Franka Panda 7-DOF arm
mounted on an Omron wheeled base with a torso lift joint. This is the default
and best-supported robot in RoboCasa.

---

## Installation

Run the install script (works on **macOS** and **WSL/Linux**):

```bash
./install.sh
```

This will:
- Create a Python virtual environment (`.venv`)
- Clone and install robosuite and robocasa
- Install all Python dependencies (PyTorch, numpy, matplotlib, etc.)
- Download RoboCasa kitchen assets (~10 GB)

After installation, activate the environment:

```bash
source .venv/bin/activate
```

Then verify everything works:

```bash
cd cabinet_door_project
python 00_verify_installation.py
```

> **macOS note:** Scripts that open a rendering window (03, 05) require
> `mjpython` instead of `python`. The install script will remind you of this.

---

## Project Structure

```
cabinet_door_project/
  00_verify_installation.py      # Check that everything is installed correctly
  01_explore_environment.py      # Create the OpenCabinet env, inspect observations/actions
  02_random_rollouts.py          # Run random actions, save video, understand the task
  03_teleop_collect_demos.py     # Teleoperate the robot to collect your own demonstrations
  04_download_dataset.py         # Download the pre-collected OpenCabinet dataset
  05_playback_demonstrations.py  # Play back demonstrations to see expert behavior
  05b_augment_handle_data.py     # Augment dataset with handle/door features
  06_train_policy.py             # Train a local low-dim Diffusion U-Net policy
  07_evaluate_policy.py          # General evaluator (BC + local low-dim checkpoints)
  07b_evaluate_policy.py         # Diffusion Policy evaluator with correct key mapping
  07c_evaluate_policy.py         # Low-dim evaluator with handle features + action remap
  07c_evaluate_policy_max.py     # Max-success evaluator (extra heuristics + clipping)
  08_visualize_policy_rollout.py # Visualize a rollout of your policy in RoboCasa
  configs/
    diffusion_policy.yaml        # Training hyperparameters
    unet_lowdim_local.yaml       # Local low-dim Diffusion U-Net hyperparameter template
  diffusion_policy/              # RoboCasa fork of Diffusion Policy (in-repo)
  notebook.ipynb                 # Interactive Jupyter notebook companion
install.sh                       # Installation script (macOS + WSL/Linux)
README.md                        # This file
```

---

## Step-by-Step Guide

### Step 0: Verify Installation

```bash
python 00_verify_installation.py
```

This checks that MuJoCo, robosuite, RoboCasa, and all dependencies are
correctly installed and that the `OpenCabinet` environment can be created.

### Step 1: Explore the Environment

```bash
python 01_explore_environment.py
```

This script creates the `OpenCabinet` environment and prints detailed
information about:
- **Observation space**: what the robot sees (camera images, joint positions,
  gripper state, base pose)
- **Action space**: what the robot can do (arm movement, gripper open/close,
  base motion, control mode)
- **Task description**: the natural language instruction for the episode
- **Success criteria**: how the environment determines task completion

### Step 2: Random Rollouts

```bash
python 02_random_rollouts.py
```

Runs the robot with random actions to see what happens (spoiler: nothing
useful, but it helps you understand the action space). Saves a video to
`/tmp/cabinet_random_rollouts.mp4`.

### Step 3: Teleoperate and Collect Demonstrations

```bash
# Mac users: use mjpython instead of python
python 03_teleop_collect_demos.py
```

Control the robot yourself using the keyboard to open cabinet doors. This
gives you intuition for the task difficulty and generates demonstration data.

**Keyboard controls:**
| Key | Action |
|-----|--------|
| Ctrl+q | Reset simulation |
| spacebar | Toggle gripper (open/close) |
| up-right-down-left | Move horizontally in x-y plane |
| .-; | Move vertically |
| o-p | Rotate (yaw) |
| y-h | Rotate (pitch) |
| e-r | Rotate (roll) |
| b | Toggle arm/base mode (if applicable) |
| s | Switch active arm (if multi-armed robot) |
| = | Switch active robot (if multi-robot environment) |              

### Step 4: Download Pre-collected Dataset

```bash
python 04_download_dataset.py
```

Downloads the official OpenCabinet demonstration dataset from the RoboCasa
servers. This includes both human demonstrations and MimicGen-expanded data
across diverse kitchen scenes.

### Step 5: Play Back Demonstrations

```bash
python 05_playback_demonstrations.py
```

Visualize the downloaded demonstrations to see how an expert opens cabinet
doors. This is the data your policy will learn from.

### Step 5b: Augment Dataset with Handle Features

```bash
python 05b_augment_handle_data.py
```

Adds state-only features that make cabinet manipulation much easier:
- `observation.handle_pos` (3): handle world position
- `observation.handle_to_eef_pos` (3): handle relative to end-effector
- `observation.door_openness` (1): door openness in [0, 1]

These are used by the low-dim Diffusion U-Net policy in Step 6.

### Step 6: Train a Policy

```bash
python 06_train_policy.py
```

Trains a **local low-dimensional Diffusion U-Net policy** on state-action pairs
from the demonstration data. This is the policy we used for most of the local
experiments in this repo.

Common options:
```bash
python 06_train_policy.py --epochs 50 --batch_size 64
```

Hyperparameter template for local runs:
- `cabinet_door_project/configs/unet_lowdim_local.yaml`

For a policy that actually works, use one of the official training repos:

```bash
# Diffusion Policy (recommended for single-task)
git clone https://github.com/robocasa-benchmark/diffusion_policy
cd diffusion_policy && pip install -e .
python train.py --config-name=train_diffusion_transformer_bs192 task=robocasa/OpenCabinet
```

See `cabinet_door_project/diffusion_policy/README.md` for up-to-date setup and eval commands.

### Step 7: Evaluate Your Policy

```bash
python 07_evaluate_policy.py --checkpoint path/to/checkpoint.pt
```

Runs your trained policy in the simulation environment and reports success
rate across multiple episodes and kitchen scenes.

**Evaluation script variants**
- `07_evaluate_policy.py`: General evaluator for local low-dim checkpoints.
- `07b_evaluate_policy.py`: Diffusion Policy evaluator with **correct observation key mapping**.
- `07c_evaluate_policy.py`: Low-dim evaluator with **handle feature reconstruction** and **action remapping**.
- `07c_evaluate_policy_max.py`: Tuned for **max success** (action clipping, optional clamp, base-to-handle assist).

---

## Key Concepts

### The OpenCabinet Task

- **Goal**: Open a kitchen cabinet door
- **Fixture**: `HingeCabinet` (a cabinet with hinged doors)
- **Initial state**: Cabinet door is closed; robot is positioned nearby
- **Success**: At least one cabinet door is open (any door joint ≥ 0.90)
- **Horizon**: 500 timesteps at 20 Hz control frequency (25 seconds)
- **Scene variety**: 2,500+ kitchen layouts/styles for generalization

### Observation Space (PandaOmron)

| Key | Shape | Description |
|-----|-------|-------------|
| `robot0_agentview_left_image` | (256, 256, 3) | Left shoulder camera |
| `robot0_agentview_right_image` | (256, 256, 3) | Right shoulder camera |
| `robot0_eye_in_hand_image` | (256, 256, 3) | Wrist-mounted camera |
| `robot0_gripper_qpos` | (2,) | Gripper finger positions |
| `robot0_base_pos` | (3,) | Base position (x, y, z) |
| `robot0_base_quat` | (4,) | Base orientation quaternion |
| `robot0_base_to_eef_pos` | (3,) | End-effector pos relative to base |
| `robot0_base_to_eef_quat` | (4,) | End-effector orientation relative to base |

**Augmented state-only features (Step 5b):**
- `observation.handle_pos` (3) — handle world position
- `observation.handle_to_eef_pos` (3) — handle relative to end-effector
- `observation.door_openness` (1) — door openness in [0, 1]

### Action Space (PandaOmron)

| Key | Dim | Description |
|-----|-----|-------------|
| `end_effector_position` | 3 | Delta (dx, dy, dz) for the end-effector |
| `end_effector_rotation` | 3 | Delta rotation (axis-angle) |
| `gripper_close` | 1 | 0 = open, 1 = close |
| `base_motion` | 4 | (forward, side, yaw, torso) |
| `control_mode` | 1 | 0 = arm control, 1 = base control |

### Dataset Format (LeRobot)

Datasets are stored in LeRobot format:
```
dataset/
  meta/           # Episode metadata (task descriptions, camera info)
  videos/         # MP4 videos from each camera
  data/           # Parquet files with actions, states, rewards
  extras/         # Per-episode metadata
```

---

## Policies Used In This Project

- **Local low-dim Diffusion U-Net**: Trained via `06_train_policy.py` and evaluated with `07c_evaluate_policy.py` or `07c_evaluate_policy_max.py`.
- **Diffusion Policy (RoboCasa fork)**: In-repo at `cabinet_door_project/diffusion_policy` and based on the upstream `real-stanford/diffusion_policy` codebase.

## Diffusion Policy (GitHub + Local Fork)

We include the RoboCasa fork of Diffusion Policy at:
- `cabinet_door_project/diffusion_policy`

Upstream repositories:
- `https://github.com/real-stanford/diffusion_policy`
- `https://github.com/robocasa-benchmark/diffusion_policy`

Compatible fork with local edits used for our trained policy:
```
https://github.com/joycechen721/diffusion_policy
```

Quick reference (from the fork README):
```bash
python train.py --config-name=train_diffusion_unet_lowdim task=robocasa/OpenCabinet
python eval_robocasa.py --checkpoint <checkpoint-path> --task_set <task-set> --split <split>
```

## Architecture Diagram

```
                    RoboCasa Stack
                    ==============

  +-------------------+     +-------------------+
  |   Kitchen Scene   |     |   OpenCabinet     |
  |  (2500+ layouts)  |     |   (Task Logic)    |
  +--------+----------+     +--------+----------+
           |                         |
           v                         v
  +------------------------------------------------+
  |              Kitchen Base Class                 |
  |  - Fixture management (cabinets, fridges, etc)  |
  |  - Object placement (bowls, cups, etc)          |
  |  - Robot positioning                            |
  +------------------------+-----------------------+
                           |
                           v
  +------------------------------------------------+
  |              robosuite (Backend)                |
  |  - MuJoCo physics simulation                   |
  |  - Robot models (PandaOmron, GR1, Spot, ...)   |
  |  - Controller framework                        |
  +------------------------+-----------------------+
                           |
                           v
  +------------------------------------------------+
  |              MuJoCo 3.3.1 (Physics)            |
  |  - Contact dynamics, rendering, sensors        |
  +------------------------------------------------+
```

---

## Troubleshooting

I'll continually update this section as students find bugs in the system. Please, let me know if you encounter issues!

| Problem | Solution |
|---------|----------|
| `MuJoCo version must be 3.3.1` | `pip install mujoco==3.3.1` |
| `numpy version must be 2.2.5` | `pip install numpy==2.2.5` |
| Rendering crashes on Mac | Use `mjpython` instead of `python` |
| `GLFW error` on headless server | Set `export MUJOCO_GL=egl` or `osmesa` |
| Out of GPU memory during training | Reduce batch size in `configs/diffusion_policy.yaml` |
| Kitchen assets not found | Run `python -m robocasa.scripts.download_kitchen_assets` |

---

## References

- [RoboCasa Paper & Website](https://robocasa.ai/)
- [RoboCasa GitHub](https://github.com/robocasa/robocasa)
- [robosuite Documentation](https://robosuite.ai/)
- [Diffusion Policy Paper](https://diffusion-policy.cs.columbia.edu/)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [LeRobot Dataset Format](https://github.com/huggingface/lerobot)
