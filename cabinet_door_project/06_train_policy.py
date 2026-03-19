"""
Step 6: Train a Diffusion U-Net Lowdim Policy on OpenCabinet
=============================================================
Trains a low-dimensional Diffusion Policy on human demonstration data from
the RoboCasa OpenCabinet task.  The model learns to predict the noise added
to an action sequence (epsilon-prediction) conditioned on a short history of
proprioceptive observations.

Architecture summary:
  - Observation: 19-D flat state vector (base pose, EEF pose, gripper, handle)
  - Action:      12-D flat action vector (base, torso, EEF, gripper, mode)
  - Model:       Conditional U-Net 1-D (ConditionalUnet1D) over the action
                 horizon, conditioned on obs via global FiLM conditioning
  - Scheduler:   DDPM for training, DDIM for fast inference

Usage:
    python 06_train_policy.py
    python 06_train_policy.py --epochs 100 --batch_size 512
"""

import argparse
import os
import sys
import yaml
import datetime as _dt

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# These imports use the in-repo fork of diffusion_policy, which lives as a
# git-submodule at ./diffusion_policy.  Make sure it is checked out and that
# its dependencies (diffusers, einops, etc.) are installed before running.
from diffusion_policy.diffusion_policy.dataset.lerobot_dataset import LerobotLowdimDataset
from diffusion_policy.diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.diffusion_policy.policy.diffusion_unet_lowdim_policy import (
    DiffusionUnetLowdimPolicy,
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler   # fast inference
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler   # training


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def print_section(title):
    """Print a formatted banner to visually separate major log stages."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def load_config(config_path):
    """
    Load a YAML configuration file from disk.

    Not currently called by the training pipeline directly — kept for
    callers that prefer file-based config over CLI flags.

    Args:
        config_path: Filesystem path to a .yaml/.yml config file.

    Returns:
        Parsed config as a Python dict.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_dataset_path():
    """
    Resolve the OpenCabinet dataset path via the RoboCasa registry.

    The registry maps dataset names to local paths set up by
    ``04_download_dataset.py``.  If the dataset has not been downloaded
    yet this function exits with a helpful error message.

    Returns:
        Absolute dataset path string if found.

    Raises:
        SystemExit: If the registry returns None or the path does not exist.
    """
    import robocasa
    from robocasa.utils.dataset_registry_utils import get_ds_path

    path = get_ds_path("OpenCabinet", source="human")
    if path is None or not os.path.exists(path):
        print("ERROR: Dataset not found. Run 04_download_dataset.py first.")
        sys.exit(1)
    return path


# ---------------------------------------------------------------------------
# Observation / action shape metadata
# ---------------------------------------------------------------------------

def _build_open_cabinet_shape_meta():
    """
    Build the ``shape_meta`` dict describing the low-dim OpenCabinet feature set.

    ``shape_meta`` is consumed by ``LerobotLowdimDataset`` and
    ``DiffusionUnetLowdimPolicy`` to know:
      - which LeRobot dataset columns to read (``lerobot_keys``)
      - how to concatenate them into a flat obs/action vector (``shape``)
      - whether a feature is low-dim state or an image (``type``)

    Observation layout (total = 19-D):
        base_pos              (3)  — world-frame robot base XYZ
        base_quat             (4)  — world-frame robot base orientation (wxyz)
        robot0_base_to_eef_pos  (3)  — base-relative EEF position
        robot0_base_to_eef_quat (4)  — base-relative EEF orientation (wxyz)
        robot0_gripper_qpos   (2)  — left/right finger joint positions
        handle_pos            (3)  — world-frame cabinet handle position
        handle_to_eef_pos     (3)  — handle → EEF offset vector
        door_openness         (1)  — scalar in [0, 1], 1 = fully open

    Action layout (total = 12-D):
        base_motion           (3)  — (Δx, Δy, Δyaw) in the base frame
        torso_lift            (1)  — torso height delta
        control_mode          (1)  — toggle between navigation / manipulation
        end_effector_position (3)  — EEF position delta
        end_effector_rotation (3)  — EEF rotation delta (axis-angle)
        gripper_close         (1)  — −1 = open, +1 = close

    Returns:
        Dict with "obs" and "action" sub-dicts compatible with
        ``LerobotLowdimDataset`` and ``DiffusionUnetLowdimPolicy``.
    """
    return {
        "obs": {
            # ---- Robot base pose ----
            "base_pos": {
                "shape": [3],
                "type": "low_dim",
                "lerobot_keys": ["state.base_position"],
            },
            "base_quat": {
                "shape": [4],
                "type": "low_dim",
                "lerobot_keys": ["state.base_rotation"],
            },
            # ---- End-effector pose (base-relative) ----
            "robot0_base_to_eef_pos": {
                "shape": [3],
                "type": "low_dim",
                "lerobot_keys": ["state.end_effector_position_relative"],
            },
            "robot0_base_to_eef_quat": {
                "shape": [4],
                "type": "low_dim",
                "lerobot_keys": ["state.end_effector_rotation_relative"],
            },
            # ---- Gripper state ----
            "robot0_gripper_qpos": {
                "shape": [2],
                "type": "low_dim",
                "lerobot_keys": ["state.gripper_qpos"],
            },
            # ---- Cabinet handle features (computed in 07_evaluate_policy.py) ----
            "handle_pos": {
                "shape": [3],
                "type": "low_dim",
                "lerobot_keys": ["state.handle_pos"],
            },
            "handle_to_eef_pos": {
                "shape": [3],
                "type": "low_dim",
                "lerobot_keys": ["state.handle_to_eef_pos"],
            },
            "door_openness": {
                "shape": [1],
                "type": "low_dim",
                "lerobot_keys": ["state.door_openness"],
            },
        },
        "action": {
            # Flat 12-D action; the dataset concatenates these columns in order
            "shape": [12],
            "lerobot_keys": [
                "action.base_motion",             # (3) Δx, Δy, Δyaw
                "action.control_mode",            # (1) nav / manip toggle
                "action.end_effector_position",   # (3) EEF position delta
                "action.end_effector_rotation",   # (3) EEF rotation delta
                "action.gripper_close",           # (1) open / close
            ],
        },
    }


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def train_unet_lowdim_policy(config):
    """
    Train a low-dim Diffusion U-Net policy on OpenCabinet human demos.

    The training loop follows the standard DDPM objective: for each batch,
    sample a random diffusion timestep t, add noise to the ground-truth action
    sequence, and train the U-Net to predict that noise (epsilon-prediction).
    At inference time the policy denoises a Gaussian sample conditioned on the
    observation history to produce an action sequence.

    Checkpoints are written every 10 epochs to a timestamped subdirectory so
    that training can be resumed or the best checkpoint selected for evaluation.

    Args:
        config: Dict of hyperparameters.  Recognised keys (all optional):
            epochs            (int,   default 50)   — training epochs
            batch_size        (int,   default 256)  — DataLoader batch size
            learning_rate     (float, default 1e-4) — AdamW learning rate
            checkpoint_dir    (str)                 — base path for checkpoints
            num_workers       (int,   default 8)    — DataLoader worker count
            val_ratio         (float, default 0.05) — fraction of data for val
            val_every         (int,   default 5)    — validate every N epochs
            num_inference_steps (int, default 20)   — DDIM steps at eval time
            horizon           (int,   default 16)   — action prediction horizon
            n_obs_steps       (int,   default 2)    — observation history length
            n_action_steps    (int,   default 8)    — steps executed per pred
            down_dims         (list,  default [...]) — U-Net channel widths

    Returns:
        None.  Writes .pt checkpoint files to ``checkpoint_dir``.
    """
    print_section("Diffusion U-Net Lowdim Policy")

    # ------------------------------------------------------------------
    # Hyperparameters (CLI / config overrides; sensible defaults included)
    # ------------------------------------------------------------------
    epochs         = int(config.get("epochs",        50))
    batch_size     = int(config.get("batch_size",   256))   # large batch suits A100/H100
    learning_rate  = float(config.get("learning_rate", 1e-4))
    checkpoint_dir = config.get("checkpoint_dir", "/tmp/cabinet_unet_checkpoints")
    num_workers    = int(config.get("num_workers",    8))    # match CPU core count

    val_ratio           = float(config.get("val_ratio",           0.05))
    val_every           = int(config.get("val_every",              5))
    num_inference_steps = int(config.get("num_inference_steps",   20))  # DDIM steps

    # Receding-horizon control parameters (must match eval script)
    horizon        = int(config.get("horizon",        16))  # total predicted actions
    n_obs_steps    = int(config.get("n_obs_steps",     2))  # obs frames fed to policy
    n_action_steps = int(config.get("n_action_steps",  8))  # actions actually executed

    # U-Net channel widths — wider = more capacity, slower training
    # [128, 256, 512] is a good trade-off for 19-D state inputs
    down_dims = config.get("down_dims", [128, 256, 512])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset_path = get_dataset_path()
    print(f"Using dataset: {dataset_path}")

    shape_meta = _build_open_cabinet_shape_meta()

    # Derive flat obs/action dimensions from shape_meta so they stay in sync
    obs_dim    = sum(v["shape"][0] for v in shape_meta["obs"].values())   # 19
    action_dim = int(shape_meta["action"]["shape"][0])                    # 12

    print(f"  obs_dim={obs_dim}  action_dim={action_dim}  horizon={horizon}")

    # LerobotLowdimDataset returns (obs_seq, action_seq) tensors per sample.
    # use_cache=True pre-loads the full dataset into RAM to avoid disk I/O
    # becoming the bottleneck during training.
    train_dataset = LerobotLowdimDataset(
        shape_meta=shape_meta,
        dataset_path=dataset_path,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        val_ratio=val_ratio,
        split="train",
        use_cache=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,       # speeds up CPU→GPU transfer
        prefetch_factor=2,     # overlap data loading with GPU compute
    )

    val_loader = None
    if val_ratio > 0:
        val_dataset = LerobotLowdimDataset(
            shape_meta=shape_meta,
            dataset_path=dataset_path,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            val_ratio=val_ratio,
            split="val",
            use_cache=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    print(f"  Train samples: {len(train_dataset)}")
    if val_loader:
        print(f"  Val   samples: {len(val_dataset)}")

    # ------------------------------------------------------------------
    # Model + noise scheduler
    # ------------------------------------------------------------------
    # ConditionalUnet1D operates over the action *time* axis (length=horizon).
    # global_cond_dim receives the flattened obs history as a FiLM conditioning
    # vector (shape: obs_dim × n_obs_steps).
    model = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * n_obs_steps,
        down_dims=down_dims,
    )

    # DDPM scheduler for training: adds noise at a random timestep and
    # computes the MSE loss between predicted and actual noise.
    # 50 timesteps keeps training fast while maintaining quality;
    # raise to 100 for a potentially smoother loss landscape.
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=50,
        beta_schedule="squaredcos_cap_v2",   # cosine schedule; more stable than linear
        prediction_type="epsilon",           # predict the noise, not x0 or v
    )

    # DiffusionUnetLowdimPolicy wraps the U-Net and scheduler, and exposes
    # compute_loss() for training and predict_action() for evaluation.
    policy = DiffusionUnetLowdimPolicy(
        model=model,
        noise_scheduler=noise_scheduler,
        horizon=horizon,
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        num_inference_steps=num_inference_steps,  # DDIM steps at eval time
        obs_as_global_cond=True,                  # flatten obs into FiLM cond
    ).to(device)

    # Fit a per-feature normalizer from the training set so the policy can
    # normalize/denormalize obs and actions during training and inference.
    policy.set_normalizer(train_dataset.get_normalizer())

    # AdamW with a small weight decay to regularise the U-Net weights
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=1e-6)

    # ------------------------------------------------------------------
    # Checkpoint directory (timestamped to avoid overwriting previous runs)
    # ------------------------------------------------------------------
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"{checkpoint_dir}_{stamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"  Checkpoints → {checkpoint_dir}")
    print(f"  Batch size: {batch_size} | U-Net dims: {down_dims}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(epochs):
        policy.train()
        losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            # Move the full batch (obs_seq + action_seq) to the training device
            batch = {k: v.to(device) for k, v in batch.items()}

            # compute_loss() internally:
            #   1. Samples a random diffusion timestep t for each item
            #   2. Adds noise to the action sequence at level t
            #   3. Runs the U-Net forward pass conditioned on obs
            #   4. Returns MSE(predicted_noise, actual_noise)
            loss = policy.compute_loss(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"  Epoch {epoch + 1:4d}/{epochs} | Train Loss: {np.mean(losses):.6f}")

        # ---- Validation (lightweight — no denoising rollout, just loss) ----
        if val_loader and (epoch + 1) % val_every == 0:
            policy.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    val_losses.append(policy.compute_loss(batch).item())
            print(f"    Validation Loss: {np.mean(val_losses):.6f}")

        # ---- Periodic checkpoint (every 10 epochs) ----
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
            torch.save(
                {
                    "model_state_dict": policy.state_dict(),
                    "shape_meta":       shape_meta,    # needed by 07_evaluate_policy.py
                    "epoch":            epoch + 1,
                    "train_loss":       float(np.mean(losses)),
                },
                ckpt_path,
            )
            print(f"    Checkpoint saved: {ckpt_path}")

    print(f"\nTraining complete.  Checkpoints in: {checkpoint_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """
    Parse command-line arguments and launch the training pipeline.

    Only the most commonly tuned hyperparameters are exposed as CLI flags;
    the rest use the defaults defined inside ``train_unet_lowdim_policy``.
    Use a YAML config file (via ``load_config``) for full control.
    """
    parser = argparse.ArgumentParser(description="Train a Diffusion U-Net lowdim policy")
    parser.add_argument("--epochs",     type=int, default=50,  help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="DataLoader batch size")
    args = parser.parse_args()

    unet_cfg = {
        "epochs":     args.epochs,
        "batch_size": args.batch_size,
    }
    train_unet_lowdim_policy(unet_cfg)


if __name__ == "__main__":
    main()