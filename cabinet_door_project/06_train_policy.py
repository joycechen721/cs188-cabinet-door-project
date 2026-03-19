import argparse
import os
import sys
import yaml
import datetime as _dt

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure local imports work correctly
from diffusion_policy.diffusion_policy.dataset.lerobot_dataset import LerobotLowdimDataset
from diffusion_policy.diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.diffusion_policy.policy.diffusion_unet_lowdim_policy import (
    DiffusionUnetLowdimPolicy,
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_dataset_path():
    import robocasa
    from robocasa.utils.dataset_registry_utils import get_ds_path
    path = get_ds_path("OpenCabinet", source="human")
    if path is None or not os.path.exists(path):
        print("ERROR: Dataset not found. Run 04_download_dataset.py first.")
        sys.exit(1)
    return path

def _build_open_cabinet_shape_meta():
    return {
        "obs": {
            "base_pos": {
                "shape": [3],
                "type": "low_dim",
                "lerobot_keys": ["state.base_position"], # Matches remapped key
            },
            "base_quat": {
                "shape": [4],
                "type": "low_dim",
                "lerobot_keys": ["state.base_rotation"],
            },
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
            "robot0_gripper_qpos": {
                "shape": [2],
                "type": "low_dim",
                "lerobot_keys": ["state.gripper_qpos"],
            },
            "handle_pos": {
                "shape": [3],
                "type": "low_dim",
                "lerobot_keys": ["state.handle_pos"], # Maps to observation.handle_pos
            },
            "handle_to_eef_pos": {
                "shape": [3],
                "type": "low_dim",
                "lerobot_keys": ["state.handle_to_eef_pos"], # Maps to observation.handle_to_eef_pos
            },
            "door_openness": {
                "shape": [1],
                "type": "low_dim",
                "lerobot_keys": ["state.door_openness"], # Maps to observation.door_openness
            },
        },
        "action": {
            "shape": [12],
            "lerobot_keys": [
                "action.base_motion",            
                "action.control_mode",           
                "action.end_effector_position",  
                "action.end_effector_rotation",  
                "action.gripper_close",          
            ],
        },
    }

def train_unet_lowdim_policy(config):
    print_section("Diffusion U-Net Lowdim Policy (Optimized for A100)")
    
    # ----------------------------
    # High-Performance Defaults
    # ----------------------------
    epochs = int(config.get("epochs", 50))
    batch_size = int(config.get("batch_size", 256)) # Increased for A100
    learning_rate = float(config.get("learning_rate", 1e-4))
    checkpoint_dir = config.get("checkpoint_dir", "/tmp/cabinet_unet_checkpoints")
    num_workers = int(config.get("num_workers", 8)) # Use Colab's multi-core CPU
    
    # Fast-Validation Settings
    val_ratio = float(config.get("val_ratio", 0.05))
    val_every = int(config.get("val_every", 5))
    num_inference_steps = int(config.get("num_inference_steps", 20)) # Sped up from 100
    
    horizon = int(config.get("horizon", 16))
    n_obs_steps = int(config.get("n_obs_steps", 2))
    n_action_steps = int(config.get("n_action_steps", 8))

    # Slim U-Net dims for low-dim states
    down_dims = config.get("down_dims", [128, 256, 512]) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ----------------------------
    # Dataset Loading (RAM Cached)
    # ----------------------------
    dataset_path = "/Users/joycechen/classes/188_robotics/cs188-cabinet-door-project/robocasa/datasets/v1.0/pretrain/atomic/OpenCabinet/20250819/lerobot_augmented"
    print(f"Using dataset: {dataset_path}")

    shape_meta = _build_open_cabinet_shape_meta()
    obs_dim = sum(v["shape"][0] for v in shape_meta["obs"].values())
    action_dim = int(shape_meta["action"]["shape"][0])

    # Dataset will now pre-load to RAM based on our new lerobot_dataset.py
    train_dataset = LerobotLowdimDataset(
        shape_meta=shape_meta, dataset_path=dataset_path,
        horizon=horizon, n_obs_steps=n_obs_steps,
        val_ratio=val_ratio, split="train", use_cache=True
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, prefetch_factor=2
    )

    val_loader = None
    if val_ratio > 0:
        val_dataset = LerobotLowdimDataset(
            shape_meta=shape_meta, dataset_path=dataset_path,
            horizon=horizon, n_obs_steps=n_obs_steps,
            val_ratio=val_ratio, split="val", use_cache=True
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    # ----------------------------
    # Policy Setup
    # ----------------------------
    model = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * n_obs_steps,
        down_dims=down_dims,
    )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=50, # speding up from 100
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    policy = DiffusionUnetLowdimPolicy(
        model=model, noise_scheduler=noise_scheduler,
        horizon=horizon, obs_dim=obs_dim, action_dim=action_dim,
        n_action_steps=n_action_steps, n_obs_steps=n_obs_steps,
        num_inference_steps=num_inference_steps,
        obs_as_global_cond=True
    ).to(device)

    policy.set_normalizer(train_dataset.get_normalizer())
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=1e-6)

    # ----------------------------
    # Training Loop
    # ----------------------------
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"{checkpoint_dir}_{stamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Batch size: {batch_size} | Model: {down_dims}")

    for epoch in range(epochs):
        policy.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = policy.compute_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"  Epoch {epoch + 1:4d} | Train Loss: {np.mean(losses):.6f}")

        # Validation and Checkpointing (standard logic)
        if (epoch + 1) % val_every == 0 and val_loader:
            policy.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    val_losses.append(policy.compute_loss(batch).item())
            print(f"    Validation Loss: {np.mean(val_losses):.6f}")

        if (epoch + 1) % 10 == 0:
            torch.save(policy.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt"))

    print(f"Final checkpoint saved to {checkpoint_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_unet_lowdim", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    unet_cfg = {"epochs": args.epochs, "batch_size": args.batch_size}
    train_unet_lowdim_policy(unet_cfg)

if __name__ == "__main__":
    main()