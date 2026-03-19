"""
Evaluate a Trained Diffusion Policy for OpenCabinet (PROPER KEY MAPPING)

This version handles observation keys correctly with explicit mapping.

Usage:
    python evaluate_diffusion_policy_proper.py \
        --checkpoint path/to/checkpoint.ckpt \
        --num_rollouts 10
"""

import argparse
import os
import sys
from collections import deque
from pathlib import Path

# Add diffusion_policy to path
_script_dir = Path(__file__).parent.absolute()
_cabinet_dir = _script_dir if (_script_dir / "diffusion_policy").exists() else _script_dir.parent
_diffusion_dir = _cabinet_dir / "diffusion_policy"

if _diffusion_dir.exists() and str(_diffusion_dir) not in sys.path:
    sys.path.insert(0, str(_diffusion_dir))

# Force osmesa (CPU offscreen renderer) on Linux/WSL2
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np
import pickle
import torch
from pathlib import Path
from omegaconf import OmegaConf
import hydra

import robocasa
from robocasa.utils.env_utils import create_env

def get_mj_model_data(env):
    """
    Robustly retrieve the MuJoCo model and data objects from a robosuite env.

    Robosuite (and its wrappers) expose these through several different paths
    depending on the version and binding (mujoco-py vs dm_control vs mujoco>=2.3):

      1. env.sim.model / env.sim.data          (mujoco-py / old robosuite)
      2. env.sim.model._model / env.sim.data   (some mixed versions)
      3. env.model / env.data                  (newer native mujoco binding)
      4. env.physics.model / env.physics.data  (dm_control style)
    """
    # Collect candidates: the env itself plus any wrapped inner envs
    candidates = [env]
    for attr in ("env", "_env", "unwrapped"):
        if hasattr(env, attr):
            candidates.append(getattr(env, attr))

    for obj in candidates:
        # Pattern 1 & 2: obj.sim exists
        if hasattr(obj, "sim") and obj.sim is not None:
            sim = obj.sim
            # mujoco-py style
            if hasattr(sim, "model") and hasattr(sim, "data"):
                return sim.model, sim.data
            # newer mujoco python bindings attach model/data directly on sim
            if hasattr(sim, "_model") and hasattr(sim, "_data"):
                return sim._model, sim._data

        # Pattern 3: model/data directly on env (robosuite >= 1.5 native mujoco)
        if hasattr(obj, "model") and hasattr(obj, "data"):
            m, d = obj.model, obj.data
            # make sure these are actually MuJoCo objects, not robosuite model XML wrappers
            if hasattr(m, "nbody") and hasattr(d, "qpos"):
                return m, d

        # Pattern 4: dm_control physics
        if hasattr(obj, "physics"):
            ph = obj.physics
            if hasattr(ph, "model") and hasattr(ph, "data"):
                return ph.model, ph.data





def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def find_fixture_handle_bodies(model, fixture_name=None):
    """
    Find MuJoCo body names for door handles.
    If fixture_name is given, filter to that fixture.
    Otherwise return ALL bodies containing 'handle' in their name.
    """
    handle_bodies = []
    for i in range(model.nbody):
        name = model.body(i).name
        if "handle" not in name:
            continue
        if fixture_name is None or fixture_name in name:
            handle_bodies.append(name)
    return handle_bodies


def find_fixture_door_joints(model, fixture_name=None):
    """
    Find door hinge joint names.
    If fixture_name is given, filter to that fixture.
    Otherwise return ALL joints containing 'door' or 'hinge' in their name.
    """
    joints = []
    for i in range(model.njnt):
        jname = model.joint(i).name
        is_door = "door" in jname or "hinge" in jname
        if not is_door:
            continue
        if fixture_name is None or fixture_name in jname:
            joints.append((jname, i))
    return joints


def compute_door_openness(model, data, door_joints):
    """Compute average normalized door openness (0=closed, 1=fully open)."""
    if not door_joints:
        return 0.0
    openness_vals = []
    for _, jidx in door_joints:
        addr = model.joint(jidx).qposadr[0]
        qpos = data.qpos[addr]
        jrange = model.jnt_range[jidx]
        jmin, jmax = jrange[0], jrange[1]
        if jmax - jmin > 1e-8:
            if abs(jmin) < abs(jmax):
                norm = abs(qpos - jmin) / (jmax - jmin)
            else:
                norm = abs(qpos - jmax) / (jmax - jmin)
        else:
            norm = 0.0
        openness_vals.append(np.clip(norm, 0.0, 1.0))
    return float(np.mean(openness_vals))


def build_handle_to_joint_map(handle_bodies, door_joints):
    """Map each handle body to its associated door joint(s)."""
    if len(handle_bodies) == 1 or len(door_joints) == 1:
        return {hb: door_joints for hb in handle_bodies}

    result = {}
    for hb in handle_bodies:
        hb_lower = hb.lower()
        if "left" in hb_lower:
            matched = [(jn, ji) for jn, ji in door_joints if "left" in jn.lower()]
        elif "right" in hb_lower:
            matched = [(jn, ji) for jn, ji in door_joints if "right" in jn.lower()]
        else:
            matched = []
        result[hb] = matched if matched else door_joints
    return result


def compute_handle_features(env, handle_ctx, open_threshold=0.90):
    """Compute handle_pos, handle_to_eef_pos, and door_openness from env."""
    model, data = get_mj_model_data(env)
    eef_pos = data.body("gripper0_right_eef").xpos.copy()

    handle_bodies = handle_ctx["handle_bodies"]
    handle_to_joint_map = handle_ctx["handle_to_joint_map"]

    per_door = {
        hb: compute_door_openness(model, data, handle_to_joint_map[hb])
        for hb in handle_bodies
    }
    active = [hb for hb in handle_bodies if per_door[hb] < open_threshold]
    candidates = active if active else handle_bodies
    dists = [np.linalg.norm(data.body(hb).xpos - eef_pos) for hb in candidates]
    target_handle = candidates[int(np.argmin(dists))]

    handle_pos = data.body(target_handle).xpos.copy()
    handle_to_eef = handle_pos - eef_pos
    openness = per_door[target_handle]

    return {
        "handle_pos": handle_pos.astype(np.float32),
        "handle_to_eef_pos": handle_to_eef.astype(np.float32),
        "door_openness": np.array([openness], dtype=np.float32),
    }


def check_any_door_open(env, threshold=0.90, handle_ctx=None):
    """Return True if any door joint is open past threshold."""
    # Use pre-found joints from handle_ctx when available (more reliable)
    if handle_ctx is not None:
        model, data = get_mj_model_data(env)
        for joints in handle_ctx["handle_to_joint_map"].values():
            if compute_door_openness(model, data, joints) >= threshold:
                return True
        return False

    fxtr = getattr(env, "fxtr", None)
    if fxtr is None or not hasattr(fxtr, "get_joint_state"):
        return env._check_success()

    joint_names = getattr(fxtr, "door_joint_names", None)
    if not joint_names:
        return env._check_success()

    try:
        joint_state = fxtr.get_joint_state(env, joint_names)
    except Exception:
        return env._check_success()

    return any(val >= threshold for val in joint_state.values())


def load_checkpoint_and_config(checkpoint_path):
    """Load checkpoint and config (if available)."""
    checkpoint_path = Path(checkpoint_path).absolute()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"Loaded checkpoint: {checkpoint_path}")

    run_dir = checkpoint_path.parent.parent
    config_path = run_dir / ".hydra" / "config.yaml"

    if not config_path.exists():
        config_path = run_dir / "config.yaml"

    if not config_path.exists():
        print("Config not found next to checkpoint. Falling back to checkpoint metadata.")
        return checkpoint, None

    cfg = OmegaConf.load(config_path)
    print(f"Loaded config: {config_path}")

    return checkpoint, cfg


def load_normalizer(normalizer_path):
    """Load normalizer."""
    normalizer_path = Path(normalizer_path).absolute()

    if not normalizer_path.exists():
        return None

    with open(normalizer_path, "rb") as f:
        normalizer = pickle.load(f)

    print(f"Loaded normalizer: {normalizer_path}")
    return normalizer


def build_normalizer_from_cfg(cfg):
    """Recompute normalizer from dataset config."""
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    return normalizer


def create_policy_from_local_checkpoint(checkpoint, normalizer, device):
    """Create policy from a local 06_train_policy.py checkpoint."""
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
    from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    cfg = checkpoint.get("config", {})
    model_cfg = checkpoint.get("model_config", {})
    horizon = int(cfg.get("horizon", 16))
    n_obs_steps = int(cfg.get("n_obs_steps", 2))
    n_action_steps = int(cfg.get("n_action_steps", 8))
    obs_dim = int(cfg.get("obs_dim", 23))
    action_dim = int(cfg.get("action_dim", 12))

    diffusion_step_embed_dim = int(model_cfg.get("diffusion_step_embed_dim", 64))
    down_dims = model_cfg.get("down_dims", [64, 128, 256])
    kernel_size = int(model_cfg.get("kernel_size", 3))
    n_groups = int(model_cfg.get("n_groups", 4))
    num_train_timesteps = int(model_cfg.get("num_train_timesteps", 100))
    num_inference_steps = int(model_cfg.get("num_inference_steps", 100))
    obs_as_global_cond = bool(model_cfg.get("obs_as_global_cond", True))
    pred_action_steps_only = bool(model_cfg.get("pred_action_steps_only", False))

    model = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * n_obs_steps,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=down_dims,
        kernel_size=kernel_size,
        n_groups=n_groups,
    )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    policy = DiffusionUnetLowdimPolicy(
        model=model,
        noise_scheduler=noise_scheduler,
        horizon=horizon,
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        num_inference_steps=num_inference_steps,
        obs_as_global_cond=obs_as_global_cond,
        pred_action_steps_only=pred_action_steps_only,
    ).to(device)

    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if normalizer is None and "normalizer_state_dict" in checkpoint:
        normalizer = LinearNormalizer()
        normalizer.load_state_dict(checkpoint["normalizer_state_dict"])
    if normalizer is not None:
        policy.set_normalizer(normalizer)

    return policy, normalizer


def create_policy_from_checkpoint(checkpoint, cfg, normalizer, device):
    """Create and load policy."""
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    policy = hydra.utils.instantiate(cfg.policy)
    policy = policy.to(device)

    print(f"\nPolicy Details:")
    print(f"  Type: {type(policy).__name__}")
    print(f"  Model parameters: {sum(p.numel() for p in policy.model.parameters()):,}")
    obs_feature_dim = getattr(policy, "obs_feature_dim", None)
    if obs_feature_dim is None:
        obs_feature_dim = getattr(policy, "obs_dim", None)
    if obs_feature_dim is not None:
        print(f"  obs_feature_dim: {obs_feature_dim}")
    print(f"  action_dim: {policy.action_dim}")
    print(f"  horizon: {policy.horizon}")

    policy.set_normalizer(normalizer)
    print(f"  ✓ Normalizer loaded")

    state_dicts = checkpoint.get("state_dicts", {})
    if "ema_model" in state_dicts:
        state_dict = state_dicts["ema_model"]
        print("  → Using EMA model weights")
    elif "model" in state_dicts:
        state_dict = state_dicts["model"]
        print("  → Using model weights")
    else:
        raise ValueError("No model weights found in checkpoint")

    policy.load_state_dict(state_dict, strict=False)
    print("  ✓ Loaded model state successfully")

    policy.eval()
    return policy


# ---------------------------------------------------------------------------
# Key mapping
# ---------------------------------------------------------------------------

KEY_MAPPING = {
    "base_pos": "robot0_base_pos",
    "base_quat": "robot0_base_quat",
    "robot0_base_to_eef_pos": "robot0_base_to_eef_pos",
    "robot0_base_to_eef_quat": "robot0_base_to_eef_quat",
    "robot0_gripper_qpos": "robot0_gripper_qpos",
    "handle_pos": "handle_pos",
    "handle_to_eef_pos": "handle_to_eef_pos",
    "door_openness": "door_openness",
}


def extract_single_obs_vec(obs_raw, training_keys, obs_meta=None, debug=False):
    """
    Extract a single flat observation vector from the environment obs dict.

    Returns a 1-D float32 array of shape (obs_dim,), or None on failure.
    The ordering matches the concatenation order used during training.
    """
    parts = []
    missing = []

    for key in training_keys:
        env_key = KEY_MAPPING.get(key, key)

        if env_key not in obs_raw:
            # Fall back to the raw training key name
            if key in obs_raw:
                env_key = key
            elif obs_meta and key in obs_meta:
                # Zero-fill with the expected shape
                shape = obs_meta[key]["shape"]
                parts.append(np.zeros(int(np.prod(shape)), dtype=np.float32))
                if debug:
                    print(f"  {key} → ZERO-FILL")
                continue
            else:
                missing.append(key)
                continue

        val = obs_raw[env_key]
        if torch.is_tensor(val):
            val = val.detach().cpu().numpy()
        val = np.atleast_1d(np.array(val, dtype=np.float32)).flatten()
        parts.append(val)

        if debug:
            print(f"  {key} → {env_key}  shape={val.shape}")

    if missing:
        avail = [k for k in obs_raw.keys() if not k.endswith("_image")]
        print(f"  ERROR: missing keys {missing}")
        print(f"  Available obs keys: {avail}")
        return None

    return np.concatenate(parts, axis=0)


def run_evaluation(
    policy,
    cfg,
    num_rollouts,
    max_steps,
    split,
    seed,
    save_video=True,
    video_dir="meow",
    video_fps=20,
    video_width=256,
    video_height=256,
    fix_action_semantics=False,
    assist_base_to_handle=False,
    assist_base_gain=0.2,
    debug=False,
):
    """Run evaluation rollouts with a rolling observation buffer and action chunking."""

    device = next(policy.parameters()).device

    # Resolve policy hyper-params
    n_obs_steps = int(getattr(policy, "n_obs_steps", 2))
    n_action_steps = int(getattr(policy, "n_action_steps", 8))

    # Get expected obs keys from config
    shape_meta = cfg.get("shape_meta", {})
    obs_meta = shape_meta.get("obs", {})
    training_keys = list(obs_meta.keys()) if obs_meta else []

    print(f"\nExpected observation keys from training: {training_keys}")
    print(f"n_obs_steps:    {n_obs_steps}")
    print(f"n_action_steps: {n_action_steps}")

    # Create environment
    print(f"Creating environment...")
    env = create_env(
        env_name="OpenCabinet",
        render_onscreen=False,
        seed=seed,
        split=split,
        camera_widths=video_width,
        camera_heights=video_height,
    )

    results = {
        "successes": [],
        "episode_lengths": [],
        "rewards": [],
    }

    # handle_ctx is built after the first env.reset() — env.sim is None until then
    handle_ctx = None
    needs_handle_ctx = any(k in training_keys for k in {"handle_pos", "handle_to_eef_pos", "door_openness"})

    if save_video:
        import os
        import imageio.v2 as imageio

        os.makedirs(video_dir, exist_ok=True)
        print(f"Saving videos to: {video_dir}")

        def render_frame():
            try:
                return env.sim.render(
                    height=video_height,
                    width=video_width,
                    camera_name="robot0_agentview_center",
                )[::-1]
            except Exception:
                try:
                    return env.sim.render(
                        width=video_width,
                        height=video_height,
                        camera_name=None,
                    )
                except Exception:
                    return None
    else:
        imageio = None
        render_frame = None

    print(f"\nRunning {num_rollouts} evaluation episodes...")

    printed_action_fix = False
    for ep in range(num_rollouts):
        obs = env.reset()
        frames = []
        if save_video and render_frame is not None:
            frame = render_frame()
            if frame is not None:
                frames.append(frame)

        # Build handle_ctx on first episode (env.sim is populated only after reset)
        if needs_handle_ctx and handle_ctx is None:
            mj_model, _ = get_mj_model_data(env)
            fxtr = getattr(env, "fxtr", None)
            fixture_name = getattr(fxtr, "name", None) if fxtr is not None else None

            handle_bodies = find_fixture_handle_bodies(mj_model, fixture_name)
            door_joints   = find_fixture_door_joints(mj_model, fixture_name)

            if not handle_bodies:
                handle_bodies = find_fixture_handle_bodies(mj_model, fixture_name=None)
            if not door_joints:
                door_joints = find_fixture_door_joints(mj_model, fixture_name=None)

            if handle_bodies:
                handle_ctx = {
                    "handle_bodies": handle_bodies,
                    "handle_to_joint_map": build_handle_to_joint_map(handle_bodies, door_joints),
                }
                print(f"  handle_ctx: bodies={handle_bodies}  joints={[j[0] for j in door_joints]}")
            else:
                print("  WARNING: No handle bodies found — handle features will be zero.")
                all_bodies = [mj_model.body(i).name for i in range(mj_model.nbody)]
                print(f"  All body names: {all_bodies}")

        # ------------------------------------------------------------------
        # FIX 1: Rolling observation buffer
        # Maintains a deque of the last n_obs_steps observation vectors so
        # the policy always sees temporally distinct frames, exactly as it
        # was trained on.  We pre-fill with the first obs (standard padding).
        # ------------------------------------------------------------------
        obs_buffer = deque(maxlen=n_obs_steps)

        def obs_to_vec(raw_obs):
            aug = raw_obs
            if handle_ctx is not None:
                try:
                    extra = compute_handle_features(env, handle_ctx)
                    aug = {**raw_obs, **extra}
                    # print("handle_pos from env:", extra["handle_pos"])
                    # print("handle_to_eef_pos:", extra["handle_to_eef_pos"])
                except Exception as e:
                    if ep == 0:
                        print(f"Warning: handle feature computation failed: {e}")
            return extract_single_obs_vec(
                aug,
                training_keys,
                obs_meta=obs_meta,
                debug=(debug and ep == 0),
            )

        # Pre-fill buffer with first observation (replicates pad_before behaviour)
        first_vec = obs_to_vec(obs)
        if first_vec is None:
            print(f"  Episode {ep + 1}: Could not extract initial observation — skipping.")
            results["successes"].append(False)
            results["episode_lengths"].append(0)
            results["rewards"].append(0.0)
            continue

        for _ in range(n_obs_steps):
            obs_buffer.append(first_vec.copy())

        ep_reward = 0.0
        success = False
        global_step = 0  # steps taken this episode

        while global_step < max_steps:
            # Build (1, n_obs_steps, obs_dim) tensor for the policy
            obs_seq = np.stack(list(obs_buffer), axis=0)          # (n_obs_steps, obs_dim)
            obs_tensor = torch.from_numpy(obs_seq).unsqueeze(0).to(device)  # (1, T, D)

            # Predict action chunk
            try:
                with torch.no_grad():
                    if type(policy).__name__ == "DiffusionUnetLowdimPolicy":
                        obs_dict_torch = {"obs": obs_tensor}
                    else:
                        # Generic key-based policy: split back into per-key tensors
                        obs_dict_torch = {}
                        cursor = 0
                        for key in training_keys:
                            dim = int(np.prod(obs_meta[key]["shape"])) if obs_meta else obs_seq.shape[-1]
                            obs_dict_torch[key] = obs_tensor[..., cursor: cursor + dim]
                            cursor += dim

                    result = policy.predict_action(obs_dict_torch)
                    # action shape: (1, n_action_steps, action_dim)  or (1, horizon, action_dim)
                    actions = result["action"][0].cpu().numpy()  # (n_action_steps, action_dim)

            except Exception as e:
                print(f"  Episode {ep + 1}, env-step {global_step}: prediction error: {repr(e)}")
                break

            # ------------------------------------------------------------------
            # FIX 2: Execute the full predicted action chunk (receding horizon)
            # Re-query the policy only after consuming all n_action_steps actions.
            # ------------------------------------------------------------------
            chunk_len = min(n_action_steps, len(actions), max_steps - global_step)
            for i in range(chunk_len):
                action_np = actions[i]

                if fix_action_semantics and action_np.shape[-1] >= 12:
                    # Action layout: eef_pos(3), eef_rot(3), gripper(1), base_motion(4), control_mode(1)
                    action_np = action_np.copy()
                    action_np[7:11] = 0.0  # zero base motion
                    action_np[11] = 1.0    # force control_mode
                    if not printed_action_fix:
                        print("  Applying action fix: zero base_motion, set control_mode=1")
                        printed_action_fix = True

                if assist_base_to_handle and handle_ctx is not None and action_np.shape[-1] >= 12:
                    try:
                        extra = compute_handle_features(env, handle_ctx)
                        to_eef = extra["handle_to_eef_pos"]
                        base_xy = np.array([to_eef[0], to_eef[1]], dtype=np.float32)
                        norm = np.linalg.norm(base_xy)
                        if norm > 1e-6:
                            base_xy = base_xy / norm
                        # nudge base towards handle in the ground plane
                        action_np = action_np.copy()
                        action_np[7] += assist_base_gain * base_xy[0]
                        action_np[8] += assist_base_gain * base_xy[1]
                    except Exception as e:
                        if debug and ep == 0:
                            print(f"  Warning: base-assist failed: {e}")

                # print(f"Raw action sample: {action_np}")
                # print(f"Action min: {action_np.min():.3f}, max: {action_np.max():.3f}, mean: {action_np.mean():.3f}")

                try:
                    obs, reward, done, info = env.step(action_np)
                except Exception as e:
                    print(f"  Episode {ep + 1}, env-step {global_step + i}: step error: {e}")
                    done = True
                    reward = 0.0

                ep_reward += reward
                global_step += 1

                if save_video and render_frame is not None:
                    frame = render_frame()
                    if frame is not None:
                        frames.append(frame)

                # Update rolling buffer with the new observation
                new_vec = obs_to_vec(obs)
                if new_vec is not None:
                    obs_buffer.append(new_vec)

                if check_any_door_open(env, handle_ctx=handle_ctx):
                    success = True
                    break

                if done:
                    break

            if success or done or global_step >= max_steps:
                break

        results["successes"].append(success)
        results["episode_lengths"].append(global_step)
        results["rewards"].append(ep_reward)

        status = "✓ SUCCESS" if success else "✗ FAIL"
        print(
            f"  Episode {ep + 1:3d}/{num_rollouts}: {status} "
            f"(steps={global_step:4d}, reward={ep_reward:.2f})"
        )

        if save_video and frames:
            out_path = os.path.join(video_dir, f"episode_{ep + 1:03d}.mp4")
            try:
                imageio.mimsave(out_path, frames, fps=video_fps)
            except Exception as e:
                print(f"  Warning: failed to save video {out_path} ({e})")

    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Diffusion Policy on OpenCabinet (Proper Key Mapping)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to policy checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--normalizer",
        type=str,
        default=None,
        help="Path to normalizer.pkl (optional, will auto-find)",
    )
    parser.add_argument(
        "--num_rollouts", type=int, default=5, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--max_steps", type=int, default=500, help="Max steps per episode"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="pretrain",
        choices=["pretrain", "target"],
        help="Kitchen scene split",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save evaluation rollouts as mp4 videos",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="eval_videos",
        help="Directory to save evaluation videos",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=20,
        help="FPS for saved videos",
    )
    parser.add_argument(
        "--video_width",
        type=int,
        default=256,
        help="Video width",
    )
    parser.add_argument(
        "--video_height",
        type=int,
        default=256,
        help="Video height",
    )
    parser.add_argument(
        "--fix_action_semantics",
        action="store_true",
        help="Zero base motion and force control_mode=1 (12D action layout)",
    )
    parser.add_argument(
        "--assist_base_to_handle",
        action="store_true",
        help="Nudge base toward handle using handle_to_eef_pos (eval-time heuristic)",
    )
    parser.add_argument(
        "--assist_base_gain",
        type=float,
        default=0.2,
        help="Strength of base-to-handle assist (eval-time heuristic)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Diffusion Policy - OpenCabinet Evaluation")
    print("=" * 60)

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    checkpoint, cfg = load_checkpoint_and_config(args.checkpoint)
    if cfg is None:
        cfg = OmegaConf.create({"shape_meta": checkpoint.get("shape_meta", {})})

    checkpoint_dir = Path(args.checkpoint).parent.parent
    normalizer_path = args.normalizer or (checkpoint_dir / "normalizer.pkl")

    normalizer = load_normalizer(normalizer_path=normalizer_path)
    if normalizer is None and "normalizer_state_dict" in checkpoint:
        from diffusion_policy.model.common.normalizer import LinearNormalizer

        normalizer = LinearNormalizer()
        normalizer.load_state_dict(checkpoint["normalizer_state_dict"])
        print("Loaded normalizer from checkpoint state.")

    if normalizer is None and cfg is not None and "task" in cfg:
        print(f"Normalizer not found at {normalizer_path}. Recomputing from dataset...")
        normalizer = build_normalizer_from_cfg(cfg)
        try:
            with open(normalizer_path, "wb") as f:
                pickle.dump(normalizer, f)
            print(f"Saved normalizer: {normalizer_path}")
        except Exception as e:
            print(f"Warning: failed to save normalizer ({e})")

    if cfg is not None and "policy" in cfg:
        policy = create_policy_from_checkpoint(checkpoint, cfg, normalizer, device)
    else:
        policy, normalizer = create_policy_from_local_checkpoint(
            checkpoint, normalizer, device
        )

    print_section(f"Evaluating on {args.split} split ({args.num_rollouts} episodes)")

    results = run_evaluation(
        policy=policy,
        cfg=cfg,
        num_rollouts=args.num_rollouts,
        max_steps=args.max_steps,
        split=args.split,
        seed=args.seed,
        save_video=args.save_video,
        video_dir=args.video_dir,
        video_fps=args.video_fps,
        video_width=args.video_width,
        video_height=args.video_height,
        fix_action_semantics=args.fix_action_semantics,
        assist_base_to_handle=args.assist_base_to_handle,
        assist_base_gain=args.assist_base_gain,
        debug=args.debug,
    )

    print_section("Evaluation Results")

    num_success = sum(results["successes"])
    success_rate = num_success / args.num_rollouts * 100
    avg_length = np.mean(results["episode_lengths"])
    avg_reward = np.mean(results["rewards"])

    print(f"  Split:          {args.split}")
    print(f"  Episodes:       {args.num_rollouts}")
    print(f"  Successes:      {num_success}/{args.num_rollouts}")
    print(f"  Success rate:   {success_rate:.1f}%")
    print(f"  Avg ep length:  {avg_length:.1f} steps")
    print(f"  Avg reward:     {avg_reward:.3f}")

    print_section("Interpretation")

    if success_rate == 0:
        if avg_length > 100:
            print("  ✓ Policy is running and taking steps!")
            print("    However, it's not solving tasks (0% success rate).")
            print("    → Consider training longer, larger down_dims, or more data.")
        else:
            print("  ✗ Policy is failing very quickly.")
            print("    → Check observation/action dimensions.")
    else:
        print(f"  ✓ Policy achieved {success_rate:.1f}% success!")

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
