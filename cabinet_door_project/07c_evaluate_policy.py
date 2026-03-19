"""
Evaluation script tuned for lowdim checkpoints trained by 06_train_policy.py.

Includes:
- Correct obs key mapping + computed handle features (handle_pos, handle_to_eef_pos, door_openness)
- Correct action remap from training layout -> env layout
- Per-dimension action clipping to controller limits (default on)
- Optional clamp to reduce saturation
- Optional base-to-handle assist (default on)
- Video recording (agentview or first-person)
"""

import argparse
import os
import sys

# Force osmesa (CPU offscreen renderer) on Linux/WSL2
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np

import robocasa  # noqa: F401
from robocasa.utils.env_utils import create_env


def print_section(title):
    """
    Print a formatted banner to separate major stages in logs.
    
    Args:
        title: Section title to display.
    
    Returns:
        None.
    """
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def get_mj_model_data(env):
    """
    Retrieve MuJoCo model/data objects across robosuite variants.
    
    Args:
        env: Robosuite/RoboCasa environment instance.
    
    Returns:
        Tuple of (model, data) or None if not found.
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
    Find door hinge joint names in the MuJoCo model.
    
    Args:
        model: MuJoCo model.
        fixture_name: Optional fixture name to filter.
    
    Returns:
        List of (joint_name, joint_index) tuples.
    """
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
    """
    Map handle bodies to their associated door joints.
    
    Args:
        handle_bodies: List of handle body names.
        door_joints: List of (joint_name, joint_index) tuples.
    
    Returns:
        Dict mapping handle body name -> list of door joints.
    """
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
    """
    Check whether any cabinet door is open past a threshold.
    
    Args:
        env: RoboCasa environment instance.
        threshold: Openness threshold.
        handle_ctx: Optional handle context for joint lookup.
    
    Returns:
        True if any door is open beyond threshold.
    """
    parts = []
    missing = []
    for key in training_keys:
        env_key = KEY_MAPPING.get(key, key)
        if env_key not in obs_raw:
            if key in obs_raw:
                env_key = key
            elif obs_meta and key in obs_meta:
                shape = obs_meta[key]["shape"]
                parts.append(np.zeros(int(np.prod(shape)), dtype=np.float32))
                if debug:
                    print(f"  {key} → ZERO-FILL (key not found in obs!)")
                continue
            else:
                missing.append(key)
                continue
        val = obs_raw[env_key]
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


def remap_action(raw):
    """
    Reorder a policy action into the env's expected 12D action layout.
    
    Args:
        raw: 1-D policy action vector.
    
    Returns:
        1-D float32 numpy array in env action order.
    """
    action_env = np.zeros(12, dtype=np.float32)
    action_env[0:6] = raw[5:11]
    action_env[6] = raw[11]
    action_env[7:10] = raw[0:3]
    action_env[10] = raw[3]
    action_env[11] = raw[4]
    return action_env


def load_unet_lowdim_policy(checkpoint_path, device):
    """Load a local U-Net lowdim policy checkpoint from 06_train_policy.py."""
    import torch
    from diffusion_policy.diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
    from diffusion_policy.diffusion_policy.policy.diffusion_unet_lowdim_policy import (
        DiffusionUnetLowdimPolicy,
    )
    from diffusion_policy.diffusion_policy.model.common.normalizer import LinearNormalizer
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    shape_meta = checkpoint.get("shape_meta", {})
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
    num_train_timesteps = int(model_cfg.get("num_train_timesteps", 25))
    num_inference_steps = int(model_cfg.get("num_inference_steps", 25))
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

    if "normalizer_state_dict" in checkpoint:
        normalizer = LinearNormalizer()
        normalizer.load_state_dict(checkpoint["normalizer_state_dict"])
        policy.set_normalizer(normalizer)

    policy.eval()
    print(f"Loaded U-Net lowdim policy from: {checkpoint_path}")
    print(f"  obs_dim: {obs_dim}, action_dim: {action_dim}")
    print(f"  horizon: {horizon}, n_obs_steps: {n_obs_steps}, n_action_steps: {n_action_steps}")
    return policy, shape_meta


def run_evaluation(
    policy,
    shape_meta,
    num_rollouts,
    max_steps,
    split,
    seed,
    video_path=None,
    video_width=256,
    video_height=256,
    video_fps=20,
    video_first_person=False,
    clamp_action=None,
    clip_action_limits=True,
    assist_base_to_handle=True,
    assist_base_gain=0.2,
    assist_distance_threshold=0.15,
    debug=False,
):
    import torch
    import imageio
    from collections import deque

    device = next(policy.parameters()).device
    n_obs_steps = int(getattr(policy, "n_obs_steps", 2))
    n_action_steps = int(getattr(policy, "n_action_steps", 8))

    obs_meta = shape_meta.get("obs", {})
    training_keys = list(obs_meta.keys()) if obs_meta else []
    expected_obs_dim = sum(int(np.prod(v["shape"])) for v in obs_meta.values())

    env = create_env(
        env_name="OpenCabinet",
        render_onscreen=False,
        seed=seed,
        split=split,
        camera_widths=video_width,
        camera_heights=video_height,
    )

    obs = env.reset()
    controller = env.robots[0].composite_controller
    action_low = controller.action_limits[0].astype(np.float32)
    action_high = controller.action_limits[1].astype(np.float32)

    handle_ctx = None
    needs_handle_ctx = any(
        k in training_keys for k in {"handle_pos", "handle_to_eef_pos", "door_openness"}
    )
    if needs_handle_ctx:
        mj_model, _ = get_mj_model_data(env)
        fxtr = getattr(env, "fxtr", None)
        fixture_name = getattr(fxtr, "name", None) if fxtr is not None else None
        handle_bodies = find_fixture_handle_bodies(mj_model, fixture_name)
        door_joints = find_fixture_door_joints(mj_model, fixture_name)
        if not handle_bodies:
            handle_bodies = find_fixture_handle_bodies(mj_model, fixture_name=None)
        if not door_joints:
            door_joints = find_fixture_door_joints(mj_model, fixture_name=None)
        if handle_bodies:
            handle_ctx = {
                "handle_bodies": handle_bodies,
                "handle_to_joint_map": build_handle_to_joint_map(handle_bodies, door_joints),
            }
        else:
            print("WARNING: No handle bodies found — handle features will be zero.")

    if handle_ctx is not None:
        obs = {**obs, **compute_handle_features(env, handle_ctx)}
    test_vec = extract_single_obs_vec(obs, training_keys, obs_meta=obs_meta, debug=debug)
    if test_vec is None or test_vec.shape[0] != expected_obs_dim:
        print("ERROR: Obs vector mismatch. Check KEY_MAPPING and handle features.")
        sys.exit(1)

    if video_path:
        os.makedirs(video_path, exist_ok=True)

    def render_frame():
        if video_first_person:
            try:
                return env.render(
                    mode="rgb_array",
                    width=video_width,
                    height=video_height,
                )
            except Exception:
                try:
                    return env.sim.render(
                        width=video_width,
                        height=video_height,
                        camera_name=None,
                    )
                except Exception:
                    return None
        try:
            return env.sim.render(
                height=video_height, width=video_width, camera_name="robot0_agentview_center"
            )[::-1]
        except Exception:
            return None

    results = {
        "successes": [],
        "episode_lengths": [],
        "rewards": [],
    }

    for ep in range(num_rollouts):
        obs = env.reset()
        frames = []
        if video_path:
            frame = render_frame()
            if frame is not None:
                frames.append(frame)

        obs_buffer = deque(maxlen=n_obs_steps)
        obs_aug = obs
        if handle_ctx is not None:
            try:
                obs_aug = {**obs, **compute_handle_features(env, handle_ctx)}
            except Exception:
                obs_aug = obs
        first_vec = extract_single_obs_vec(obs_aug, training_keys, obs_meta=obs_meta)
        if first_vec is None:
            results["successes"].append(False)
            results["episode_lengths"].append(0)
            results["rewards"].append(0.0)
            continue
        for _ in range(n_obs_steps):
            obs_buffer.append(first_vec.copy())

        ep_reward = 0.0
        success = False
        global_step = 0

        while global_step < max_steps:
            obs_seq = np.stack(list(obs_buffer), axis=0)
            obs_tensor = torch.from_numpy(obs_seq).unsqueeze(0).to(device)
            with torch.no_grad():
                result = policy.predict_action({"obs": obs_tensor})
                actions = result["action"][0].cpu().numpy()

            chunk_len = min(n_action_steps, len(actions), max_steps - global_step)
            for i in range(chunk_len):
                raw = actions[i].copy()
                action_env = remap_action(raw)

                if assist_base_to_handle and handle_ctx is not None:
                    try:
                        extra = compute_handle_features(env, handle_ctx)
                        to_eef = extra["handle_to_eef_pos"]
                        base_xy = np.array([to_eef[0], to_eef[1]], dtype=np.float32)
                        dist = np.linalg.norm(base_xy)
                        if dist > assist_distance_threshold:
                            base_xy = base_xy / max(dist, 1e-6)
                            action_env[7] += assist_base_gain * base_xy[0]
                            action_env[8] += assist_base_gain * base_xy[1]
                    except Exception:
                        pass

                if clamp_action is not None:
                    action_env = np.clip(action_env, -clamp_action, clamp_action)
                if clip_action_limits:
                    action_env = np.clip(action_env, action_low, action_high)

                obs, reward, done, info = env.step(action_env)
                ep_reward += reward
                global_step += 1

                if video_path:
                    frame = render_frame()
                    if frame is not None:
                        frames.append(frame)

                obs_aug = obs
                if handle_ctx is not None:
                    try:
                        obs_aug = {**obs, **compute_handle_features(env, handle_ctx)}
                    except Exception:
                        obs_aug = obs
                new_vec = extract_single_obs_vec(obs_aug, training_keys, obs_meta=obs_meta)
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

        status = "SUCCESS" if success else "FAIL"
        print(
            f"  Episode {ep + 1:3d}/{num_rollouts}: {status:7s} "
            f"(steps={global_step:4d}, reward={ep_reward:.2f})"
        )

        if video_path and frames:
            out_path = os.path.join(video_path, f"episode_{ep + 1:03d}.mp4")
            try:
                imageio.mimsave(out_path, frames, fps=video_fps)
            except Exception as e:
                print(f"  Warning: failed to save video {out_path} ({e})")

    env.close()
    return results


def main():
    """
    Parse CLI args and launch evaluation/training.
    """
    parser = argparse.ArgumentParser(description="Max-success evaluation for OpenCabinet")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--num_rollouts", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=600, help="Max steps per episode")
    parser.add_argument(
        "--split",
        type=str,
        default="pretrain",
        choices=["pretrain", "target"],
        help="Kitchen scene split to evaluate on",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--video_path", type=str, default=None, help="Directory to save videos")
    parser.add_argument("--video_width", type=int, default=256, help="Video width")
    parser.add_argument("--video_height", type=int, default=256, help="Video height")
    parser.add_argument("--video_fps", type=int, default=20, help="Video FPS")
    parser.add_argument(
        "--video_first_person",
        action="store_true",
        help="Render from env first-person view (fallback to default camera)",
    )
    parser.add_argument(
        "--clamp_action",
        type=float,
        default=0.25,
        help="Clamp env action to +/- this value (set None to disable)",
    )
    parser.add_argument(
        "--no_clip_action_limits",
        action="store_true",
        help="Disable per-dimension clipping to controller limits",
    )
    parser.add_argument(
        "--assist_base_to_handle",
        action="store_true",
        help="Enable base-to-handle assist (default off unless flag is set)",
    )
    parser.add_argument(
        "--assist_base_gain",
        type=float,
        default=0.2,
        help="Strength of base-to-handle assist",
    )
    parser.add_argument(
        "--assist_distance_threshold",
        type=float,
        default=0.15,
        help="Distance threshold before applying base assist",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug prints")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    print_section("OpenCabinet - Max-Success Evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    policy, shape_meta = load_unet_lowdim_policy(args.checkpoint, device)

    results = run_evaluation(
        policy=policy,
        shape_meta=shape_meta,
        num_rollouts=args.num_rollouts,
        max_steps=args.max_steps,
        split=args.split,
        seed=args.seed,
        video_path=args.video_path,
        video_width=args.video_width,
        video_height=args.video_height,
        video_fps=args.video_fps,
        video_first_person=args.video_first_person,
        clamp_action=args.clamp_action,
        clip_action_limits=not args.no_clip_action_limits,
        assist_base_to_handle=args.assist_base_to_handle,
        assist_base_gain=args.assist_base_gain,
        assist_distance_threshold=args.assist_distance_threshold,
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


if __name__ == "__main__":
    main()
