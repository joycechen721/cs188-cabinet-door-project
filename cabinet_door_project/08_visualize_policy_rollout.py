"""
Step 8: Visualize a Policy Rollout
=====================================
Loads a trained policy checkpoint from 06_train_policy.py and runs it
live in the OpenCabinet environment so you can watch the robot.

This is your primary debugging tool: watch exactly where and why the policy
fails — does it reach for the handle? Does it grasp? Does it pull correctly?

Two rendering modes:
  On-screen  (default)  — interactive MuJoCo viewer window, real-time
  Off-screen (--offscreen) — renders to a video file, works without a display

Usage:
    # Watch live in a window (WSL/Linux) + save video
    python 08_visualize_policy_rollout.py --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt

    # Save to video only (no display needed — works headless / in notebooks)
    python 08_visualize_policy_rollout.py --checkpoint ... --offscreen

    # Run 3 episodes, slow down playback so you can follow along
    python 08_visualize_policy_rollout.py --checkpoint ... --num_episodes 3 --max_steps 200

    # Mac users must use mjpython for the on-screen window
    mjpython 08_visualize_policy_rollout.py --checkpoint ...
"""

import os
import sys

# ── Rendering mode detection ────────────────────────────────────────────────
# We peek at sys.argv *before* argparse so we can configure the GL backend
# before any library is imported.  Wrong GL backend = gladLoadGL error.
_OFFSCREEN = "--offscreen" in sys.argv

if _OFFSCREEN:
    # Off-screen mode: use Mesa's software osmesa renderer.
    # EGL is the default on headless Linux but fails on WSL2 (no /dev/dri).
    if sys.platform == "linux":
        os.environ.setdefault("MUJOCO_GL", "osmesa")
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
else:
    # On-screen mode: re-exec with correct display vars baked into the OS
    # environment so Mesa (GLFW) sees them before any C library initializes.
    # On WSLg the .bashrc often sets a stale VcXsrv-style DISPLAY that
    # breaks GLFW; os.execve() restarts the process cleanly.
    if sys.platform == "linux" and "__TELEOP_DISPLAY_OK" not in os.environ:
        _env = dict(os.environ)
        _changed = False
        if _env.get("WAYLAND_DISPLAY"):
            if not _env.get("DISPLAY", "").startswith(":"):
                _env["DISPLAY"] = ":0"
                _changed = True
            if _env.get("GALLIUM_DRIVER") != "llvmpipe":
                _env["GALLIUM_DRIVER"] = "llvmpipe"
                _changed = True
            if _env.get("MESA_GL_VERSION_OVERRIDE") != "4.5":
                _env["MESA_GL_VERSION_OVERRIDE"] = "4.5"
                _changed = True
        if _changed:
            _env["__TELEOP_DISPLAY_OK"] = "1"
            os.execve(sys.executable, [sys.executable] + sys.argv, _env)
        else:
            os.environ["__TELEOP_DISPLAY_OK"] = "1"
# ────────────────────────────────────────────────────────────────────────────

import argparse
import time

import numpy as np
import robocasa  # noqa: F401 — registers OpenCabinet environment
from robocasa.utils.env_utils import create_env
from robosuite.wrappers import VisualizationWrapper


# ── Policy loading (matches 07_evaluate_policy.py) ──────────────────────────

def load_unet_lowdim_policy(checkpoint, device):
    """Create and load a local U-Net lowdim policy from checkpoint dict."""
    import torch
    from diffusion_policy.diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
    from diffusion_policy.diffusion_policy.policy.diffusion_unet_lowdim_policy import (
        DiffusionUnetLowdimPolicy,
    )
    from diffusion_policy.diffusion_policy.model.common.normalizer import LinearNormalizer
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

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
    noise_scheduler = DDIMScheduler(
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

    normalizer = None
    if "normalizer_state_dict" in checkpoint:
        normalizer = LinearNormalizer()
        normalizer.load_state_dict(checkpoint["normalizer_state_dict"])
        policy.set_normalizer(normalizer)

    policy.eval()
    return policy, shape_meta


def load_policy(checkpoint_path, device):
    """Load either a simple MLP policy or a U-Net lowdim policy."""
    import torch
    import torch.nn as nn

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "shape_meta" in ckpt or "model_config" in ckpt:
        policy, shape_meta = load_unet_lowdim_policy(ckpt, device)
        return {
            "mode": "unet",
            "policy": policy,
            "shape_meta": shape_meta,
            "ckpt": ckpt,
        }

    state_dim = ckpt["state_dim"]
    action_dim = ckpt["action_dim"]

    class SimplePolicy(nn.Module):
        def __init__(self, state_dim, action_dim, hidden_dim=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh(),
            )

        def forward(self, state):
            return self.net(state)

    model = SimplePolicy(state_dim, action_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return {
        "mode": "simple",
        "model": model,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "ckpt": ckpt,
    }


def extract_state(obs, state_dim):
    """Flatten non-image observations into a state vector of length state_dim."""
    parts = []
    for key in sorted(obs.keys()):
        val = obs[key]
        if isinstance(val, np.ndarray) and not key.endswith("_image"):
            parts.append(val.flatten())
    if not parts:
        return np.zeros(state_dim, dtype=np.float32)
    state = np.concatenate(parts).astype(np.float32)
    if len(state) < state_dim:
        state = np.pad(state, (0, state_dim - len(state)))
    elif len(state) > state_dim:
        state = state[:state_dim]
    return state


KEY_MAPPING = {
    "base_pos": "robot0_base_pos",
    "base_quat": "robot0_base_quat",
    "robot0_base_to_eef_pos": "robot0_base_to_eef_pos",
    "robot0_base_to_eef_quat": "robot0_base_to_eef_quat",
    "robot0_gripper_qpos": "robot0_gripper_qpos",
    "handle_pos": "door_obj_pos",
    "handle_to_eef_pos": "door_obj_to_robot0_eef_pos",
    "door_openness": "door_openness",
}

def get_mj_model_data(env):
    """Robustly retrieve MuJoCo model/data across robosuite versions."""
    candidates = [env]
    for attr in ("env", "_env", "unwrapped"):
        if hasattr(env, attr):
            candidates.append(getattr(env, attr))

    for obj in candidates:
        if hasattr(obj, "sim") and obj.sim is not None:
            sim = obj.sim
            if hasattr(sim, "model") and hasattr(sim, "data"):
                return sim.model, sim.data
            if hasattr(sim, "_model") and hasattr(sim, "_data"):
                return sim._model, sim._data

        if hasattr(obj, "model") and hasattr(obj, "data"):
            m, d = obj.model, obj.data
            if hasattr(m, "nbody") and hasattr(d, "qpos"):
                return m, d

        if hasattr(obj, "physics"):
            ph = obj.physics
            if hasattr(ph, "model") and hasattr(ph, "data"):
                return ph.model, ph.data


def find_fixture_handle_bodies(model, fixture_name=None):
    """Find MuJoCo body names for door handles."""
    handle_bodies = []
    for i in range(model.nbody):
        name = model.body(i).name
        if "handle" not in name:
            continue
        if fixture_name is None or fixture_name in name:
            handle_bodies.append(name)
    return handle_bodies


def find_fixture_door_joints(model, fixture_name=None):
    """Find door hinge joint names."""
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


def extract_single_obs_vec(obs_raw, training_keys, obs_meta=None, debug=False):
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
    """Remap policy action (12-dim) to env action (12-dim)."""
    action_env = np.zeros(12, dtype=np.float32)
    action_env[0:6] = raw[5:11]    # eef_pos + eef_rot
    action_env[6] = raw[11]        # gripper
    action_env[7:10] = raw[0:3]    # base_motion x,y,yaw → base
    action_env[10] = raw[3]        # base_motion[3] → torso lift
    action_env[11] = raw[4]        # control_mode → control mode toggle
    return action_env


# ── On-screen rollout ────────────────────────────────────────────────────────

def run_onscreen(policy_bundle, args):
    """
    Run the policy with an interactive MuJoCo viewer window.

    The viewer opens automatically; you can pan/zoom/rotate the camera
    with the mouse while the robot executes the policy.
    """
    import torch

    if policy_bundle["mode"] == "simple":
        device = next(policy_bundle["model"].parameters()).device
    else:
        device = next(policy_bundle["policy"].parameters()).device

    env = create_env(
        env_name="OpenCabinet",
        render_onscreen=True,
        seed=args.seed,
        split=args.split,
    )
    env = VisualizationWrapper(env)
    controller = env.robots[0].composite_controller
    action_low = controller.action_limits[0].astype(np.float32)
    action_high = controller.action_limits[1].astype(np.float32)

    successes = 0
    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep + 1}/{args.num_episodes} ---")
        obs = env.reset()
        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", "")
        print(f"  Task:    {lang}")
        print(f"  Layout:  {env.layout_id}   Style: {env.style_id}")
        print(f"  Running for up to {args.max_steps} steps...")
        print(f"  (Watch the viewer window — use mouse to orbit the camera)\n")

        success = False

        if policy_bundle["mode"] == "simple":
            state_dim = policy_bundle["state_dim"]
            model = policy_bundle["model"]
            for step in range(args.max_steps):
                state = extract_state(obs, state_dim)
                with torch.no_grad():
                    action = model(
                        torch.from_numpy(state).unsqueeze(0).to(device)
                    ).cpu().numpy().squeeze(0)

                env_dim = env.action_dim
                if len(action) < env_dim:
                    action = np.pad(action, (0, env_dim - len(action)))
                elif len(action) > env_dim:
                    action = action[:env_dim]

                if args.clamp_action is not None:
                    action = np.clip(action, -args.clamp_action, args.clamp_action)
                if not args.no_clip_action_limits and action.shape == action_low.shape:
                    action = np.clip(action, action_low, action_high)

                obs, reward, done, info = env.step(action)

                if step % 20 == 0:
                    checking = env._check_success()
                    status = "cabinet OPEN" if checking else "in progress"
                    act_mag = float(np.abs(action).mean())
                    print(
                        f"  step {step:4d}  reward={reward:+.3f}  "
                        f"action_mag={act_mag:.3f}  [{status}]"
                    )

                if check_any_door_open(env):
                    success = True
                    break

                time.sleep(1.0 / args.max_fr)
        else:
            from collections import deque

            policy = policy_bundle["policy"]
            shape_meta = policy_bundle["shape_meta"]
            n_obs_steps = int(getattr(policy, "n_obs_steps", 2))
            n_action_steps = int(getattr(policy, "n_action_steps", 8))
            obs_meta = shape_meta.get("obs", {})
            training_keys = list(obs_meta.keys()) if obs_meta else []
            expected_obs_dim = sum(int(np.prod(v["shape"])) for v in obs_meta.values())

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
                        "handle_to_joint_map": build_handle_to_joint_map(
                            handle_bodies, door_joints
                        ),
                    }
                else:
                    print("WARNING: No handle bodies found — handle features will be zero.")

            obs_aug = obs
            if handle_ctx is not None:
                try:
                    obs_aug = {**obs, **compute_handle_features(env, handle_ctx)}
                except Exception:
                    obs_aug = obs

            test_vec = extract_single_obs_vec(
                obs_aug, training_keys, obs_meta=obs_meta, debug=args.debug
            )
            if test_vec is None or test_vec.shape[0] != expected_obs_dim:
                print("ERROR: Obs vector mismatch. Check KEY_MAPPING.")
                env.close()
                return

            obs_buffer = deque(maxlen=n_obs_steps)
            for _ in range(n_obs_steps):
                obs_buffer.append(test_vec.copy())

            global_step = 0
            while global_step < args.max_steps:
                obs_seq = np.stack(list(obs_buffer), axis=0)
                obs_tensor = torch.from_numpy(obs_seq).unsqueeze(0).to(device)
                with torch.no_grad():
                    result = policy.predict_action({"obs": obs_tensor})
                    actions = result["action"][0].cpu().numpy()

                chunk_len = min(n_action_steps, len(actions), args.max_steps - global_step)
                for i in range(chunk_len):
                    raw = actions[i].copy()
                    action_env = remap_action(raw)

                    if args.assist_base_to_handle and handle_ctx is not None:
                        try:
                            extra = compute_handle_features(env, handle_ctx)
                            to_eef = extra["handle_to_eef_pos"]
                            base_xy = np.array([to_eef[0], to_eef[1]], dtype=np.float32)
                            dist = np.linalg.norm(base_xy)
                            if dist > args.assist_distance_threshold:
                                base_xy = base_xy / max(dist, 1e-6)
                                action_env[7] += args.assist_base_gain * base_xy[0]
                                action_env[8] += args.assist_base_gain * base_xy[1]
                        except Exception:
                            pass

                    if args.clamp_action is not None:
                        action_env = np.clip(action_env, -args.clamp_action, args.clamp_action)
                    if not args.no_clip_action_limits:
                        action_env = np.clip(action_env, action_low, action_high)

                    obs, reward, done, info = env.step(action_env)
                    global_step += 1

                    if global_step % 20 == 0:
                        checking = check_any_door_open(env)
                        status = "cabinet OPEN" if checking else "in progress"
                        act_mag = float(np.abs(action_env).mean())
                        print(
                            f"  step {global_step:4d}  reward={reward:+.3f}  "
                            f"action_mag={act_mag:.3f}  [{status}]"
                        )

                    if check_any_door_open(env, handle_ctx=handle_ctx):
                        success = True
                        break

                    obs_aug = obs
                    if handle_ctx is not None:
                        try:
                            obs_aug = {**obs, **compute_handle_features(env, handle_ctx)}
                        except Exception:
                            obs_aug = obs
                    new_vec = extract_single_obs_vec(
                        obs_aug, training_keys, obs_meta=obs_meta
                    )
                    if new_vec is not None:
                        obs_buffer.append(new_vec)

                    time.sleep(1.0 / args.max_fr)

                    if done or success or global_step >= args.max_steps:
                        break
                if done or success or global_step >= args.max_steps:
                    break

        result = "SUCCESS" if success else "did not open cabinet"
        print(f"\n  Result: {result}")
        if success:
            successes += 1

    env.close()
    print(f"\nFinal: {successes}/{args.num_episodes} episodes succeeded.")


# ── Off-screen rollout with video ────────────────────────────────────────────

def run_offscreen(policy_bundle, args):
    """
    Run the policy headlessly and save a side-by-side annotated video.

    Each frame shows the robot from the front-view camera; per-step
    diagnostics (step count, reward, success flag) are printed to the
    terminal.
    """
    import torch
    import imageio
    if policy_bundle["mode"] == "simple":
        device = next(policy_bundle["model"].parameters()).device
    else:
        device = next(policy_bundle["policy"].parameters()).device

    video_dir = os.path.dirname(args.video_path)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    successes = 0
    all_frames = []  # collect frames across episodes

    env = create_env(
        env_name="OpenCabinet",
        render_onscreen=False,
        seed=args.seed,
        split=args.split,
        camera_widths=args.video_width,
        camera_heights=args.video_height,
    )
    controller = env.robots[0].composite_controller
    action_low = controller.action_limits[0].astype(np.float32)
    action_high = controller.action_limits[1].astype(np.float32)

    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep + 1}/{args.num_episodes} ---")
        obs = env.reset()
        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", "")
        print(f"  Task:    {lang}")
        print(f"  Layout:  {env.layout_id}   Style: {env.style_id}")

        success = False
        ep_frames = []

        if policy_bundle["mode"] == "simple":
            state_dim = policy_bundle["state_dim"]
            model = policy_bundle["model"]
            for step in range(args.max_steps):
                state = extract_state(obs, state_dim)
                with torch.no_grad():
                    action = model(
                        torch.from_numpy(state).unsqueeze(0).to(device)
                    ).cpu().numpy().squeeze(0)

                env_dim = env.action_dim
                if len(action) < env_dim:
                    action = np.pad(action, (0, env_dim - len(action)))
                elif len(action) > env_dim:
                    action = action[:env_dim]

                if args.clamp_action is not None:
                    action = np.clip(action, -args.clamp_action, args.clamp_action)
                if not args.no_clip_action_limits and action.shape == action_low.shape:
                    action = np.clip(action, action_low, action_high)

                obs, reward, done, info = env.step(action)

                frame = env.sim.render(
                    height=args.video_height,
                    width=args.video_width,
                    camera_name="robot0_agentview_center",
                )[::-1]
                ep_frames.append(frame)

                if step % 20 == 0:
                    checking = env._check_success()
                    status = "cabinet OPEN" if checking else "in progress"
                    print(
                        f"  step {step:4d}  reward={reward:+.3f}  [{status}]"
                    )

                if check_any_door_open(env):
                    success = True
                    break
        else:
            from collections import deque

            policy = policy_bundle["policy"]
            shape_meta = policy_bundle["shape_meta"]
            n_obs_steps = int(getattr(policy, "n_obs_steps", 2))
            n_action_steps = int(getattr(policy, "n_action_steps", 8))
            obs_meta = shape_meta.get("obs", {})
            training_keys = list(obs_meta.keys()) if obs_meta else []
            expected_obs_dim = sum(int(np.prod(v["shape"])) for v in obs_meta.values())

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
                        "handle_to_joint_map": build_handle_to_joint_map(
                            handle_bodies, door_joints
                        ),
                    }
                else:
                    print("WARNING: No handle bodies found — handle features will be zero.")

            obs_aug = obs
            if handle_ctx is not None:
                try:
                    obs_aug = {**obs, **compute_handle_features(env, handle_ctx)}
                except Exception:
                    obs_aug = obs

            test_vec = extract_single_obs_vec(obs_aug, training_keys, obs_meta=obs_meta, debug=False)
            if test_vec is None or test_vec.shape[0] != expected_obs_dim:
                print("ERROR: Obs vector mismatch. Check KEY_MAPPING.")
                env.close()
                return

            obs_buffer = deque(maxlen=n_obs_steps)
            for _ in range(n_obs_steps):
                obs_buffer.append(test_vec.copy())

            global_step = 0
            while global_step < args.max_steps:
                obs_seq = np.stack(list(obs_buffer), axis=0)
                obs_tensor = torch.from_numpy(obs_seq).unsqueeze(0).to(device)
                with torch.no_grad():
                    result = policy.predict_action({"obs": obs_tensor})
                    actions = result["action"][0].cpu().numpy()

                chunk_len = min(n_action_steps, len(actions), args.max_steps - global_step)
                for i in range(chunk_len):
                    raw = actions[i].copy()
                    action_env = remap_action(raw)

                    if args.assist_base_to_handle and handle_ctx is not None:
                        try:
                            extra = compute_handle_features(env, handle_ctx)
                            to_eef = extra["handle_to_eef_pos"]
                            base_xy = np.array([to_eef[0], to_eef[1]], dtype=np.float32)
                            dist = np.linalg.norm(base_xy)
                            if dist > args.assist_distance_threshold:
                                base_xy = base_xy / max(dist, 1e-6)
                                action_env[7] += args.assist_base_gain * base_xy[0]
                                action_env[8] += args.assist_base_gain * base_xy[1]
                        except Exception:
                            pass

                    if args.clamp_action is not None:
                        action_env = np.clip(action_env, -args.clamp_action, args.clamp_action)
                    if not args.no_clip_action_limits:
                        action_env = np.clip(action_env, action_low, action_high)

                    obs, reward, done, info = env.step(action_env)
                    global_step += 1

                    frame = env.sim.render(
                        height=args.video_height,
                        width=args.video_width,
                        camera_name="robot0_agentview_center",
                    )[::-1]
                    ep_frames.append(frame)

                    if global_step % 20 == 0:
                        checking = check_any_door_open(env)
                        status = "cabinet OPEN" if checking else "in progress"
                        print(
                            f"  step {global_step:4d}  reward={reward:+.3f}  [{status}]"
                        )

                    if check_any_door_open(env, handle_ctx=handle_ctx):
                        success = True
                        break

                    obs_aug = obs
                    if handle_ctx is not None:
                        try:
                            obs_aug = {**obs, **compute_handle_features(env, handle_ctx)}
                        except Exception:
                            obs_aug = obs
                    new_vec = extract_single_obs_vec(
                        obs_aug, training_keys, obs_meta=obs_meta
                    )
                    if new_vec is not None:
                        obs_buffer.append(new_vec)

                    if done or success or global_step >= args.max_steps:
                        break
                if done or success or global_step >= args.max_steps:
                    break

        result = "SUCCESS" if success else "did not open cabinet"
        print(f"  Result: {result}  ({len(ep_frames)} frames)")
        if success:
            successes += 1

        all_frames.extend(ep_frames)
    env.close()

    # Write video
    print(f"\nWriting {len(all_frames)} frames to {args.video_path} ...")
    with imageio.get_writer(args.video_path, fps=args.fps) as writer:
        for frame in all_frames:
            writer.append_data(frame)
    print(f"Video saved: {args.video_path}")

    print(f"\nFinal: {successes}/{args.num_episodes} episodes succeeded.")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained policy rollout in OpenCabinet"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/tmp/cabinet_policy_checkpoints/best_policy.pt",
        help="Path to policy checkpoint (.pt) saved by 06_train_policy.py",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=300,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="pretrain",
        choices=["pretrain", "target"],
        help="Kitchen scene split to evaluate on",
    )
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="Render to video file instead of opening a viewer window",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="/tmp/policy_rollout.mp4",
        help="Output video path (used with --offscreen)",
    )
    parser.add_argument(
        "--video_width",
        type=int,
        default=256,
        help="Video width (used with --offscreen)",
    )
    parser.add_argument(
        "--video_height",
        type=int,
        default=256,
        help="Video height (used with --offscreen)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second for the saved video",
    )
    parser.add_argument(
        "--max_fr",
        type=int,
        default=20,
        help="On-screen playback rate cap (frames/second); lower = slower",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for environment layout/style selection",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug prints for obs mapping",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - Policy Rollout Visualizer")
    print("=" * 60)
    print()

    # Load policy
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required.  Run: pip install torch")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Train a policy first with:  python 06_train_policy.py")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_bundle = load_policy(args.checkpoint, device)

    print(f"Checkpoint: {args.checkpoint}")
    if "epoch" in policy_bundle["ckpt"] and "loss" in policy_bundle["ckpt"]:
        print(f"  Epoch {policy_bundle['ckpt']['epoch']}, loss {policy_bundle['ckpt']['loss']:.6f}")
    if policy_bundle["mode"] == "simple":
        print(
            f"  Policy: Simple MLP  (state_dim={policy_bundle['state_dim']}, "
            f"action_dim={policy_bundle['action_dim']})"
        )
    else:
        policy = policy_bundle["policy"]
        print(
            f"  Policy: U-Net lowdim  (obs_dim={policy.obs_dim}, action_dim={policy.action_dim})"
        )
        print(
            f"  horizon={policy.horizon}, n_obs_steps={policy.n_obs_steps}, "
            f"n_action_steps={policy.n_action_steps}"
        )
    print(f"  Device: {device}")
    print()

    mode = "off-screen (video)" if args.offscreen else "on-screen (viewer window)"
    print(f"Mode:     {mode}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Max steps/ep: {args.max_steps}")
    if args.offscreen:
        print(f"Output:   {args.video_path}")
    print()

    if args.offscreen:
        run_offscreen(policy_bundle, args)
    else:
        print("Opening viewer window...")
        print("  Tip: orbit the camera with the mouse to see the gripper.\n")
        run_onscreen(policy_bundle, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
