"""
Step 7: Evaluate a Trained Policy
===================================
Runs a trained policy in the OpenCabinet environment and reports
success rate across multiple episodes and kitchen scenes.

Usage:
    # Evaluate the simple BC policy from Step 6
    python 07_evaluate_policy.py --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt

    # Evaluate with more episodes
    python 07_evaluate_policy.py --checkpoint path/to/policy.pt --num_rollouts 50

    # Evaluate on target (held-out) kitchen scenes
    python 07_evaluate_policy.py --checkpoint path/to/policy.pt --split target

    # Save evaluation videos
    python 07_evaluate_policy.py --checkpoint path/to/policy.pt --video_path /tmp/eval_videos.mp4

For evaluating official Diffusion Policy / pi-0 / GR00T checkpoints,
use the evaluation scripts from those repos instead (see 06_train_policy.py).
This script also supports the local U-Net lowdim checkpoints from 06_train_policy.py.
"""

import argparse
import os
import sys
import time

# MuJoCo requires an offscreen renderer on headless machines (e.g. WSL2).
# OSMesa is a CPU-based fallback; EGL is GPU-accelerated but needs /dev/dri,
# which is typically unavailable inside WSL2.
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np

import robocasa  # noqa: F401  — registers RoboCasa envs with robosuite
from robocasa.utils.env_utils import create_env


# ---------------------------------------------------------------------------
# MuJoCo introspection helpers
# ---------------------------------------------------------------------------

def get_mj_model_data(env):
    """
    Retrieve the MuJoCo (model, data) pair from a robosuite-wrapped env.

    Robosuite's wrapper hierarchy has changed across versions, so this function
    probes several common attribute paths rather than assuming a fixed structure.

    Args:
        env: A robosuite / RoboCasa environment (possibly multi-wrapped).

    Returns:
        (model, data) tuple if found, or None if the MuJoCo objects cannot
        be located through any known attribute path.
    """
    # Collect the env itself plus any inner envs it wraps
    candidates = [env]
    for attr in ("env", "_env", "unwrapped"):
        if hasattr(env, attr):
            candidates.append(getattr(env, attr))

    for obj in candidates:
        # Path 1: robosuite ≥ 1.4 — env.sim holds a MjSim with .model/.data
        if hasattr(obj, "sim") and obj.sim is not None:
            sim = obj.sim
            if hasattr(sim, "model") and hasattr(sim, "data"):
                return sim.model, sim.data
            # Some bindings expose private attributes instead
            if hasattr(sim, "_model") and hasattr(sim, "_data"):
                return sim._model, sim._data

        # Path 2: dm_control-style — env.model / env.data directly
        if hasattr(obj, "model") and hasattr(obj, "data"):
            m, d = obj.model, obj.data
            # Sanity-check: real MuJoCo model/data have these attributes
            if hasattr(m, "nbody") and hasattr(d, "qpos"):
                return m, d

        # Path 3: dm_control physics wrapper — env.physics.model/data
        if hasattr(obj, "physics"):
            ph = obj.physics
            if hasattr(ph, "model") and hasattr(ph, "data"):
                return ph.model, ph.data

    return None


def find_fixture_handle_bodies(model, fixture_name=None):
    """
    Find door/hinge joints in the MuJoCo model that belong to a cabinet fixture.

    Joints whose names contain "door" or "hinge" are treated as cabinet door
    joints.  An optional ``fixture_name`` substring filter lets callers
    restrict results to a single fixture when the scene has multiple cabinets.

    Args:
        model:        MuJoCo model object.
        fixture_name: If provided, only joints whose name contains this
                      substring are returned.

    Returns:
        List of (joint_name, joint_index) tuples for matching joints.
    """
    joints = []
    for i in range(model.njnt):
        jname = model.joint(i).name
        # Keep only joints that look like door/hinge actuators
        is_door = "door" in jname or "hinge" in jname
        if not is_door:
            continue
        # Optionally narrow to a specific fixture (e.g. "cabinet_0")
        if fixture_name is None or fixture_name in jname:
            joints.append((jname, i))
    return joints


def compute_door_openness(model, data, door_joints):
    """
    Compute the average normalized openness of one or more door joints.

    Each joint's position is read from ``data.qpos`` and normalised by the
    joint's range so that 0.0 = fully closed and 1.0 = fully open.

    Args:
        model:       MuJoCo model object.
        data:        MuJoCo data object (contains current qpos).
        door_joints: List of (joint_name, joint_index) tuples to evaluate.

    Returns:
        Float in [0, 1] representing the average openness across all supplied
        joints.  Returns 0.0 if ``door_joints`` is empty.
    """
    if not door_joints:
        return 0.0

    openness_values = []
    for jname, ji in door_joints:
        joint = model.joint(ji)
        qpos_val = data.qpos[joint.qposadr]          # current joint position
        lo, hi = joint.range                          # joint limits
        span = hi - lo
        if span > 0:
            normalized = np.clip((qpos_val - lo) / span, 0.0, 1.0)
        else:
            normalized = 0.0
        openness_values.append(normalized)

    return float(np.mean(openness_values))


def build_handle_to_joint_map(handle_bodies, door_joints):
    """
    Map each handle body to the door joint(s) it controls.

    When a cabinet has two doors (left + right) the handle names encode which
    side they belong to.  This function matches "left" handles to "left"
    joints and "right" handles to "right" joints; unmatched handles fall back
    to the full joint list.

    Args:
        handle_bodies: List of handle body name strings.
        door_joints:   List of (joint_name, joint_index) tuples.

    Returns:
        Dict mapping each handle body name → list of (joint_name, joint_index).
    """
    # Single-door shortcut — every handle maps to all joints
    if len(handle_bodies) == 1 or len(door_joints) == 1:
        return {hb: door_joints for hb in handle_bodies}

    result = {}
    for hb in handle_bodies:
        hb_lower = hb.lower()
        if "left" in hb_lower:
            # Match to joints that are also labelled "left"
            matched = [(jn, ji) for jn, ji in door_joints if "left" in jn.lower()]
        elif "right" in hb_lower:
            matched = [(jn, ji) for jn, ji in door_joints if "right" in jn.lower()]
        else:
            matched = []
        # Fall back to the complete joint list if side-matching failed
        result[hb] = matched if matched else door_joints
    return result


def compute_handle_features(env, handle_ctx, open_threshold=0.90):
    """
    Compute handle-related low-dim observation features for a policy.

    These features supplement the raw env observation dict with quantities
    that the policy was trained on but that the env does not expose directly:
    handle world-position, handle-to-end-effector offset vector, and a scalar
    door-openness value.

    Args:
        env:            RoboCasa environment instance (post-reset).
        handle_ctx:     Dict produced by ``run_evaluation_unet`` on first reset,
                        containing keys:
                          "handle_bodies"       – list of body name strings
                          "handle_to_joint_map" – dict from body name → joints
        open_threshold: Openness value considered "fully open" (unused here,
                        kept for API compatibility with ``check_any_door_open``).

    Returns:
        Dict with keys "handle_pos", "handle_to_eef_pos", "door_openness",
        each mapping to a float32 numpy array.
    """
    model, data = get_mj_model_data(env)
    handle_bodies    = handle_ctx["handle_bodies"]
    handle_joint_map = handle_ctx["handle_to_joint_map"]

    # --- Handle world position (mean across all handles) ---
    handle_positions = []
    for hb in handle_bodies:
        body_id = model.body(hb).id
        # xpos is the 3-D world-frame position of the body origin
        handle_positions.append(data.xpos[body_id].copy())
    handle_pos = np.mean(handle_positions, axis=0).astype(np.float32)  # shape (3,)

    # --- End-effector world position ---
    # robot0_eef_pos is the standard robosuite observation key for the EEF
    eef_pos = np.array(env.robots[0]._hand_pos, dtype=np.float32)       # shape (3,)

    # --- Handle-to-EEF offset (used as a reaching signal by the policy) ---
    handle_to_eef_pos = (eef_pos - handle_pos).astype(np.float32)       # shape (3,)

    # --- Scalar door openness (mean across all handle-associated joints) ---
    all_openness = []
    for hb in handle_bodies:
        joints = handle_joint_map.get(hb, [])
        all_openness.append(compute_door_openness(model, data, joints))
    door_openness = np.array([np.mean(all_openness)], dtype=np.float32)  # shape (1,)

    return {
        "handle_pos":        handle_pos,
        "handle_to_eef_pos": handle_to_eef_pos,
        "door_openness":     door_openness,
    }


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def load_unet_lowdim_policy(checkpoint_path, device):
    """
    Load a low-dim Diffusion U-Net policy from a .pt checkpoint.

    The checkpoint is expected to be a dict saved by ``06_train_policy.py``
    with at minimum the keys:
        "model_state_dict"  – ``state_dict`` compatible with ``UnetLowdimPolicy``
        "shape_meta"        – obs/action shape metadata dict used at training time

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device:          ``torch.device`` to load the policy onto.

    Returns:
        Tuple (policy, shape_meta) where ``policy`` is a callable
        ``UnetLowdimPolicy`` in eval mode and ``shape_meta`` is the raw
        metadata dict stored in the checkpoint.

    Raises:
        SystemExit: If the checkpoint cannot be loaded or the model definition
                    cannot be imported.
    """
    import torch

    print(f"  Loading checkpoint: {checkpoint_path}")
    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"ERROR: Could not load checkpoint — {e}")
        sys.exit(1)

    shape_meta = ckpt.get("shape_meta", {})

    # Derive obs/action dims from the stored shape metadata
    obs_meta    = shape_meta.get("obs", {})
    action_meta = shape_meta.get("action", {})
    obs_dim     = sum(int(np.prod(v["shape"])) for v in obs_meta.values())
    action_dim  = int(np.prod(action_meta.get("shape", [12])))

    # n_obs_steps / n_action_steps default to the values used during training
    n_obs_steps    = shape_meta.get("n_obs_steps", 2)
    n_action_steps = shape_meta.get("n_action_steps", 8)
    n_pred_steps   = shape_meta.get("n_pred_steps", 16)

    print(f"  obs_dim={obs_dim}  action_dim={action_dim}")
    print(f"  n_obs_steps={n_obs_steps}  n_action_steps={n_action_steps}")

    try:
        # Import the policy class from diffusion_policy (must be installed)
        from diffusion_policy.policy.diffusion_unet_lowdim_policy import (
            DiffusionUnetLowdimPolicy,
        )
        from diffusion_policy.model.diffusion.conditional_unet1d import (
            ConditionalUnet1D,
        )
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    except ImportError as e:
        print(f"ERROR: diffusion_policy package not found — {e}")
        print("Install it from https://github.com/real-stanford/diffusion_policy")
        sys.exit(1)

    # Reconstruct the noise-prediction U-Net with the same architecture used
    # during training (input dim = obs * n_obs_steps + action)
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * n_obs_steps,
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    policy = DiffusionUnetLowdimPolicy(
        model=noise_pred_net,
        noise_scheduler=noise_scheduler,
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        n_pred_steps=n_pred_steps,
    )

    # Restore trained weights; strict=False tolerates minor architecture drifts
    policy.load_state_dict(ckpt["model_state_dict"], strict=False)
    policy.to(device)
    policy.eval()

    print("  Checkpoint loaded successfully.")
    return policy, shape_meta


# ---------------------------------------------------------------------------
# Observation utilities
# ---------------------------------------------------------------------------

# Translate training-time feature names to the keys used in the live env obs
# dict.  Add entries here whenever a new computed feature is introduced.
KEY_MAPPING = {
    "base_pos":                    "robot0_base_pos",
    "base_quat":                   "robot0_base_quat",
    "robot0_base_to_eef_pos":      "robot0_base_to_eef_pos",
    "robot0_base_to_eef_quat":     "robot0_base_to_eef_quat",
    "robot0_gripper_qpos":         "robot0_gripper_qpos",
    # Computed handle features injected into obs by compute_handle_features()
    "handle_pos":                  "handle_pos",
    "handle_to_eef_pos":           "handle_to_eef_pos",
    "door_openness":               "door_openness",
}


def extract_single_obs_vec(obs_raw, training_keys, obs_meta=None, debug=False):
    """
    Assemble a flat observation vector from a raw env obs dict.

    Iterates over ``training_keys`` in the same order used at training time,
    looks up each key in ``obs_raw`` (via ``KEY_MAPPING`` if needed), and
    concatenates the resulting arrays into a single 1-D float32 vector.

    Missing keys that have known shapes in ``obs_meta`` are zero-filled so
    evaluation can continue even when a feature is temporarily unavailable
    (e.g. before ``handle_ctx`` is populated).

    Args:
        obs_raw:       Raw observation dict returned by ``env.step`` / ``env.reset``.
        training_keys: Ordered list of feature names used during training
                       (i.e. ``list(shape_meta["obs"].keys())``).
        obs_meta:      Optional dict mapping feature name → {"shape": ...}
                       used to zero-fill missing keys.
        debug:         If True, prints a per-key resolution log.

    Returns:
        1-D float32 numpy array of length == sum of all feature dimensions,
        or None if any key is missing and cannot be zero-filled.
    """
    parts   = []
    missing = []

    for key in training_keys:
        # First try the mapped env key, then the raw key, then zero-fill
        env_key = KEY_MAPPING.get(key, key)

        if env_key not in obs_raw:
            if key in obs_raw:
                # The training key happens to match the env key directly
                env_key = key
            elif obs_meta and key in obs_meta:
                # Key is absent from this obs — fill with zeros to avoid crash
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
        # Report which keys could not be resolved — caller should fix KEY_MAPPING
        avail = [k for k in obs_raw.keys() if not k.endswith("_image")]
        print(f"  ERROR: missing keys {missing}")
        print(f"  Available obs keys: {avail}")
        return None

    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Success detection
# ---------------------------------------------------------------------------

def check_any_door_open(env, threshold=0.90, handle_ctx=None):
    """
    Return True if at least one cabinet door is open past the success threshold.

    Three detection strategies are tried in order of accuracy:
      1. handle_ctx  — precise per-joint openness via MuJoCo qpos (preferred).
      2. fxtr API    — the fixture object's own joint-state getter.
      3. env fallback — env._check_success(), which may use a different metric.

    Args:
        env:          RoboCasa environment instance.
        threshold:    Fraction of full range considered "open" (default 0.90).
        handle_ctx:   Optional dict built in ``run_evaluation_unet`` containing
                      "handle_to_joint_map".  If None the fixture API is used.

    Returns:
        bool — True if any door exceeds the threshold.
    """
    # Strategy 1: use precomputed joint map for precise openness
    if handle_ctx is not None:
        model, data = get_mj_model_data(env)
        for joints in handle_ctx["handle_to_joint_map"].values():
            if compute_door_openness(model, data, joints) >= threshold:
                return True
        return False

    # Strategy 2: ask the fixture object for its joint states
    fxtr = getattr(env, "fxtr", None)
    if fxtr is None or not hasattr(fxtr, "get_joint_state"):
        return env._check_success()   # Strategy 3 fallback

    joint_names = getattr(fxtr, "door_joint_names", None)
    if not joint_names:
        return env._check_success()   # Strategy 3 fallback

    try:
        joint_state = fxtr.get_joint_state(env, joint_names)
    except Exception:
        return env._check_success()   # Strategy 3 fallback — fxtr API failed

    return any(val >= threshold for val in joint_state.values())


# ---------------------------------------------------------------------------
# Action remapping
# ---------------------------------------------------------------------------

def remap_action(raw):
    """
    Reorder a policy action vector to match the environment's expected layout.

    The policy was trained with a particular action dimension ordering that
    differs from what robosuite's composite controller expects.  This function
    performs the one-time remapping required at inference time.

    Policy action layout (12-D, from shape_meta):
        raw[0:3]   base motion (x, y, yaw)
        raw[3]     torso lift
        raw[4]     control mode toggle
        raw[5:8]   end-effector position delta (x, y, z)
        raw[8:11]  end-effector rotation delta (roll, pitch, yaw)
        raw[11]    gripper close (−1 = open, +1 = close)

    Env action layout (12-D, per 01_explore_environment.py):
        env[0:3]   eef position delta
        env[3:6]   eef rotation delta
        env[6]     gripper close
        env[7:10]  base motion (x, y, yaw)
        env[10]    torso lift
        env[11]    control mode toggle

    Args:
        raw: 12-D float32 numpy array produced by the policy.

    Returns:
        12-D float32 numpy array in the env's expected dimension order.
    """
    action_env = np.zeros(12, dtype=np.float32)
    action_env[0:6]  = raw[5:11]   # eef_pos (3) + eef_rot (3)
    action_env[6]    = raw[11]     # gripper close
    action_env[7:10] = raw[0:3]    # base motion (x, y, yaw)
    action_env[10]   = raw[3]      # torso lift
    action_env[11]   = raw[4]      # control mode toggle
    return action_env


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_section(title):
    """Print a formatted section header to stdout."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation_unet(
    policy,
    shape_meta,
    num_rollouts,
    max_steps,
    split,
    video_path,
    video_width,
    video_height,
    video_first_person,
    seed,
    debug=False,
    clamp_action=None,
    clip_action_limits=False,
    zero_base_motion=False,
    fixed_control_mode=None,
    force_n_action_steps=None,
):
    """
    Run closed-loop evaluation of a local U-Net lowdim policy.

    Implements a receding-horizon control loop:
      1. The policy observes the last ``n_obs_steps`` obs vectors.
      2. It predicts an action *sequence* of length ``n_pred_steps``.
      3. Only the first ``n_action_steps`` actions are executed.
      4. The obs buffer is updated after each env step.

    An episode is marked successful when any cabinet door exceeds the
    open-threshold (see ``check_any_door_open``).

    Args:
        policy:               Loaded ``DiffusionUnetLowdimPolicy`` in eval mode.
        shape_meta:           Obs/action shape metadata dict from the checkpoint.
        num_rollouts:         Number of episodes to evaluate.
        max_steps:            Hard step limit per episode.
        split:                Kitchen scene split ("pretrain" or "target").
        video_path:           Directory for per-episode MP4 files, or None.
        video_width:          Rendered frame width in pixels.
        video_height:         Rendered frame height in pixels.
        video_first_person:   If True, render from the env's first-person cam.
        seed:                 RNG seed for reproducibility.
        debug:                Extra logging for obs/action diagnostics.
        clamp_action:         If set, clip all env action dims to ±this value.
        clip_action_limits:   Clip to per-dimension controller limits if True.
        zero_base_motion:     Force base motion components to zero (debug).
        fixed_control_mode:   Override control mode dim to a constant (debug).
        force_n_action_steps: Override the policy's ``n_action_steps`` (debug).

    Returns:
        Dict with keys:
            "successes"        – list of bool, one per episode
            "episode_lengths"  – list of int step counts
            "rewards"          – list of cumulative float rewards
    """
    import torch
    import imageio
    from collections import deque

    device = next(policy.parameters()).device

    # --- Policy hyperparameters ---
    n_obs_steps    = int(getattr(policy, "n_obs_steps",    2))
    n_action_steps = int(getattr(policy, "n_action_steps", 8))
    if force_n_action_steps is not None:
        n_action_steps = int(force_n_action_steps)

    # Extract ordered key list and expected obs vector length from checkpoint
    obs_meta        = shape_meta.get("obs", {})
    training_keys   = list(obs_meta.keys()) if obs_meta else []
    expected_obs_dim = sum(int(np.prod(v["shape"])) for v in obs_meta.values())

    # --- Build the environment ---
    env = create_env(
        env_name="OpenCabinet",
        render_onscreen=False,   # headless — frames are captured manually
        seed=seed,
        split=split,
        camera_widths=video_width,
        camera_heights=video_height,
    )

    # --- Inspect the action space before the episode loop ---
    obs = env.reset()
    print("Env action dim:", env.action_dim)
    controller = env.robots[0].composite_controller
    for part_name, part_controller in controller.part_controllers.items():
        print(f"  {part_name}: action_dim={part_controller.control_dim}")

    # Per-dimension limits used for optional action clipping (--clip_action_limits)
    action_low  = controller.action_limits[0].astype(np.float32)
    action_high = controller.action_limits[1].astype(np.float32)

    # List any obs keys that relate to the door / openness state for sanity-check
    door_keys = [k for k in obs.keys() if any(w in k.lower() for w in ("door", "open", "cabinet"))]
    print(f"  Door-related obs keys: {door_keys}")

    # --- Build handle_ctx (needed when the policy uses computed handle features) ---
    # These features are not part of the raw env obs dict; they are computed
    # from MuJoCo model/data after each step via compute_handle_features().
    handle_ctx = None
    needs_handle_ctx = any(
        k in training_keys for k in {"handle_pos", "handle_to_eef_pos", "door_openness"}
    )
    if needs_handle_ctx:
        mj_model, _ = get_mj_model_data(env)
        fxtr         = getattr(env, "fxtr", None)
        fixture_name = getattr(fxtr, "name", None) if fxtr is not None else None

        # find_fixture_handle_bodies / find_fixture_door_joints scan the model
        # for joints/bodies whose names contain "handle"/"door"/"hinge"
        handle_bodies = find_fixture_handle_bodies(mj_model, fixture_name)
        door_joints   = find_fixture_door_joints(mj_model, fixture_name)

        # Broaden the search if the fixture-name filter returned nothing
        if not handle_bodies:
            handle_bodies = find_fixture_handle_bodies(mj_model, fixture_name=None)
        if not door_joints:
            door_joints   = find_fixture_door_joints(mj_model, fixture_name=None)

        if handle_bodies:
            handle_ctx = {
                "handle_bodies":      handle_bodies,
                "handle_to_joint_map": build_handle_to_joint_map(handle_bodies, door_joints),
            }
            print(f"  handle_ctx: bodies={handle_bodies}  joints={[j[0] for j in door_joints]}")
        else:
            print("  WARNING: No handle bodies found — handle features will be zero.")

    # --- Validate the obs vector size against the training config ---
    # Do this once before the episode loop to catch key-mapping errors early.
    if handle_ctx is not None:
        obs = {**obs, **compute_handle_features(env, handle_ctx)}
    test_vec = extract_single_obs_vec(obs, training_keys, obs_meta=obs_meta, debug=debug)
    if test_vec is None:
        print("ERROR: Could not build obs vector on first reset. Check KEY_MAPPING.")
        sys.exit(1)
    if test_vec.shape[0] != expected_obs_dim:
        print(
            f"ERROR: Obs dim mismatch! Got {test_vec.shape[0]}, "
            f"expected {expected_obs_dim}. Check KEY_MAPPING for zero-filled keys."
        )
        sys.exit(1)
    print(f"  Obs vec shape OK: {test_vec.shape[0]} == {expected_obs_dim}")

    if debug:
        print(f"  n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}")
        if clamp_action is not None:
            print(f"  clamp_action={clamp_action}")
        if clip_action_limits:
            print("  clip_action_limits=True (per-dimension controller limits)")
        if video_first_person and video_path:
            print("  video_first_person=True")
        if zero_base_motion:
            print("  zero_base_motion=True")
        if fixed_control_mode is not None:
            print(f"  fixed_control_mode={fixed_control_mode}")

    # Ensure the video output directory exists before the first episode
    if video_path:
        os.makedirs(video_path, exist_ok=True)

    def render_frame():
        """
        Render a single RGB frame for the current simulation state.

        Tries the first-person view if requested; falls back to the standard
        "robot0_agentview_center" camera.  Returns None if rendering fails
        (e.g. OSMesa not available), allowing video to be silently disabled.
        """
        if video_first_person:
            try:
                return env.render(mode="rgb_array", width=video_width, height=video_height)
            except Exception:
                try:
                    return env.sim.render(width=video_width, height=video_height, camera_name=None)
                except Exception:
                    return None
        try:
            # [::-1] flips the image vertically (MuJoCo renders bottom-up)
            return env.sim.render(
                height=video_height, width=video_width, camera_name="robot0_agentview_center"
            )[::-1]
        except Exception:
            return None

    # Accumulators for summary statistics
    results = {
        "successes":       [],
        "episode_lengths": [],
        "rewards":         [],
    }

    disable_video = False   # set to True on first render error to suppress further warnings

    # ==========================================================================
    # Episode loop
    # ==========================================================================
    for ep in range(num_rollouts):
        obs = env.reset()

        # Open a new per-episode video file (e.g. episode_001.mp4)
        video_writer = None
        if video_path:
            ep_video_path = os.path.join(video_path, f"episode_{ep + 1:03d}.mp4")
            video_writer  = imageio.get_writer(ep_video_path, fps=20, format="ffmpeg")

        ep_reward = 0.0
        success   = False

        # Rolling buffer of obs vectors — the policy consumes n_obs_steps at once
        obs_buffer = deque(maxlen=n_obs_steps)

        # Augment the initial obs with computed handle features (if required)
        obs_aug = obs
        if handle_ctx is not None:
            try:
                obs_aug = {**obs, **compute_handle_features(env, handle_ctx)}
            except Exception as e:
                if ep == 0:
                    print(f"Warning: handle feature computation failed: {e}")

        # Convert the first obs to a flat vector and fill the buffer
        first_vec = extract_single_obs_vec(obs_aug, training_keys, obs_meta=obs_meta)
        if first_vec is None:
            # Cannot build the obs vector — skip this episode
            results["successes"].append(False)
            results["episode_lengths"].append(0)
            results["rewards"].append(0.0)
            continue
        for _ in range(n_obs_steps):
            obs_buffer.append(first_vec.copy())   # pre-fill with the first obs

        # -----------------------------------------------------------------------
        # Step loop (receding-horizon control)
        # -----------------------------------------------------------------------
        global_step = 0
        while global_step < max_steps:
            # Stack the obs buffer into a (n_obs_steps, obs_dim) sequence
            obs_seq    = np.stack(list(obs_buffer), axis=0)
            # Add a batch dimension and move to device: (1, n_obs_steps, obs_dim)
            obs_tensor = torch.from_numpy(obs_seq).unsqueeze(0).to(device)

            # Run the diffusion policy forward pass to get an action sequence
            with torch.no_grad():
                result  = policy.predict_action({"obs": obs_tensor})
                actions = result["action"][0].cpu().numpy()  # (n_pred_steps, action_dim)

            # Log the raw and remapped first action on the very first step
            if ep == 0 and global_step == 0:
                print(f"  Raw action[0]: {actions[0]}")
                print(f"  Remapped:      {remap_action(actions[0])}")
                if debug:
                    print(
                        f"  Raw action stats: mean={actions.mean():+.3f} "
                        f"abs_mean={np.abs(actions).mean():.3f} "
                        f"max={actions.max():+.3f} min={actions.min():+.3f}"
                    )

            # Execute up to n_action_steps actions from the predicted sequence
            chunk_len = min(n_action_steps, len(actions), max_steps - global_step)
            for i in range(chunk_len):
                raw = actions[i].copy()

                # Optional debug overrides applied before remapping
                if zero_base_motion:
                    raw[0:3] = 0.0               # disable base translation/rotation
                if fixed_control_mode is not None:
                    raw[4] = float(fixed_control_mode)

                # Reorder dims from policy layout → env layout
                action_env = remap_action(raw)

                # Optional action-space clipping for safety / debugging
                if clamp_action is not None:
                    action_env = np.clip(action_env, -clamp_action, clamp_action)
                if clip_action_limits:
                    action_env = np.clip(action_env, action_low, action_high)

                obs, reward, done, info = env.step(action_env)
                ep_reward  += reward
                global_step += 1

                # Capture video frame (silently disabled on the first render error)
                if video_writer is not None and not disable_video:
                    try:
                        frame = render_frame()
                        if frame is not None:
                            video_writer.append_data(frame)
                    except Exception as e:
                        print(f"  Video disabled due to render error: {e}")
                        disable_video = True

                # Update obs buffer with the latest observation
                obs_aug = obs
                if handle_ctx is not None:
                    try:
                        obs_aug = {**obs, **compute_handle_features(env, handle_ctx)}
                    except Exception:
                        obs_aug = obs   # fall back to raw obs if handle features fail
                new_vec = extract_single_obs_vec(obs_aug, training_keys, obs_meta=obs_meta)
                if new_vec is not None:
                    obs_buffer.append(new_vec)

                # Early-exit on success or env termination
                if check_any_door_open(env, handle_ctx=handle_ctx):
                    success = True
                    break
                if done:
                    break

            if success or done or global_step >= max_steps:
                break

        # Record episode outcomes
        results["successes"].append(success)
        results["episode_lengths"].append(global_step)
        results["rewards"].append(ep_reward)

        status = "SUCCESS" if success else "FAIL"
        print(
            f"  Episode {ep + 1:3d}/{num_rollouts}: {status:7s} "
            f"(steps={global_step:4d}, reward={ep_reward:.1f})"
        )

        if video_writer is not None:
            video_writer.close()

    # Close the environment and release any GPU resources
    env.close()
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """
    Parse command-line arguments and run the evaluation pipeline.

    The function loads the requested checkpoint, selects the appropriate
    evaluation backend (currently only "unet" for local checkpoints), runs
    ``num_rollouts`` episodes, and prints a summary with expected baselines.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained OpenCabinet policy")

    # --- Required ---
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to policy checkpoint (.pt file)",
    )

    # --- Episode settings ---
    parser.add_argument("--num_rollouts", type=int, default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--max_steps",   type=int, default=500,
                        help="Hard step limit per episode")
    parser.add_argument(
        "--split",
        type=str,
        default="pretrain",
        choices=["pretrain", "target"],
        help="Kitchen scene split: 'pretrain' (seen) or 'target' (held-out)",
    )

    # --- Video ---
    parser.add_argument("--video_path",       type=str, default=None,
                        help="Directory to save per-episode MP4 videos (optional)")
    parser.add_argument("--video_width",      type=int, default=256,
                        help="Video frame width in pixels (smaller = faster)")
    parser.add_argument("--video_height",     type=int, default=256,
                        help="Video frame height in pixels")
    parser.add_argument("--video_first_person", action="store_true",
                        help="Render video from the robot's first-person camera")

    # --- Reproducibility ---
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")

    # --- Debug / ablation flags ---
    parser.add_argument("--debug", action="store_true",
                        help="Print per-key obs mapping and action statistics")
    parser.add_argument("--clamp_action", type=float, default=None,
                        help="Clamp all env action dims to ±this value")
    parser.add_argument("--clip_action_limits", action="store_true",
                        help="Clip env actions per-dimension to controller limits")
    parser.add_argument("--zero_base_motion", action="store_true",
                        help="Zero out base motion components (disables navigation)")
    parser.add_argument("--fixed_control_mode", type=float, default=None,
                        help="Force the control_mode action dim to a constant value")
    parser.add_argument("--force_n_action_steps", type=int, default=None,
                        help="Override the policy's n_action_steps at eval time")

    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    print_section("OpenCabinet - Policy Evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load and prepare the policy from the checkpoint
    policy, shape_meta = load_unet_lowdim_policy(args.checkpoint, device)
    mode = "unet"   # only local U-Net checkpoints are supported by this script

    print_section(f"Evaluating on '{args.split}' split ({args.num_rollouts} episodes)")

    if mode == "unet":
        results = run_evaluation_unet(
            policy=policy,
            shape_meta=shape_meta,
            num_rollouts=args.num_rollouts,
            max_steps=args.max_steps,
            split=args.split,
            video_path=args.video_path,
            video_width=args.video_width,
            video_height=args.video_height,
            video_first_person=args.video_first_person,
            seed=args.seed,
            debug=args.debug,
            clamp_action=args.clamp_action,
            clip_action_limits=args.clip_action_limits,
            zero_base_motion=args.zero_base_motion,
            fixed_control_mode=args.fixed_control_mode,
            force_n_action_steps=args.force_n_action_steps,
        )

    # --- Summary statistics ---
    print_section("Evaluation Results")

    num_success  = sum(results["successes"])
    success_rate = num_success / args.num_rollouts * 100
    avg_length   = np.mean(results["episode_lengths"])
    avg_reward   = np.mean(results["rewards"])

    print(f"  Split:          {args.split}")
    print(f"  Episodes:       {args.num_rollouts}")
    print(f"  Successes:      {num_success}/{args.num_rollouts}")
    print(f"  Success rate:   {success_rate:.1f}%")
    print(f"  Avg ep length:  {avg_length:.1f} steps")
    print(f"  Avg reward:     {avg_reward:.3f}")

    if args.video_path:
        print(f"\n  Videos saved to: {args.video_path}")

    # --- Contextual baselines from the RoboCasa benchmark ---
    print_section("Performance Context")
    print(
        "Expected success rates from the RoboCasa benchmark:\n"
        "\n"
        "  Method            | Pretrain | Target\n"
        "  ------------------|----------|-------\n"
        "  Random actions    |    ~0%   |   ~0%\n"
        "  Diffusion Policy  |  ~30-60% | ~20-50%\n"
        "  pi-0              |  ~40-70% | ~30-60%\n"
        "  GR00T N1.5        |  ~35-65% | ~25-55%\n"
        "\n"
        "Note: The simple MLP policy from Step 6 is not expected to\n"
        "achieve meaningful success rates.  Use the official Diffusion\n"
        "Policy repo for real results."
    )


if __name__ == "__main__":
    main()