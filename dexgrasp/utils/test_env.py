import torch
from utils.torch_jit_utils import *
import numpy as np
import json
import os
import time


# remove(56,84), (149-179), [84-149]->(5,13)->cut->(5,7)->(1,35), (216-222) : 222-28-30-30-6=94
def obs_process(observation,pro_dim=128):
    """
    observation =(N,2582) or (n_obs_steps, N, 2582)
    """
    if pro_dim == 77:
        if observation.shape[-1] == 1757:
            return observation
        if observation.shape[-1] != 1902:
            raise ValueError(
                "O6 compact obs expects raw O6 DexRep dim 1902 or compact dim 1757, got {}".format(
                    observation.shape[-1]
                )
            )
        keep_indices = torch.cat(
            [
                torch.arange(0, 17),
                torch.arange(51, 86),
                torch.arange(146, 152),
                torch.arange(152, 164),
                torch.arange(176, 183),
                torch.arange(222, observation.shape[-1]),
            ]
        ).to(device=observation.device)
        return observation[..., keep_indices]

    reshape84_149_part =torch.arange(84,149).reshape(5,13) #
    remove_indice_part = reshape84_149_part[:,-6:].reshape(-1) #fingertips pos&ang vel 30

    remmove28_56 = torch.arange(28,56) # shadow vel 28
    remove56_84 = torch.arange(56,84) # shadow force 28

    remove149_179 = torch.arange(149, 179) #finger force 30
    remove216_222 = torch.arange(216, 222) #obj pos&ang vel 6

    index_to_remove = torch.cat([remove56_84,remove149_179])
    if pro_dim==134:
        final_indice_remove = torch.cat([index_to_remove, remove_indice_part])
    elif pro_dim==128:
        final_indice_remove = torch.cat([index_to_remove, remove_indice_part, remove216_222])
    elif pro_dim==100:
        final_indice_remove = torch.cat([index_to_remove, remove_indice_part, remove216_222, remmove28_56])
    elif pro_dim==222:
        final_indice_remove = torch.empty(0, dtype=torch.long)
    else:
        raise ValueError("unsupported pro_dim for obs_process: {}".format(pro_dim))


    all_indices = torch.arange(observation.shape[-1])

    mask = ~torch.isin(all_indices, final_indice_remove)
    filter_observation = observation[...,mask]
    return filter_observation


def extract_tactile_force(observation):
    force_indices = torch.as_tensor(
        [149, 150, 151, 155, 156, 157, 161, 162, 163, 167, 168, 169, 173, 174, 175],
        device=observation.device,
        dtype=torch.long,
    )
    return observation.index_select(-1, force_indices).reshape(*observation.shape[:-1], 5, 3)


def env_flag(name, default="0"):
    return str(os.environ.get(name, default)).lower() in {"1", "true", "yes", "on"}


def cfg_infer_runtime(task):
    return task.cfg.get("env", {}).get("infer_runtime", {}) or {}


def task_hand_dof_limits(task):
    lower = getattr(task, "o6_hand_dof_lower_limits", None)
    upper = getattr(task, "o6_hand_dof_upper_limits", None)
    if lower is None or upper is None:
        lower = task.shadow_hand_dof_lower_limits
        upper = task.shadow_hand_dof_upper_limits
    return lower, upper


def tactile_probe_output_path():
    path = os.environ.get("DEXGRASP_TACTILE_PROBE_PATH")
    if path:
        return path
    if env_flag("DEXGRASP_RECORD_TACTILE_PROBE", "0"):
        return os.path.join(os.getcwd(), "results", "tactile_probe.jsonl")
    return ""


def object_code_for_env(task, env_id, fallback):
    object_idxs = getattr(task, "object_idxs", None)
    object_code_list = getattr(task, "object_code_list", None)
    if object_code_list is None:
        object_code_list = task.cfg.get("env", {}).get("object_code_dict", [])
    object_index = None
    if object_idxs is not None and env_id < len(object_idxs):
        object_index = object_idxs[env_id]
        if torch.is_tensor(object_index):
            object_index = int(object_index.detach().cpu().item())
        else:
            object_index = int(object_index)
    if object_index is not None and object_code_list and 0 <= object_index < len(object_code_list):
        return str(object_code_list[object_index]), object_index
    return str(fallback), object_index


def _tensor_row_to_list(tensor, env_id, default=None):
    if tensor is None:
        return default
    try:
        return tensor[env_id].detach().cpu().numpy().tolist()
    except Exception:
        return default


def _debug_early_state(task, obj_id, label, step_idx=None, force=False, current_action=None):
    if not hasattr(task, "object_pos") or not hasattr(task, "object_linvel"):
        return False

    object_pos = task.object_pos.detach()
    object_linvel = task.object_linvel.detach()
    object_init = getattr(task, "object_init_state", None)
    object_init_pos = object_init[:, 0:3] if object_init is not None else None
    saved_object_init = None
    if hasattr(task, "saved_root_tensor") and hasattr(task, "object_indices"):
        saved_object_init = task.saved_root_tensor[task.object_indices].detach()
    saved_object_init_pos = saved_object_init[:, 0:3] if saved_object_init is not None else None
    speed = object_linvel.norm(dim=-1)
    delta_z = object_pos[:, 2] - object_init_pos[:, 2] if object_init_pos is not None else torch.zeros_like(speed)
    high_or_fast = (object_pos[:, 2] > 1.05) | (object_pos[:, 2] < 0.45) | (speed > 3.0) | (delta_z.abs() > 0.08)
    bad_ids = torch.arange(task.num_envs, device=task.device) if force else torch.where(high_or_fast)[0]
    if bad_ids.numel() == 0:
        return False

    if step_idx is None:
        print("{}: bad_env_ids={}".format(label, bad_ids.detach().cpu().numpy().tolist()))
    else:
        print("{}: step={}, bad_env_ids={}".format(label, step_idx, bad_ids.detach().cpu().numpy().tolist()))

    first_action_dof = None
    if hasattr(task, "o6_actions_to_dof_targets") and hasattr(task, "grasp_seqs"):
        first_offset = task.initial_wrist_z_offset(-1) if hasattr(task, "initial_wrist_z_offset") else 0.0
        first_xy_offset = task.initial_wrist_xy_away_offset(-1) if hasattr(task, "initial_wrist_xy_away_offset") else 0.0
        first_action_dof = task.o6_actions_to_dof_targets(
            task.grasp_seqs[:, 0],
            wrist_z_offset=first_offset,
            wrist_xy_away_offset=first_xy_offset,
        )
    current_action_dof = None
    if current_action is not None and hasattr(task, "o6_actions_to_dof_targets"):
        current_step_id = 0 if step_idx is None else step_idx + 1
        current_offset = task.initial_wrist_z_offset(current_step_id) if hasattr(task, "initial_wrist_z_offset") else 0.0
        current_xy_offset = task.initial_wrist_xy_away_offset(current_step_id) if hasattr(task, "initial_wrist_xy_away_offset") else 0.0
        current_action_dof = task.o6_actions_to_dof_targets(
            current_action,
            wrist_z_offset=current_offset,
            wrist_xy_away_offset=current_xy_offset,
        )

    for bad_id in bad_ids[:5]:
        env_id = int(bad_id.detach().cpu().item())
        object_code, object_index = object_code_for_env(task, env_id, obj_id)
        grasp_seqs = getattr(task, "grasp_seqs", None)
        first_action = _tensor_row_to_list(grasp_seqs[:, 0] if grasp_seqs is not None else None, env_id, default=[])
        current_action_list = _tensor_row_to_list(current_action, env_id, default=[])
        first_dof = _tensor_row_to_list(first_action_dof, env_id, default=[])
        current_dof = _tensor_row_to_list(current_action_dof, env_id, default=[])
        cur_dof = _tensor_row_to_list(getattr(task, "o6_hand_dof_pos", None), env_id, default=[])
        obj_init = _tensor_row_to_list(object_init, env_id, default=[])
        saved_obj_init = _tensor_row_to_list(saved_object_init, env_id, default=[])
        obj_init_p = object_init_pos[env_id] if object_init_pos is not None else None
        saved_obj_init_p = saved_object_init_pos[env_id] if saved_object_init_pos is not None else None
        pos = object_pos[env_id]
        vel = object_linvel[env_id]
        delta = (pos - obj_init_p) if obj_init_p is not None else torch.zeros_like(pos)
        saved_delta = (pos - saved_obj_init_p) if saved_obj_init_p is not None else torch.zeros_like(pos)

        hand_root = _tensor_row_to_list(getattr(task, "root_state_tensor", None), int(task.hand_indices[env_id].detach().cpu().item()), default=[]) if hasattr(task, "hand_indices") and hasattr(task, "root_state_tensor") else []
        hand_dof_pos = getattr(task, "o6_hand_dof_pos", None)
        hand_dof_xyz = _tensor_row_to_list(hand_dof_pos[:, 0:3] if hand_dof_pos is not None else None, env_id, default=[])
        palm_state = getattr(task, "o6_palm_pos", None)
        if palm_state is None:
            palm_state = getattr(task, "right_hand_pos", None)
        palm_pos = _tensor_row_to_list(palm_state, env_id, default=[])
        fingertip_pos = _tensor_row_to_list(getattr(task, "fingertip_pos", None), env_id, default=[])
        rigid_palm_pos = []
        rigid_fingertip_pos = []
        rigid_palm = None
        rigid_tips = None
        if hasattr(task, "rigid_body_states"):
            palm_idx = None
            hand_body_idx_dict = getattr(task, "hand_body_idx_dict", {})
            if isinstance(hand_body_idx_dict, dict):
                palm_idx = hand_body_idx_dict.get("palm")
            if palm_idx is not None and int(palm_idx) >= 0:
                rigid_palm = task.rigid_body_states[env_id, int(palm_idx), 0:3]
                rigid_palm_pos = rigid_palm.detach().cpu().numpy().tolist()
            fingertip_handles = getattr(task, "fingertip_handles", None)
            if fingertip_handles is not None:
                rigid_tips = task.rigid_body_states[env_id, fingertip_handles, 0:3]
                rigid_fingertip_pos = rigid_tips.detach().cpu().numpy().tolist()

        palm_dist = None
        tip_min_dist = None
        rigid_palm_dist = None
        rigid_tip_min_dist = None
        palm_state = getattr(task, "o6_palm_pos", None)
        if palm_state is None:
            palm_state = getattr(task, "right_hand_pos", None)
        if palm_state is not None:
            palm_dist = float(torch.norm(palm_state[env_id] - pos).detach().cpu().item())
        if hasattr(task, "fingertip_pos"):
            tip_min_dist = float(torch.norm(task.fingertip_pos[env_id] - pos.unsqueeze(0), dim=-1).min().detach().cpu().item())
        if rigid_palm_pos:
            rigid_palm_dist = float(torch.norm(rigid_palm - pos).detach().cpu().item())
        if rigid_tips is not None:
            rigid_tip_min_dist = float(torch.norm(rigid_tips - pos.unsqueeze(0), dim=-1).min().detach().cpu().item())

        print(
            "{}_detail: env_id={}, object_index={}, object_code={}, pos={}, vel={}, delta_from_init={}, "
            "delta_from_saved_root={}, speed={:.4f}, init_state={}, saved_init_state={}, first_action={}, "
            "current_action={}, first_dof_xyzrpy={}, current_dof_xyzrpy={}, cur_dof_xyzrpy={}, "
            "hand_root_state={}, hand_dof_xyz={}, palm_pos={}, palm_obj_dist={}, fingertip_min_obj_dist={}, "
            "fingertip_pos={}, rigid_palm_pos={}, rigid_palm_obj_dist={}, rigid_fingertip_min_obj_dist={}, "
            "rigid_fingertip_pos={}".format(
                label,
                env_id,
                object_index,
                object_code,
                pos.detach().cpu().numpy().tolist(),
                vel.detach().cpu().numpy().tolist(),
                delta.detach().cpu().numpy().tolist(),
                saved_delta.detach().cpu().numpy().tolist(),
                float(speed[env_id].detach().cpu().item()),
                obj_init,
                saved_obj_init,
                first_action,
                current_action_list,
                first_dof[:6] if first_dof else [],
                current_dof[:6] if current_dof else [],
                cur_dof[:6] if cur_dof else [],
                hand_root,
                hand_dof_xyz,
                palm_pos,
                palm_dist,
                tip_min_dist,
                fingertip_pos,
                rigid_palm_pos,
                rigid_palm_dist,
                rigid_tip_min_dist,
                rigid_fingertip_pos,
            )
        )
    return True

@torch.no_grad()
def test_env(args, task, env, model, bc_model_name,obj_id, global_feat=None,use_gt=False, use_part_gt=False):
    runtime_cfg = cfg_infer_runtime(task)
    video_recorders = []
    video_record_paths = []
    record_stride = int(os.environ.get("DEXGRASP_RECORD_VIDEO_STRIDE", "2"))
    record_video = env_flag("DEXGRASP_RECORD_VIDEO", "1" if runtime_cfg.get("record_video", False) else "0")
    if record_video:
        from isaacgym import gymapi
        import imageio

        out_dir = os.environ.get("DEXGRASP_RECORD_VIDEO_DIR", "./results/videos")
        os.makedirs(out_dir, exist_ok=True)
        env_ids = [
            int(x) for x in os.environ.get("DEXGRASP_RECORD_ENV_IDS", "0").split(",")
            if x.strip()
        ]
        width = int(os.environ.get("DEXGRASP_RECORD_WIDTH", "640"))
        height = int(os.environ.get("DEXGRASP_RECORD_HEIGHT", "480"))
        for env_id in env_ids:
            if env_id < 0 or env_id >= task.num_envs:
                continue
            cam_props = gymapi.CameraProperties()
            cam_props.width = width
            cam_props.height = height
            cam_handle = task.gym.create_camera_sensor(task.envs[env_id], cam_props)
            cam_pos = gymapi.Vec3(0.75, -0.85, 0.95)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.62)
            task.gym.set_camera_location(cam_handle, task.envs[env_id], cam_pos, cam_target)
            safe_obj_id = obj_id.replace("/", "_")
            video_path = os.path.join(out_dir, f"{safe_obj_id}_env{env_id}.mp4")
            writer = imageio.get_writer(video_path, fps=int(os.environ.get("DEXGRASP_RECORD_FPS", "30")))
            video_recorders.append((env_id, cam_handle, writer, width, height))
            video_record_paths.append(video_path)
        if video_record_paths:
            print("record_video_paths:", video_record_paths)

    if use_gt or use_part_gt:
        gt_actions = torch.tensor(task.obj_trajs_info['grasp_seqs']).to('cuda')
    # traj_obs = torch.tensor(task.obj_trajs_info['obs']).to('cuda')

    model.eval()
    if model.__class__.__name__ == 'LitBCModel' and hasattr(model.model, 'reset_history'):
        model.model.reset_history()
    pro_dim = task.cfg['env']['obs_dim']['prop']
    maxlen = 50000
    cur_reward_sum = torch.zeros(task.num_envs, dtype=torch.float, device=args.rl_device)
    cur_episode_length = torch.zeros(task.num_envs, dtype=torch.float, device=args.rl_device)
    cur_success = torch.zeros(task.num_envs, dtype=torch.float, device=args.rl_device)
    cur_success_done_sum = torch.zeros(task.num_envs, dtype=torch.float, device=args.rl_device)
    metric_mask = torch.ones(task.num_envs, dtype=torch.bool, device=args.rl_device)
    if runtime_cfg.get("exclude_first_env_from_metrics", task.cfg['env'].get("exclude_first_env_from_metrics", False)) and task.num_envs > 0:
        metric_mask[0] = False
        print("exclude_first_env_from_metrics: env0 is simulated but ignored in success metrics")
    metric_num_envs = int(metric_mask.sum().detach().cpu().item())
    if metric_num_envs > 0:
        maxlen = metric_num_envs

    task.reset_buf = torch.ones(task.num_envs, device=task.device, dtype=torch.long)
    task.progress_buf = torch.zeros(task.num_envs, device=task.device, dtype=torch.long)
    if runtime_cfg.get("direct_initial_reset", False) and hasattr(task, "direct_reset_to_first_frame"):
        task.direct_reset_to_first_frame()
    else:
        env.reset()
    init_fly_debug = bool(runtime_cfg.get("init_fly_debug", False))
    if init_fly_debug:
        task.compute_observations()
        _debug_early_state(task, obj_id, "post_reset_fly_debug")
    current_obs_state = task.obs_buf.clone()
    o6_policy_obs_mode = task.cfg['env'].get('o6_policy_obs_mode', '')
    use_o6_compact_obs = o6_policy_obs_mode and o6_policy_obs_mode != "dexrep"
    if use_o6_compact_obs:
        obj_rotmat = torch.as_tensor(task.obj_trajs_info['obj_rotmat'], device=args.rl_device, dtype=torch.float32)
        obj_rot_flat = obj_rotmat.reshape(task.num_envs, 9)
        current_obs_state = torch.zeros((task.num_envs, 21), dtype=torch.float32, device=args.rl_device)
        current_obs_state[:, :12] = task.grasp_seqs[:, 0, :].to(args.rl_device)
        current_obs_state[:, 12:21] = obj_rot_flat

    tactile_probe_path = tactile_probe_output_path()
    tactile_probe_enabled = bool(tactile_probe_path)
    probe_count = torch.zeros(task.num_envs, dtype=torch.float, device=args.rl_device)
    probe_delta_count = torch.zeros(task.num_envs, dtype=torch.float, device=args.rl_device)
    probe_force_sum = torch.zeros(task.num_envs, dtype=torch.float, device=args.rl_device)
    probe_force_max = torch.zeros(task.num_envs, dtype=torch.float, device=args.rl_device)
    probe_delta_sum = torch.zeros(task.num_envs, dtype=torch.float, device=args.rl_device)
    probe_delta_max = torch.zeros(task.num_envs, dtype=torch.float, device=args.rl_device)
    probe_finger_sum = torch.zeros(task.num_envs, 5, dtype=torch.float, device=args.rl_device)
    probe_finger_max = torch.zeros(task.num_envs, 5, dtype=torch.float, device=args.rl_device)
    probe_success_ever = torch.zeros(task.num_envs, dtype=torch.bool, device=args.rl_device)
    probe_prev_force = None

    def tactile_probe_update(force):
        nonlocal probe_prev_force
        if not tactile_probe_enabled or force is None:
            return
        force = force.detach()
        force_norm = force.norm(dim=-1)
        env_force_mean = force_norm.mean(dim=-1)
        env_force_max = force_norm.max(dim=-1).values
        probe_count.add_(1.0)
        probe_force_sum.add_(env_force_mean)
        probe_force_max.copy_(torch.maximum(probe_force_max, env_force_max))
        probe_finger_sum.add_(force_norm)
        probe_finger_max.copy_(torch.maximum(probe_finger_max, force_norm))
        if probe_prev_force is not None:
            delta_norm = (force - probe_prev_force).norm(dim=-1)
            env_delta_mean = delta_norm.mean(dim=-1)
            env_delta_max = delta_norm.max(dim=-1).values
            probe_delta_count.add_(1.0)
            probe_delta_sum.add_(env_delta_mean)
            probe_delta_max.copy_(torch.maximum(probe_delta_max, env_delta_max))
        probe_prev_force = force.clone()

    def tactile_probe_write(final_iter, succ_sum, succ_rate):
        if not tactile_probe_enabled:
            return
        count = probe_count.clamp_min(1.0)
        delta_count = probe_delta_count.clamp_min(1.0)
        force_mean = (probe_force_sum / count).detach().cpu().numpy()
        force_max = probe_force_max.detach().cpu().numpy()
        delta_mean = (probe_delta_sum / delta_count).detach().cpu().numpy()
        delta_max = probe_delta_max.detach().cpu().numpy()
        finger_mean = (probe_finger_sum / count.unsqueeze(-1)).detach().cpu().numpy()
        finger_max = probe_finger_max.detach().cpu().numpy()
        success_ever = probe_success_ever.detach().cpu().numpy().astype(np.int64)
        final_success = (task.successes > 0).detach().cpu().numpy().astype(np.int64)
        rows = []
        for env_id in range(task.num_envs):
            object_code, object_index = object_code_for_env(task, env_id, obj_id)
            rows.append(
                {
                    "env_id": int(env_id),
                    "object_index": None if object_index is None else int(object_index),
                    "object_code": object_code,
                    "success": int(success_ever[env_id]),
                    "final_success": int(final_success[env_id]),
                    "force_mean_norm": float(force_mean[env_id]),
                    "force_max_norm": float(force_max[env_id]),
                    "delta_mean_norm": float(delta_mean[env_id]),
                    "delta_max_norm": float(delta_max[env_id]),
                    "force_finger_mean_norm": [float(x) for x in finger_mean[env_id].tolist()],
                    "force_finger_max_norm": [float(x) for x in finger_max[env_id].tolist()],
                    "num_force_samples": int(probe_count[env_id].detach().cpu().item()),
                    "num_delta_samples": int(probe_delta_count[env_id].detach().cpu().item()),
                }
            )
        record = {
            "run_label": os.environ.get("DEXGRASP_RESULT_SUFFIX", ""),
            "obj_id_arg": obj_id,
            "bc_model_name": bc_model_name,
            "num_envs": int(task.num_envs),
            "final_iter": int(final_iter),
            "reported_success_sum": float(succ_sum.detach().cpu().item() if torch.is_tensor(succ_sum) else succ_sum),
            "reported_success_rate": float(succ_rate.detach().cpu().item() if torch.is_tensor(succ_rate) else succ_rate),
            "rows": rows,
        }
        probe_dir = os.path.dirname(tactile_probe_path)
        if probe_dir:
            os.makedirs(probe_dir, exist_ok=True)
        with open(tactile_probe_path, "a", encoding="utf-8") as f:
            json.dump(record, f, sort_keys=True)
            f.write("\n")
        print("tactile_probe_wrote:", tactile_probe_path)

    if tactile_probe_enabled and current_obs_state.shape[-1] == 2582:
        tactile_probe_update(extract_tactile_force(current_obs_state))
    elif tactile_probe_enabled:
        print("tactile_probe_disabled: raw 2582-D observation unavailable at reset")
    # current_obs_state= traj_obs[:,0]
    if current_obs_state.shape[-1] in {2582, 1902}:
        current_obs_state = obs_process(current_obs_state,pro_dim=pro_dim)
    if len(current_obs_state.size())==3 or len(current_obs_state.size())==4:
        current_obs_state = current_obs_state.transpose(0,1) #(N, 2, 2488)
    a=1
    if bc_model_name=='ActorCriticPNG':
        current_obs_state = torch.cat([current_obs_state[:,:pro_dim], global_feat.repeat(task.num_envs,1)], dim=-1)
        a=1

    reward_sum = []
    episode_length = []

    success_sum_list = []
    success_done_list = []

    i=0
    rollout_log_every = int(os.environ.get("DEXGRASP_ROLLOUT_LOG_EVERY", runtime_cfg.get("rollout_log_every", 10)))
    init_fly_reported = False
    profile_enabled = os.environ.get("DEXGRASP_ROLLOUT_PROFILE", "0") == "1"
    profile_target_ms = float(os.environ.get("DEXGRASP_ROLLOUT_PROFILE_TARGET_MS", "70"))
    profile_warmup_steps = int(os.environ.get("DEXGRASP_ROLLOUT_PROFILE_WARMUP_STEPS", "5"))
    profile_stats = {
        "policy_ms": [],
        "env_step_ms": [],
        "obs_process_ms": [],
        "bookkeeping_ms": [],
        "loop_ms": [],
    }

    def profile_now():
        if profile_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def profile_add(key, start, step_idx):
        if not profile_enabled:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if step_idx >= profile_warmup_steps:
            profile_stats[key].append((time.perf_counter() - start) * 1000.0)

    def profile_summary(label):
        if not profile_enabled:
            return
        print(
            "rollout_profile_start: label={}, target_ms={}, warmup_steps={}, measured_steps={}".format(
                label, profile_target_ms, profile_warmup_steps, len(profile_stats.get("loop_ms", []))
            )
        )
        for key in ["policy_ms", "env_step_ms", "obs_process_ms", "bookkeeping_ms", "loop_ms"]:
            values = profile_stats.get(key, [])
            if len(values) == 0:
                print("rollout_profile {}: no_samples".format(key))
                continue
            arr = np.asarray(values, dtype=np.float64)
            mean_ms = float(np.mean(arr))
            print(
                "rollout_profile {}: mean_ms={:.3f}, p50_ms={:.3f}, p95_ms={:.3f}, p99_ms={:.3f}, max_ms={:.3f}, effective_hz={:.2f}, deadline_miss_rate={:.4f}".format(
                    key,
                    mean_ms,
                    float(np.percentile(arr, 50)),
                    float(np.percentile(arr, 95)),
                    float(np.percentile(arr, 99)),
                    float(np.max(arr)),
                    1000.0 / mean_ms if mean_ms > 0 else float("inf"),
                    float(np.mean(arr > profile_target_ms)),
                )
            )

    while len(reward_sum) < maxlen:
        loop_t0 = profile_now()
        policy_t0 = loop_t0
        # print(f'Frame {i+1} ---------------')
        # ep_infos = []
        # actions, state, obs_feat = actor.act_inference(current_obs_state)

        if model.__class__.__name__ != 'LitBCModel':
            raise ValueError("bc branch rollout expects LitBCModel, got {}".format(model.__class__.__name__))
        policy_obs = current_obs_state
        actions = model.model.act_inference(policy_obs)
        if getattr(task, "actions_are_normalized", True):
            actions = actions.clamp(-1, 1)
        else:
            lower = torch.tensor(
                [-10.0, -10.0, -10.0, -np.pi, -np.pi, -np.pi,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                device=actions.device,
                dtype=actions.dtype,
            )
            upper = torch.tensor(
                [10.0, 10.0, 10.0, np.pi, np.pi, np.pi,
                 1.36, 0.58, 1.60, 1.60, 1.60, 1.60],
                device=actions.device,
                dtype=actions.dtype,
            )
            actions = torch.max(torch.min(actions, upper), lower)
        profile_add("policy_ms", policy_t0, i)
        if use_gt or use_part_gt:
            if use_part_gt and i>5:
                pass
            elif getattr(task, "actions_are_normalized", True) is False:
                actions = gt_actions[:, i]
            elif len(current_obs_state.size())==3:
                hand_dof_lower_limits, hand_dof_upper_limits = task_hand_dof_limits(task)
                actions_gt = unscale(gt_actions[:, i*3:i*3+3], hand_dof_lower_limits, hand_dof_upper_limits)
                actions = actions_gt.transpose(0,1)
            else:
                hand_dof_lower_limits, hand_dof_upper_limits = task_hand_dof_limits(task)
                actions_gt = unscale(gt_actions[:, i], hand_dof_lower_limits, hand_dof_upper_limits)
                actions = actions_gt
            a=1

        # actions = torch.zeros_like(actions).to('cuda')

        # Step the vec_environment
        # next_obs, rews, dones, infos = vec_env.step(actions)

        env_step_t0 = profile_now()
        if i>30 and env.task.cfg['env']['use_pre_fixed_actions']: #40
            actions = env.task.get_pre_target_actions(actions,i)
        env.task.step(actions, i+1)
        if init_fly_debug and not init_fly_reported and i < 8:
            if _debug_early_state(env.task, obj_id, "init_fly_debug", step_idx=i, current_action=actions):
                init_fly_reported = True
        next_obs = env.task.obs_buf.clone()# (B,2580)
        if tactile_probe_enabled and next_obs.shape[-1] == 2582:
            tactile_probe_update(extract_tactile_force(next_obs))
        # next_obs = traj_obs[:, i]
        profile_add("env_step_ms", env_step_t0, i)

        if video_recorders and i % record_stride == 0:
            task.gym.fetch_results(task.sim, True)
            task.gym.step_graphics(task.sim)
            task.gym.render_all_camera_sensors(task.sim)
            from isaacgym import gymapi
            for env_id, cam_handle, writer, width, height in video_recorders:
                img = task.gym.get_camera_image(task.sim, task.envs[env_id], cam_handle, gymapi.IMAGE_COLOR)
                img = np.asarray(img, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
                writer.append_data(img)

        # next_obs_gt = obs_process(traj_obs[:,i+1],pro_dim=pro_dim)
        # next_obs[:,185:209] = scale(next_obs[:,185:209], task.o6_hand_dof_lower_limits[:24], task.o6_hand_dof_upper_limits[:24])
        obs_process_t0 = profile_now()
        if next_obs.shape[-1] in {2582, 1902}:
            next_obs = obs_process(next_obs,pro_dim=pro_dim) #2582->2488
        # next_obs_gt = obs_process(next_obs_gt,pro_dim=pro_dim)
        a=1
        profile_add("obs_process_ms", obs_process_t0, i)

        bookkeeping_t0 = profile_now()
        if bc_model_name == 'ActorCriticPNG':
            next_obs = torch.cat([next_obs[:,:pro_dim], global_feat.repeat(task.num_envs,1)], dim=-1)
        rews = env.task.rew_buf.clone() #(B, )
        dones = env.task.reset_buf.clone() #(B, )

        # next_states = vec_env.get_state()
        # Record the transition
        # storage.add_transitions(state, obs_feat, actions_expert, rews, dones)
        if use_o6_compact_obs:
            current_obs_state[:, :12] = actions.detach().to(current_obs_state.device)
            current_obs_state[:, 12:21] = obj_rot_flat
        else:
            current_obs_state.copy_(next_obs) #(B, 2582)

        # current_obs_state = traj_obs[:,i+1]
        # current_obs_state = obs_process(current_obs_state,pro_dim=pro_dim)  # 2582->2488

        # Book keeping
        # ep_infos.append(infos)
        cur_reward_sum[:] += rews
        cur_episode_length[:] += 1

        new_ids = (dones > 0).nonzero(as_tuple=False)
        if (
            model.__class__.__name__ == 'LitBCModel'
            and hasattr(model.model, 'reset_history')
            and new_ids.numel() > 0
        ):
            model.model.reset_history(new_ids.reshape(-1))
        success_ = torch.where(task.successes==1)[0]*1 #(B, )

        cur_success[:] = task.successes #torch.where(task.successes==1)[0]*1
        if tactile_probe_enabled:
            probe_success_ever.copy_(torch.logical_or(probe_success_ever, task.successes > 0))
        cur_success_done_sum[new_ids] = cur_success[new_ids]
        profile_add("bookkeeping_ms", bookkeeping_t0, i)
        profile_add("loop_ms", loop_t0, i)
        should_log_rollout = rollout_log_every > 0 and (i % rollout_log_every == 0 or i > 100)
        if should_log_rollout:
            masked_success_sum = task.successes[metric_mask].sum()
            if masked_success_sum > 0:
                print('name={}, iter={}, N_seq={}, success_num:{}'.format(obj_id, i, metric_num_envs, task.successes[metric_mask].sum()))
                a=1
            else:
                print('name={}, iter={}, N_seq={}, None'.format(obj_id, i, metric_num_envs))
                a=1
        flat_new_ids = new_ids.reshape(-1)
        metric_new_ids = flat_new_ids[metric_mask[flat_new_ids]] if flat_new_ids.numel() > 0 else flat_new_ids
        reward_sum.extend(cur_reward_sum[metric_new_ids].detach().cpu().numpy().tolist())
        episode_length.extend(cur_episode_length[metric_new_ids].detach().cpu().numpy().tolist())
        # successes.extend(infos['successes'][new_ids.cpu()][:, 0].cpu().numpy().tolist())

        success_sum_list.append(cur_success[metric_mask].detach().cpu().numpy().sum())
        success_done_list.append(cur_success[new_ids].cpu().numpy())

        # if len(new_ids) > 0:
        if i > 100:
            print('-' * 20)
            # donw_succ = torch.zeros(task.num_envs)
            # succ_rate = statistics.mean(cur_success.cpu().numpy())
            # donw_succ_rate = cur_success_done_sum.sum()/len(new_ids)
            succ_sum = max(success_sum_list)
            succ_rate = succ_sum / max(metric_num_envs, 1)
            succ_sum_float = float(succ_sum.item() if torch.is_tensor(succ_sum) else succ_sum)

            succ_rate_float = float(succ_rate.item() if torch.is_tensor(succ_rate) else succ_rate)
            print(f'Mean Success {obj_id}::> {succ_rate_float * 100:.2f}')

            result_desc = 'name={}, N_seq={}, success_num:{},success_rate:{}'.format(obj_id, metric_num_envs, succ_sum_float, succ_rate_float)
            result_info = {
                "name": obj_id,
                "N_seq": int(metric_num_envs),
                "success_num": succ_sum_float,
                "success_rate": succ_rate_float,
            }
            print(result_desc)
            success_env_ids = torch.where((task.successes > 0) & metric_mask)[0].detach().cpu().numpy().tolist()
            print("success_env_ids:", success_env_ids)
            tactile_probe_write(i, succ_sum, succ_rate)
            if video_record_paths:
                summary_path = os.path.join(os.path.dirname(video_record_paths[0]), f"{obj_id.replace('/', '_')}_summary.txt")
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(result_desc + "\n")
                    f.write("success_env_ids: " + ",".join(map(str, success_env_ids)) + "\n")
                    f.write("video_paths:\n" + "\n".join(video_record_paths) + "\n")
            break
            # print(f'Done Mean Success {obj_id}::> {donw_succ_rate * 100:.2f}')
        i+=1

        cur_reward_sum[new_ids] = 0
        cur_episode_length[new_ids] = 0
        cur_success_done_sum[new_ids] = 0
        if i>300:
            break

    # pred_actions = torch.stack(pred_actions, dim=0).transpose(0, 1)[:, :100]
    # sim_actions = torch.stack(sim_actions,dim=0).transpose(0,1)[:, :100]
    # data_dict = {'pred_actions': pred_actions, 'sim_actions': sim_actions}
    # if args.save_traj:
    #     np.save('./results/{}.npy'.format(obj_id), data_dict, allow_pickle=True)


    for _, _, writer, _, _ in video_recorders:
        writer.close()

    profile_summary("{}:{}".format(bc_model_name, obj_id))

    return succ_rate, result_desc, result_info
    # return  obj_id, task.num_envs, task.successes.mean().cpu().numpy().item(),task.successes.sum().cpu().numpy().item()
