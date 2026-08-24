import os
import os.path as osp
from glob import glob
import sys
sys.path.append('../')
import numpy as np
import yaml
from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
# from utils.process_sarl import *
from utils.process_marl import process_MultiAgentRL, get_AgentIndex
# from utils.logger import DataLog
import torch
from isaacgym.torch_utils import *
import gc
import time
from multiprocessing import Process
import copy
import tempfile

THIS_DEXGRASP_DIR = osp.realpath(osp.dirname(__file__))
THIS_REPO_DIR = osp.realpath(osp.join(THIS_DEXGRASP_DIR, ".."))
_PREPROCESS_GIF_WRITTEN = False

def _env_flag(name, default="0"):
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}

def _env_nonnegative_int(name, default=0):
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError:
        raise ValueError("{} must be an integer, got {!r}".format(name, raw_value))
    if value < 0:
        raise ValueError("{} must be non-negative, got {}".format(name, value))
    return value

def _select_preprocess_paths(npy_paths, object_code=None, max_files=0):
    paths = sorted(npy_paths)
    if object_code:
        normalized_code = object_code[:-4] if object_code.endswith(".npy") else object_code
        paths = [
            path
            for path in paths
            if osp.splitext(osp.basename(path))[0] == normalized_code
        ]
    if max_files > 0:
        paths = paths[:max_files]
    return paths

def _preprocess_gif_path():
    path = os.environ.get("DEXGRASP_PREPROCESS_GIF_PATH")
    if path:
        return path if osp.isabs(path) else osp.realpath(osp.join(os.getcwd(), path))
    return osp.join(THIS_REPO_DIR, "assets", "Grasping.gif")

def _preprocess_gif_mode():
    return os.environ.get("DEXGRASP_PREPROCESS_GIF_MODE", "viewer").strip().lower()

def _setup_preprocess_gif(task, proc_id):
    global _PREPROCESS_GIF_WRITTEN
    if not _env_flag("DEXGRASP_PREPROCESS_RECORD_GIF"):
        return None
    if _PREPROCESS_GIF_WRITTEN and not _env_flag("DEXGRASP_PREPROCESS_GIF_OVERWRITE_EACH_WORKER"):
        return None
    gif_proc_id = int(os.environ.get("DEXGRASP_PREPROCESS_GIF_PROC_ID", "0"))
    if proc_id != gif_proc_id:
        return None

    try:
        import imageio
    except Exception as exc:
        print("preprocess_gif_skipped: failed to import image writer dependency: {}".format(exc))
        return None

    mode = _preprocess_gif_mode()
    if mode not in {"viewer", "sensor_grid"}:
        raise ValueError(
            "unsupported DEXGRASP_PREPROCESS_GIF_MODE: {}. Expected viewer or sensor_grid".format(mode)
        )

    fps = int(os.environ.get("DEXGRASP_PREPROCESS_GIF_FPS", "20"))
    gif_path = _preprocess_gif_path()
    os.makedirs(osp.dirname(gif_path), exist_ok=True)

    if mode == "viewer":
        if task.viewer is None:
            print(
                "preprocess_gif_skipped: viewer mode requires a non-headless IsaacGym viewer. "
                "Run with DEXGRASP_PREPROCESS_RECORD_GIF=1 before config/env creation."
            )
            return None
        viewer_width = int(os.environ.get("DEXGRASP_PREPROCESS_GIF_VIEWER_WIDTH", "800"))
        viewer_height = int(os.environ.get("DEXGRASP_PREPROCESS_GIF_VIEWER_HEIGHT", "450"))
        if hasattr(task.gym, "set_viewer_size"):
            try:
                task.gym.set_viewer_size(task.viewer, viewer_width, viewer_height)
            except Exception:
                pass
        if _env_flag("DEXGRASP_PREPROCESS_GIF_SET_VIEWER_CAMERA", "0"):
            from isaacgym import gymapi
            cam_pos = gymapi.Vec3(
                float(os.environ.get("DEXGRASP_PREPROCESS_GIF_VIEWER_CAM_X", "3.3")),
                float(os.environ.get("DEXGRASP_PREPROCESS_GIF_VIEWER_CAM_Y", "3.0")),
                float(os.environ.get("DEXGRASP_PREPROCESS_GIF_VIEWER_CAM_Z", "4.0")),
            )
            cam_target = gymapi.Vec3(
                float(os.environ.get("DEXGRASP_PREPROCESS_GIF_VIEWER_TARGET_X", "3.8")),
                float(os.environ.get("DEXGRASP_PREPROCESS_GIF_VIEWER_TARGET_Y", "4.5")),
                float(os.environ.get("DEXGRASP_PREPROCESS_GIF_VIEWER_TARGET_Z", "0.7")),
            )
            task.gym.viewer_camera_look_at(task.viewer, None, cam_pos, cam_target)
        frame_dir = os.environ.get("DEXGRASP_PREPROCESS_GIF_FRAME_DIR")
        if frame_dir:
            os.makedirs(frame_dir, exist_ok=True)
        else:
            frame_dir = tempfile.mkdtemp(prefix="dexgrasp_preprocess_gif_")
        print("preprocess_gif_path:", gif_path)
        print("preprocess_gif_mode: viewer")
        print("preprocess_gif_frame_dir:", frame_dir)
        return {
            "mode": mode,
            "path": gif_path,
            "fps": fps,
            "frame_dir": frame_dir,
            "frame_paths": [],
            "stride": int(os.environ.get("DEXGRASP_PREPROCESS_GIF_STRIDE", "1")),
        }

    from isaacgym import gymapi

    env_ids_raw = os.environ.get("DEXGRASP_PREPROCESS_GIF_ENV_IDS")
    if env_ids_raw:
        env_ids = [int(x.strip()) for x in env_ids_raw.split(",") if x.strip()]
    else:
        num_envs = int(os.environ.get("DEXGRASP_PREPROCESS_GIF_NUM_ENVS", "9"))
        env_ids = list(range(min(num_envs, task.num_envs)))
    env_ids = [env_id for env_id in env_ids if 0 <= env_id < task.num_envs]
    if not env_ids:
        print("preprocess_gif_skipped: no valid env ids for {} envs".format(task.num_envs))
        return None

    tile_width = int(os.environ.get("DEXGRASP_PREPROCESS_GIF_TILE_WIDTH", "320"))
    tile_height = int(os.environ.get("DEXGRASP_PREPROCESS_GIF_TILE_HEIGHT", "240"))
    grid_cols = int(os.environ.get("DEXGRASP_PREPROCESS_GIF_GRID_COLS", "0"))
    if grid_cols <= 0:
        grid_cols = int(np.ceil(np.sqrt(len(env_ids))))
    grid_rows = int(np.ceil(float(len(env_ids)) / float(grid_cols)))

    cam_props = gymapi.CameraProperties()
    cam_props.width = tile_width
    cam_props.height = tile_height
    cam_pos = gymapi.Vec3(
        float(os.environ.get("DEXGRASP_PREPROCESS_GIF_CAM_X", "0.75")),
        float(os.environ.get("DEXGRASP_PREPROCESS_GIF_CAM_Y", "-0.85")),
        float(os.environ.get("DEXGRASP_PREPROCESS_GIF_CAM_Z", "0.95")),
    )
    cam_target = gymapi.Vec3(
        float(os.environ.get("DEXGRASP_PREPROCESS_GIF_TARGET_X", "0.0")),
        float(os.environ.get("DEXGRASP_PREPROCESS_GIF_TARGET_Y", "0.0")),
        float(os.environ.get("DEXGRASP_PREPROCESS_GIF_TARGET_Z", "0.62")),
    )
    cameras = []
    for env_id in env_ids:
        cam_handle = task.gym.create_camera_sensor(task.envs[env_id], cam_props)
        task.gym.set_camera_location(cam_handle, task.envs[env_id], cam_pos, cam_target)
        cameras.append((env_id, cam_handle))
    writer = imageio.get_writer(gif_path, mode="I", fps=fps)
    print("preprocess_gif_path:", gif_path)
    print("preprocess_gif_mode: sensor_grid")
    print("preprocess_gif_env_ids:", env_ids)
    return {
        "mode": mode,
        "cameras": cameras,
        "writer": writer,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "grid_cols": grid_cols,
        "grid_rows": grid_rows,
        "path": gif_path,
        "stride": int(os.environ.get("DEXGRASP_PREPROCESS_GIF_STRIDE", "1")),
    }

def _record_preprocess_gif_frame(task, recorder, step_idx):
    if recorder is None:
        return
    stride = max(1, int(recorder["stride"]))
    if step_idx % stride != 0:
        return
    if recorder["mode"] == "viewer":
        if task.viewer is None:
            return
        task.gym.fetch_results(task.sim, True)
        task.gym.step_graphics(task.sim)
        task.gym.draw_viewer(task.viewer, task.sim, True)
        frame_path = osp.join(recorder["frame_dir"], "frame_{:06d}.png".format(len(recorder["frame_paths"])))
        task.gym.write_viewer_image_to_file(task.viewer, frame_path)
        recorder["frame_paths"].append(frame_path)
        return

    from isaacgym import gymapi
    task.gym.fetch_results(task.sim, True)
    task.gym.step_graphics(task.sim)
    task.gym.render_all_camera_sensors(task.sim)
    tile_height = recorder["tile_height"]
    tile_width = recorder["tile_width"]
    grid = np.zeros(
        (
            recorder["grid_rows"] * tile_height,
            recorder["grid_cols"] * tile_width,
            3,
        ),
        dtype=np.uint8,
    )
    for tile_idx, (env_id, cam_handle) in enumerate(recorder["cameras"]):
        img = task.gym.get_camera_image(task.sim, task.envs[env_id], cam_handle, gymapi.IMAGE_COLOR)
        img = np.asarray(img, dtype=np.uint8).reshape(tile_height, tile_width, 4)[:, :, :3]
        row = tile_idx // recorder["grid_cols"]
        col = tile_idx % recorder["grid_cols"]
        y0 = row * tile_height
        x0 = col * tile_width
        grid[y0:y0 + tile_height, x0:x0 + tile_width] = img
    recorder["writer"].append_data(grid)

def _close_preprocess_gif(recorder):
    global _PREPROCESS_GIF_WRITTEN
    if recorder is None:
        return
    if recorder["mode"] == "viewer":
        try:
            import imageio
            writer = imageio.get_writer(recorder["path"], mode="I", fps=recorder["fps"])
            for frame_path in recorder["frame_paths"]:
                writer.append_data(imageio.imread(frame_path))
            writer.close()
        finally:
            if not _env_flag("DEXGRASP_PREPROCESS_GIF_KEEP_FRAMES"):
                for frame_path in recorder.get("frame_paths", []):
                    try:
                        os.remove(frame_path)
                    except OSError:
                        pass
                try:
                    os.rmdir(recorder["frame_dir"])
                except OSError:
                    pass
    else:
        recorder["writer"].close()
    _PREPROCESS_GIF_WRITTEN = True
    print("preprocess_gif_saved:", recorder["path"])

def _default_preprocess_roots():
    input_root = os.environ.get("DEXGRASP_PREPROCESS_INPUT_ROOT", "./dataset_o6_raw")
    output_root = os.environ.get("DEXGRASP_PREPROCESS_OUTPUT_ROOT", "./dataset_o6_preproc")
    train_root = os.environ.get("DEXGRASP_PREPROCESS_TRAIN_DIR", osp.join(input_root, "train"))
    valid_root = os.environ.get("DEXGRASP_PREPROCESS_VALID_DIR", osp.join(input_root, "valid"))
    train_out = os.environ.get("DEXGRASP_PREPROCESS_TRAIN_OUT_DIR", osp.join(output_root, "train"))
    valid_out = os.environ.get("DEXGRASP_PREPROCESS_VALID_OUT_DIR", osp.join(output_root, "valid"))
    return train_root, valid_root, train_out, valid_out

def worker_run(args, proc_id, npy_list, cfg, cfg_train,sim_params, agent_index, folder_name, save_npy):
    if _env_flag("DEXGRASP_PREPROCESS_RECORD_GIF") and _preprocess_gif_mode() == "sensor_grid":
        cfg.setdefault("env", {}).setdefault("infer_runtime", {})["record_video"] = True
        cfg["env"]["infer_runtime"]["enable_camera_sensors"] = True
    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index, npy_list=npy_list)
    gif_recorder = _setup_preprocess_gif(task, proc_id)
    seq_actions = task.grasp_seqs
    N, T, D = seq_actions.size()
    all_obs = None
    gt_all_actions = torch.zeros(task.num_envs, T, D, device=seq_actions.device)
    sim_all_actions = torch.zeros(task.num_envs, T, task.num_o6_hand_dofs, device=seq_actions.device)
    obj_all_state = torch.zeros(task.num_envs, T, 7)
    debug_hand_motion = _env_flag("DEXGRASP_PREPROCESS_DEBUG_HAND_MOTION")
    debug_env_id = int(os.environ.get("DEXGRASP_PREPROCESS_DEBUG_ENV_ID", "0"))
    debug_dof0 = None

    # start rollout trajs
    for i in range(seq_actions.shape[1]):
        # actions = torch.zeros((task.num_envs, task.num_actions))
        # actions[:, 2] = 1.0
        if i == 0:
            task.reset_buf = torch.ones(task.num_envs, device=task.device, dtype=torch.long)
            task.progress_buf = torch.zeros(task.num_envs, device=task.device, dtype=torch.long)
            env.reset()
        else:
            actions = seq_actions[:, i, :]
            env.task.step(actions, i)
            gt_all_actions[:, i - 1, :] = actions.clone()
            sim_all_actions[:, i - 1, :] = env.task.o6_hand_dof_pos.clone()
            # obj_all_state[:, i-1, :] = env.task.get_object_state()

            # obj_state, hand_state = env.task.get_object_and_hand_state() #CPU运行才能用
        obs_now = env.task.obs_buf.clone()
        if all_obs is None:
            all_obs = torch.zeros(task.num_envs, T, obs_now.shape[-1], device=obs_now.device, dtype=obs_now.dtype)
            print("preprocess_obs_dim: {}".format(obs_now.shape[-1]))
        elif all_obs.shape[-1] != obs_now.shape[-1]:
            raise RuntimeError(
                "observation dim changed during preprocessing: {} -> {}".format(
                    all_obs.shape[-1],
                    obs_now.shape[-1],
                )
            )
        _record_preprocess_gif_frame(env.task, gif_recorder, i)
        all_obs[:, i, :] = obs_now
        obj_all_state[:, i, :] = env.task.get_object_state()
        if debug_hand_motion and debug_env_id < task.num_envs:
            cur_dof = env.task.o6_hand_dof_pos[debug_env_id].detach().clone()
            if debug_dof0 is None:
                debug_dof0 = cur_dof
            if i in {0, 1, 2, seq_actions.shape[1] - 1}:
                delta = torch.norm(cur_dof - debug_dof0).item()
                print(
                    "preprocess_hand_motion_debug: step={}, env_id={}, dof_delta_from_step0={:.6f}, dof_xyz={}".format(
                        i,
                        debug_env_id,
                        delta,
                        cur_dof[:3].detach().cpu().numpy().tolist(),
                    )
                )

    # success_idx_all = torch.where(task.successes==1)[0].to(all_obs.device)
    flag = (task.successes==1).to(all_obs.device)
    total_successes=[]
    all_data = []
    object_idxs_tensor = torch.as_tensor(task.object_idxs, device=flag.device, dtype=torch.long)
    for obj_id in np.unique(task.object_idxs):
        obj_id = int(obj_id)
        save_dict = {}
        id_mask = object_idxs_tensor == obj_id
        flag_id = flag*id_mask
        success_idx_id = torch.where(flag_id==1)[0].to(all_obs.device)
        mean_success = torch.mean(task.successes[id_mask])
        total_successes.append(mean_success.data)
        selected_idx_id = success_idx_id
        if selected_idx_id.numel() == 0 and _env_flag("DEXGRASP_PREPROCESS_ALLOW_NO_SUCCESS_FALLBACK"):
            selected_idx_id = torch.where(id_mask)[0].to(all_obs.device)
            print(
                "preprocess_no_success_fallback: obj_id={}, using {} expert trajectories".format(
                    int(obj_id), selected_idx_id.numel()
                )
            )
        if selected_idx_id.numel() == 0:
            print("preprocess_skip_no_success: obj_id={}, no successful replay trajectories".format(int(obj_id)))
            continue

        save_dict["obs"] = all_obs[selected_idx_id].cpu().numpy()
        gt_all_actions_select = gt_all_actions[selected_idx_id].cpu()
        if D == task.num_o6_hand_dofs:
            vis_unscale_actions = unscale(
                gt_all_actions_select,
                task.o6_hand_dof_lower_limits.cpu(),
                task.o6_hand_dof_upper_limits.cpu(),
            )
            save_dict['vis_unscale_actions'] = torch.clamp(vis_unscale_actions, min=-1.0, max=1.0).numpy()
        else:
            save_dict['vis_unscale_actions'] = gt_all_actions_select.numpy()

        # save_dict['sim_unscale_actions'] = unscale(sim_all_actions_select, task.o6_hand_dof_lower_limits.cpu(),
        #                                            task.o6_hand_dof_upper_limits.cpu())
        # save_dict['sim_unscale_actions'] = torch.clamp(save_dict['sim_unscale_actions'], min=-1.0, max=1.0).numpy()

        if _env_flag("DEXGRASP_PREPROCESS_SAVE_PCDS") and len(selected_idx_id)>0:
            # h2o_vec = env.task.get_h2o_vector(sim_all_actions.cpu(), obj_all_state.cpu(), success_idx) #(N,T,21, 3)
            # save_dict['h2o_vec'] = h2o_vec.numpy()

            hand_pcds, hand_joints = env.task.get_seq_hand_pcd(sim_all_actions.cpu(), selected_idx_id)  # (N,T,512,3)
            try:
                obj_pcds = env.task.get_seq_obj_pcd(obj_all_state.cpu(), selected_idx_id, obj_id)  # ShadowHand signature
            except TypeError:
                obj_pcds = env.task.get_seq_obj_pcd(obj_all_state.cpu(), selected_idx_id)  # O6 signature
            save_dict['hand_pcds'] = hand_pcds.numpy()
            save_dict['obj_pcds'] = obj_pcds.numpy()


        success_idx = success_idx_id.cpu().numpy()
        select_idx = selected_idx_id.cpu().numpy()
        save_dict['success_idx'] = success_idx
        save_dict['selected_idx'] = select_idx
        save_dict['obj_rotmat'] = env.task.obj_trajs_info['obj_rotmat'][select_idx]
        save_dict['obj_scale'] = env.task.obj_trajs_info['obj_scale'][select_idx]
        save_dict['grasp_seqs'] = env.task.obj_trajs_info['grasp_seqs'][select_idx]
        obj_code_idx = npy_list[obj_id].get('obj_code_idx', obj_id)
        if isinstance(obj_code_idx, (list, tuple, np.ndarray)):
            obj_code_idx = np.asarray(obj_code_idx).reshape(-1)
            obj_code_idx = obj_code_idx[0] if obj_code_idx.size > 0 else obj_id
        try:
            obj_code_idx = int(obj_code_idx)
        except (TypeError, ValueError):
            obj_code_idx = obj_id
        save_dict['obj_code_idx'] = obj_code_idx
        all_data.append(save_dict)

        print(f"Success Rate: {mean_success}")

        # save obs and actions
        if save_npy:
            np.save(osp.join(folder_name, f"{task.object_code_list[obj_id]}.npy"), save_dict)

    _close_preprocess_gif(gif_recorder)
    env.task.clean_sim()
    del env, task
    gc.collect()
    torch.cuda.empty_cache()
    if total_successes:
        print(f"Process {proc_id} finished!  Mean SR={torch.stack(total_successes).mean()}")
    else:
        print(f"Process {proc_id} finished!  Mean SR=nan")
    a=1
    return all_data

def configure_o6_args(args):
    args.task = "o6HandGraspDexRepIjrr"
    args.cfg_env = "cfg/o6_hand_grasp_dexrep_ijrr.yaml"
    return args


def data_preprocess_online(data):
    set_np_formatting()

    args = configure_o6_args(get_args())
    # args.algo = "ppo1"
    args.seed = 0
    args.rl_device = "cuda:0"
    args.sim_device = "cuda:0"
    args.logdir = "logs/dexrep_dexgrasp"
    args.headless = True
    if _env_flag("DEXGRASP_PREPROCESS_RECORD_GIF") and _preprocess_gif_mode() == "viewer":
        args.headless = False

    cfg, cfg_train, logdir = load_cfg(args)
    cfg['env']['o6_policy_obs_mode'] = 'dexrep'
    cfg['env']['observationType'] = 'DexRep'
    cfg['env']['obs_dim']['prop'] = int(os.environ.get("DEXGRASP_O6_PREPROCESS_PROP_DIM", "77"))
    cfg['env']['obs_dim']['dexrep_sensor'] = int(os.environ.get("DEXGRASP_O6_PREPROCESS_DEXREP_SENSOR_DIM", "1040"))
    cfg['env']['obs_dim']['dexrep_pnl'] = int(os.environ.get("DEXGRASP_O6_PREPROCESS_DEXREP_PNL_DIM", "640"))
    cfg['env']['seq_start_rot_uniform'] = False
    if args.num_objs != -1:
        cfg['env']['num_objs'] = args.num_objs
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))

    agent_index = get_AgentIndex(cfg)
    folder_name = './dataset/train'
    save_npy = False

    cfg['env']['seq_start_rot_uniform'] = False
    all_npy = copy.deepcopy(data)

    n_per_proc = int(os.environ.get("DEXGRASP_PREPROCESS_N_PER_PROC", "20"))
    npy_groups = [all_npy[i:i + n_per_proc] for i in range(0, len(all_npy), n_per_proc)]
    # worker_run(0, npy_groups[0], cfg,cfg_train,sim_params, agent_index, folder_name, val_folder_name)

    all_data = []

    for proc_id, npy_list in enumerate(npy_groups):
        batch_data = worker_run(args, proc_id, npy_list, cfg, cfg_train, sim_params, agent_index, folder_name, save_npy)
        all_data.extend(batch_data)
    return all_data


def run():
    print("Algorithm: ", args.algo)
    agent_index = get_AgentIndex(cfg)

    required_keys = {'obs', 'vis_unscale_actions', 'success_idx'}
    replay_processed = _env_flag("DEXGRASP_PREPROCESS_REPLAY_PROCESSED")
    save_default = "0" if replay_processed else "1"
    save_npy = _env_flag("DEXGRASP_PREPROCESS_SAVE_OUTPUT", save_default)
    object_code_filter = os.environ.get("DEXGRASP_PREPROCESS_OBJECT_CODE", "").strip()
    max_files = _env_nonnegative_int("DEXGRASP_PREPROCESS_MAX_FILES", 0)

    if replay_processed:
        # The saved grasp sequences and object rotations already include any
        # preprocessing-time augmentation. Reapplying it would not replay the
        # exact data stored in the processed dataset.
        cfg['env']['seq_start_pos_uniform'] = False
        cfg['env']['seq_start_rot_uniform'] = False
        print(
            "Processed replay enabled: existing processed files will be simulated; "
            "trajectory augmentation is disabled; save_output={}.".format(save_npy)
        )

    train_root, valid_root, train_out, valid_out = _default_preprocess_roots()
    split = os.environ.get("DEXGRASP_PREPROCESS_SPLIT", "all").lower()
    if split == "train":
        split_roots = [(train_root, train_out)]
    elif split in {"valid", "val"}:
        split_roots = [(valid_root, valid_out)]
    elif split == "all":
        split_roots = [(train_root, train_out), (valid_root, valid_out)]
    else:
        raise ValueError("unsupported DEXGRASP_PREPROCESS_SPLIT: {}".format(split))

    selected_input_count = 0
    for traj_root_path, out_root_path in split_roots:
        if save_npy:
            os.makedirs(out_root_path, exist_ok=True)
        npy_paths = _select_preprocess_paths(
            glob(osp.join(traj_root_path, "*.npy")),
            object_code=object_code_filter,
            max_files=max_files,
        )
        print(
            "Selected {} input file(s) from {}{}".format(
                len(npy_paths),
                traj_root_path,
                " for object {}".format(object_code_filter) if object_code_filter else "",
            )
        )
        selected_input_count += len(npy_paths)

        all_npy = []
        for i, npy_path in enumerate(npy_paths):
            obj_code = osp.basename(npy_path)[:-len('.npy')]
            obj_trajs_info = np.load(npy_path, allow_pickle=True).item()
            if required_keys.issubset(obj_trajs_info.keys()):
                if not replay_processed:
                    print(f"Skipping already processed file: {npy_path}")
                    continue
                print(f"Replaying processed file: {npy_path}")
            obj_trajs_info = {
                key: val
                for key, val in obj_trajs_info.items()
                if key in {"grasp_seqs", "obj_rotmat", "obj_scale", "obj_code_idx"}
            }
                
            # if len(obj_trajs_info['grasp_seqs']) > 150:
            #     for key, val in obj_trajs_info.items():
            #         obj_trajs_info[key] = val[:150]
            obj_trajs_info['obj_code'] = obj_code
            all_npy.append(obj_trajs_info)

        n_per_proc = int(os.environ.get("DEXGRASP_PREPROCESS_N_PER_PROC", "5"))
        npy_groups = [all_npy[i:i + n_per_proc] for i in range(0, len(all_npy), n_per_proc)]

        for proc_id, npy_list in enumerate(npy_groups):
            worker_run(args, proc_id, npy_list, cfg, cfg_train, sim_params,
                       agent_index, out_root_path, save_npy=save_npy)

    if selected_input_count == 0:
        filter_description = (
            " matching object {!r}".format(object_code_filter) if object_code_filter else ""
        )
        raise FileNotFoundError(
            "No NPY input files{} were found for split {!r}.".format(filter_description, split)
        )

    print("Finish ALL !!!!!")



if __name__ == '__main__':
    set_np_formatting()
    args = configure_o6_args(get_args())
    if _env_flag("DEXGRASP_PREPROCESS_RECORD_GIF") and _preprocess_gif_mode() == "viewer":
        args.headless = False
    cfg, cfg_train, logdir = load_cfg(args)
    cfg['env']['o6_policy_obs_mode'] = 'dexrep'
    cfg['env']['observationType'] = 'DexRep'
    cfg['env']['obs_dim']['prop'] = int(os.environ.get("DEXGRASP_O6_PREPROCESS_PROP_DIM", "77"))
    cfg['env']['obs_dim']['dexrep_sensor'] = int(os.environ.get("DEXGRASP_O6_PREPROCESS_DEXREP_SENSOR_DIM", "1040"))
    cfg['env']['obs_dim']['dexrep_pnl'] = int(os.environ.get("DEXGRASP_O6_PREPROCESS_DEXREP_PNL_DIM", "640"))
    if args.num_objs != -1:
        cfg['env']['num_objs'] = args.num_objs
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    run()
