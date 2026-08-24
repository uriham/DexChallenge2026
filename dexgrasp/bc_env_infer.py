import gc
import os
import os.path as osp
import subprocess
import sys
import time
import numpy as np
import yaml

THIS_DEXGRASP_DIR = osp.realpath(osp.dirname(__file__))
THIS_REPO_DIR = osp.realpath(osp.join(THIS_DEXGRASP_DIR, ".."))
for _path in [THIS_REPO_DIR, THIS_DEXGRASP_DIR]:
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from utils.process_marl import get_AgentIndex

from ActionDiffusion.bc.model.policy.lhm_policy import LitBCModel
import dexgrasp.utils.test_env as test_env_module
from dexgrasp.utils.test_env import test_env
from utils.info_summary_print import save_results_summary

import torch

TEST_ENV_MODULE_PATH = osp.realpath(test_env_module.__file__)


def list_object_codes_from_dir(trajs_path):
    npy_paths = sorted(
        osp.join(trajs_path, name)
        for name in os.listdir(trajs_path)
        if name.endswith(".npy")
    )
    if not npy_paths:
        raise FileNotFoundError("no npy files found in trajs_path: {}".format(trajs_path))
    return [osp.splitext(osp.basename(path))[0] for path in npy_paths]


def resolve_repo_path(path):
    path = str(path)
    if osp.isabs(path):
        return path
    candidates = [
        osp.realpath(osp.join(THIS_DEXGRASP_DIR, path)),
        osp.realpath(osp.join(THIS_REPO_DIR, path)),
    ]
    for candidate in candidates:
        if osp.exists(candidate):
            return candidate
    return candidates[0]


def should_scan_object_codes(object_code_dict):
    return (
        object_code_dict is None
        or object_code_dict == "auto"
        or object_code_dict == ["auto"]
        or len(object_code_dict) == 0
    )


def env_bool(name, default=False):
    if os.environ.get(name) is None:
        return bool(default)
    return os.environ[name].lower() in {"1", "true", "yes", "on"}


def env_int(name, default):
    if os.environ.get(name) is None:
        return int(default)
    return int(os.environ[name])


def env_float(name, default):
    if os.environ.get(name) is None:
        return float(default)
    return float(os.environ[name])


def infer_runtime_cfg():
    return cfg.get('env', {}).get('infer_runtime', {}) or {}


def get_mode_object_codes(mode):
    if mode in ['seen', 'one']:
        obj_id_list = cfg['env'].get('seen_object_code_dict', [])
        trajs_path = cfg['trajs_path']['train']
    else:
        obj_id_list = cfg['env'].get('unseen_object_code_dict', [])
        trajs_path = cfg['trajs_path']['valid']
    if should_scan_object_codes(obj_id_list):
        obj_id_list = list_object_codes_from_dir(trajs_path)
    return [obj_id[:-4] if obj_id.endswith('.npy') else obj_id for obj_id in obj_id_list]


def assert_clean_runtime_paths():
    if osp.realpath(os.getcwd()) != THIS_DEXGRASP_DIR:
        raise RuntimeError("bc_env_infer.py must run from its clean worktree dexgrasp directory")
    expected_prefix = osp.realpath(osp.join(THIS_DEXGRASP_DIR, "utils"))
    if not TEST_ENV_MODULE_PATH.startswith(expected_prefix + os.sep):
        raise RuntimeError(
            "wrong test_env module imported: {} expected under {}".format(
                TEST_ENV_MODULE_PATH,
                expected_prefix,
            )
        )
    if env_bool("DEXGRASP_DEBUG_IMPORTS", False):
        print("test_env_module_path:", TEST_ENV_MODULE_PATH)


def create_env():
    runtime_cfg = infer_runtime_cfg()
    if os.environ.get("DEXGRASP_TASK"):
        args.task = os.environ["DEXGRASP_TASK"]
    else:
        args.task = "o6HandGraspDexRepIjrr"
    if args.num_objs != -1:
        cfg['env']['num_objs'] = args.num_objs
    if os.environ.get("DEXGRASP_NUM_ENVS"):
        cfg['env']['numEnvs'] = int(os.environ["DEXGRASP_NUM_ENVS"])
    sim_params = parse_sim_params(args, cfg, cfg_train)
    max_gpu_contact_pairs = runtime_cfg.get("max_gpu_contact_pairs")
    if os.environ.get("DEXGRASP_MAX_GPU_CONTACT_PAIRS") or max_gpu_contact_pairs:
        sim_params.physx.max_gpu_contact_pairs = env_int(
            "DEXGRASP_MAX_GPU_CONTACT_PAIRS",
            max_gpu_contact_pairs,
        )
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    cfg['env']['env_mode'] = 'bc_env_infer'

    cfg["env"]["observationType"] = 'DexRep'
    if os.environ.get("DEXGRASP_HAND_ROOT_Z"):
        hand_root_pos = list(cfg["env"].get("hand_root_pos", [0.0, 0.0, cfg["env"].get("table_height", 0.6)]))
        hand_root_pos[2] = float(os.environ["DEXGRASP_HAND_ROOT_Z"])
        cfg["env"]["hand_root_pos"] = hand_root_pos
    if os.environ.get("DEXGRASP_HAND_ROOT_POS"):
        values = [float(v.strip()) for v in os.environ["DEXGRASP_HAND_ROOT_POS"].split(",") if v.strip()]
        if len(values) != 3:
            raise ValueError("DEXGRASP_HAND_ROOT_POS expects three comma-separated floats")
        cfg["env"]["hand_root_pos"] = values
    if os.environ.get("DEXGRASP_O6_WRIST_Z_OFFSET"):
        cfg["env"]["o6_wrist_z_offset"] = float(os.environ["DEXGRASP_O6_WRIST_Z_OFFSET"])
    if os.environ.get("DEXGRASP_O6_INITIAL_WRIST_Z_OFFSET"):
        cfg["env"]["o6_initial_wrist_z_offset"] = float(os.environ["DEXGRASP_O6_INITIAL_WRIST_Z_OFFSET"])
    if os.environ.get("DEXGRASP_RESET_WARMUP_STEPS"):
        cfg["env"].setdefault("infer_runtime", {})
        cfg["env"]["infer_runtime"]["reset_warmup_steps"] = int(os.environ["DEXGRASP_RESET_WARMUP_STEPS"])
    runtime_bool_overrides = {
        "DEXGRASP_RESET_OBJECTS_AFTER_ENV_RESET": "reset_objects_after_env_reset",
        "DEXGRASP_SYNC_HAND_BEFORE_OBJECT_RESET": "sync_hand_before_object_reset",
        "DEXGRASP_DIRECT_INITIAL_RESET": "direct_initial_reset",
        "DEXGRASP_REFRESH_AFTER_RESET": "refresh_after_reset",
    }
    for env_name, cfg_name in runtime_bool_overrides.items():
        if os.environ.get(env_name) is not None:
            cfg["env"].setdefault("infer_runtime", {})
            cfg["env"]["infer_runtime"][cfg_name] = env_bool(env_name)

    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)

    return task, env

def create_bc_model(base_path):
    from omegaconf import OmegaConf
    import os.path as osp

    bc_config_name = os.environ.get("DEXGRASP_BC_CONFIG", "lhm_bc_o6_dexrep_full.yaml")
    if not bc_config_name.endswith(".yaml"):
        bc_config_name = "{}.yaml".format(bc_config_name)
    bc_args = OmegaConf.load("{}/{}".format(osp.join(base_path, 'ActionDiffusion/bc/config'), bc_config_name))
    env_args = OmegaConf.load("{}/o6_hand_grasp_dexrep_ijrr.yaml".format(osp.join(base_path, 'dexgrasp/cfg')))
    if os.environ.get("DEXGRASP_POLICY_ACTOR_CRITIC"):
        bc_args.policy.actor_critic = os.environ["DEXGRASP_POLICY_ACTOR_CRITIC"]
    if os.environ.get("DEXGRASP_TEMPORAL_HISTORY"):
        bc_args.encoder.n_obs_steps = int(os.environ["DEXGRASP_TEMPORAL_HISTORY"])
    if os.environ.get("DEXGRASP_TCN_DROPOUT"):
        bc_args.encoder.tcn_dropout = float(os.environ["DEXGRASP_TCN_DROPOUT"])
    if os.environ.get("DEXGRASP_TCN_KERNEL_SIZE"):
        bc_args.encoder.tcn_kernel_size = int(os.environ["DEXGRASP_TCN_KERNEL_SIZE"])
    if os.environ.get("DEXGRASP_TCN_DILATIONS"):
        bc_args.encoder.tcn_dilations = [
            int(value.strip())
            for value in os.environ["DEXGRASP_TCN_DILATIONS"].split(",")
            if value.strip()
        ]
    if os.environ.get("DEXGRASP_OBS_DIM"):
        bc_args.obs_dim = int(os.environ["DEXGRASP_OBS_DIM"])
    if os.environ.get("DEXGRASP_O6_OBS_MODE"):
        bc_args.o6_obs_mode = os.environ["DEXGRASP_O6_OBS_MODE"]
    if os.environ.get("DEXGRASP_DEXREP_SENSOR_DIM"):
        bc_args.dexrep_sensor_dim = int(os.environ["DEXGRASP_DEXREP_SENSOR_DIM"])
    if os.environ.get("DEXGRASP_DEXREP_PNL_DIM"):
        bc_args.dexrep_pnl_dim = int(os.environ["DEXGRASP_DEXREP_PNL_DIM"])
    bc_model_name = bc_args.policy.actor_critic

    if bc_model_name=='ActorCriticPNG':
        env_args.env.obs_dim.pop('dexrep_sensor')
        env_args.env.obs_dim.pop('dexrep_pnl')

    elif bc_model_name in {
        'ActorCriticDexRep',
        'ActorCriticDexRepTemporal',
        'ActorCriticDexRepTemporalTCN',
    }:
        if getattr(bc_args, "o6_obs_mode", None) == "dexrep":
            env_args.env.o6_policy_obs_mode = "dexrep"
            env_args.env.obs_dim.prop = int(bc_args.obs_dim)
            if getattr(bc_args, "dexrep_sensor_dim", None) is not None:
                env_args.env.obs_dim.dexrep_sensor = int(bc_args.dexrep_sensor_dim)
            if getattr(bc_args, "dexrep_pnl_dim", None) is not None:
                env_args.env.obs_dim.dexrep_pnl = int(bc_args.dexrep_pnl_dim)
            cfg['env']['o6_policy_obs_mode'] = "dexrep"
        env_args.env.obs_dim.pop('pnG')
    else:
        raise ValueError("unsupported BC actor_critic: {}".format(bc_model_name))

    bc_model = LitBCModel(bc_args, env_args.env)
    bc_model = bc_model.to(args.rl_device)
    bc_args.policy.checkpoints = resolve_repo_path(
        bc_args.policy.checkpoints.split('DexRep_Isaac_ijrr/')[-1]
    )
    if os.environ.get("DEXGRASP_POLICY_CKPT"):
        bc_args.policy.checkpoints = resolve_repo_path(os.environ["DEXGRASP_POLICY_CKPT"])
    ckpt = torch.load(bc_args.policy.checkpoints, map_location=torch.device(args.rl_device))
    bc_model.load_state_dict(ckpt['state_dict'])
    print("checkpoint_load path={} actor_critic={}".format(bc_args.policy.checkpoints, bc_model_name))
	
    checkpoint_stem = osp.splitext(osp.basename(bc_args.policy.checkpoints))[0]
    experiment_name = osp.basename(osp.dirname(bc_args.policy.checkpoints))
    snapshot_names = '{}_{}'.format(checkpoint_stem, experiment_name)
    bc_info_name= cfg['env']['obj_type']+'_'+snapshot_names #+'_'bc_model_name

    return bc_model, bc_model_name, bc_info_name


def release_cuda_memory():
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception as exc:
        print("cuda_memory_release_warning:", exc)


def cleanup_task_env(task, env):
    cleanup_target = None
    if env is not None and getattr(env, "task", None) is not None:
        cleanup_target = env.task
    elif task is not None:
        cleanup_target = task

    if cleanup_target is not None:
        try:
            cleanup_target.clean_sim()
        except Exception as exc:
            print("clean_sim_warning:", exc)


def result_filename_for_info(bc_info_name):
    filename = bc_info_name if bc_info_name.endswith('.yaml') else bc_info_name + '.yaml'
    return osp.join(THIS_DEXGRASP_DIR, "results", filename)


def parse_result_detail(result_desc):
    parsed = {}
    for part in str(result_desc).split(", "):
        if "=" in part:
            key, value = part.split("=", 1)
        elif ":" in part:
            key, value = part.split(":", 1)
        else:
            continue
        parsed[key.strip()] = value.strip()
    try:
        return {
            "name": parsed.get("name", ""),
            "N_seq": int(float(parsed.get("N_seq", 0))),
            "success_num": float(parsed.get("success_num", 0.0)),
            "success_rate": float(parsed.get("success_rate", 0.0)),
        }
    except Exception:
        return {
            "name": str(result_desc),
            "N_seq": 0,
            "success_num": 0.0,
            "success_rate": 0.0,
        }


def load_result_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "detail_info" in data:
        details = data.get("detail_info") or []
    else:
        details = []
        for item in data.get("detail", []) or []:
            if isinstance(item, list) and item:
                details.append(parse_result_detail(item[0]))
            else:
                details.append(parse_result_detail(item))
    total_success = float(data.get("total_success_num", sum(d.get("success_num", 0.0) for d in details)))
    total_trials = int(data.get("total_trials", sum(d.get("N_seq", 0) for d in details)))
    return {
        "details": details,
        "total_success_num": total_success,
        "total_trials": total_trials,
    }


def save_aggregate_summary(mode, detail_info, filename):
    total_success = float(sum(item.get("success_num", 0.0) for item in detail_info))
    total_trials = int(sum(item.get("N_seq", 0) for item in detail_info))
    weighted_rate = total_success / total_trials if total_trials > 0 else 0.0
    mean_rate = float(np.mean([item.get("success_rate", 0.0) for item in detail_info])) if detail_info else 0.0
    results = {
        "dataset_name": mode,
        "detail_info": detail_info,
        "total_success_num": total_success,
        "total_trials": total_trials,
        "weighted_success_rate": weighted_rate,
        "mean_object_success_rate": mean_rate,
    }
    os.makedirs(osp.join(THIS_DEXGRASP_DIR, "results"), exist_ok=True)
    path = result_filename_for_info(filename)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(results, f, allow_unicode=True, sort_keys=False)
    print(
        "All Trajectory Success Rate: {:.4f} ({:.1f}/{})".format(
            weighted_rate,
            total_success,
            total_trials,
        )
    )
    print("Mean Object Success Rate: {:.4f}".format(mean_rate))
    print("aggregate_result_path:", path)
    return results


def run(mode='seen'):
    base_path = '../'
    obj_id_list = get_mode_object_codes(mode)

    bc_model, bc_model_name, bc_info_name = create_bc_model(base_path)
    bc_info_name=bc_info_name+'_'+'test_num{}'.format(cfg['env']['test_num'])
    if os.environ.get("DEXGRASP_RESULT_SUFFIX"):
        bc_info_name = bc_info_name + '_' + os.environ["DEXGRASP_RESULT_SUFFIX"]
    cfg['env']['bc_model_name'] = bc_model_name

    results = {
        'total_succ_rates':[],
        'dataset_name':mode,
        'detail':[],
        'detail_info':[],
        'total_success_num': 0.0,
        'total_trials': 0,
    }
    print('bc info: {}'.format(bc_info_name))
    batch_size = cfg['env']['infer_batch_size']
    for i in range(0, len(obj_id_list), batch_size):
        batch = obj_id_list[i:i + batch_size]
        processed_batch = [obj_id[:-4] if obj_id.endswith('.npy') else obj_id for obj_id in batch]
        cfg['env']['object_code_dict'] = processed_batch
        obj_glob_feat = None
        task = None
        env = None
        try:
            task, env = create_env()

            use_gt = os.environ.get("DEXGRASP_USE_GT", "0").lower() in {"1", "true", "yes", "on"}
            use_part_gt = os.environ.get("DEXGRASP_USE_PART_GT", "0").lower() in {"1", "true", "yes", "on"}
            if use_gt or use_part_gt:
                print("rollout_action_source: use_gt={} use_part_gt={}".format(use_gt, use_part_gt))
            succ_rate, result_desc, result_info = test_env(
                args,
                task,
                env,
                bc_model,
                bc_model_name,
                processed_batch[-1],
                obj_glob_feat,
                use_gt=use_gt,
                use_part_gt=use_part_gt,
            )
            results['total_succ_rates'].append(succ_rate)
            results['detail'].append([result_desc])
            results['detail_info'].append(result_info)
            results['total_success_num'] += result_info['success_num']
            results['total_trials'] += result_info['N_seq']
        finally:
            cleanup_task_env(task, env)
            del task, env, obj_glob_feat
            release_cuda_memory()
            gc.collect()
        gc.collect()

    save_results_summary(results,filename=bc_info_name, to_yaml=True)
    print('---------------------finish {}--------------------------\n'.format(bc_info_name))


def run_subprocess_batches(mode='seen'):
    runtime_cfg = infer_runtime_cfg()
    obj_id_list = get_mode_object_codes(mode)
    chunk_size = env_int("DEXGRASP_SUBPROCESS_BATCH_SIZE", runtime_cfg.get("subprocess_batch_size", 1))
    sleep_secs = env_float("DEXGRASP_SUBPROCESS_SLEEP_SECS", runtime_cfg.get("subprocess_sleep_secs", 2))
    base_suffix = os.environ.get("DEXGRASP_RESULT_SUFFIX", "")
    command = [sys.executable, osp.realpath(__file__)] + sys.argv[1:]
    bc_model, _, bc_info_name = create_bc_model('../')
    del bc_model
    release_cuda_memory()
    gc.collect()
    bc_info_name=bc_info_name+'_'+'test_num{}'.format(cfg['env']['test_num'])
    if base_suffix:
        bc_info_name = bc_info_name + '_' + base_suffix
    child_result_paths = []

    if chunk_size <= 0:
        raise ValueError("DEXGRASP_SUBPROCESS_BATCH_SIZE must be positive")

    print(
        "subprocess_eval_start: mode={}, total_objects={}, chunk_size={}, sleep_secs={}".format(
            mode, len(obj_id_list), chunk_size, sleep_secs
        )
    )
    for start in range(0, len(obj_id_list), chunk_size):
        chunk = obj_id_list[start:start + chunk_size]
        child_env = os.environ.copy()
        child_env["DEXGRASP_EVAL_CHILD"] = "1"
        child_env["DEXGRASP_EVAL_OBJECTS"] = ",".join(chunk)
        child_env["DEXGRASP_OBJ_TYPE"] = mode
        child_env["DEXGRASP_INFER_BATCH_SIZE"] = str(len(chunk))
        suffix_parts = [part for part in [base_suffix, "subproc{:04d}".format(start // chunk_size)] if part]
        child_suffix = "_".join(suffix_parts)
        child_env["DEXGRASP_RESULT_SUFFIX"] = child_suffix
        child_info_name = bc_info_name
        if base_suffix:
            child_info_name = child_info_name.rsplit("_" + base_suffix, 1)[0]
        child_info_name = child_info_name + "_" + child_suffix
        child_result_paths.append(result_filename_for_info(child_info_name))

        print(
            "subprocess_eval_chunk: index={}, objects={}".format(
                start // chunk_size, ",".join(chunk)
            )
        )
        completed = subprocess.run(command, cwd=THIS_DEXGRASP_DIR, env=child_env)
        if completed.returncode != 0:
            raise RuntimeError(
                "subprocess eval failed: index={}, returncode={}, objects={}".format(
                    start // chunk_size, completed.returncode, ",".join(chunk)
                )
            )
        release_cuda_memory()
        gc.collect()
        if sleep_secs > 0 and start + chunk_size < len(obj_id_list):
            time.sleep(sleep_secs)

    detail_info = []
    for path in child_result_paths:
        if not osp.exists(path):
            print("aggregate_warning_missing_child_result:", path)
            continue
        summary = load_result_summary(path)
        detail_info.extend(summary["details"])

    save_aggregate_summary(mode, detail_info, bc_info_name + "_aggregate")
    print("subprocess_eval_finish: mode={}, total_objects={}".format(mode, len(obj_id_list)))


if __name__ == '__main__':
    assert_clean_runtime_paths()
    args = get_args()
    args.task = "o6HandGraspDexRepIjrr"
    args.cfg_env = "cfg/o6_hand_grasp_dexrep_ijrr.yaml"
    cfg, cfg_train, logdir = load_cfg(args)
    agent_index = get_AgentIndex(cfg)

    if os.environ.get("DEXGRASP_EVAL_DATA_DIR"):
        cfg['trajs_path']['train'] = os.environ["DEXGRASP_EVAL_DATA_DIR"]
        cfg['trajs_path']['valid'] = os.environ["DEXGRASP_EVAL_DATA_DIR"]
    if os.environ.get("DEXGRASP_EVAL_ASSET_DIR"):
        asset_dir = os.environ["DEXGRASP_EVAL_ASSET_DIR"].strip()
        if not asset_dir.startswith("/"):
            asset_dir = "/" + asset_dir
        if not asset_dir.endswith("/"):
            asset_dir = asset_dir + "/"
        cfg['env']['asset']['assetFileNameObj'] = asset_dir
        cfg['env']['asset']['assetFileNameObj_raw'] = asset_dir

    if cfg['env']['obj_type'] in ['seen', 'one']:
        cfg['env'].setdefault('seen_object_code_dict', 'auto')
    else:
        cfg['env'].setdefault('unseen_object_code_dict', 'auto')

    if os.environ.get("DEXGRASP_EVAL_OBJECTS"):
        cfg['env']['obj_type'] = os.environ.get("DEXGRASP_OBJ_TYPE", cfg['env'].get('obj_type', 'seen'))
        object_code_key = (
            'seen_object_code_dict'
            if cfg['env']['obj_type'] in ['seen', 'one']
            else 'unseen_object_code_dict'
        )
        cfg['env'][object_code_key] = [
            code.strip()
            for code in os.environ["DEXGRASP_EVAL_OBJECTS"].split(",")
            if code.strip()
        ]
    if os.environ.get("DEXGRASP_INFER_BATCH_SIZE"):
        cfg['env']['infer_batch_size'] = int(os.environ["DEXGRASP_INFER_BATCH_SIZE"])
    if os.environ.get("DEXGRASP_TEST_NUM"):
        cfg['env']['test_num'] = int(os.environ["DEXGRASP_TEST_NUM"])

    runtime_cfg = infer_runtime_cfg()
    if (
        env_bool("DEXGRASP_EVAL_SUBPROCESS", runtime_cfg.get("use_subprocess", False))
        and os.environ.get("DEXGRASP_EVAL_CHILD", "0") != "1"
    ):
        run_subprocess_batches(mode=cfg['env']['obj_type'])
    else:
        run(mode=cfg['env']['obj_type'])
