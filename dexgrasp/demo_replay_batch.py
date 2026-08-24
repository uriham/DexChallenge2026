"""
G3: 시연 1개를 num_envs개의 서로 다른 오브젝트에 동시에 재배치해서 재생하고,
워프(학습된 보정) 없이 순수 재배치만으로 몇 %나 성공하는지 측정한다.

--------------------------------------------------------------------------
8/21 수정 — 물체 회전 정렬 추가 (ALIGN_OBJECT_ROTATION)

기존 버전은 물체 **위치만** 평행이동하고 물체 회전을 무시했다(성공률 3/21 ≈ 14%).
verify_demo_convention.py로 확인해보니 오브젝트마다 스폰 회전이 크게 다르다
(||R-I||_F: 평균 2.06, 최대 2.83 = 이론상 최대값). 즉 "회전 랜덤화가 없으니
시연 rpy를 그대로 써도 된다"는 기존 가정이 틀렸고, 손이 엉뚱한 각도로 접근하고 있었다.

이제 demograsp_warp.DemoReference가 시연을 물체 프레임(위치+회전)에 정렬해서 재배치한다.
ALIGN_OBJECT_ROTATION=False로 두면 기존(위치만) 동작을 재현해 비교할 수 있다.
--------------------------------------------------------------------------

사용법 (dexgrasp/ 에서 실행):
    python demo_replay_batch.py --headless
"""
import os
import os.path as osp
import pickle

import numpy as np

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from utils.process_marl import get_AgentIndex

import torch

from demograsp_warp import DemoReference, build_full_plan

THIS_DEXGRASP_DIR = osp.realpath(osp.dirname(__file__))

# ---- 설정 ----
DEMO_PKL = osp.join(THIS_DEXGRASP_DIR, "demos", "core-bottle-be16ada66829940a451786f3cbfd6769_traj0.pkl")
SOURCE_PREPROC_DIR = "/data/DexGraspMotionChallenge2026/dataset_shinwoo_preproc/train"
ALIGN_OBJECT_ROTATION = True  # False면 기존(위치만 정렬) 동작 재현

TARGET_OBJECT_CODES = [
    "core-bottle-be16ada66829940a451786f3cbfd6769",  # 진단용: 데모를 만든 그 병 자기 자신
    "core-bottle-7565e6eeee63174757354938178674b",
    "core-bottle-b6261066a2e8e4212c528d33bca1ac2",
    "core-jar-12ec19e85b31e274725f67267e31c89",
    "core-jar-f2cb6d5160ad3850ccc0a0f55f0bc5c2",
    "core-mug-43e1cabc5dd2fa91fffc97a61124b1a9",
    "core-mug-1bc5d303ff4d6e7e1113901b72a68e7c",
    "core-can-baaa4b9538caa7f06e20028ed3cb196e",
    "core-can-9b1f0ddd23357e01a81ec39fd9664e9b",
    "sem-Vase-aa080060dcfd22b3265d1076b4b6c5c",
    "sem-Vase-6e0036ac8c75b63dc6b7f2129135b672",
    "core-bowl-4227b58665eadcefc0dc3ed657ab97f0",
    "core-bowl-afb6bf20c56e86f3d8fdbcba78c84028",
    "core-cellphone-3a6a3db4a0174fddd2789f496481c83e",
    "core-cellphone-8f049b65309d8390f5304dc8cfbb76e1",
    "sem-Book-20dec770602a8ac2331999dc8823fe0d",
    "sem-Book-7899800a75e1ed45c7c51d4ea74651a7",
    "core-pistol-e3619c5b8d8ad37cf4de29b99f103946",
    "core-camera-1ab3abb5c090d9b68e940c4e64a94e1e",
    "sem-FoodItem-e5a4666423674a889f7bbf181f9c6d08",
    "ddg-bigbird_krylon_short_cuts",
]
# ---------------------------------------------------


def assert_clean_runtime_paths():
    if osp.realpath(os.getcwd()) != THIS_DEXGRASP_DIR:
        raise RuntimeError("demo_replay_batch.py must run from dexgrasp/ directory (cd dexgrasp 후 실행)")


def configure_o6_args(args):
    args.task = "o6HandGraspDexRepIjrr"
    args.cfg_env = "cfg/o6_hand_grasp_dexrep_ijrr.yaml"
    return args


def load_multi_object_npy_list(object_codes):
    npy_list = []
    for code in object_codes:
        src_path = osp.join(SOURCE_PREPROC_DIR, code + ".npy")
        if not osp.exists(src_path):
            raise FileNotFoundError("타겟 오브젝트 파일을 못 찾음: {}".format(src_path))
        data = np.load(src_path, allow_pickle=True).item()
        npy_list.append({
            "grasp_seqs": data["grasp_seqs"][0:1],
            "obj_rotmat": data["obj_rotmat"][0:1],
            "obj_scale": data["obj_scale"][0:1],
            "obj_code_idx": data.get("obj_code_idx", 0),
            "obj_code": code,
        })
    return npy_list


def main():
    set_np_formatting()
    args = configure_o6_args(get_args())
    args.seed = 0
    args.rl_device = "cuda:0"
    args.sim_device = "cuda:0"

    cfg, cfg_train, logdir = load_cfg(args)
    cfg["env"]["o6_policy_obs_mode"] = "dexrep"
    cfg["env"]["observationType"] = "DexRep"
    cfg["env"]["obs_dim"]["prop"] = 77
    cfg["env"]["obs_dim"]["dexrep_sensor"] = 1040
    cfg["env"]["obs_dim"]["dexrep_pnl"] = 640
    cfg["env"]["seq_start_pos_uniform"] = False
    cfg["env"]["seq_start_rot_uniform"] = False

    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    agent_index = get_AgentIndex(cfg)

    with open(DEMO_PKL, "rb") as f:
        demo = pickle.load(f)
    print("loaded demo: object={} success={} t_lift={}".format(
        demo["object_code"], demo["success"], demo["t_lift"]
    ))
    print("align_object_rotation =", ALIGN_OBJECT_ROTATION)

    npy_list = load_multi_object_npy_list(TARGET_OBJECT_CODES)
    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index, npy_list=npy_list)
    print("task.num_envs =", task.num_envs, "(기대값:", len(TARGET_OBJECT_CODES), ")")
    if task.num_envs != len(TARGET_OBJECT_CODES):
        raise RuntimeError("num_envs가 오브젝트 개수와 안 맞음 - npy_list 구성 확인 필요")

    task.reset_buf = torch.ones(task.num_envs, device=task.device, dtype=torch.long)
    task.progress_buf = torch.zeros(task.num_envs, device=task.device, dtype=torch.long)
    env.reset()

    for i in range(task.num_envs):
        loaded_code = task.object_code_list[task.object_idxs[i]]
        if loaded_code != TARGET_OBJECT_CODES[i]:
            print("경고: env {} 예상={} 실제={} (순서 불일치)".format(i, TARGET_OBJECT_CODES[i], loaded_code))

    palm_idx = task.hand_body_idx_dict["palm"]
    obj_state = task.get_object_state().clone().cpu().numpy()  # (N,7)
    obj_pos0, obj_quat0 = obj_state[:, 0:3], obj_state[:, 3:7]
    start_wrist6 = task.o6_hand_dof_pos[:, 0:6].clone().cpu().numpy()  # (N,6)

    # palm_pos_world - virtual_xyz = 상수 c (verify_demo_convention.py에서 (0,0,0.6) 확인)
    palm_pos = task.rigid_body_states[:, palm_idx, 0:3].clone().cpu().numpy()
    hand_root_offset = (palm_pos - start_wrist6[:, 0:3]).mean(axis=0)
    print("hand_root_offset c =", np.round(hand_root_offset, 6))

    if not ALIGN_OBJECT_ROTATION:
        # 기존 동작 재현: 물체 회전을 무시(전부 항등 쿼터니언)하고 위치만 정렬
        demo = dict(demo)
        demo["obj_quat_world"] = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]]), (demo["T"], 1))
        obj_quat0 = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]]), (task.num_envs, 1))

    demo_ref = DemoReference(demo, hand_root_offset)
    finger_limits = (
        task.o6_hand_dof_lower_limits[task.actuated_dof_indices].cpu().numpy(),
        task.o6_hand_dof_upper_limits[task.actuated_dof_indices].cpu().numpy(),
    )
    full_actions = build_full_plan(
        demo_ref, obj_pos0, obj_quat0, start_wrist6,
        actions=None, finger_limits=finger_limits, settle_steps=10,
    )
    n_steps = full_actions.shape[0]
    print("replay length: {} steps total".format(n_steps))

    actions_t = torch.as_tensor(full_actions, dtype=torch.float32, device=task.device)

    obj_z0 = obj_pos0[:, 2].copy()
    obj_z_max = obj_z0.copy()
    for i in range(1, n_steps):
        env.task.step(actions_t[i], i)
        cur_z = task.get_object_state()[:, 2].clone().cpu().numpy()
        obj_z_max = np.maximum(obj_z_max, cur_z)

    successes = task.successes.clone().cpu().numpy()
    print()
    print("=== 오브젝트별 결과 ===")
    for i, code in enumerate(TARGET_OBJECT_CODES):
        print("{:55s} success={} z_rise={:.4f}".format(
            code, bool(successes[i] == 1), obj_z_max[i] - obj_z0[i]
        ))

    print()
    print("=== 요약 ===")
    print("align_object_rotation={}  성공: {}/{} ({:.1f}%)".format(
        ALIGN_OBJECT_ROTATION, int(successes.sum()), len(successes), 100.0 * successes.mean()
    ))

    task.clean_sim()


if __name__ == "__main__":
    assert_clean_runtime_paths()
    main()
