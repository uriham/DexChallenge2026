"""
M3 3단계: 기록해둔 시연(record_demo.py 산출물)을 "다른 물체 위치"에 재배치해서
reach(현재 손 위치 -> 데모 시작점) + track(데모 그대로 재생) 두 단계로 재생한다.

핵심 근거 (o6_hand_grasp_dexrep_ijrr.py:1435-1480 o6_actions_to_dof_targets 확인):
- 손목 액션 6개는 world quaternion이 아니라 o6_hand_dof_pos[:,0:6]
  (virtual_joint x/y/z/roll/pitch/yaw)에 스케일 변환 없이 1:1로 그대로 들어간다.
- 따라서 "물체 위치만 큼 x,y,z를 평행이동"하는 것만으로 재배치가 되고,
  DemoGrasp처럼 쿼터니언/SE(3) 계산이 필요 없다 (1차 버전: 물체 회전 랜덤화 없음 가정).
- 손가락은 능동 관절 6개(hand_dof 컬럼 [6,7,9,11,13,15])만 사용 (record_demo.py에서
  URDF로 검증됨). actions_are_normalized=False(cfg 기본값)라 raw radian 값을 그대로 액션으로 써도 됨.

사용법 (dexgrasp/ 에서 실행):
    python demo_replay.py --headless
TARGET_OBJECT_CODE 를 바꿔서 다른 오브젝트에 재배치해볼 수 있다.
(1차 sanity check: 시연을 만든 병 자기 자신으로 먼저 돌려보는 걸 권장)
"""
import os
import os.path as osp
import pickle

import numpy as np

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from utils.process_marl import get_AgentIndex

import torch

THIS_DEXGRASP_DIR = osp.realpath(osp.dirname(__file__))

# ---- 여기만 바꿔서 재사용 ----
DEMO_PKL = osp.join(THIS_DEXGRASP_DIR, "demos", "core-bottle-be16ada66829940a451786f3cbfd6769_traj0.pkl")
TARGET_OBJECT_CODE = "core-bottle-be16ada66829940a451786f3cbfd6769"  # 1차: 자기 자신 재배치 sanity check
SOURCE_PREPROC_DIR = "/data/DexGraspMotionChallenge2026/dataset_shinwoo_preproc/train"
REACH_STEPS = 20
FINGER_INDICES_IN_HAND_DOF = [6, 7, 9, 11, 13, 15]  # record_demo.py/URDF로 검증된 능동 관절 인덱스
# ---------------------------------------------------


def assert_clean_runtime_paths():
    if osp.realpath(os.getcwd()) != THIS_DEXGRASP_DIR:
        raise RuntimeError("demo_replay.py must run from dexgrasp/ directory (cd dexgrasp 후 실행)")


def configure_o6_args(args):
    args.task = "o6HandGraspDexRepIjrr"
    args.cfg_env = "cfg/o6_hand_grasp_dexrep_ijrr.yaml"
    return args


def load_target_object_npy_list():
    """재배치 대상 오브젝트를 스폰하기 위한 npy_list. grasp_seqs 내용 자체는 안 쓰고
    obj_rotmat/obj_scale로 정확한 스폰만 필요하므로 그 오브젝트의 아무 궤적(0번)이나 사용."""
    src_path = osp.join(SOURCE_PREPROC_DIR, TARGET_OBJECT_CODE + ".npy")
    if not osp.exists(src_path):
        raise FileNotFoundError("타겟 오브젝트 파일을 못 찾음: {}".format(src_path))
    data = np.load(src_path, allow_pickle=True).item()
    obj_trajs_info = {
        "grasp_seqs": data["grasp_seqs"][0:1],
        "obj_rotmat": data["obj_rotmat"][0:1],
        "obj_scale": data["obj_scale"][0:1],
        "obj_code_idx": data.get("obj_code_idx", 0),
        "obj_code": TARGET_OBJECT_CODE,
    }
    return [obj_trajs_info]


def build_replay_actions(demo, new_obj_pos0, start_wrist6, reach_steps):
    # hand_dof(결과 상태)가 아니라 action_applied(원본 명령값)를 재생 기준으로 써야 함 —
    # hand_dof[i]는 action_applied[i]를 적용한 "결과"라서, hand_dof를 그대로 다시 액션으로
    # 쓰면 전체가 한 스텝씩 밀려서 재생된다 (off-by-one 버그, 8/21 self-replay 실패로 발견).
    action_applied = demo["action_applied"]  # (T, 12), index 0은 reset용 더미(전부 0)라 미사용
    T = action_applied.shape[0]
    demo_obj_pos0 = demo["obj_pos_world"][0]  # (3,)

    wrist_xyz_rel = action_applied[:, 0:3] - demo_obj_pos0  # (T,3) 물체 기준 상대 위치
    wrist_rpy = action_applied[:, 3:6]  # (T,3) 1차 버전: 물체 회전 랜덤화 없음 -> 그대로 사용
    finger = action_applied[:, 6:12]  # (T,6) action_applied는 이미 능동 6관절 포맷(원본 12D 액션)

    track_xyz = wrist_xyz_rel + new_obj_pos0  # 새 오브젝트 위치로 재배치
    track_actions_full = np.concatenate([track_xyz, wrist_rpy, finger], axis=1).astype(np.float32)  # (T,12)
    track_actions = track_actions_full[1:]  # 인덱스0은 더미라서 제외, 실제 재생은 1..T-1

    first_real_target = track_actions[0]
    reach_wrist = np.linspace(start_wrist6, first_real_target[0:6], reach_steps + 1)[1:]  # (reach_steps,6)
    reach_finger = np.tile(first_real_target[6:12], (reach_steps, 1))
    reach_actions = np.concatenate([reach_wrist, reach_finger], axis=1).astype(np.float32)  # (reach_steps,12)

    dummy_row = reach_actions[0:1]  # index 0은 안 쓰임(reset 직후라서), 채우기용
    full_actions = np.concatenate([dummy_row, reach_actions, track_actions], axis=0)
    return full_actions, T


def main():
    set_np_formatting()
    args = configure_o6_args(get_args())
    args.seed = 0
    args.rl_device = "cuda:0"
    args.sim_device = "cuda:0"
    args.headless = True

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

    npy_list = load_target_object_npy_list()
    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index, npy_list=npy_list)
    if task.num_envs != 1:
        raise RuntimeError("num_envs가 1이 아님({})".format(task.num_envs))

    # reset으로 타겟 오브젝트/손 초기 상태 확보
    task.reset_buf = torch.ones(task.num_envs, device=task.device, dtype=torch.long)
    task.progress_buf = torch.zeros(task.num_envs, device=task.device, dtype=torch.long)
    env.reset()

    new_obj_pos0 = task.get_object_state()[0, 0:3].clone().cpu().numpy()
    start_wrist6 = task.o6_hand_dof_pos[0, 0:6].clone().cpu().numpy()
    print("target object initial pos:", new_obj_pos0)
    print("current wrist dof (post-reset):", start_wrist6)

    full_actions, T = build_replay_actions(demo, new_obj_pos0, start_wrist6, REACH_STEPS)
    n_steps = full_actions.shape[0]
    print("replay length: reach={} + track={} (+1 reset frame) = {} steps".format(
        REACH_STEPS, T - 1, n_steps
    ))

    actions_t = torch.as_tensor(full_actions, dtype=torch.float32, device=task.device)

    obj_z_trace = []
    for i in range(n_steps):
        if i == 0:
            continue  # 이미 위에서 reset 함
        env.task.step(actions_t[i : i + 1], i)
        obj_z_trace.append(task.get_object_state()[0, 2].item())

    success = bool(task.successes[0].item() == 1)
    z0 = obj_z_trace[0] if obj_z_trace else float("nan")
    print("replay finished. success={} obj_z0={:.4f} obj_z_max={:.4f}".format(
        success, z0, max(obj_z_trace) if obj_z_trace else float("nan")
    ))

    task.clean_sim()


if __name__ == "__main__":
    assert_clean_runtime_paths()
    main()
