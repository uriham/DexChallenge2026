"""
M3 1단계: 성공이 검증된 궤적 하나를 IsaacGym에서 다시 리플레이하면서
"실제로 시뮬레이터가 계산한 상태"(월드 프레임 팜 pose, 관절값, 물체 pose)를
매 스텝 기록해 시연(demo) pkl로 저장한다.

- data_preprocess.py 의 worker_run()/run() 과 동일한 env 생성·롤아웃 패턴을 재사용한다.
- num_envs=1이 되도록, 오브젝트 1개 + 궤적 1개짜리 npy_list만 parse_task에 넘긴다.
- 이미 dataset_shinwoo_preproc 전처리에서 "성공"으로 검증된 궤적을 그대로 재생하므로
  seq_start_pos_uniform/seq_start_rot_uniform 랜덤 증강은 끈다 (그때와 동일 조건 재현).

사용법 (dexgrasp/ 디렉터리에서 실행):
    python record_demo.py
필요하면 아래 OBJECT_CODE / DEMO_TRAJ_INDEX / SOURCE_PREPROC_DIR 상수만 바꿔서 재사용.
"""
import os
import os.path as osp
import pickle

import numpy as np


from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from utils.process_marl import get_AgentIndex

import torch
from isaacgym.torch_utils import quat_conjugate, quat_mul, quat_rotate_inverse

THIS_DEXGRASP_DIR = osp.realpath(osp.dirname(__file__))

# ---- 여기만 바꿔서 다른 오브젝트/궤적에 재사용 ----
OBJECT_CODE = "core-bottle-be16ada66829940a451786f3cbfd6769"
DEMO_TRAJ_INDEX = 0  # dataset_shinwoo_preproc 안에서 "성공한 궤적" 중 몇 번째를 쓸지 (0 = 첫번째)
SOURCE_PREPROC_DIR = "/data/DexGraspMotionChallenge2026/dataset_shinwoo_preproc/train"
OUTPUT_DIR = osp.join(THIS_DEXGRASP_DIR, "demos")
LIFT_Z_THRESHOLD = 0.02  # 물체 z가 초기값보다 이만큼(m) 이상 오르면 "들어올리기 시작"으로 판단
# ---------------------------------------------------


def assert_clean_runtime_paths():
    if osp.realpath(os.getcwd()) != THIS_DEXGRASP_DIR:
        raise RuntimeError("record_demo.py must run from dexgrasp/ directory (cd dexgrasp 후 실행)")


def configure_o6_args(args):
    args.task = "o6HandGraspDexRepIjrr"
    args.cfg_env = "cfg/o6_hand_grasp_dexrep_ijrr.yaml"
    return args


def load_single_traj_npy_list():
    src_path = osp.join(SOURCE_PREPROC_DIR, OBJECT_CODE + ".npy")
    if not osp.exists(src_path):
        raise FileNotFoundError("전처리된 오브젝트 파일을 못 찾음: {}".format(src_path))
    data = np.load(src_path, allow_pickle=True).item()
    n_success = data["grasp_seqs"].shape[0]
    if DEMO_TRAJ_INDEX >= n_success:
        raise IndexError(
            "DEMO_TRAJ_INDEX={} 지만 이 오브젝트엔 성공 궤적이 {}개뿐임".format(DEMO_TRAJ_INDEX, n_success)
        )

    obj_trajs_info = {
        "grasp_seqs": data["grasp_seqs"][DEMO_TRAJ_INDEX : DEMO_TRAJ_INDEX + 1],
        "obj_rotmat": data["obj_rotmat"][DEMO_TRAJ_INDEX : DEMO_TRAJ_INDEX + 1],
        "obj_scale": data["obj_scale"][DEMO_TRAJ_INDEX : DEMO_TRAJ_INDEX + 1],
        "obj_code_idx": data.get("obj_code_idx", 0),
        "obj_code": OBJECT_CODE,
    }
    print(
        "loaded demo source: object={} traj_index={}/{} T={}".format(
            OBJECT_CODE, DEMO_TRAJ_INDEX, n_success, obj_trajs_info["grasp_seqs"].shape[1]
        )
    )
    return [obj_trajs_info]


def to_object_frame(world_pos, world_quat, obj_pos0, obj_quat0):
    """world_pos(3,)/world_quat(4,)를 물체 초기 pose(obj_pos0, obj_quat0) 기준 상대좌표로 변환."""
    obj_quat0_conj = quat_conjugate(obj_quat0.unsqueeze(0))
    rel_pos = quat_rotate_inverse(obj_quat0.unsqueeze(0), (world_pos - obj_pos0).unsqueeze(0))[0]
    rel_quat = quat_mul(obj_quat0_conj, world_quat.unsqueeze(0))[0]
    return rel_pos, rel_quat


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
    # 전처리 때 검증된 그 조건 그대로 재현 (랜덤 증강 끔)
    cfg["env"]["seq_start_pos_uniform"] = False
    cfg["env"]["seq_start_rot_uniform"] = False

    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    agent_index = get_AgentIndex(cfg)

    npy_list = load_single_traj_npy_list()
    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index, npy_list=npy_list)
    if task.num_envs != 1:
        raise RuntimeError("num_envs가 1이 아님({}) - npy_list 구성을 확인할 것".format(task.num_envs))

    palm_idx = task.hand_body_idx_dict["palm"]
    seq_actions = task.grasp_seqs  # (1, T, 12)
    T = seq_actions.shape[1]

    record = {
        "hand_dof": [],       # (T, num_dof) 관절값 (손목 6 + 손가락)
        "palm_pos_world": [], # (T, 3)
        "palm_quat_world": [],# (T, 4)
        "obj_pos_world": [],  # (T, 3)
        "obj_quat_world": [], # (T, 4)
        "action_applied": [], # (T, 12) 그 스텝에 실제로 흘려보낸 명령값 (i=0은 전부 0)
    }

    for i in range(T):
        if i == 0:
            task.reset_buf = torch.ones(task.num_envs, device=task.device, dtype=torch.long)
            task.progress_buf = torch.zeros(task.num_envs, device=task.device, dtype=torch.long)
            env.reset()
            applied = torch.zeros(1, seq_actions.shape[-1])
        else:
            actions = seq_actions[:, i, :]
            env.task.step(actions, i)
            applied = actions.clone().cpu()

        record["hand_dof"].append(task.o6_hand_dof_pos[0].clone().cpu().numpy())
        record["palm_pos_world"].append(task.rigid_body_states[0, palm_idx, 0:3].clone().cpu().numpy())
        record["palm_quat_world"].append(task.rigid_body_states[0, palm_idx, 3:7].clone().cpu().numpy())
        obj_state = task.get_object_state()[0].clone().cpu()
        record["obj_pos_world"].append(obj_state[0:3].numpy())
        record["obj_quat_world"].append(obj_state[3:7].numpy())
        record["action_applied"].append(applied[0].numpy())

    success = bool(task.successes[0].item() == 1)
    print("replay finished. success={}".format(success))

    obj_pos_world = np.stack(record["obj_pos_world"])  # (T,3)
    z0 = obj_pos_world[0, 2]
    lift_mask = obj_pos_world[:, 2] > (z0 + LIFT_Z_THRESHOLD)
    t_lift = int(np.argmax(lift_mask)) if lift_mask.any() else None
    print("T_lift = {} (z0={:.4f}, z_max={:.4f})".format(t_lift, z0, obj_pos_world[:, 2].max()))
    if t_lift is None:
        print("경고: LIFT_Z_THRESHOLD={}m 만큼 오른 시점을 못 찾음. success={} 확인 필요.".format(
            LIFT_Z_THRESHOLD, success
        ))

    # 물체 "초기" pose 기준 상대좌표로 팜 pose 변환 (DemoGrasp 식 물체 프레임화)
    obj_pos0 = torch.as_tensor(obj_pos_world[0])
    obj_quat0 = torch.as_tensor(record["obj_quat_world"][0])
    palm_pos_obj_frame = []
    palm_quat_obj_frame = []
    for t in range(T):
        p, q = to_object_frame(
            torch.as_tensor(record["palm_pos_world"][t]),
            torch.as_tensor(record["palm_quat_world"][t]),
            obj_pos0,
            obj_quat0,
        )
        palm_pos_obj_frame.append(p.numpy())
        palm_quat_obj_frame.append(q.numpy())

    demo = {
        "object_code": OBJECT_CODE,
        "demo_traj_index": DEMO_TRAJ_INDEX,
        "success": success,
        "T": T,
        "t_lift": t_lift,
        "hand_dof": np.stack(record["hand_dof"]),
        "palm_pos_world": np.stack(record["palm_pos_world"]),
        "palm_quat_world": np.stack(record["palm_quat_world"]),
        "obj_pos_world": obj_pos_world,
        "obj_quat_world": np.stack(record["obj_quat_world"]),
        "palm_pos_obj_frame": np.stack(palm_pos_obj_frame),
        "palm_quat_obj_frame": np.stack(palm_quat_obj_frame),
        "action_applied": np.stack(record["action_applied"]),
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = osp.join(OUTPUT_DIR, "{}_traj{}.pkl".format(OBJECT_CODE, DEMO_TRAJ_INDEX))
    with open(out_path, "wb") as f:
        pickle.dump(demo, f)
    print("saved demo to: {}".format(out_path))

    task.clean_sim()


if __name__ == "__main__":
    assert_clean_runtime_paths()
    main()
