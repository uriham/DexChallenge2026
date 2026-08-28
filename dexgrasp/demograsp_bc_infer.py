"""
DemoGrasp one-step PPO 체크포인트를 대회 공식 bc_env_infer.py 파이프라인(test_env)으로
평가하기 위한 어댑터 실행 스크립트.

--------------------------------------------------------------------------
왜 필요한가

utils/test_env.py는 두 곳에서 model.__class__.__name__ == 'LitBCModel'을 하드 요구하고
(:307, :511-512), model.model.act_inference(obs)를 "매 시뮬레이션 스텝마다" 부르는
closed-loop 롤아웃이다(부분 리셋 지원 — env마다 끝나는 시점이 달라도 계속 진행).
BC 정책(매 스텝 관측을 보고 액션을 내는 정책) 전용 인터페이스다.

반면 DemoGrasp 정책(ppo_onestep.ActorCritic, train_demograsp.py로 학습한 체크포인트)은
에피소드(리셋) 당 "1회만" 호출돼 12D 워프 벡터 하나를 내고, demograsp_warp.build_full_plan
으로 그걸 전체 궤적으로 미리 펼친 뒤 재생하는 open-loop 구조다. 그대로는 test_env에 못 끼운다.

이 스크립트는 test_env가 매 스텝 부르는 act_inference를 "reset_history(env_ids)가 불린
env만 그 순간 정책을 1회 호출해 새 궤적을 계산해두고, 그 외 env는 이미 계산해둔 궤적을
스텝 카운터로 인덱싱만 한다"로 재해석하는 어댑터(_DemoGraspInnerModel/LitBCModel)를 만들어
끼워 넣는다. task/env 생성, 오브젝트 목록 산출, 배치 루프, 결과 집계는 bc_env_infer.py의
기존 함수(create_env, test_env, get_mode_object_codes, save_results_summary 등)를
그대로 재사용한다 — 이 스크립트가 실제로 새로 만드는 건 어댑터 두 클래스뿐이다.

액션 적용 경로 호환성(사전 확인 완료): cfg/o6_hand_grasp_dexrep_ijrr.yaml:80의
o6_control_wrist_each_step: True가 기본값이고 bc_env_infer.py가 이를 덮어쓰지 않으므로,
env_mode='bc_env_infer'에서도 매 스텝 12D 전체가 그대로 적용된다
(tasks/o6_hand_grasp_dexrep_ijrr.py:2054) — 우리 절대좌표 궤적 재생과 충돌 없음.
성공 판정(task.successes)도 동일 함수를 공유하므로 우리 내부 PPO test-loop 수치
(71%/59.9%)와 여기서 나오는 수치는 같은 기준으로 비교 가능하다.

--------------------------------------------------------------------------
사용법 (dexgrasp/ 에서 실행)

    DEMOGRASP_CKPT=/data/DexGraspMotionChallenge2026/runs_demograsp/<run>/model_2000.pt \
    DEXGRASP_EVAL_DATA_DIR=/data/DexGraspMotionChallenge2026/dataset_shinwoo_preproc/train \
    DEXGRASP_EVAL_OBJECTS=core-bottle-...,core-jar-...,... \
    python demograsp_bc_infer.py --headless

DEXGRASP_EVAL_OBJECTS를 생략하면 (README에 문서화된 bc_env_infer.py 기본 동작과 동일하게)
DEXGRASP_EVAL_DATA_DIR 경로의 .npy 전체를 스캔해서 그 안의 모든 오브젝트를 검증한다
(bc_env_infer.py의 get_mode_object_codes/list_object_codes_from_dir 재사용, split_o6_dataset_by_*.py가
오브젝트 단위가 아니라 궤적 단위로 train/valid를 나누므로 "오브젝트 개수 제한"은 더 이상 기본이 아님).
DEMOGRASP_EXCLUDE_TRAIN_RUN으로 학습 run 디렉터리를 주면 그 결과에서 학습에 쓴 오브젝트만 뺀다.
DEXGRASP_HELD_OUT_N(선택)을 주면 스캔 결과를 그 개수로 잘라 스모크 테스트에 쓸 수 있다.
"""
import os
import os.path as osp

import numpy as np

THIS_DEXGRASP_DIR = osp.realpath(osp.dirname(__file__))


def env_str(name, default):
    return os.environ.get(name, default)


def env_bool(name, default=False):
    if os.environ.get(name) is None:
        return bool(default)
    return os.environ[name].lower() in {"1", "true", "yes", "on"}


def env_int(name, default):
    if os.environ.get(name) is None:
        return int(default)
    return int(os.environ[name])


import bc_env_infer as bci  # noqa: E402  (isaacgym import-order 준수 위해 여기서 import)
from utils.config import get_args, load_cfg  # noqa: E402
from utils.process_marl import get_AgentIndex  # noqa: E402

from demograsp_warp import DemoReference, build_full_plan  # noqa: E402
from demograsp_env import OBS_DIM, NUM_PCL_POINTS  # noqa: E402  (관측 레이아웃 재사용)
from demograsp_port.ppo_onestep.module import ActorCritic  # noqa: E402
from train_demograsp import build_train_param  # noqa: E402

import torch  # noqa: E402  (isaacgym이 위 import들에서 먼저 로드된 뒤에 torch를 import해야 함)


DEFAULT_DEMO_PKL = osp.join(
    THIS_DEXGRASP_DIR, "demos", "core-bottle-be16ada66829940a451786f3cbfd6769_traj0.pkl"
)


class _DemoGraspInnerModel:
    """LitBCModel.model 이 제공해야 하는 인터페이스(act_inference, reset_history)만 구현.

    demograsp_env.DemoGraspO6Env와 계산 로직은 동일하나, 여기서는 "배치 전체 동시 리셋"이
    아니라 "reset_history(env_ids)로 통보받은 일부 env만" 궤적을 새로 계산해야 한다는 점이
    다르다(test_env.py의 부분 리셋 롤아웃 구조 때문).
    """

    def __init__(self, task, actor_critic, demo_pkl, pcl_seed=0, align_rotation=False, settle_steps=10):
        self.task = task
        self.actor_critic = actor_critic
        self.device = task.device
        self.num_envs = int(task.num_envs)
        self.settle_steps = int(settle_steps)

        self.palm_idx = task.hand_body_idx_dict["palm"]
        self.finger_limits = (
            task.o6_hand_dof_lower_limits[task.actuated_dof_indices].cpu().numpy(),
            task.o6_hand_dof_upper_limits[task.actuated_dof_indices].cpu().numpy(),
        )

        palm_pos = task.rigid_body_states[:, self.palm_idx, 0:3].detach().cpu().numpy()
        virtual_xyz = task.o6_hand_dof_pos[:, 0:3].detach().cpu().numpy()
        self.hand_root_offset = (palm_pos - virtual_xyz).mean(axis=0)

        import pickle
        with open(demo_pkl, "rb") as f:
            demo = pickle.load(f)
        self.demo_ref = DemoReference(demo, self.hand_root_offset, align_rotation=align_rotation)

        pcds = task.obj_init_obj_pcds  # (B, N, 3) 물체 로컬 좌표
        n_pts = pcds.shape[1]
        if n_pts < NUM_PCL_POINTS:
            raise ValueError("점군 개수 {} < {}".format(n_pts, NUM_PCL_POINTS))
        rng = np.random.default_rng(pcl_seed)
        idx = rng.choice(n_pts, NUM_PCL_POINTS, replace=False)
        self.pcl_local = pcds[:, torch.as_tensor(idx, device=pcds.device), :].contiguous().to(self.device)

        self._plans = [None] * self.num_envs  # env별 (T_i, 12) 텐서
        self._step_idx = np.zeros(self.num_envs, dtype=np.int64)

    def _compute_policy_obs(self, env_ids_t):
        palm_pose = self.task.rigid_body_states[env_ids_t, self.palm_idx, 0:7]
        obj_pose = self.task.get_object_state()[env_ids_t]
        pcl = self.pcl_local[env_ids_t].reshape(env_ids_t.numel(), -1)
        obs = torch.cat([palm_pose, obj_pose, pcl], dim=-1)
        assert obs.shape[-1] == OBS_DIM, (obs.shape, OBS_DIM)
        return obs

    def reset_history(self, env_ids=None):
        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_t = env_ids.to(self.device).long().reshape(-1)
        if env_ids_t.numel() == 0:
            return

        obs = self._compute_policy_obs(env_ids_t)
        states = torch.zeros(env_ids_t.numel(), 0, device=self.device)
        with torch.no_grad():
            actions = self.actor_critic(obs, states, inference=True)

        obj_state = self.task.get_object_state()[env_ids_t].detach().cpu().numpy()
        start_wrist6 = self.task.o6_hand_dof_pos[env_ids_t, 0:6].detach().cpu().numpy()
        acts_np = actions.detach().cpu().numpy()

        plan = build_full_plan(
            self.demo_ref,
            obj_state[:, 0:3], obj_state[:, 3:7],
            start_wrist6,
            actions=acts_np,
            finger_limits=self.finger_limits,
            settle_steps=self.settle_steps,
            verbose=False,
        )  # (T, len(env_ids), 12)
        print("reset_history: {} env(s), plan_len={} steps (test_env는 i>100에서 강제 종료함 - "
              "plan_len이 101을 넘으면 lift/settle 구간이 잘릴 수 있음)".format(
                  env_ids_t.numel(), plan.shape[0]))
        plan_t = torch.as_tensor(plan, dtype=torch.float32, device=self.device)
        for local_i, env_id in enumerate(env_ids_t.detach().cpu().numpy().tolist()):
            self._plans[env_id] = plan_t[:, local_i, :]
            self._step_idx[env_id] = 0

    def act_inference(self, obs):
        # obs(=test_env가 넘기는 매 스텝 관측)는 쓰지 않는다 - 우리 정책은 reset_history
        # 시점에 이미 자체 관측으로 워프를 계산해뒀고, 여기서는 미리 계산된 궤적을 재생만 한다.
        actions = torch.zeros(self.num_envs, 12, device=self.device)
        for env_id in range(self.num_envs):
            plan = self._plans[env_id]
            if plan is None:
                # reset_history가 아직 한번도 안 불린 env(있어선 안 되지만 방어적으로 0 반환)
                continue
            idx = min(self._step_idx[env_id], plan.shape[0] - 1)
            actions[env_id] = plan[idx]
            self._step_idx[env_id] += 1
        return actions


class LitBCModel:
    """test_env.py의 `model.__class__.__name__ == 'LitBCModel'` 검사를 만족시키기 위해
    이름만 맞춘 클래스. ActionDiffusion.bc.model.policy.lhm_policy.LitBCModel과 상속
    관계는 없다 - 그쪽 체크는 덕타이핑(클래스 이름 문자열 비교)이라 이름만 같으면 된다."""

    def __init__(self, inner_model):
        self.model = inner_model

    def eval(self):
        self.model.actor_critic.eval()
        return self


def build_demograsp_actor_critic(ckpt_path, device):
    train_param = build_train_param("/tmp/demograsp_bc_infer_unused")
    actor_critic = ActorCritic(
        (OBS_DIM,), (0,), (12,),
        train_param.init_noise_std, train_param.policy,
        asymmetric=False, use_pcl=train_param.is_vision,
    )
    state_dict = torch.load(ckpt_path, map_location=device)
    actor_critic.load_state_dict(state_dict)
    actor_critic.to(device)
    actor_critic.eval()
    for p in actor_critic.parameters():
        p.requires_grad_(False)
    return actor_critic


def resolve_eval_objects(mode):
    """DEXGRASP_EVAL_OBJECTS가 있으면 그 목록을 그대로 쓰고, 없으면 bc_env_infer.py의
    get_mode_object_codes(mode)를 그대로 재사용해 DEXGRASP_EVAL_DATA_DIR 경로의 .npy
    전체를 스캔한다 (README가 문서화한 bc_env_infer.py 기본 동작과 동일 - "경로만 주면
    그 안의 모든 파일을 검증"). 호출 시점에 bci.cfg가 이미 채워져 있어야 한다.

    split_o6_dataset_by_object.py/by_distribution.py를 보면 train/valid는 오브젝트
    단위가 아니라 "오브젝트마다의 궤적"을 나눈 것이라, 오브젝트 자체는 양쪽에 다 존재할 수
    있다. 그래서 DEMOGRASP_EXCLUDE_TRAIN_RUN으로 학습 run의 config.json을 주면, 스캔
    결과에서 그 학습에 쓴 오브젝트 코드만 제외한다(궤적이 아니라 오브젝트 단위 제외)."""
    explicit = env_str("DEXGRASP_EVAL_OBJECTS", "")
    if explicit:
        codes = [c.strip() for c in explicit.split(",") if c.strip()]
    else:
        codes = bci.get_mode_object_codes(mode)

    exclude_run = env_str("DEMOGRASP_EXCLUDE_TRAIN_RUN", "")
    if exclude_run:
        import json
        config_path = osp.join(exclude_run, "config.json")
        with open(config_path, "r") as f:
            exclude_set = set(json.load(f).get("object_codes", []))
        before = len(codes)
        codes = [c for c in codes if c not in exclude_set]
        print("exclude_codes: {}개 제외 대상 중 {}개가 후보 풀에서 제거됨 ({} -> {})".format(
            len(exclude_set), before - len(codes), before, len(codes)
        ))

    n = env_int("DEXGRASP_HELD_OUT_N", 0)
    if n > 0:
        codes = codes[:n]
    return codes


def main():
    if osp.realpath(os.getcwd()) != THIS_DEXGRASP_DIR:
        raise RuntimeError("demograsp_bc_infer.py must run from dexgrasp/ directory (cd dexgrasp 후 실행)")

    ckpt_path = env_str("DEMOGRASP_CKPT", "")
    if not ckpt_path:
        raise ValueError("DEMOGRASP_CKPT 환경변수가 필요함 (예: .../runs_demograsp/<run>/model_2000.pt)")
    demo_pkl = env_str("DEMOGRASP_DEMO_PKL", DEFAULT_DEMO_PKL)
    align_rotation = env_bool("DEMOGRASP_ALIGN_ROTATION", False)

    # ---- bc_env_infer.py의 if __name__=='__main__' 블록과 동일한 절차로 전역 상태를 채움 ----
    bci.assert_clean_runtime_paths()
    args = get_args()
    args.task = "o6HandGraspDexRepIjrr"
    args.cfg_env = "cfg/o6_hand_grasp_dexrep_ijrr.yaml"
    cfg, cfg_train, logdir = load_cfg(args)
    agent_index = get_AgentIndex(cfg)

    # yaml 기본값은 o6_policy_obs_mode: "dexrep"(:88). 원래 bc_env_infer.py는 create_bc_model()이
    # ActorCriticDexRep 계열 정책일 때 이 값을 세팅해주는데, 우리는 create_bc_model() 대신
    # build_demograsp_actor_critic()을 쓰므로 아무도 이걸 안 건드려 yaml 기본값이 그대로 남는다.
    # train_demograsp.py:build_env()(검증된 57.27% 결과가 나온 경로)와 동일하게 맞춘다.
    cfg['env']['o6_policy_obs_mode'] = 'prev_action_obj_rot'

    if os.environ.get("DEXGRASP_EVAL_DATA_DIR"):
        cfg['trajs_path']['train'] = os.environ["DEXGRASP_EVAL_DATA_DIR"]
        cfg['trajs_path']['valid'] = os.environ["DEXGRASP_EVAL_DATA_DIR"]

    cfg['env']['obj_type'] = env_str("DEXGRASP_OBJ_TYPE", cfg['env'].get('obj_type', 'unseen'))
    object_code_key = (
        'seen_object_code_dict' if cfg['env']['obj_type'] in ['seen', 'one']
        else 'unseen_object_code_dict'
    )
    cfg['env'].setdefault(object_code_key, 'auto')

    # bc_env_infer 모듈의 전역을 먼저 채운다 - resolve_eval_objects()가 bci.get_mode_object_codes()로
    # bci.cfg(trajs_path/obj_type)를 읽어 디렉터리를 스캔하므로, 이 시점 이전에 채워져 있어야 함.
    # agent_index도 bci.create_env()(내부 parse_task 호출)가 모듈 전역으로 참조하므로 같이 채움
    # (안 채우면 NameError: name 'agent_index' is not defined - bc_env_infer.py:162).
    bci.args, bci.cfg, bci.cfg_train, bci.agent_index = args, cfg, cfg_train, agent_index

    eval_objects = resolve_eval_objects(cfg['env']['obj_type'])
    cfg['env'][object_code_key] = eval_objects

    if os.environ.get("DEXGRASP_INFER_BATCH_SIZE"):
        cfg['env']['infer_batch_size'] = int(os.environ["DEXGRASP_INFER_BATCH_SIZE"])
    if os.environ.get("DEXGRASP_TEST_NUM"):
        cfg['env']['test_num'] = int(os.environ["DEXGRASP_TEST_NUM"])

    device = args.rl_device
    print("loading DemoGrasp checkpoint:", ckpt_path)
    actor_critic = build_demograsp_actor_critic(ckpt_path, device)

    mode = cfg['env']['obj_type']
    obj_id_list = eval_objects
    batch_size = cfg['env']['infer_batch_size']

    bc_info_name = "demograsp_{}_test_num{}".format(
        osp.splitext(osp.basename(ckpt_path))[0], cfg['env']['test_num']
    )
    if os.environ.get("DEXGRASP_RESULT_SUFFIX"):
        bc_info_name += "_" + os.environ["DEXGRASP_RESULT_SUFFIX"]
    cfg['env']['bc_model_name'] = "DemoGraspActorCritic"

    results = {
        'total_succ_rates': [], 'dataset_name': mode, 'detail': [], 'detail_info': [],
        'total_success_num': 0.0, 'total_trials': 0,
    }
    print("demograsp_bc_infer: mode={} n_objects={} batch_size={} demo={} align_rotation={}".format(
        mode, len(obj_id_list), batch_size, osp.basename(demo_pkl), align_rotation
    ))

    for i in range(0, len(obj_id_list), batch_size):
        batch = obj_id_list[i:i + batch_size]
        cfg['env']['object_code_dict'] = batch
        task = None
        env = None
        try:
            task, env = bci.create_env()
            inner = _DemoGraspInnerModel(task, actor_critic, demo_pkl, align_rotation=align_rotation)
            bc_model = LitBCModel(inner)
            succ_rate, result_desc, result_info = bci.test_env(
                args, task, env, bc_model, "DemoGraspActorCritic", batch[-1], None,
            )
            results['total_succ_rates'].append(succ_rate)
            results['detail'].append([result_desc])
            results['detail_info'].append(result_info)
            results['total_success_num'] += result_info['success_num']
            results['total_trials'] += result_info['N_seq']
        finally:
            bci.cleanup_task_env(task, env)
            del task, env
            bci.release_cuda_memory()

    bci.save_results_summary(results, filename=bc_info_name, to_yaml=True)
    print("---------------------finish {}--------------------------".format(bc_info_name))


if __name__ == "__main__":
    main()
