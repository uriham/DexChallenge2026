"""
DemoGrasp one-step PPO <-> 대회 O6 태스크 어댑터.

demograsp_port/ppo_onestep/ppo.py 의 학습 루프가 요구하는 인터페이스만 제공하는 얇은 래퍼.
(reference/run_rl_grasp.py 의 build_runner가 env에 기대하는 것과 동일한 계약)

PPO 루프가 실제로 부르는 것 (ppo.py:164-183):
    obs = vec_env.reset_idx(arange(num_envs))["obs"]
    states = vec_env.get_state()
    actions, ... = actor_critic(obs, states)
    vec_env.generate_reaching_plan_idx(arange(num_envs), actions=actions)
    for t in range(vec_env.max_episode_length):
        env_action = vec_env.compute_reference_actions()
        obs, reward, reset, extras = vec_env.step(env_action)
        if t == vec_env.max_episode_length - 2:
            rews = vec_env.successes ; rews[vec_env.has_hit_table] = 0 ; break

--------------------------------------------------------------------------
설계 근거 (8/21 코드 조사로 확인)

- **액션 적용**: cfg의 env_mode='extract_obs'면 pre_physics_step(:2048,2054)이
  cur_targets에 전체 DOF를 그대로 반영한다. 새 env_mode를 만들 필요 없음.
- **보상/종료**: PPO는 step()의 reward를 안 쓰고 task.successes를 직접 읽으며,
  종료도 자체 for문으로 관리한다. 따라서 태스크의 reward/done 배선을 고칠 필요 없음.
  successes는 compute_hand_reward(:2141-2150)에서 이미 올바르게 계산되고
  (0.3m 리프트 or goal_dist<=0.12), reset()에서 per-env 0으로 초기화된다.
- **전체 리셋**: one-step PPO는 매 iteration 전체 env를 동시에 리셋하므로
  pre_physics_step의 "액션 전체 교체"(:2039-2041)가 문제되지 않는다.
- **VecTaskPython.step()/reset() 우회**: obs가 21D면 None을 반환하는 버그가 있어
  (vec_task.py:131-134의 else가 주석 처리됨) task.step()을 직접 부른다.
- **초기 손 pose**: task.reset()(:1941)이 손을 각 env의 grasp_seqs[:,0,:]로 초기화하므로,
  env마다 다른 궤적을 넣으면 초기 pose 랜덤화가 자동으로 얻어진다(대회 규정 요구사항).
--------------------------------------------------------------------------
"""
import pickle

import numpy as np
import torch
from gym import spaces

from demograsp_warp import DemoReference, build_full_plan

# 관측 구성: 팜 pose(7) + 물체 초기 pose(7) + 물체 점군(512*3)
# 점군은 반드시 맨 뒤여야 한다 (module.py:132 pc_start_idx = num_obs - prod(pc_shape))
NUM_PCL_POINTS = 512
OBS_STATE_DIM = 14
OBS_DIM = OBS_STATE_DIM + NUM_PCL_POINTS * 3  # 1550


class DemoGraspO6Env:
    def __init__(self, task, env, demo_pkl, action_dim=12, settle_steps=10,
                 pcl_seed=0, enable_hit_table_penalty=False):
        self.task = task
        self.env = env
        self.device = task.device
        self.num_envs = int(task.num_envs)
        self.action_dim = int(action_dim)
        self.settle_steps = int(settle_steps)
        self.enable_hit_table_penalty = bool(enable_hit_table_penalty)

        self.num_states = 0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

        self.palm_idx = task.hand_body_idx_dict["palm"]
        self.finger_limits = (
            task.o6_hand_dof_lower_limits[task.actuated_dof_indices].cpu().numpy(),
            task.o6_hand_dof_upper_limits[task.actuated_dof_indices].cpu().numpy(),
        )

        # palm_pos_world - virtual_xyz = 상수 c. 하드코딩 대신 런타임 측정.
        palm_pos = task.rigid_body_states[:, self.palm_idx, 0:3].detach().cpu().numpy()
        virtual_xyz = task.o6_hand_dof_pos[:, 0:3].detach().cpu().numpy()
        self.hand_root_offset = (palm_pos - virtual_xyz).mean(axis=0)

        with open(demo_pkl, "rb") as f:
            demo = pickle.load(f)
        if not demo.get("success", False):
            print("경고: 이 시연은 success=False로 기록되어 있음 -> {}".format(demo_pkl))
        self.demo_ref = DemoReference(demo, self.hand_root_offset)
        print("demo loaded: object={} T={} t_lift={} hand_root_offset={}".format(
            demo["object_code"], self.demo_ref.T, self.demo_ref.t_lift,
            np.round(self.hand_root_offset, 5),
        ))

        # 물체 점군: obj_init_obj_pcds는 (B,N,3)로 물체 회전·스케일이 이미 반영된
        # 물체 로컬(중심) 좌표. 우리 경로에선 N=2048이라 512개를 **고정 인덱스로** 뽑는다
        # (스텝마다 재추출하면 관측이 흔들림).
        pcds = task.obj_init_obj_pcds  # (B, N, 3)
        n_pts = pcds.shape[1]
        if n_pts < NUM_PCL_POINTS:
            raise ValueError("점군 개수 {} < {}".format(n_pts, NUM_PCL_POINTS))
        rng = np.random.default_rng(pcl_seed)
        idx = rng.choice(n_pts, NUM_PCL_POINTS, replace=False)
        self.pcl_local = pcds[:, torch.as_tensor(idx, device=pcds.device), :].contiguous().to(self.device)
        print("obj point cloud: {} -> {} points, local frame (회전/스케일 반영됨)".format(n_pts, NUM_PCL_POINTS))

        self._plan = None
        self._step_count = 0
        self._last_obs = torch.zeros(self.num_envs, OBS_DIM, device=self.device)
        self.obs_dict = {"obs": self._last_obs}

    # ------------------------------------------------------------------ #
    # PPO가 참조하는 속성들
    # ------------------------------------------------------------------ #
    @property
    def max_episode_length(self):
        """플랜 길이(리치 거리에 따라 iteration마다 달라짐). PPO는 이 값을 매 rollout
        시작 시 읽으므로 동적이어도 안전하다(ppo.py:173)."""
        return int(self._plan.shape[0]) if self._plan is not None else 2

    @property
    def successes(self):
        return self.task.successes

    @property
    def has_hit_table(self):
        if not self.enable_hit_table_penalty:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        palm_z = self.task.rigid_body_states[:, self.palm_idx, 2]
        return palm_z < float(self.task.table_height)

    def get_state(self):
        return torch.zeros(self.num_envs, 0, device=self.device)

    # ------------------------------------------------------------------ #
    def _compute_obs(self):
        palm_pose = self.task.rigid_body_states[:, self.palm_idx, 0:7]  # (N,7)
        obj_pose = self.task.get_object_state()  # (N,7)
        pcl = self.pcl_local.reshape(self.num_envs, -1)  # (N, 512*3)
        return torch.cat([palm_pose, obj_pose, pcl], dim=-1)

    def reset_idx(self, env_ids=None):
        """one-step PPO는 항상 전체 env를 리셋한다. 부분 리셋은 지원하지 않는다
        (태스크의 pre_physics_step이 리셋 시 액션 배치를 통째로 덮어쓰기 때문)."""
        if env_ids is not None and len(env_ids) != self.num_envs:
            raise NotImplementedError("부분 리셋 미지원 - one-step PPO는 전체 리셋만 사용")

        self.task.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.task.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.env.reset()  # 내부에서 task.step(zeros, id=-1) -> pre_physics_step이 리셋 수행

        self._plan = None
        self._step_count = 0
        self._last_obs = self._compute_obs()
        self.obs_dict = {"obs": self._last_obs}
        return self.obs_dict

    def generate_reaching_plan_idx(self, env_ids=None, actions=None):
        """정책 액션으로 시연을 워프하고, 현재 손 위치에서 시연 시작점까지의 reach 구간을
        붙여 에피소드 전체 궤적을 미리 계산한다. 에피소드당 1회만 호출된다."""
        obj_state = self.task.get_object_state().detach().cpu().numpy()  # (N,7)
        start_wrist6 = self.task.o6_hand_dof_pos[:, 0:6].detach().cpu().numpy()  # (N,6)
        acts = None if actions is None else actions.detach().cpu().numpy()

        plan = build_full_plan(
            self.demo_ref,
            obj_state[:, 0:3], obj_state[:, 3:7],
            start_wrist6,
            actions=acts,
            finger_limits=self.finger_limits,
            settle_steps=self.settle_steps,
            verbose=False,
        )
        self._plan = torch.as_tensor(plan, dtype=torch.float32, device=self.device)
        self._step_count = 0

    def compute_reference_actions(self):
        """미리 계산된 플랜을 진행도로 인덱싱만 한다 (정책 재호출 없음).
        플랜 끝을 넘어가면 마지막 자세를 유지."""
        if self._plan is None:
            raise RuntimeError("generate_reaching_plan_idx()를 먼저 호출해야 함")
        idx = min(self._step_count + 1, self._plan.shape[0] - 1)  # index 0은 더미
        return self._plan[idx]

    def step(self, action):
        self.task.step(action, self._step_count + 1)
        self._step_count += 1
        # one-step PPO는 스텝별 obs를 쓰지 않는다(정책은 리셋 직후 1회만 호출됨).
        # 매 스텝 점군까지 다시 이어붙이는 건 낭비라 캐시를 반환한다.
        return self._last_obs, self.task.rew_buf, self.task.reset_buf, self.task.extras
