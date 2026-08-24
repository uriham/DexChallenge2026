# DemoGrasp 이식 자료 (8/21 준비)

`/home/user/DemoGrasp`와 왔다갔다하지 않아도 되도록, 필요한 파일만 여기 미리 옮겨뒀다.
전체 계획은 `/home/user/.claude/plans/curried-tickling-dragon.md` 참고.

## ⚠️ 여기 있는 파일 중 그대로 실행되는 건 없다

`DemoGrasp/tasks/grasp.py`를 원본 그대로 이 리포에서 import/실행하는 건 **불가능**하다 (확인됨, 8/21):
- `from isaacgymenvs.tasks.base.vec_task import VecTask` — 이 챌린지 conda env(`DexGraspMotionChallenge2026`)엔 `isaacgymenvs` 패키지 자체가 없음.
- 설사 설치해도 대회 태스크는 `isaacgymenvs.VecTask`가 아니라 자체 `tasks/hand_base/base_task.py::BaseTask`를 상속하므로 클래스 구조가 안 맞음.

그래서 아래 두 폴더는 역할이 다르다.

## `reference/` — 읽기 전용, import 금지

`DemoGrasp/tasks/{grasp.py, utils.py, reward.py}` + `DemoGrasp/run_rl_grasp.py` 원본 그대로 복사한 것. **목적은 오직 "다음 세션이 다른 프로젝트 폴더로 안 건너가고 여기서 grep/Read로 바로 참고하게" 하는 것뿐.** import해서 쓰지 말 것.

**`run_rl_grasp.py`의 `build_runner`(15-58줄)가 "env와 `ppo_onestep.PPO`를 실제로 어떻게 연결하는지" 보여주는 배선도**다. M5에서 이식판 PPO를 우리 env 래퍼에 연결할 때 이 패턴 그대로 따라가면 됨:
```python
# run_rl_grasp.py:34-46
assert env.randomize_tracking_reference      # ← 래퍼가 반드시 가진 속성
act_dim = 6
if env.randomize_grasp_pose:                 # ← 래퍼가 반드시 가진 속성
    act_dim += env.num_active_hand_dofs      # ← 래퍼가 반드시 가진 속성 (O6는 6)

runner = ppo_onestep.PPO(
    vec_env=env,                             # ← 우리 챌린지 env 래퍼가 여기 들어감
    actor_critic_class=ppo_onestep.ActorCritic,
    train_param=train_param,                 # PPOOneStep.yaml 하이퍼파라미터 (참고용 값은 계획서 Context 참고)
    log_dir=log_dir,
    apply_reset=False,
    action_dim=act_dim,
)
```
즉 우리가 만들 env 래퍼는 최소한 `randomize_tracking_reference`, `randomize_grasp_pose`, `num_active_hand_dofs`, `device`, `num_envs`, `reset_idx()`, `generate_reaching_plan_idx()`, `compute_reference_actions()`, `max_episode_length`, `step()`을 갖춰야 이 생성자 호출이 그대로 통한다.

여기서 실제로 새 코드를 짤 때 봐야 할 핵심 함수 (파일 안에서 이름으로 검색):
- `grasp.py` `generate_reaching_plan_idx` — 시연 워핑 + reaching 모션플래닝. **이걸 O6/GraspM3용으로 재구현하는 게 이식의 핵심.**
- `grasp.py` `compute_reference_actions` — 워핑된 궤적을 매 스텝 읽어서 저수준 액션으로 변환.
- `grasp.py` `compute_required_observations` / `transform_obj_pcl_2_world` — 정책 관측(손목pose+물체초기pose+물체pcl world변환) 조립. M5의 "관측 벡터 조립" 항목이 이 부분.
- `utils.py` `batch_linear_interpolate_poses` — reaching phase가 쓰는 보간 함수. 로직만 참고해서 새로 쓸 것(이 파일 자체를 import하지 말고).
- `utils.py` `transform_points` — 쿼터니언으로 점 회전시키는 3줄짜리 헬퍼. 물체 포인트클라우드를 world로 옮길 때 필요.
- `reward.py` `reward_binary`의 51-58번째 줄 근처(`has_hit_table` 계산부) — 손이 테이블에 닿으면 보상을 0으로 만드는 안전장치. 대회 쪽 `compute_hand_reward_rl`에 비슷한 게 있는지 먼저 확인하고, 없으면 이 로직만 참고해서 추가할지 판단(필수 아님, M5 참고사항).

## `ppo_onestep/` — 실사용 대상, 단 2가지 수정 필요

`DemoGrasp/algo/ppo_onestep/{__init__,module,ppo,storage}.py` 그대로 복사. 이건 환경과 완전히 분리된 순수 학습기 코드라 대부분 그대로 재사용 가능.

**수정 필요 ①: import 경로 (2줄, 기계적 수정)**
```
module.py:10, ppo.py:20
from isaacgymenvs.utils.torch_jit_utils import *
```
→ 대회 리포 자체의 `utils/torch_jit_utils.py`로 교체 (같은 함수를 `isaacgym.torch_utils`에서 그대로 재노출하고 있어서 함수 목록은 동일함, 확인됨).

**수정 필요 ②: PointNet 백본 (설계 판단 필요, 아직 미해결)**

`module.py`가 물체 포인트클라우드 인코더로 `from ..pn_utils.maniskill_learn.networks.backbones.pointnet import getPointNet`를 쓰는데, 이 함수는 `maniskill_learn`이라는 벤더링된 미니 프레임워크(문자열 기반 레지스트리 시스템, `Registry`/`build_from_cfg`)에 걸쳐 있어서 `pointnet.py` 파일 하나만 떼어올 수 없다(builder.py, utils/meta/*, utils/data/*, modules/activation.py 등 10여 개 파일 필요).

**권장: 통째로 옮기지 말고, 아래 스펙대로 작은 표준 PointNet을 새로 짤 것** (`getPointNet()`이 실제로 만드는 네트워크의 정확한 구조, `DemoGrasp/algo/pn_utils/maniskill_learn/networks/backbones/pointnet.py:368-403` + `SimplePointNetV0.forward_raw` 확인 완료):

```
입력: (B, 512, 3) 물체 표면점

1. subtract_mean_coords 트릭:
   mean_xyz = 전체 점의 평균 (B, 1, 3)
   각 점 = [mean_xyz(3), xyz - mean_xyz(3)]  →  (B, 512, 6)

2. per-point MLP (ConvMLP, mlp_spec=[6, 128, 256]):
   Conv1d(6→128) → 활성화 → Conv1d(128→256) → 활성화
   (또는 Linear를 점마다 동일하게 적용해도 동치)
   출력: (B, 512, 256)

3. max_mean_mix_aggregation 트릭 (256를 반으로 나눠서):
   앞 128채널 → 점 축으로 max-pool  → (B, 128)
   뒤 128채널 → 점 축으로 mean-pool → (B, 128)
   concat → (B, 256)

4. global MLP (LinearMLP, mlp_spec=[256, 256, 128]):
   Linear(256→256) → 활성화 → Linear(256→128)
   출력: (B, 128)  ← 이게 module.py의 pc_emb_dim=128과 일치
```

이렇게 하면 `module.py`의 `PointNetBackbone` 클래스(32-53번째 줄)에서 `self.backbone = getPointNet(...)` 한 줄만 이 새 클래스로 바꾸면 되고, `forward` 인터페이스(`(B,512,3) 입력 → (B,128) 출력`)만 맞으면 나머지 `ActorCritic` 코드는 무수정으로 동작한다.

## 여기 없는 것 — 일부러 안 옮김

| 안 옮긴 것 | 이유 |
|---|---|
| `assets/` (URDF·메시·포인트클라우드, 수 GB) | 대회 쪽엔 다른 손(O6)·다른 물체(GraspM3)를 쓰므로 무관 |
| `ckpt/*.pt` (학습된 체크포인트 7개) | **로드 금지** — Rules상 대회 데이터셋 외 학습 데이터 사용은 실격. 손도 다름(O6 아님) |
| `tasks/hand/*.yaml` (손 설명서 7개) | O6는 이미 대회 태스크에 다 기술돼 있음, 새로 안 만듦 |
| `tasks/grasp_ref_*.pkl` (다른 손 6개 시연) | O6 전용 시연은 `dexgrasp/demos/`에 이미 새로 만드는 중 |
| `data/lerobot`, `docs/` | RL 학습과 무관 |
