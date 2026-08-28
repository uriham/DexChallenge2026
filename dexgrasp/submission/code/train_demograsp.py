"""
DemoGrasp one-step PPO 학습 엔트리포인트 (O6/GraspM3 판).

reference/run_rl_grasp.py 의 build_runner(:15-58) 구조를 따르되 hydra/isaacgymenvs 의존을 걷어내고,
대회 리포의 task 생성 경로(utils.config + utils.parse_task)를 쓴다.

사용법 (dexgrasp/ 에서):
    # 1) 무워프 동치성 확인 (정책 없이 시연 재배치만 재생 - G3와 같은 결과가 나와야 정상)
    DEMOGRASP_DEBUG=replay DEMOGRASP_NUM_OBJECTS=8 DEMOGRASP_TRAJ_PER_OBJECT=1 \
        python train_demograsp.py --headless

    # 2) 스모크런 (3 iteration만 돌려 크래시 여부 확인)
    DEMOGRASP_NUM_OBJECTS=4 DEMOGRASP_TRAJ_PER_OBJECT=2 DEMOGRASP_MAX_ITERATIONS=3 \
        python train_demograsp.py --headless

    # 3) 본 학습
    DEMOGRASP_NUM_OBJECTS=64 DEMOGRASP_TRAJ_PER_OBJECT=4 DEMOGRASP_MAX_ITERATIONS=2000 \
        python train_demograsp.py --headless

    # 4) 전체 오브젝트 평가 (청크 단위 순회, OOM 방지, demograsp_bc_infer.py 경유 없음)
    DEMOGRASP_MODE=eval_all DEMOGRASP_CKPT=/data/.../model_2000.pt \
    DEMOGRASP_EXCLUDE_TRAIN_RUN=/data/.../config.json \
    DEMOGRASP_CHUNK_SIZE=1500 DEXGRASP_RESULTS_PATH=./results/eval_all.json \
        python train_demograsp.py --headless

환경변수:
    DEMOGRASP_DEBUG            replay | (미설정=학습)
    DEMOGRASP_MODE             eval | eval_all | (미설정=학습)
    DEMOGRASP_NUM_OBJECTS      학습에 쓸 오브젝트 수 (기본 64, eval_all에선 미사용)
    DEMOGRASP_TRAJ_PER_OBJECT  오브젝트당 궤적 수 (기본 4, eval_all 기본은 1). num_envs = 둘의 곱
    DEMOGRASP_CHUNK_SIZE       eval_all 전용 - 청크당 오브젝트 수 (기본 1500)
    DEMOGRASP_EVAL_ALL_LIMIT   eval_all 전용 - 앞에서부터 N개 오브젝트로만 소규모 테스트 (기본 0=제한없음)
    DEMOGRASP_ROUNDS_PER_CHUNK eval_all 전용 - 청크당 반복 라운드 수 (기본 1)
    DEXGRASP_RESULTS_PATH      eval_all 전용 - 결과 JSON 저장 경로
    DEMOGRASP_MAX_ITERATIONS   PPO iteration 수 (기본 2000)
    DEMOGRASP_RUN_NAME         런 이름 (기본: 타임스탬프)
    DEMOGRASP_CKPT             이어서 학습할(또는 eval/eval_all에서 평가할) 체크포인트 경로
    DEMOGRASP_DEMO_PKL         시연 pkl 경로
    DEMOGRASP_LOG_ROOT         로그/체크포인트 루트 (기본 /data/... , 홈 디스크 여유 없음)
"""
import glob
import json
import os
import os.path as osp
from datetime import datetime

import numpy as np

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from utils.process_marl import get_AgentIndex

import torch
from omegaconf import OmegaConf

from demograsp_env import DemoGraspO6Env
from demograsp_port.ppo_onestep import PPO, ActorCritic

THIS_DEXGRASP_DIR = osp.realpath(osp.dirname(__file__))

SOURCE_PREPROC_DIR = os.environ.get(
    "DEMOGRASP_SOURCE_DIR", "/data/DexGraspMotionChallenge2026/dataset_shinwoo_preproc/train"
)
DEFAULT_DEMO_PKL = osp.join(THIS_DEXGRASP_DIR, "demos", "core-bottle-be16ada66829940a451786f3cbfd6769_traj0.pkl")
DEFAULT_LOG_ROOT = "/data/DexGraspMotionChallenge2026/runs_demograsp"


def env_int(name, default):
    return int(os.environ.get(name, default))


def env_str(name, default):
    return os.environ.get(name, default)


def assert_clean_runtime_paths():
    if osp.realpath(os.getcwd()) != THIS_DEXGRASP_DIR:
        raise RuntimeError("train_demograsp.py must run from dexgrasp/ directory (cd dexgrasp 후 실행)")


def configure_o6_args(args):
    args.task = "o6HandGraspDexRepIjrr"
    args.cfg_env = "cfg/o6_hand_grasp_dexrep_ijrr.yaml"
    return args


def build_train_param(log_root):
    """DemoGrasp/tasks/train/PPOOneStep.yaml 을 그대로 옮기되 2가지만 변경:
    - log_dir: 홈 디스크 여유가 없어 /data 로
    - is_vision: True (원본 False). False면 ActorCritic이 점군을 슬라이스하지 않아
      관측이 팜/물체 pose 14차원뿐이 되고 물체 형상 정보가 정책에 전혀 안 들어간다.
      논문의 state-based 정책도 PointNet(512x3)->128을 쓴다.
    """
    return OmegaConf.create({
        "name": "ppo_onestep",
        "log_dir": log_root,
        "is_vision": True,
        "policy": {
            "backbone_type": "pn",
            "freeze_backbone": False,  # backbone 내부는 이미 freeze, proj만 학습됨
            "pi_hid_sizes": [1024, 1024, 512, 512],
            "vf_hid_sizes": [1024, 1024, 512, 512],
            "activation": "elu",
            "pc_shape": [512, 3],
            "pc_emb_dim": 128,
        },
        "test": False,
        "resume": 0,
        "save_interval": 100,
        "print_log": True,
        "max_iterations": env_int("DEMOGRASP_MAX_ITERATIONS", 2000),
        "cliprange": 0.2,
        "ent_coef": 0,
        "nsteps": 1,          # one-step MDP
        "noptepochs": 5,
        "nminibatches": 4,
        "max_grad_norm": 1,
        "optim_stepsize": 3.0e-4,
        "schedule": "adaptive",
        "desired_kl": 0.016,
        "gamma": 0.96,
        "lam": 0.95,
        "init_noise_std": 0.8,
        "surrogate_loss_coef": 1.0,
        "value_loss_coef": 2.0,
        "discard_invalid_resets": False,
    })


def sample_training_objects(num_objects, traj_per_object, seed=0, object_codes=None, exclude_codes=None):
    """전처리된 train 셋에서 오브젝트를 샘플링해 npy_list를 만든다.
    env i = npy_list를 펼친 순서. num_envs = sum(각 오브젝트의 궤적 수).
    오브젝트마다 궤적 시작 프레임이 다르므로 초기 손 pose 다양성이 자동으로 확보된다.

    object_codes가 주어지면 랜덤 샘플링 대신 그 목록을 그대로 씀 (순서 보존) —
    demo_replay_batch.py(G3)와 정확히 같은 오브젝트로 A/B 비교할 때 오브젝트 샘플링
    자체가 변수가 되는 걸 막기 위함 (8/25 무워프 동치성 확인 중 발견).

    exclude_codes가 주어지면 그 오브젝트들은 후보에서 아예 제외 - 학습에 쓴 오브젝트를
    빼고 순수 미사용 오브젝트로만 held-out 평가할 때 씀 (8/26)."""
    if object_codes:
        files = [osp.join(SOURCE_PREPROC_DIR, code + ".npy") for code in object_codes]
        missing = [f for f in files if not osp.exists(f)]
        if missing:
            raise FileNotFoundError("다음 오브젝트 파일을 못 찾음: {}".format(missing))
    else:
        files = sorted(glob.glob(osp.join(SOURCE_PREPROC_DIR, "*.npy")))
        if not files:
            raise FileNotFoundError("전처리 데이터를 못 찾음: {}".format(SOURCE_PREPROC_DIR))
        if exclude_codes:
            exclude_set = set(exclude_codes)
            before = len(files)
            files = [f for f in files if osp.basename(f)[:-4] not in exclude_set]
            print("exclude_codes: {}개 제외 대상 중 {}개가 후보 풀에서 제거됨 ({} -> {})".format(
                len(exclude_set), before - len(files), before, len(files)
            ))
        rng = np.random.default_rng(seed)
        rng.shuffle(files)

    npy_list = []
    n_envs = 0
    for path in files:
        if len(npy_list) >= num_objects:
            break
        data = np.load(path, allow_pickle=True).item()
        n_avail = data["grasp_seqs"].shape[0]
        if n_avail < 1:
            continue
        k = min(traj_per_object, n_avail)
        npy_list.append({
            "grasp_seqs": data["grasp_seqs"][:k],
            "obj_rotmat": data["obj_rotmat"][:k],
            "obj_scale": data["obj_scale"][:k],
            "obj_code_idx": data.get("obj_code_idx", 0),
            "obj_code": osp.basename(path)[:-4],
        })
        n_envs += k

    print("training objects: {} (요청 {}), num_envs = {}".format(len(npy_list), num_objects, n_envs))
    return npy_list


def build_env(args, npy_list):
    cfg, cfg_train, _ = load_cfg(args)
    # DexRep 관측 경로를 끈다 -> compute_observations가 21D 경량 경로로 분기(:1698)하고
    # numObservations도 자동으로 21이 된다(:144). 정책 관측은 우리가 따로 조립하므로 무관.
    cfg["env"]["o6_policy_obs_mode"] = "prev_action_obj_rot"
    cfg["env"]["observationType"] = "DexRep"
    # env_mode는 extract_obs 유지 - pre_physics_step(:2054)이 액션을 전체 DOF에 반영하는 경로
    cfg["env"]["env_mode"] = "extract_obs"
    cfg["env"]["seq_start_pos_uniform"] = False
    cfg["env"]["seq_start_rot_uniform"] = False

    # bc_env_infer.py의 DEXGRASP_EVAL_ASSET_DIR와 동일한 오버라이드 - 채점 측 private set이
    # 궤적 npy(SOURCE_PREPROC_DIR)뿐 아니라 별도 오브젝트 메쉬 경로를 함께 줄 수 있으므로 재사용.
    if os.environ.get("DEXGRASP_EVAL_ASSET_DIR"):
        asset_dir = os.environ["DEXGRASP_EVAL_ASSET_DIR"].strip()
        if not asset_dir.startswith("/"):
            asset_dir = "/" + asset_dir
        if not asset_dir.endswith("/"):
            asset_dir = asset_dir + "/"
        cfg["env"]["asset"]["assetFileNameObj"] = asset_dir
        cfg["env"]["asset"]["assetFileNameObj_raw"] = asset_dir

    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    agent_index = get_AgentIndex(cfg)

    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index, npy_list=npy_list)
    print("task.num_envs =", task.num_envs)
    return task, env


def debug_replay(vec_env):
    """무워프 동치성 확인: 정책 없이(액션=None) 시연 재배치만 재생.
    demo_replay_batch.py(G3)와 같은 성공률이 나와야 워프/관측 조립이 정상이라는 뜻.
    reference/run_rl_grasp.py:test_demo_replay(:85-94)에 대응."""
    print("\n=== debug: 무워프 재생 (액션 없음) ===")
    vec_env.reset_idx(torch.arange(vec_env.num_envs))
    vec_env.generate_reaching_plan_idx(torch.arange(vec_env.num_envs), actions=None)
    print("episode length =", vec_env.max_episode_length)
    for _ in range(vec_env.max_episode_length - 1):
        vec_env.step(vec_env.compute_reference_actions())
    succ = vec_env.successes.detach().cpu().numpy()
    print("성공: {}/{} ({:.1f}%)".format(int(succ.sum()), len(succ), 100.0 * succ.mean()))
    return succ


def run_eval_all(args, exclude_codes):
    """SOURCE_PREPROC_DIR의 오브젝트 전체를 청크 단위로 순회하며 평가한다.

    demograsp_bc_infer.py(bc_env_infer.py 경유) 경로가 원인 불명 버그로 같은 체크포인트/
    같은 오브젝트에서도 성공률을 크게 낮게 보고하는 게 확인돼서(8/27, 200개 오브젝트로
    직접 대조: train_demograsp.py 52.81% vs demograsp_bc_infer.py 10.5%), 새 어댑터를
    더 파는 대신 이미 검증된 이 파일의 단일-배치 로직(reset_idx -> generate_reaching_plan_idx
    -> step 루프, ppo_onestep.PPO.run()의 is_testing 분기와 동일)을 청크마다 새 task/env로
    반복 호출한다. 청크가 끝날 때마다 task.clean_sim()으로 완전히 해제하므로 오브젝트
    총량과 무관하게 한 번에 필요한 VRAM은 청크 크기만큼으로 고정된다(OOM 방지).

    청크 크기 기본값 1500은 지난 VRAM 스케일링 실측(env당 ~10.4MiB, 고정비 ~2.3GB, 안전
    상한 ~1,800envs)에 근거함."""
    ckpt = env_str("DEMOGRASP_CKPT", "")
    if not ckpt:
        raise ValueError("DEMOGRASP_MODE=eval_all 이려면 DEMOGRASP_CKPT=<model_XXXX.pt 경로>가 필요함")

    files = sorted(glob.glob(osp.join(SOURCE_PREPROC_DIR, "*.npy")))
    if not files:
        raise FileNotFoundError("전처리 데이터를 못 찾음: {}".format(SOURCE_PREPROC_DIR))
    if exclude_codes:
        exclude_set = set(exclude_codes)
        before = len(files)
        files = [f for f in files if osp.basename(f)[:-4] not in exclude_set]
        print("exclude_codes: {}개 제외 대상 중 {}개가 후보 풀에서 제거됨 ({} -> {})".format(
            len(exclude_set), before - len(files), before, len(files)
        ))
    codes_all = [osp.basename(f)[:-4] for f in files]
    limit = env_int("DEMOGRASP_EVAL_ALL_LIMIT", 0)
    if limit > 0:
        codes_all = codes_all[:limit]
        print("DEMOGRASP_EVAL_ALL_LIMIT={} - 소규모 테스트용으로 앞에서부터 {}개만 사용".format(limit, len(codes_all)))

    chunk_size = env_int("DEMOGRASP_CHUNK_SIZE", 1500)
    traj_per_object = env_int("DEMOGRASP_TRAJ_PER_OBJECT", 1)
    rounds_per_chunk = env_int("DEMOGRASP_ROUNDS_PER_CHUNK", 1)
    demo_pkl = env_str("DEMOGRASP_DEMO_PKL", DEFAULT_DEMO_PKL)
    train_param = build_train_param(env_str("DEMOGRASP_LOG_ROOT", DEFAULT_LOG_ROOT))
    train_param.test = True

    total_success = 0.0
    total_trials = 0
    chunk_results = []
    num_chunks = (len(codes_all) + chunk_size - 1) // chunk_size
    print("run_eval_all: {}개 오브젝트, 청크 크기 {}, {}개 청크".format(len(codes_all), chunk_size, num_chunks))

    for chunk_id, start in enumerate(range(0, len(codes_all), chunk_size)):
        chunk_codes = codes_all[start:start + chunk_size]
        npy_list = sample_training_objects(
            num_objects=len(chunk_codes), traj_per_object=traj_per_object, object_codes=chunk_codes,
        )
        task, env = build_env(args, npy_list)
        vec_env = DemoGraspO6Env(task=task, env=env, demo_pkl=demo_pkl, action_dim=12)
        runner = PPO(
            vec_env=vec_env, actor_critic_class=ActorCritic,
            train_param=train_param, log_dir=None, apply_reset=False, action_dim=12,
        )
        runner.test(ckpt)

        chunk_success = 0.0
        chunk_trials = 0
        for _ in range(rounds_per_chunk):
            current_obs = vec_env.reset_idx(torch.arange(vec_env.num_envs))["obs"]
            current_states = vec_env.get_state()
            actions = runner.actor_critic(current_obs, current_states, inference=True)
            vec_env.generate_reaching_plan_idx(torch.arange(vec_env.num_envs), actions=actions)
            for t in range(vec_env.max_episode_length):
                env_action = vec_env.compute_reference_actions()
                vec_env.step(env_action)
                if t == vec_env.max_episode_length - 2:
                    succ = vec_env.successes.clone().cpu().numpy()
                    break
            chunk_success += float(succ.sum())
            chunk_trials += len(succ)

        total_success += chunk_success
        total_trials += chunk_trials
        chunk_rate = chunk_success / max(chunk_trials, 1)
        chunk_results.append({
            "chunk_id": chunk_id, "num_objects": len(chunk_codes),
            "success": chunk_success, "trials": chunk_trials, "success_rate": chunk_rate,
        })
        print("chunk {}/{}: {}개 오브젝트, success_rate={:.4f} | 누적 {}/{} = {:.4f}".format(
            chunk_id + 1, num_chunks, len(chunk_codes), chunk_rate,
            int(total_success), total_trials, total_success / max(total_trials, 1),
        ))

        task.clean_sim()
        del task, env, vec_env, runner
        torch.cuda.empty_cache()

    final_rate = total_success / max(total_trials, 1)
    print("=== run_eval_all 완료: {}/{} = {:.4f} ({}개 오브젝트, {}개 청크) ===".format(
        int(total_success), total_trials, final_rate, len(codes_all), num_chunks
    ))

    results_path = env_str("DEXGRASP_RESULTS_PATH", "")
    if results_path:
        os.makedirs(osp.dirname(results_path) or ".", exist_ok=True)
        with open(results_path, "w") as f:
            json.dump({
                "total_success": total_success, "total_trials": total_trials,
                "success_rate": final_rate, "num_objects": len(codes_all),
                "chunk_size": chunk_size, "chunks": chunk_results,
            }, f, indent=2)
        print("results saved to:", results_path)


def main():
    set_np_formatting()
    args = configure_o6_args(get_args())
    args.seed = 0
    args.rl_device = "cuda:0"
    args.sim_device = "cuda:0"

    object_codes_raw = env_str("DEMOGRASP_OBJECT_CODES", "")
    object_codes = [c.strip() for c in object_codes_raw.split(",") if c.strip()] or None

    # 학습에 쓴 오브젝트를 제외하고 싶으면, 그 학습 런의 config.json(main()이 이미 저장해둠,
    # object_codes 키)에서 목록을 읽어와서 후보 풀에서 뺀다.
    exclude_run_config = env_str("DEMOGRASP_EXCLUDE_TRAIN_RUN", "")
    exclude_codes = None
    if exclude_run_config:
        with open(exclude_run_config) as f:
            exclude_codes = json.load(f)["object_codes"]
        print("제외할 학습 런: {} ({}개 오브젝트)".format(exclude_run_config, len(exclude_codes)))

    if env_str("DEMOGRASP_MODE", "") == "eval_all":
        run_eval_all(args, exclude_codes)
        return

    npy_list = sample_training_objects(
        num_objects=env_int("DEMOGRASP_NUM_OBJECTS", 64),
        traj_per_object=env_int("DEMOGRASP_TRAJ_PER_OBJECT", 4),
        object_codes=object_codes,
        exclude_codes=exclude_codes,
    )
    task, env = build_env(args, npy_list)

    vec_env = DemoGraspO6Env(
        task=task,
        env=env,
        demo_pkl=env_str("DEMOGRASP_DEMO_PKL", DEFAULT_DEMO_PKL),
        action_dim=12,  # run_rl_grasp.py:35-37과 동일: 6(wrist) + num_active_hand_dofs(6)
    )

    if env_str("DEMOGRASP_DEBUG", "") == "replay":
        debug_replay(vec_env)
        task.clean_sim()
        return

    if env_str("DEMOGRASP_MODE", "") == "eval":
        # 원본 run_rl_grasp.py의 test=True 경로 재사용 (PPO.run()에 이미 구현되어 있음,
        # inference=True 결정론적 정책으로 10라운드 성공률 출력 후 종료).
        ckpt = env_str("DEMOGRASP_CKPT", "")
        if not ckpt:
            raise ValueError("DEMOGRASP_MODE=eval 이려면 DEMOGRASP_CKPT=<model_XXXX.pt 경로>가 필요함")
        train_param = build_train_param(env_str("DEMOGRASP_LOG_ROOT", DEFAULT_LOG_ROOT))
        train_param.test = True
        runner = PPO(
            vec_env=vec_env, actor_critic_class=ActorCritic,
            train_param=train_param, log_dir=None, apply_reset=False, action_dim=12,
        )
        runner.test(ckpt)  # load_state_dict + eval()
        runner.run()  # is_testing=True 분기: 10라운드 성공률 출력 후 exit(0)
        return

    log_root = env_str("DEMOGRASP_LOG_ROOT", DEFAULT_LOG_ROOT)
    run_name = env_str("DEMOGRASP_RUN_NAME", "o6_demograsp_{}".format(
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    log_dir = osp.join(log_root, run_name)
    os.makedirs(log_dir, exist_ok=True)

    train_param = build_train_param(log_root)
    with open(osp.join(log_dir, "config.json"), "w") as f:
        json.dump({
            "train_param": OmegaConf.to_container(train_param),
            "num_envs": int(task.num_envs),
            "num_objects": len(npy_list),
            "object_codes": [d["obj_code"] for d in npy_list],
            "demo_pkl": env_str("DEMOGRASP_DEMO_PKL", DEFAULT_DEMO_PKL),
        }, f, indent=2)
    print("log_dir:", log_dir)

    runner = PPO(
        vec_env=vec_env,
        actor_critic_class=ActorCritic,
        train_param=train_param,
        log_dir=log_dir,
        apply_reset=False,
        action_dim=12,
    )

    ckpt = env_str("DEMOGRASP_CKPT", "")
    if ckpt:
        print("loading checkpoint:", ckpt)
        runner.load(ckpt)

    runner.run()
    task.clean_sim()


if __name__ == "__main__":
    assert_clean_runtime_paths()
    main()
