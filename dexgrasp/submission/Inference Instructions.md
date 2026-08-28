# Inference Instructions for Model_UVLL_HanDex

This document outlines the required steps to evaluate and run inference using
the submitted `Model_UVLL_HanDex.ckpt` weights.

## 1. Environment Setup

Before running inference, please ensure your environment matches the
configuration detailed in the provided `RUNTIME_VERSIONS.txt` file. You must
install Isaac Gym Preview 4 and all dependencies described in the official
repository's README (`bash install.sh`).

## 2. Restore the Model

Unlike the baseline BC policy, `Model_UVLL_HanDex.ckpt` is a plain
`ActorCritic` `state_dict()` (see `demograsp_port/ppo_onestep/module.py`) —
it does not need to sit in a specific Lightning `saved_models/` directory
structure, just somewhere readable.

First, copy the contents of `code/` (this submission) into the challenge
repo's `dexgrasp/` directory, preserving relative paths — this adds the new
files listed in section 5 below, including the single demonstration
`demos/core-bottle-be16ada66829940a451786f3cbfd6769_traj0.pkl` that the
policy warps per object:

```bash
cp -r code/* DexGraspMotionChallenge2026/dexgrasp/
```

Then place the checkpoint anywhere accessible, e.g.:

```bash
ckpt_dir="DexGraspMotionChallenge2026/dexgrasp/checkpoints"
mkdir -p "$ckpt_dir"
cp Model_UVLL_HanDex.ckpt "$ckpt_dir/"
```

## 3. Running Inference

```bash
cd DexGraspMotionChallenge2026/dexgrasp

DEMOGRASP_MODE=eval_all \
DEMOGRASP_CKPT=$ckpt_dir/Model_UVLL_HanDex.ckpt \
DEMOGRASP_SOURCE_DIR=/absolute/path/to/private/trajectory/npy/files \
DEXGRASP_EVAL_ASSET_DIR=/absolute/path/to/private/object/meshes \
DEMOGRASP_CHUNK_SIZE=1500 \
DEXGRASP_RESULTS_PATH=./results/private_eval.json \
python train_demograsp.py --headless
```

- `DEMOGRASP_SOURCE_DIR` — directory of preprocessed `.npy` trajectory files
  (same schema `data_preprocess.py` produces: `grasp_seqs`/`obj_rotmat`/
  `obj_scale` per object). Every object found here is evaluated automatically
  — no need to specify how many.
- `DEXGRASP_EVAL_ASSET_DIR` — object mesh root, only needed if the private
  set's objects live under a different mesh path than this repo's default
  (`/meshdata/object_5048meshdata/`); omit if meshes are already in place.
- `DEMOGRASP_CHUNK_SIZE` (default 1500) — objects per simulation batch. The
  simulation is torn down and rebuilt between chunks, so peak GPU memory
  never depends on the total number of objects (chosen from our own scaling
  measurements: ~10.4 MiB/env marginal, ~2.3 GB fixed, well under a single
  GPU's memory — lower it on smaller hardware if needed).
- Prints a running success total after every chunk; `DEXGRASP_RESULTS_PATH`
  writes the final per-chunk + overall summary to JSON.

For a smaller/faster single-batch run (e.g. reproducing this submission's own
`LOCAL_EVALUATION.yaml`) use `DEMOGRASP_MODE=eval` with `DEMOGRASP_NUM_OBJECTS`/
`DEMOGRASP_TRAJ_PER_OBJECT` instead — see section 6.

## 4. Method summary

Single successful GraspM3 demonstration + a policy that outputs one 12-D "warp"
(Δwrist-xyz, Δwrist-rpy, Δ6-finger) per episode. The warp is applied once to the
demonstration to re-target it to the current object, expanded into a full
reach + track + settle trajectory, then replayed open-loop for the episode.
This is a port of DemoGrasp's method (single-demo + RL-learned warp) onto this
challenge's O6 task and GraspM3 data; see `demograsp_warp.py` for the warp math
and `demograsp_env.py` for the task adapter.

Training data: GraspM3 only (`dataset_shinwoo_preproc/train`, itself
preprocessed from the officially-downloaded `GraspM3.tar.gz`). No objects
outside the provided dataset were used for training or pretraining anything we
trained ourselves (see section 7 for the one pretrained third-party component
we reused, not trained).

**The demonstration itself is an existing GraspM3 trajectory, not synthesized
or recorded by us from scratch.** It is trajectory index 0 of GraspM3 object
`core-bottle-be16ada66829940a451786f3cbfd6769`, taken as-is from the dataset
and merely replayed once in Isaac Gym to confirm it succeeds (`record_demo.py`)
before being saved to `demos/*.pkl`. No new grasp motion was authored — the
only degrees of freedom we introduce are the learned per-object warp and the
reach segment that connects a new episode's starting hand pose to this same
GraspM3 trajectory's start.

## 5. Code package

`code/` in this submission contains every file we created or modified beyond
the baseline repo (everything else needed — `utils/`, `dexrep/`, task code,
etc. — is assumed already present, per the challenge wiki: "training scripts,
evaluation interfaces, and sample simulation pipelines" are provided):

```
record_demo.py                             (new  - records the one demonstration
                                             trajectory below by replaying a
                                             GraspM3 trajectory in sim and
                                             verifying it succeeds; not needed to
                                             run inference, included for
                                             reproducibility of demos/*.pkl)
demograsp_warp.py                          (new  - warp/reach math)
demograsp_env.py                           (new  - PPO <-> task adapter)
train_demograsp.py                         (new  - entrypoint, see sections 2-3, 6)
demograsp_port/ppo_onestep/__init__.py
demograsp_port/ppo_onestep/module.py       (modified - swapped in this repo's
                                             own frozen ShapeNet PointNet, see section 7)
demograsp_port/ppo_onestep/ppo.py          (modified - removed an isaacgymenvs
                                             import unused in this file; added
                                             optional video recording to the
                                             eval loop)
demograsp_port/ppo_onestep/storage.py
demos/core-bottle-be16ada66829940a451786f3cbfd6769_traj0.pkl  (the one
                                             demonstration trajectory the
                                             method warps per object; required)
```

To regenerate `demos/*.pkl` (e.g. with a different source trajectory instead
of the one we picked), edit the `OBJECT_CODE`/`DEMO_TRAJ_INDEX` constants at
the top of `record_demo.py` and run it from `dexgrasp/`:
```
python record_demo.py --headless
```
It replays that GraspM3 trajectory in Isaac Gym, confirms the object actually
gets lifted (`success=True`), and writes the resulting simulated states to
`demos/<object_code>_traj<N>.pkl`.

## 6. Reproducing `LOCAL_EVALUATION.yaml` (fixed-size single batch)

```bash
cd dexgrasp

DEMOGRASP_MODE=eval \
DEMOGRASP_CKPT=$ckpt_dir/Model_UVLL_HanDex.ckpt \
DEMOGRASP_EXCLUDE_TRAIN_RUN=</path/to training run dir>/config.json \
DEMOGRASP_NUM_OBJECTS=64 DEMOGRASP_TRAJ_PER_OBJECT=4 \
DEXGRASP_RESULTS_PATH=./results/local_eval.json \
python train_demograsp.py --headless
```

- `DEMOGRASP_EXCLUDE_TRAIN_RUN`: path to the training run's `config.json`
  (lists the 64 objects seen during training, so evaluation objects are drawn
  from the remaining, unseen pool). Omit to evaluate on a plain random sample
  instead of a strictly held-out one.
- `DEMOGRASP_NUM_OBJECTS` / `DEMOGRASP_TRAJ_PER_OBJECT`: how many objects /
  trajectories per object to evaluate on, all in one simulation batch (unlike
  section 3's `eval_all`, which chunks — use this only for batches small
  enough to fit in memory at once).
- Runs 10 rounds of deterministic (`inference=True`) rollout and prints
  per-round + mean/min/max success rate; `DEXGRASP_RESULTS_PATH` additionally
  dumps a JSON summary (this is the source of `LOCAL_EVALUATION.yaml`).

Video of a successful rollout (optional): add
`DEXGRASP_RECORD_VIDEO=1 DEXGRASP_RECORD_ENV_IDS=0,1,2,3` to the same command;
mp4s land under `./results/videos_demograsp_heldout/`, and the console prints
`video_success_env_ids` telling you which of those env ids actually succeeded.

**Result**: mean success rate **57.27%** over 10 rounds (min 53.88%, max
61.22%) on 64 GraspM3 objects not seen during training (245 envs, up to 4
trajectories/object) — see `LOCAL_EVALUATION.yaml`. For reference, the
challenge's documented BC baseline is 23.93% and the leaderboard entry cited
in the README is 25.56%.

## 7. Environment/config modifications made (relative to the repo's default cfg)

All changes are made by the training/eval entrypoint (`train_demograsp.py`),
not by editing the shared task file, and are listed here per the challenge's
disclosure requirement:

| cfg key | default | our value | why |
|---|---|---|---|
| `env_mode` | (baseline uses `bc_env_infer`) | `extract_obs` | Applies the full 12-D action to `cur_targets` every step (needed to replay our precomputed absolute-pose trajectory); functionally equivalent to `bc_env_infer` for action application since `o6_control_wrist_each_step: True` is the shared default either way. |
| `o6_policy_obs_mode` | `dexrep` | `prev_action_obj_rot` (21-D lightweight, DexRep off) | Our policy is called once per episode, not per step, so the heavy per-step DexRep feature pipeline is unnecessary; the policy's own observation (palm pose + object pose + 512-pt object point cloud) is assembled separately in `demograsp_env.py`. |
| `is_vision` (train_param, not task cfg) | `False` | `True` | Without it, `ActorCritic` never sees the object point cloud, i.e. it could not learn an object-shape-dependent warp at all. |
| `seq_start_pos_uniform` | `True` | `False` | This flag (only active when `env_mode=='extract_obs'`) canonicalizes each trajectory's approach direction onto one fixed axis — it is a deterministic normalization, not randomization, and the baseline's own `env_mode='bc_env_infer'` path never triggers it either. Turning it off keeps our path's behavior consistent with the baseline's on this point. |
| `seq_start_rot_uniform` | `False` | `False` (unchanged) | n/a |

Initial hand pose randomization requirement: unaffected by any of the above —
`reset()` sets the hand to `grasp_seqs[:, 0, :]` (each assigned trajectory's
own recorded start frame) regardless of `env_mode`, which is where the
required pose randomization actually comes from (verified: GraspM3's own
recorded start positions vary by ~0.20–0.27m from origin per trajectory,
consistent with README's "15–20cm away" description).

Success criterion (0.3m lift or ≤0.12m goal distance) is computed by the
shared `compute_hand_reward()` function, called identically regardless of
`env_mode` — no divergence from baseline on this point.

## 8. Third-party pretrained component (disclosure)

The point-cloud encoder backbone (`PointNetBackbone` in
`demograsp_port/ppo_onestep/module.py`) reuses this repo's own
`dexrep.pointnet_model.model_rec.ShapeNetAutoEncoder`, loaded from the
checkpoint `dexrep/pointnet_model/epo_180_REC_SPnetDenseEncoder_shapenet55_normrot512.pt`
that ships with this repository and is already used by the official DexRep
baseline. It is kept **frozen** (no gradient, `requires_grad_(False)`,
`eval()`); only a small linear projection head on top of it is trained. We did
not pretrain, fine-tune, or otherwise train this backbone ourselves, and it
was not trained on any object outside what this repo already ships with.
