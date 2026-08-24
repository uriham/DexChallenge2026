#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: $0"
    echo "Runs a fresh 200-epoch demo training job on GPU 0, then evaluates best and last checkpoints."
    exit 0
fi
if [[ "$#" -ne 0 ]]; then
    echo "Unexpected arguments. Use --help for usage." >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
challenge_env_prefix="${CHALLENGE_ENV_PREFIX:-/home/user/anaconda3/envs/DexGraspMotionChallenge2026}"
experiment_name="1obj_seq2000_DexRep_pro77_start_uniform_vis_action_dsam_mod_o6_75preprocessed_dexrep_200ep_seed42_bestlast"
experiment_dir="${repo_dir}/ActionDiffusion/bc/saved_models/${experiment_name}"

if [[ ! -x "${challenge_env_prefix}/bin/python" ]]; then
    echo "Challenge Python not found: ${challenge_env_prefix}/bin/python" >&2
    exit 2
fi
if [[ -e "${experiment_dir}" ]]; then
    echo "Refusing to reuse a non-clean experiment directory: ${experiment_dir}" >&2
    exit 3
fi

mkdir -p "${experiment_dir}"

export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME="${challenge_env_prefix}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${challenge_env_prefix}/lib:${challenge_env_prefix}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CC=/usr/bin/gcc-8
export CXX=/usr/bin/g++-8
export TORCH_CUDA_ARCH_LIST="8.6+PTX"
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

export DEXGRASP_BC_CONFIG=lhm_bc_o6_dexrep_full.yaml
export DEXGRASP_EXP_SUFFIX=o6_75preprocessed_dexrep_200ep_seed42_bestlast
export DEXGRASP_NUM_EPOCHS=200
export DEXGRASP_BATCH_SIZE=128
export DEXGRASP_NUM_WORKERS=8
export DEXGRASP_TRAIN_SEED=42
export DEXGRASP_DETERMINISTIC=1
export DEXGRASP_CHECK_VAL_EVERY_N_EPOCH=1
export DEXGRASP_SAVE_BEST_CKPT=1
export DEXGRASP_BEST_MONITOR=val_loss
export DEXGRASP_BEST_MODE=min
export DEXGRASP_SAVE_EPOCH_CKPT=0
export DEXGRASP_SAVE_FINAL_CKPT=1
unset DEXGRASP_RESUME_CKPT
unset DEXGRASP_CKPT_EVERY_N_EPOCHS

cd "${script_dir}"

"${challenge_env_prefix}/bin/python" train_bc_lighting_dexrep.py \
    2>&1 | tee "${experiment_dir}/training.log"

"${challenge_env_prefix}/bin/python" evaluate_best_last.py \
    "${experiment_dir}" \
    --eval-data-dir "${script_dir}/dataset_o6_75preproc/valid" \
    --gpu 0 \
    --seed 42 \
    --test-num 40 \
    --subprocess-batch-size 10
