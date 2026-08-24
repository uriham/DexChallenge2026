#!/usr/bin/env python3
"""Evaluate a training run's validation-best and final checkpoints identically."""

import argparse
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime

import yaml


THIS_DIR = Path(__file__).resolve().parent


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_checkpoint(experiment_dir, key):
    manifest_path = experiment_dir / "checkpoint_selection.yaml"
    if not manifest_path.is_file():
        return None
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream) or {}
    raw_path = (manifest.get(key) or {}).get("path")
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if path.is_file():
        return path.resolve()
    local_path = experiment_dir / path.name
    return local_path.resolve() if local_path.is_file() else None


def resolve_checkpoints(experiment_dir):
    best = manifest_checkpoint(experiment_dir, "best")
    if best is None:
        best_candidates = sorted(experiment_dir.glob("best*.ckpt"))
        if len(best_candidates) != 1:
            raise RuntimeError(
                "expected exactly one best checkpoint in {}, found {}".format(
                    experiment_dir,
                    len(best_candidates),
                )
            )
        best = best_candidates[0].resolve()

    last = manifest_checkpoint(experiment_dir, "last")
    if last is None:
        last = (experiment_dir / "last.ckpt").resolve()
    if not last.is_file():
        raise FileNotFoundError("last checkpoint not found: {}".format(last))
    return best, last


def run_evaluation(label, checkpoint, args, run_id, log_dir):
    suffix = "best_last_{}_{}".format(run_id, label)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["DEXGRASP_BC_CONFIG"] = args.bc_config
    env["DEXGRASP_POLICY_CKPT"] = str(checkpoint)
    env["DEXGRASP_EVAL_DATA_DIR"] = str(args.eval_data_dir)
    env["DEXGRASP_TEST_NUM"] = str(args.test_num)
    env["DEXGRASP_EVAL_SUBPROCESS"] = "1"
    env["DEXGRASP_SUBPROCESS_BATCH_SIZE"] = str(args.subprocess_batch_size)
    env["DEXGRASP_RESULT_SUFFIX"] = suffix
    if args.eval_asset_dir:
        env["DEXGRASP_EVAL_ASSET_DIR"] = args.eval_asset_dir

    command = [
        sys.executable,
        str(THIS_DIR / "bc_env_infer.py"),
        "--headless",
        "--seed",
        str(args.seed),
        "--torch_deterministic",
        "--sim_device",
        "cuda:0",
        "--rl_device",
        "cuda:0",
        "--pipeline",
        "gpu",
    ]
    print("{} command: {}".format(label, shlex.join(command)))
    print("{} checkpoint: {}".format(label, checkpoint))
    if args.dry_run:
        return None

    log_path = log_dir / "{}.log".format(label)
    aggregate_path = None
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=str(THIS_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_stream.write(line)
            if line.startswith("aggregate_result_path:"):
                aggregate_path = Path(line.split(":", 1)[1].strip()).resolve()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            "{} evaluation failed with exit code {}; see {}".format(
                label,
                return_code,
                log_path,
            )
        )
    if aggregate_path is None or not aggregate_path.is_file():
        matches = sorted(
            (THIS_DIR / "results").glob("*{}*aggregate.yaml".format(suffix)),
            key=lambda path: path.stat().st_mtime,
        )
        if not matches:
            raise FileNotFoundError(
                "{} evaluation did not report an aggregate result".format(label)
            )
        aggregate_path = matches[-1].resolve()

    with aggregate_path.open("r", encoding="utf-8") as stream:
        result = yaml.safe_load(stream) or {}
    required = {
        "total_success_num",
        "total_trials",
        "weighted_success_rate",
        "mean_object_success_rate",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise KeyError("{} result is missing keys: {}".format(label, missing))
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "result_path": str(aggregate_path),
        "total_success_num": float(result["total_success_num"]),
        "total_trials": int(result["total_trials"]),
        "weighted_success_rate": float(result["weighted_success_rate"]),
        "mean_object_success_rate": float(result["mean_object_success_rate"]),
    }


def comparison_result(best, last, args):
    weighted_delta = best["weighted_success_rate"] - last["weighted_success_rate"]
    mean_delta = best["mean_object_success_rate"] - last["mean_object_success_rate"]
    if weighted_delta > 0:
        winner = "best"
    elif weighted_delta < 0:
        winner = "last"
    elif mean_delta > 0:
        winner = "best"
    elif mean_delta < 0:
        winner = "last"
    else:
        winner = "tie"
    return {
        "evaluation_protocol": {
            "scope": "local_evaluator",
            "seed": args.seed,
            "test_num": args.test_num,
            "eval_data_dir": str(args.eval_data_dir),
            "eval_asset_dir": args.eval_asset_dir,
        },
        "best": best,
        "last": last,
        "delta_best_minus_last": {
            "weighted_success_rate": weighted_delta,
            "mean_object_success_rate": mean_delta,
        },
        "rollout_winner": winner,
        "warning": (
            "This comparison uses the local evaluator. It is not an official "
            "private-test or human-likeness score."
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument(
        "--eval-data-dir",
        type=Path,
        default=THIS_DIR / "dataset_o6_75preproc" / "valid",
    )
    parser.add_argument(
        "--eval-asset-dir",
        default="",
        help="Optional object subdirectory under the configured assetRoot.",
    )
    parser.add_argument("--bc-config", default="lhm_bc_o6_dexrep_full.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-num", type=int, default=40)
    parser.add_argument("--subprocess-batch-size", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.experiment_dir = args.experiment_dir.expanduser().resolve()
    args.eval_data_dir = args.eval_data_dir.expanduser().resolve()
    if not args.experiment_dir.is_dir():
        raise NotADirectoryError(args.experiment_dir)
    if not args.eval_data_dir.is_dir():
        raise NotADirectoryError(args.eval_data_dir)

    best_checkpoint, last_checkpoint = resolve_checkpoints(args.experiment_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.experiment_dir / "eval_logs" / "best_last_{}".format(run_id)
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=False)

    best_result = run_evaluation(
        "best",
        best_checkpoint,
        args,
        run_id,
        log_dir,
    )
    last_result = run_evaluation(
        "last",
        last_checkpoint,
        args,
        run_id,
        log_dir,
    )
    if args.dry_run:
        return

    comparison = comparison_result(best_result, last_result, args)
    output_path = log_dir / "comparison.yaml"
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(comparison, stream, sort_keys=False)

    for label, result in (("best", best_result), ("last", last_result)):
        print(
            "{}: {}/{} weighted={:.4%} mean_object={:.4%}".format(
                label,
                result["total_success_num"],
                result["total_trials"],
                result["weighted_success_rate"],
                result["mean_object_success_rate"],
            )
        )
    print("rollout_winner:", comparison["rollout_winner"])
    print("comparison_path:", output_path)


if __name__ == "__main__":
    main()
