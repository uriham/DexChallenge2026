#!/usr/bin/env python3
import argparse
import json
import os
import os.path as osp
import shutil
from glob import glob

import numpy as np


DEFAULT_INPUT_ROOT = "./linkerhand_small_75"
DEFAULT_OUTPUT_ROOT = "./dataset_o6_75"


def split_counts(num_items, train_ratio):
    if num_items <= 0:
        return 0
    if num_items == 1:
        return 1

    train_count = int(np.floor(num_items * train_ratio))
    train_count = max(1, train_count)
    train_count = min(num_items - 1, train_count)
    return train_count


def index_value(value, indices, num_items):
    if isinstance(value, np.ndarray) and value.shape[:1] == (num_items,):
        return value[indices].copy()
    if isinstance(value, list) and len(value) == num_items:
        return [value[i] for i in indices.tolist()]
    return value


def sequence_features(array, prefix):
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim < 3:
        return [], []

    num_trajs, num_steps = arr.shape[:2]
    flat = arr.reshape(num_trajs, num_steps, -1)
    frame_ids = [0, min(5, num_steps - 1), num_steps - 1]

    chunks = []
    names = []
    for frame_id in frame_ids:
        chunks.append(flat[:, frame_id, :])
        names.extend(["{}_frame{}_dim{}".format(prefix, frame_id, i) for i in range(flat.shape[2])])

    delta = flat[:, -1, :] - flat[:, 0, :]
    chunks.append(delta)
    names.extend(["{}_last_minus_first_dim{}".format(prefix, i) for i in range(flat.shape[2])])

    mean = flat.mean(axis=1)
    chunks.append(mean)
    names.extend(["{}_mean_dim{}".format(prefix, i) for i in range(flat.shape[2])])

    std = flat.std(axis=1)
    chunks.append(std)
    names.extend(["{}_std_dim{}".format(prefix, i) for i in range(flat.shape[2])])
    return chunks, names


def trajectory_features(data):
    if "grasp_seqs" not in data:
        raise KeyError("npy data must contain grasp_seqs")

    grasp = np.asarray(data["grasp_seqs"])
    num_trajs = int(grasp.shape[0])
    chunks = []
    names = []

    seq_chunks, seq_names = sequence_features(grasp, "grasp_seqs")
    chunks.extend(seq_chunks)
    names.extend(seq_names)

    if "seq_params_base" in data:
        seq_params = np.asarray(data["seq_params_base"])
        if seq_params.shape[:1] == (num_trajs,):
            seq_chunks, seq_names = sequence_features(seq_params, "seq_params_base")
            chunks.extend(seq_chunks)
            names.extend(seq_names)

    if "obj_rotmat" in data:
        rot = np.asarray(data["obj_rotmat"], dtype=np.float64)
        if rot.shape[:1] == (num_trajs,):
            rot = rot.reshape(num_trajs, -1)
            chunks.append(rot)
            names.extend(["obj_rotmat_dim{}".format(i) for i in range(rot.shape[1])])

    if "obj_scale" in data:
        scale = np.asarray(data["obj_scale"], dtype=np.float64)
        if scale.shape[:1] == (num_trajs,):
            scale = scale.reshape(num_trajs, -1)
            chunks.append(scale)
            names.extend(["obj_scale_dim{}".format(i) for i in range(scale.shape[1])])

    if not chunks:
        raise ValueError("no trajectory features could be extracted")
    return np.concatenate(chunks, axis=1), names


def normalize_features(features):
    features = np.asarray(features, dtype=np.float64)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-8] = 1.0
    normalized = (features - mean) / std
    normalized = np.nan_to_num(normalized, copy=False)
    return normalized


def distribution_score(features, valid_indices):
    num_trajs = int(features.shape[0])
    valid_indices = np.asarray(valid_indices, dtype=np.int64)
    valid_mask = np.zeros(num_trajs, dtype=bool)
    valid_mask[valid_indices] = True
    train_mask = ~valid_mask

    train = features[train_mask]
    valid = features[valid_mask]
    if len(train) == 0 or len(valid) == 0:
        return 0.0, {
            "mean_abs_diff": 0.0,
            "std_abs_diff": 0.0,
            "quantile_abs_diff": 0.0,
        }

    mean_abs_diff = float(np.abs(train.mean(axis=0) - valid.mean(axis=0)).mean())
    std_abs_diff = float(np.abs(train.std(axis=0) - valid.std(axis=0)).mean())
    train_q = np.quantile(train, [0.1, 0.5, 0.9], axis=0)
    valid_q = np.quantile(valid, [0.1, 0.5, 0.9], axis=0)
    quantile_abs_diff = float(np.abs(train_q - valid_q).mean())
    score = mean_abs_diff + 0.5 * std_abs_diff + 0.25 * quantile_abs_diff
    return float(score), {
        "mean_abs_diff": mean_abs_diff,
        "std_abs_diff": std_abs_diff,
        "quantile_abs_diff": quantile_abs_diff,
    }


def greedy_representative_valid(features, valid_count, rng, num_restarts):
    num_trajs = int(features.shape[0])
    if valid_count <= 0:
        return np.empty((0,), dtype=np.int64), 0.0, {}
    if valid_count >= num_trajs:
        valid_indices = np.arange(num_trajs, dtype=np.int64)
        score, metrics = distribution_score(features, valid_indices)
        return valid_indices, score, metrics

    target = features.mean(axis=0)
    best_indices = None
    best_score = None
    best_metrics = None

    for restart_id in range(max(1, num_restarts)):
        selected = []
        remaining = np.arange(num_trajs, dtype=np.int64)
        selected_sum = np.zeros(features.shape[1], dtype=np.float64)

        if restart_id > 0:
            first = int(rng.integers(0, num_trajs))
            selected.append(first)
            selected_sum += features[first]
            remaining = remaining[remaining != first]

        while len(selected) < valid_count:
            step_count = len(selected) + 1
            candidate_means = (selected_sum[None, :] + features[remaining]) / float(step_count)
            errors = np.mean((candidate_means - target[None, :]) ** 2, axis=1)
            best_pos = int(np.argmin(errors))
            best_candidate = int(remaining[best_pos])
            selected.append(best_candidate)
            selected_sum += features[best_candidate]
            remaining = np.delete(remaining, best_pos)

        valid_indices = np.sort(np.asarray(selected, dtype=np.int64))
        score, metrics = distribution_score(features, valid_indices)
        if best_score is None or score < best_score:
            best_indices = valid_indices
            best_score = score
            best_metrics = metrics

    return best_indices, float(best_score), best_metrics


def split_file(path, train_dir, valid_dir, args, rng):
    data = np.load(path, allow_pickle=True).item()
    features, feature_names = trajectory_features(data)
    num_trajs = int(np.asarray(data["grasp_seqs"]).shape[0])
    train_count = split_counts(num_trajs, args.train_ratio)
    valid_count = num_trajs - train_count

    normalized_features = normalize_features(features)
    valid_indices, score, metrics = greedy_representative_valid(
        normalized_features, valid_count, rng, args.restarts
    )
    valid_mask = np.zeros(num_trajs, dtype=bool)
    valid_mask[valid_indices] = True
    train_indices = np.where(~valid_mask)[0].astype(np.int64)

    basename = osp.basename(path)
    train_data = {key: index_value(value, train_indices, num_trajs) for key, value in data.items()}
    valid_data = {key: index_value(value, valid_indices, num_trajs) for key, value in data.items()}

    split_info = {
        "source_file": osp.abspath(path),
        "num_trajs": num_trajs,
        "train_count": int(len(train_indices)),
        "valid_count": int(len(valid_indices)),
        "train_indices": train_indices.tolist(),
        "valid_indices": valid_indices.tolist(),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "method": "per_object_distribution_matching",
        "feature_dim": int(features.shape[1]),
        "feature_names": feature_names if args.store_feature_names else [],
        "score": score,
        "metrics": metrics,
    }
    train_data["split_info"] = dict(split_info, split="train")
    valid_data["split_info"] = dict(split_info, split="valid")

    np.save(osp.join(train_dir, basename), train_data, allow_pickle=True)
    np.save(osp.join(valid_dir, basename), valid_data, allow_pickle=True)
    return split_info


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split each O6 object npy into train/valid trajectory subsets while matching "
            "initial, temporal, and object-state feature distributions."
        )
    )
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--store-feature-names", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")

    input_root = osp.abspath(args.input_root)
    output_root = osp.abspath(args.output_root)
    if not osp.isdir(input_root):
        raise FileNotFoundError("input root not found: {}".format(input_root))
    if osp.abspath(input_root) == osp.abspath(output_root):
        raise ValueError("output root must be different from input root")
    if osp.exists(output_root):
        if not args.overwrite:
            raise FileExistsError("output root exists, pass --overwrite: {}".format(output_root))
        shutil.rmtree(output_root)

    train_dir = osp.join(output_root, "train")
    valid_dir = osp.join(output_root, "valid")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    paths = sorted(glob(osp.join(input_root, "*.npy")))
    if not paths:
        raise FileNotFoundError("no npy files found in {}".format(input_root))

    rng = np.random.default_rng(args.seed)
    summary = []
    total_train = 0
    total_valid = 0
    weighted_score = 0.0
    weighted_mean_diff = 0.0
    weighted_std_diff = 0.0
    weighted_quantile_diff = 0.0

    for file_id, path in enumerate(paths, start=1):
        info = split_file(path, train_dir, valid_dir, args, rng)
        summary.append(info)
        total_train += info["train_count"]
        total_valid += info["valid_count"]
        weight = float(info["num_trajs"])
        weighted_score += info["score"] * weight
        weighted_mean_diff += info["metrics"]["mean_abs_diff"] * weight
        weighted_std_diff += info["metrics"]["std_abs_diff"] * weight
        weighted_quantile_diff += info["metrics"]["quantile_abs_diff"] * weight
        if not args.quiet and (file_id <= 20 or file_id % 100 == 0 or file_id == len(paths)):
            print(
                "{}/{} {}: {} traj -> train {}, valid {}, score {:.6f}".format(
                    file_id,
                    len(paths),
                    osp.basename(path),
                    info["num_trajs"],
                    info["train_count"],
                    info["valid_count"],
                    info["score"],
                )
            )

    total_weight = max(1.0, float(total_train + total_valid))
    summary_data = {
        "input_root": input_root,
        "output_root": output_root,
        "num_objects": len(paths),
        "total_train_trajs": int(total_train),
        "total_valid_trajs": int(total_valid),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "restarts": args.restarts,
        "method": "per_object_distribution_matching",
        "features": [
            "grasp_seqs: first frame, frame 5, last frame, last-first, temporal mean, temporal std",
            "seq_params_base: first frame, frame 5, last frame, last-first, temporal mean, temporal std",
            "obj_rotmat flattened",
            "obj_scale",
        ],
        "global_score": float(weighted_score / total_weight),
        "global_metrics": {
            "mean_abs_diff": float(weighted_mean_diff / total_weight),
            "std_abs_diff": float(weighted_std_diff / total_weight),
            "quantile_abs_diff": float(weighted_quantile_diff / total_weight),
        },
        "files": summary,
    }
    summary_path = osp.join(output_root, "distribution_split_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, sort_keys=True)

    legacy_summary_path = osp.join(output_root, "split_summary.json")
    with open(legacy_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, sort_keys=True)

    print(
        "done: {} objects, train {}, valid {}, global_score {:.6f}, summary {}".format(
            len(paths),
            total_train,
            total_valid,
            summary_data["global_score"],
            summary_path,
        )
    )


if __name__ == "__main__":
    main()
