#!/usr/bin/env python3
import argparse
import json
import os
import os.path as osp
import random
import shutil
from glob import glob

import numpy as np


DEFAULT_INPUT_ROOT = "./dataset_o6_small41"
DEFAULT_OUTPUT_ROOT = "./dataset_o6"


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


def split_file(path, train_dir, valid_dir, args, rng):
    data = np.load(path, allow_pickle=True).item()
    if "grasp_seqs" not in data:
        raise KeyError("{} must contain grasp_seqs".format(path))

    num_trajs = int(np.asarray(data["grasp_seqs"]).shape[0])
    train_count = split_counts(num_trajs, args.train_ratio)
    all_indices = np.arange(num_trajs, dtype=np.int64)
    if args.shuffle:
        rng.shuffle(all_indices)

    train_indices = np.sort(all_indices[:train_count])
    valid_indices = np.sort(all_indices[train_count:])

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
        "shuffle": args.shuffle,
        "seed": args.seed,
    }
    train_data["split_info"] = dict(split_info, split="train")
    valid_data["split_info"] = dict(split_info, split="valid")

    np.save(osp.join(train_dir, basename), train_data, allow_pickle=True)
    np.save(osp.join(valid_dir, basename), valid_data, allow_pickle=True)
    return split_info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split each object npy into train/valid trajectory subsets, keeping files separated by object."
    )
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(shuffle=True)
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

    rng = random.Random(args.seed)
    summary = []
    total_train = 0
    total_valid = 0
    for path in paths:
        info = split_file(path, train_dir, valid_dir, args, rng)
        summary.append(info)
        total_train += info["train_count"]
        total_valid += info["valid_count"]
        print(
            "{}: {} traj -> train {}, valid {}".format(
                osp.basename(path), info["num_trajs"], info["train_count"], info["valid_count"]
            )
        )

    summary_data = {
        "input_root": input_root,
        "output_root": output_root,
        "num_objects": len(paths),
        "total_train_trajs": total_train,
        "total_valid_trajs": total_valid,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "shuffle": args.shuffle,
        "files": summary,
    }
    summary_path = osp.join(output_root, "split_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, sort_keys=True)
    print(
        "done: {} objects, train {}, valid {}, summary {}".format(
            len(paths), total_train, total_valid, summary_path
        )
    )


if __name__ == "__main__":
    main()
