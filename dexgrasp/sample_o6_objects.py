#!/usr/bin/env python3
import argparse
import json
import os
import os.path as osp
import random
import shutil
from collections import defaultdict
from glob import glob


DEFAULT_TRAJ_ROOT = "./linkerhand_small_all"
DEFAULT_MESH_ROOT = "../assets/object_5048meshdata"
DEFAULT_TRAJ_OUT = "./linkerhand_small_75"
DEFAULT_MESH_OUT = "../assets/object_75meshdata"


def object_code_from_npy(path):
    return osp.splitext(osp.basename(path))[0]


def diversity_key(object_code):
    parts = object_code.split("-")
    if len(parts) >= 2 and parts[0] in {"core", "sem"}:
        return "{}-{}".format(parts[0], parts[1])
    if len(parts) >= 2 and parts[0] == "ddg":
        return "{}-{}".format(parts[0], parts[1].split("_")[0])
    if len(parts) >= 2 and parts[0] == "mujoco":
        return "{}-{}".format(parts[0], parts[1].split("_")[0])
    return parts[0]


def prepare_output_dir(path, overwrite):
    if osp.exists(path):
        if not overwrite:
            raise FileExistsError("output path exists, pass --overwrite: {}".format(path))
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def choose_diverse_objects(object_codes, count, seed):
    groups = defaultdict(list)
    for code in object_codes:
        groups[diversity_key(code)].append(code)

    rng = random.Random(seed)
    for codes in groups.values():
        rng.shuffle(codes)

    group_keys = list(groups.keys())
    rng.shuffle(group_keys)
    group_keys.sort(key=lambda key: (-len(groups[key]), key))

    selected = []
    selected_set = set()
    while len(selected) < count:
        progressed = False
        for key in group_keys:
            if len(selected) >= count:
                break
            while groups[key] and groups[key][-1] in selected_set:
                groups[key].pop()
            if not groups[key]:
                continue
            code = groups[key].pop()
            selected.append(code)
            selected_set.add(code)
            progressed = True
        if not progressed:
            break

    if len(selected) < count:
        raise RuntimeError("only selected {} objects, requested {}".format(len(selected), count))
    return selected


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample diverse O6 object trajectory files and matching mesh folders."
    )
    parser.add_argument("--traj-root", default=DEFAULT_TRAJ_ROOT)
    parser.add_argument("--mesh-root", default=DEFAULT_MESH_ROOT)
    parser.add_argument("--traj-out", default=DEFAULT_TRAJ_OUT)
    parser.add_argument("--mesh-out", default=DEFAULT_MESH_OUT)
    parser.add_argument("--count", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    traj_root = osp.abspath(args.traj_root)
    mesh_root = osp.abspath(args.mesh_root)
    traj_out = osp.abspath(args.traj_out)
    mesh_out = osp.abspath(args.mesh_out)

    if not osp.isdir(traj_root):
        raise FileNotFoundError("trajectory root not found: {}".format(traj_root))
    if not osp.isdir(mesh_root):
        raise FileNotFoundError("mesh root not found: {}".format(mesh_root))
    if args.count <= 0:
        raise ValueError("--count must be positive")

    traj_paths = sorted(glob(osp.join(traj_root, "*.npy")))
    traj_by_code = {object_code_from_npy(path): path for path in traj_paths}
    mesh_codes = {
        name
        for name in os.listdir(mesh_root)
        if osp.isdir(osp.join(mesh_root, name))
    }
    available_codes = sorted(set(traj_by_code) & mesh_codes)
    missing_mesh = sorted(set(traj_by_code) - mesh_codes)
    if len(available_codes) < args.count:
        raise RuntimeError(
            "not enough matched trajectory/mesh objects: matched {}, requested {}, missing_mesh {}".format(
                len(available_codes), args.count, len(missing_mesh)
            )
        )

    selected_codes = choose_diverse_objects(available_codes, args.count, args.seed)
    prepare_output_dir(traj_out, args.overwrite)
    prepare_output_dir(mesh_out, args.overwrite)

    copied = []
    for code in selected_codes:
        src_traj = traj_by_code[code]
        dst_traj = osp.join(traj_out, osp.basename(src_traj))
        src_mesh = osp.join(mesh_root, code)
        dst_mesh = osp.join(mesh_out, code)
        shutil.copy2(src_traj, dst_traj)
        shutil.copytree(src_mesh, dst_mesh)
        copied.append(
            {
                "object_code": code,
                "diversity_key": diversity_key(code),
                "trajectory_file": osp.basename(dst_traj),
                "mesh_dir": code,
            }
        )

    summary = {
        "traj_root": traj_root,
        "mesh_root": mesh_root,
        "traj_out": traj_out,
        "mesh_out": mesh_out,
        "count": len(copied),
        "seed": args.seed,
        "matched_source_objects": len(available_codes),
        "source_trajectories": len(traj_paths),
        "source_mesh_dirs": len(mesh_codes),
        "missing_mesh_count": len(missing_mesh),
        "num_diversity_keys": len(set(item["diversity_key"] for item in copied)),
        "objects": copied,
    }
    summary_path = osp.join(traj_out, "sample_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    shutil.copy2(summary_path, osp.join(mesh_out, "sample_summary.json"))

    print(
        "sampled {} objects from {} matched objects; diversity_keys {}; traj_out {}; mesh_out {}".format(
            len(copied),
            len(available_codes),
            summary["num_diversity_keys"],
            traj_out,
            mesh_out,
        )
    )
    print("summary {}".format(summary_path))


if __name__ == "__main__":
    main()
