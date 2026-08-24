#!/usr/bin/env python3
import argparse
import json
import os
import os.path as osp
import shutil
from glob import glob

import numpy as np


DEFAULT_INPUT_ROOT = "./dataset_o6"
DEFAULT_OUTPUT_ROOT = "./dataset_o6_aug_oldstyle_only"
DEFAULT_ASSET_ROOT = "../assets/meshdata"


def load_obj_z_extent(obj_path):
    z_values = []
    with open(obj_path, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) >= 4:
                z_values.append(float(parts[3]))
    if not z_values:
        raise ValueError("no OBJ vertices found: {}".format(obj_path))
    z_values = np.asarray(z_values, dtype=np.float64)
    return float(z_values.max() - z_values.min())


def object_code_from_path(path):
    return osp.splitext(osp.basename(path))[0]


def get_raw_height(obj_code, asset_root, cache):
    if obj_code in cache:
        return cache[obj_code]
    obj_path = osp.join(asset_root, obj_code, "coacd", "decomposed.obj")
    if not osp.exists(obj_path):
        raise FileNotFoundError("object mesh not found: {}".format(obj_path))
    height = load_obj_z_extent(obj_path)
    cache[obj_code] = height
    return height


def repeat_slow_window(seq, start, end, repeat_extra):
    if repeat_extra <= 0:
        return seq
    t0 = max(0, int(start))
    t1 = min(seq.shape[1] - 1, int(end))
    if t0 >= seq.shape[1] or t0 > t1:
        return seq

    repeated_frames = []
    slow_repeats = int(repeat_extra) + 1
    for t in range(seq.shape[1]):
        repeats = slow_repeats if t0 <= t <= t1 else 1
        frame = seq[:, t : t + 1, :]
        repeated_frames.extend([frame] * repeats)
    return np.concatenate(repeated_frames, axis=1)


def apply_oldstyle_o6_adjustments(
    grasp_seqs,
    obj_scale,
    raw_height,
    scale_factor,
    table_z_extra,
    tighten_start,
    extra_bend,
    slow_start,
    slow_end,
    repeat_extra,
):
    aug = np.asarray(grasp_seqs, dtype=np.float32).copy()
    orig_scale = np.asarray(obj_scale, dtype=np.float32)
    new_scale = orig_scale * np.float32(scale_factor)

    z_offsets = -raw_height * (orig_scale - new_scale) / 2.0
    aug[:, :, 2] += z_offsets[:, None].astype(np.float32)
    aug[:, :, 2] += np.float32(table_z_extra)

    if extra_bend != 0.0 and tighten_start < aug.shape[1]:
        start = max(0, int(tighten_start))
        aug[:, start:, 8:12] += np.float32(extra_bend)
        aug[:, start:, 8:12] = np.minimum(aug[:, start:, 8:12], np.float32(1.60))

    aug = repeat_slow_window(aug, slow_start, slow_end, repeat_extra)

    return aug.astype(np.float32), new_scale.astype(np.float32), z_offsets.astype(np.float32)


def reorder_grasp_to_seq_params_base(grasp_seqs):
    grasp = np.asarray(grasp_seqs, dtype=np.float32)
    return np.concatenate([grasp[:, :, 6:12], grasp[:, :, 0:6]], axis=-1).astype(np.float32)


def copy_augmented_only_value(value, nseq, original_num_frames, slow_start, slow_end, repeat_extra):
    if isinstance(value, np.ndarray) and value.shape[:1] == (nseq,):
        copied = value.copy()
        if value.ndim >= 3 and value.shape[1] == original_num_frames:
            copied = repeat_slow_window(copied, slow_start, slow_end, repeat_extra)
        return copied
    if isinstance(value, list) and len(value) == nseq:
        return list(value)
    return value


def augment_file(path, out_path, args, height_cache):
    obj_code = object_code_from_path(path)
    data = np.load(path, allow_pickle=True).item()
    if "grasp_seqs" not in data or "obj_scale" not in data:
        raise KeyError("{} must contain grasp_seqs and obj_scale".format(path))

    grasp = np.asarray(data["grasp_seqs"], dtype=np.float32)
    obj_scale = np.asarray(data["obj_scale"], dtype=np.float32)
    nseq = grasp.shape[0]
    original_num_frames = grasp.shape[1]
    if obj_scale.shape[0] != nseq:
        raise ValueError("obj_scale length mismatch in {}".format(path))

    raw_height = get_raw_height(obj_code, args.asset_root, height_cache)
    aug_grasp, aug_scale, z_offsets = apply_oldstyle_o6_adjustments(
        grasp,
        obj_scale,
        raw_height,
        args.scale_factor,
        args.table_z_extra,
        args.tighten_start,
        args.extra_bend,
        args.slow_start,
        args.slow_end,
        args.pause_steps,
    )

    out = {}
    for key, value in data.items():
        if key == "grasp_seqs":
            out[key] = aug_grasp.astype(np.float32)
        elif key == "obj_scale":
            out[key] = aug_scale.astype(np.float32)
        elif key == "seq_params_base":
            out[key] = reorder_grasp_to_seq_params_base(aug_grasp)
        else:
            out[key] = copy_augmented_only_value(
                value, nseq, original_num_frames, args.slow_start, args.slow_end, args.pause_steps
            )

    out["augment_info"] = {
        "source_file": osp.abspath(path),
        "mode": "oldstyle_o6_augmented_only",
        "raw_height": raw_height,
        "scale_factor": args.scale_factor,
        "table_z_extra": args.table_z_extra,
        "z_offset_min": float(z_offsets.min()),
        "z_offset_max": float(z_offsets.max()),
        "tighten_start": args.tighten_start,
        "extra_bend": args.extra_bend,
        "slow_start": args.slow_start,
        "slow_end": args.slow_end,
        "repeat_extra_per_slow_frame": args.pause_steps,
        "original_num_sequences": int(nseq),
        "output_num_sequences": int(out["grasp_seqs"].shape[0]),
        "original_num_frames": int(original_num_frames),
        "output_num_frames": int(out["grasp_seqs"].shape[1]),
    }

    os.makedirs(osp.dirname(out_path), exist_ok=True)
    np.save(out_path, out, allow_pickle=True)
    return out["augment_info"]


def process_split(split, args, height_cache):
    in_dir = osp.join(args.input_root, split)
    out_dir = osp.join(args.output_root, split)
    if not osp.isdir(in_dir):
        raise FileNotFoundError("input split dir not found: {}".format(in_dir))
    if osp.exists(out_dir):
        if args.overwrite:
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError("output dir exists, pass --overwrite: {}".format(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    infos = []
    for path in sorted(glob(osp.join(in_dir, "*.npy"))):
        if osp.getsize(path) < 1024:
            continue
        out_path = osp.join(out_dir, osp.basename(path))
        info = augment_file(path, out_path, args, height_cache)
        info["split"] = split
        info["output_file"] = osp.abspath(out_path)
        infos.append(info)
        print(
            "{} {}: {} -> {} seq, {} -> {} frames, scale_factor={}, z_offset=[{:.6f}, {:.6f}]".format(
                split,
                osp.basename(path),
                info["original_num_sequences"],
                info["output_num_sequences"],
                info["original_num_frames"],
                info["output_num_frames"],
                info["scale_factor"],
                info["z_offset_min"],
                info["z_offset_max"],
            )
        )
    return infos


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write old screening-style o6 augmented-only BC npy files."
    )
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--asset-root", default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--splits", default="train,valid")
    parser.add_argument("--scale-factor", type=float, default=0.833)
    parser.add_argument("--table-z-extra", type=float, default=0.0025)
    parser.add_argument("--tighten-start", type=int, default=33)
    parser.add_argument("--extra-bend", type=float, default=0.05)
    parser.add_argument("--slow-start", type=int, default=28)
    parser.add_argument("--slow-end", type=int, default=50)
    parser.add_argument(
        "--pause-steps",
        type=int,
        default=2,
        help="Extra copies for each frame in [slow-start, slow-end]. Default 2 makes one frame become three frames.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.input_root = osp.abspath(args.input_root)
    args.output_root = osp.abspath(args.output_root)
    args.asset_root = osp.abspath(args.asset_root)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    height_cache = {}
    all_infos = []
    for split in splits:
        all_infos.extend(process_split(split, args, height_cache))

    summary_path = osp.join(args.output_root, "augment_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_infos, f, indent=2, sort_keys=True)
    print("wrote summary: {}".format(summary_path))


if __name__ == "__main__":
    main()
