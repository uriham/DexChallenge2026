#!/usr/bin/env python3
"""Randomly copy a category-diverse subset of objects from an NPY dataset."""

import argparse
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path


MANIFEST_NAME = "selection_manifest.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Randomly select objects while preferring distinct categories. An object can "
            "be one NPY file or one immediate child directory containing NPY files."
        )
    )
    parser.add_argument("--source", type=Path, required=True, help="source dataset directory")
    parser.add_argument("--output", type=Path, required=True, help="directory for selected objects")
    parser.add_argument(
        "--count", type=int, default=100, help="maximum objects to select (default: 100)"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument(
        "--mode",
        choices=("auto", "files", "directories"),
        default="auto",
        help=(
            "files: each NPY is an object; directories: each immediate child directory is "
            "an object; auto: use files when source has NPY files directly, otherwise directories"
        ),
    )
    parser.add_argument(
        "--category-regex",
        help=(
            "optional regex used to extract a category from each object name; use a named "
            "group 'category' or the first capture group, e.g. '^(?P<category>[^_]+)_'"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="delete an existing output directory before copying",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the selection without creating or copying anything",
    )
    return parser.parse_args()


def default_category(object_name):
    """Infer a best-effort category from common GraspM3/O6 object IDs."""
    parts = object_name.split("-")
    if len(parts) >= 2 and parts[0] in {"core", "sem"}:
        # Treat sem-Bottle and core-bottle as the same semantic category.
        return parts[1].lower()
    if len(parts) >= 2 and parts[0] in {"ddg", "mujoco"}:
        return parts[1].split("_")[0].lower()

    # Generic IDs commonly look like bottle_001, bottle-12, or bottle_<uuid>.
    return re.split(r"[-_]", object_name, maxsplit=1)[0].lower()


def category_from_name(object_name, category_pattern):
    if category_pattern is None:
        return default_category(object_name), True

    match = category_pattern.search(object_name)
    if match is None:
        # An unmatched object remains selectable without being merged into an arbitrary group.
        return object_name, False
    if "category" in match.groupdict():
        category = match.group("category")
    elif match.lastindex:
        category = match.group(1)
    else:
        category = match.group(0)
    if not category:
        raise ValueError("category regex produced an empty category for {!r}".format(object_name))
    return category, True


def contains_npy(directory):
    return any(path.is_file() and path.suffix.lower() == ".npy" for path in directory.rglob("*"))


def discover_objects(source, requested_mode):
    direct_npy_files = sorted(
        path for path in source.iterdir() if path.is_file() and path.suffix.lower() == ".npy"
    )
    mode = requested_mode
    if mode == "auto":
        mode = "files" if direct_npy_files else "directories"

    if mode == "files":
        objects = direct_npy_files
    else:
        objects = sorted(
            path
            for path in source.iterdir()
            if path.is_dir() and not path.is_symlink() and contains_npy(path)
        )
    return mode, objects


def select_diverse(objects, count, seed, category_pattern):
    groups = defaultdict(list)
    matched_count = 0
    object_categories = {}
    for path in objects:
        object_name = path.stem if path.is_file() else path.name
        category, matched = category_from_name(object_name, category_pattern)
        groups[category].append(path)
        object_categories[path] = category
        matched_count += int(matched)

    rng = random.Random(seed)
    categories = list(groups)
    rng.shuffle(categories)
    for paths in groups.values():
        rng.shuffle(paths)

    selected = []
    # Round-robin sampling takes one random object from every category before
    # taking a second one from any category.
    while len(selected) < count:
        made_progress = False
        for category in categories:
            if len(selected) >= count:
                break
            if groups[category]:
                selected.append(groups[category].pop())
                made_progress = True
        if not made_progress:
            break

    return selected, object_categories, len(groups), matched_count


def validate_paths(source, output, count):
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if count <= 0:
        raise ValueError("--count must be greater than zero")
    if not source.is_dir():
        raise FileNotFoundError("source directory does not exist: {}".format(source))
    if source == output:
        raise ValueError("--source and --output must be different directories")
    if source in output.parents:
        raise ValueError("--output must not be inside --source")
    return source, output


def prepare_output(output, overwrite):
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                "output already exists; choose another path or pass --overwrite: {}".format(output)
            )
        if not output.is_dir():
            raise NotADirectoryError("output exists but is not a directory: {}".format(output))
        shutil.rmtree(str(output))
    output.mkdir(parents=True)


def copy_objects(selected, output, mode, categories):
    copied = []
    for index, source_path in enumerate(selected, start=1):
        destination = output / source_path.name
        if mode == "files":
            shutil.copy2(str(source_path), str(destination))
        else:
            shutil.copytree(str(source_path), str(destination), symlinks=True)
        object_name = source_path.stem if mode == "files" else source_path.name
        copied.append(
            {
                "index": index,
                "object": object_name,
                "category": categories[source_path],
                "source": str(source_path),
                "destination": str(destination),
            }
        )
        print("[{}/{}] {} -> {}".format(index, len(selected), source_path, destination))
    return copied


def main():
    args = parse_args()
    source, output = validate_paths(args.source, args.output, args.count)
    try:
        category_pattern = re.compile(args.category_regex) if args.category_regex else None
    except re.error as error:
        raise ValueError("invalid --category-regex: {}".format(error))

    mode, objects = discover_objects(source, args.mode)
    if not objects:
        description = "NPY files" if mode == "files" else "object directories containing NPY files"
        raise FileNotFoundError("no {} found directly under {}".format(description, source))

    selected, categories, available_categories, matched_count = select_diverse(
        objects, min(args.count, len(objects)), args.seed, category_pattern
    )
    selected_categories = len({categories[path] for path in selected})
    print(
        "Found {} objects in {} mode and {} inferred categories; selecting {} objects "
        "from {} categories (seed {}).".format(
            len(objects), mode, available_categories, len(selected), selected_categories, args.seed
        )
    )
    if len(objects) < args.count:
        print(
            "Warning: requested {} objects, but only {} are available; all will be selected.".format(
                args.count, len(objects)
            ),
            file=sys.stderr,
        )
    if category_pattern is not None and matched_count < len(objects):
        print(
            "Warning: category regex did not match {} object names; each unmatched name was "
            "treated as its own category.".format(len(objects) - matched_count),
            file=sys.stderr,
        )

    if args.dry_run:
        for index, path in enumerate(selected, start=1):
            print(
                "[{}/{}] category={} object={}".format(
                    index, len(selected), categories[path], path.name
                )
            )
        print("Dry run complete; no files were copied.")
        return

    prepare_output(output, args.overwrite)
    copied = copy_objects(selected, output, mode, categories)
    manifest = {
        "source": str(source),
        "output": str(output),
        "mode": mode,
        "requested_count": args.count,
        "selected_count": len(copied),
        "seed": args.seed,
        "category_regex": args.category_regex,
        "available_object_count": len(objects),
        "available_category_count": available_categories,
        "selected_category_count": selected_categories,
        "objects": copied,
    }
    manifest_path = output / MANIFEST_NAME
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print("Done. Selection manifest: {}".format(manifest_path))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, NotADirectoryError, ValueError) as error:
        print("Error: {}".format(error), file=sys.stderr)
        sys.exit(2)
