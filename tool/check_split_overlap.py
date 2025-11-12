#!/usr/bin/env python3
"""
Utility script to inspect val/test JSON files and report duplicated subject IDs.

Usage:
    python tool/check_split_overlap.py \
        --train-json path/to/train.json \
        --val-json path/to/val.json \
        --test-json path/to/test.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

SUB_EQU_PATTERN = re.compile(r"Sub[_-]?(\d+).*?Equ", re.IGNORECASE)


def _load_json(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected a dict at top-level in {path}, got {type(data)}")
    normalized: Dict[str, List[str]] = {}
    for key, values in data.items():
        if not isinstance(values, list):
            raise ValueError(f"Expected list for key '{key}' in {path}, got {type(values)}")
        normalized[str(key)] = [str(v) for v in values]
    return normalized


def _extract_subject(value: str) -> str | None:
    if value is None:
        return None
    match = SUB_EQU_PATTERN.search(str(value))
    return match.group(1) if match else None


def _find_internal_duplicates(split_dict: Mapping[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
    duplicates: Dict[str, Dict[str, List[str]]] = {}
    for key, entries in split_dict.items():
        subjects = defaultdict(list)
        for entry in entries:
            sid = _extract_subject(entry)
            if sid:
                subjects[sid].append(entry)
        dup = {sid: vals for sid, vals in subjects.items() if len(vals) > 1}
        if dup:
            duplicates[key] = dup
    return duplicates


def _find_pair_overlap(
    left: Mapping[str, List[str]], right: Mapping[str, List[str]]
) -> Dict[str, List[str]]:
    overlap: Dict[str, List[str]] = {}
    shared_keys = left.keys() & right.keys()
    for key in shared_keys:
        left_subjects = {_extract_subject(v) for v in left[key]}
        right_subjects = {_extract_subject(v) for v in right[key]}
        left_subjects.discard(None)
        right_subjects.discard(None)
        dup = sorted(left_subjects & right_subjects)
        if dup:
            overlap[key] = dup
    return overlap


def _find_triple_overlap(
    first: Mapping[str, List[str]],
    second: Mapping[str, List[str]],
    third: Mapping[str, List[str]],
) -> Dict[str, List[str]]:
    overlap: Dict[str, List[str]] = {}
    shared_keys = first.keys() & second.keys() & third.keys()
    for key in shared_keys:
        subjects = [
            {_extract_subject(v) for v in first[key]},
            {_extract_subject(v) for v in second[key]},
            {_extract_subject(v) for v in third[key]},
        ]
        for s in subjects:
            s.discard(None)
        dup = sorted(set.intersection(*subjects)) if all(subjects) else []
        if dup:
            overlap[key] = dup
    return overlap


def _print_duplicates(title: str, data: Mapping[str, Mapping[str, List[str]]]) -> None:
    if not data:
        print(f"[OK] No duplicates found within {title}.")
        return
    print(f"[WARN] Duplicates inside {title}:")
    for key, subject_map in sorted(data.items()):
        print(f"  - {key}:")
        for sid, entries in sorted(subject_map.items()):
            print(f"      Sub ID {sid} appears {len(entries)} times: {entries}")


def _print_overlap(label: str, overlap: Mapping[str, List[str]]) -> None:
    if not overlap:
        print(f"[OK] No overlapping Sub IDs between {label}.")
        return
    print(f"[WARN] Subjects appearing in {label}:")
    for key, subject_ids in sorted(overlap.items()):
        print(f"  - {key}: {', '.join(subject_ids)}")


def _print_compact_summary(
    train_val: Mapping[str, List[str]],
    train_test: Mapping[str, List[str]],
    val_test: Mapping[str, List[str]],
    all_three: Mapping[str, List[str]],
    split_keys: Mapping[str, List[str]] | None = None,
) -> None:
    key_pool = set(train_val) | set(train_test) | set(val_test) | set(all_three)
    if split_keys:
        key_pool |= set(split_keys)
    all_keys = sorted(key_pool)
    if not all_keys:
        print("No overlapping Sub IDs across train/val/test.")
        return
    for key in all_keys:
        tv = train_val.get(key, [])
        tt = train_test.get(key, [])
        vt = val_test.get(key, [])
        aa = all_three.get(key, [])
        print(key)
        if not any((tv, tt, vt, aa)):
            print("  No overlaps across train/val/test.")
        else:
            print(f"  train-val : {', '.join(tv) if tv else '(none)'}")
            print(f"  train-test: {', '.join(tt) if tt else '(none)'}")
            print(f"  val-test  : {', '.join(vt) if vt else '(none)'}")
            print(f"  all-three : {', '.join(aa) if aa else '(none)'}")
        print("-")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Sub/Equ subject overlap in train/val/test JSON files.")
    parser.add_argument("--train-json", default="dataset/split2/class/digital_camera/1_trainset_info.json", type=Path, help="Path to *_trainset_info.json")
    parser.add_argument("--val-json", default="dataset/split2/class/digital_camera/1_valset_info.json", type=Path, help="Path to *_valset_info.json")
    parser.add_argument("--test-json", default="dataset/split2/class/digital_camera/1_testset_info.json", type=Path, help="Path to *_testset_info.json")
    parser.add_argument(
        "--compact",
        action="store_false",
        help="Print only overlapping Sub IDs per key (train/val/test intersections).",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    train_dict = _load_json(args.train_json)
    val_dict = _load_json(args.val_json)
    test_dict = _load_json(args.test_json)

    train_dups = _find_internal_duplicates(train_dict)
    val_dups = _find_internal_duplicates(val_dict)
    test_dups = _find_internal_duplicates(test_dict)
    overlap_train_val = _find_pair_overlap(train_dict, val_dict)
    overlap_train_test = _find_pair_overlap(train_dict, test_dict)
    overlap_val_test = _find_pair_overlap(val_dict, test_dict)
    overlap_all = _find_triple_overlap(train_dict, val_dict, test_dict)
    split_keys = set(train_dict) | set(val_dict) | set(test_dict)

    if args.compact:
        _print_compact_summary(
            overlap_train_val,
            overlap_train_test,
            overlap_val_test,
            overlap_all,
            split_keys=split_keys,
        )
    else:
        _print_duplicates("train split", train_dups)
        _print_duplicates("val split", val_dups)
        _print_duplicates("test split", test_dups)
        _print_overlap("train & val", overlap_train_val)
        _print_overlap("train & test", overlap_train_test)
        _print_overlap("val & test", overlap_val_test)
        _print_overlap("train & val & test", overlap_all)

    has_issue = bool(
        train_dups
        or val_dups
        or test_dups
        or overlap_train_val
        or overlap_train_test
        or overlap_val_test
        or overlap_all
    )
    return 1 if has_issue else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
