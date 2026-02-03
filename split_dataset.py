#!/usr/bin/env python3
"""
Split a dataset of class subfolders into train/eval/test sets.
Defaults: 75% train, 2.8% eval, 22.2% test.
Preserves class folder names and copies files into destination root under
`train/`, `eval/`, and `test/` subfolders.

Usage:
    python split_dataset.py <source_root> <dest_root> [--train 0.75 --eval 0.028 --test 0.222 --seed 42]

Examples:
    python split_dataset.py "D:/Study materials/Year 2/SEGP/Code/Apple dataset" "D:/datasets/apple_splits"

"""
from pathlib import Path
import argparse
import random
import shutil
import sys
from typing import Tuple

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.heic'}


def parse_args():
    p = argparse.ArgumentParser(description='Split dataset into train/eval/test preserving class folders')
    p.add_argument('source', help='Source dataset root (contains class subfolders)')
    p.add_argument('dest', help='Destination root where train/eval/test folders will be created')
    p.add_argument('--train', type=float, default=0.75, help='Train fraction (default: 0.75)')
    p.add_argument('--eval', type=float, default=0.028, help='Eval fraction (default: 0.028)')
    p.add_argument('--test', type=float, default=0.222, help='Test fraction (default: 0.222)')
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    p.add_argument('--move', action='store_true', help='Move files instead of copying')
    return p.parse_args()


def validate_fractions(train: float, eval_f: float, test: float) -> None:
    total = train + eval_f + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f'Fractions must sum to 1.0; got {total}')


def _gather_files(class_dir: Path):
    return [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def compute_counts(n: int, train_f: float, eval_f: float, test_f: float) -> Tuple[int,int,int]:
    if n == 0:
        return 0,0,0
    n_eval = int(round(n * eval_f))
    n_test = int(round(n * test_f))
    n_train = n - n_eval - n_test
    # Fix rounding issues if negative or off by 1
    if n_train < 0:
        # push negative into train by reducing eval or test
        deficit = -n_train
        if n_eval >= deficit:
            n_eval -= deficit
        elif n_test >= deficit:
            n_test -= deficit
        else:
            # as a last resort, set train to 0 and adjust others proportionally
            n_eval = int(round(n * eval_f))
            n_test = n - n_eval
        n_train = n - n_eval - n_test
    return n_train, n_eval, n_test


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def split_class(class_dir: Path, dest_root: Path, train_f: float, eval_f: float, test_f: float, rng: random.Random, move: bool) -> Tuple[int,int,int]:
    files = _gather_files(class_dir)
    rng.shuffle(files)
    n = len(files)
    n_train, n_eval, n_test = compute_counts(n, train_f, eval_f, test_f)

    train_dir = dest_root / 'train' / class_dir.name
    eval_dir = dest_root / 'eval' / class_dir.name
    test_dir = dest_root / 'test' / class_dir.name
    ensure_dir(train_dir)
    ensure_dir(eval_dir)
    ensure_dir(test_dir)

    idx = 0
    ops = shutil.move if move else shutil.copy2
    for i in range(n_train):
        src = files[idx]
        dst = train_dir / src.name
        if dst.exists():
            dst = unique_dest(dst.parent, src.name)
        ops(str(src), str(dst))
        idx += 1
    for i in range(n_eval):
        src = files[idx]
        dst = eval_dir / src.name
        if dst.exists():
            dst = unique_dest(dst.parent, src.name)
        ops(str(src), str(dst))
        idx += 1
    for i in range(n_test):
        src = files[idx]
        dst = test_dir / src.name
        if dst.exists():
            dst = unique_dest(dst.parent, src.name)
        ops(str(src), str(dst))
        idx += 1

    return n_train, n_eval, n_test


def unique_dest(dest_dir: Path, name: str) -> Path:
    base = Path(name).stem
    ext = Path(name).suffix
    i = 1
    candidate = dest_dir / f"{base}_{i}{ext}"
    while candidate.exists():
        i += 1
        candidate = dest_dir / f"{base}_{i}{ext}"
    return candidate


def main():
    args = parse_args()
    src = Path(args.source)
    dest = Path(args.dest)
    if not src.exists():
        print(f"Source path not found: {src}")
        sys.exit(1)
    validate_fractions(args.train, args.eval, args.test)
    rng = random.Random(args.seed)

    class_dirs = [p for p in sorted(src.iterdir()) if p.is_dir()]
    if not class_dirs:
        print('No class subfolders found in source; aborting')
        sys.exit(1)

    summary = []
    total_files = 0
    total_train = total_eval = total_test = 0

    for c in class_dirs:
        n_files = len(_gather_files(c))
        total_files += n_files
        n_train, n_eval, n_test = split_class(c, dest, args.train, args.eval, args.test, rng, args.move)
        summary.append((c.name, n_files, n_train, n_eval, n_test))
        total_train += n_train
        total_eval += n_eval
        total_test += n_test

    print('Split complete')
    print(f'Total files found: {total_files}')
    print(f'Train: {total_train}, Eval: {total_eval}, Test: {total_test}')
    print('\nPer-class breakdown:')
    for name, n_files, n_train, n_eval, n_test in summary:
        print(f' - {name}: total={n_files} -> train={n_train}, eval={n_eval}, test={n_test}')


if __name__ == '__main__':
    main()
