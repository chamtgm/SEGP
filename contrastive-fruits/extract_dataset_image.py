#!/usr/bin/env python3
"""
Scan a folder recursively and copy all .jpg/.jpeg files into a destination folder.
Default source is the attached folder path; destination defaults to a new folder named
"I03_iPhone7" in the current working directory.
"""
import os
import shutil
import argparse
from pathlib import Path


def unique_dest(dest: Path, name: str) -> Path:
    candidate = dest / name
    if not candidate.exists():
        return candidate
    stem = Path(name).stem
    suffix = Path(name).suffix
    i = 1
    while True:
        candidate = dest / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def copy_jpgs(src: str, dst: str) -> int:
    src_p = Path(src)
    if not src_p.exists():
        raise FileNotFoundError(f"Source path not found: {src}")
    dst_p = Path(dst)
    dst_p.mkdir(parents=True, exist_ok=True)

    exts = {'.jpg', '.jpeg', '.heic'}
    count = 0
    for p in src_p.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts:
            dest_path = dst_p / p.name
            if dest_path.exists():
                dest_path = unique_dest(dst_p, p.name)
            shutil.copy2(p, dest_path)
            count += 1
    return count


if __name__ == '__main__':
    default_src = r"E:\FloreView_Dataset\D37_Apple_iPhone12"
    parser = argparse.ArgumentParser(description='Copy all JPG/JPEG files from source to destination folder')
    parser.add_argument('source', nargs='?', default=default_src, help='Source folder to scan (default: attached folder)')
    parser.add_argument('dest', nargs='?', default=os.path.join(os.getcwd(), 'D37_Apple_iPhone12'), help='Destination folder (default: ./D37_Apple_iPhone12)')
    args = parser.parse_args()

    try:
        n = copy_jpgs(args.source, args.dest)
        print(f"Copied {n} JPG/JPEG files to: {args.dest}")
    except Exception as e:
        print(f"Error: {e}")
        raise
