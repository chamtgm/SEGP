import os
import argparse

def find_images(root_dir):
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    matches = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                matches.append(os.path.join(root, f))
    return matches


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--real-root', required=True)
    p.add_argument('--cf-root', required=True)
    args = p.parse_args()

    real_root = args.real_root
    cf_root = args.cf_root

    real_paths = find_images(real_root)
    cf_paths = find_images(cf_root)

    print(f"Found {len(real_paths)} real images and {len(cf_paths)} counterfactual images")

    cf_by_basename = {os.path.basename(p): p for p in cf_paths}
    cf_by_class = {}
    for p in cf_paths:
        rel = os.path.relpath(p, cf_root)
        parts = rel.split(os.sep)
        if len(parts) > 0:
            cls = parts[0]
            cf_by_class.setdefault(cls, []).append(p)

    basename_count = 0
    relpath_count = 0
    fallback_count = 0
    skipped = []

    for rp in real_paths:
        basename = os.path.basename(rp)
        if basename in cf_by_basename:
            basename_count += 1
            continue
        rel = os.path.relpath(rp, real_root)
        cand = os.path.join(cf_root, rel)
        if os.path.exists(cand):
            relpath_count += 1
            continue
        parts = rel.split(os.sep)
        cls = parts[0] if len(parts) > 0 else None
        if cls in cf_by_class and len(cf_by_class[cls]) > 0:
            fallback_count += 1
        else:
            skipped.append(rp)

    print(f"Matches by basename: {basename_count}")
    print(f"Matches by relpath: {relpath_count}")
    print(f"Matches by fallback (same class random): {fallback_count}")
    print(f"Skipped (no CF found): {len(skipped)}")
    if len(skipped) > 0:
        print('\nSample skipped real images:')
        for s in skipped[:20]:
            print(' -', s)


if __name__ == '__main__':
    main()
