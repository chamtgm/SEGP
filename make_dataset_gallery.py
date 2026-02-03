import argparse
import os
from PIL import Image
import math


def find_images(root):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    out = []
    for cls in sorted(os.listdir(root)):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for p in sorted(os.listdir(cls_dir)):
            if p.lower().endswith(exts):
                out.append((cls, os.path.join(cls_dir, p)))
    return out


def make_grid(rows, cols, thumb_w, thumb_h, padding=8):
    W = cols * thumb_w + (cols + 1) * padding
    H = rows * thumb_h + (rows + 1) * padding
    out = Image.new('RGB', (W, H), color=(255,255,255))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', type=str, required=True)
    p.add_argument('--out-dir', type=str, default='hvae_generated_gallery')
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--max-per-class', type=int, default=20, help='Max samples per class to include (unless --all-content).')
    p.add_argument('--all-content', action='store_true')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # collect class -> list(paths)
    classes = [d for d in sorted(os.listdir(args.dataset_root)) if os.path.isdir(os.path.join(args.dataset_root, d))]
    class_images = {}
    for c in classes:
        pth = os.path.join(args.dataset_root, c)
        imgs = [os.path.join(pth, f) for f in sorted(os.listdir(pth)) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff'))]
        class_images[c] = imgs

    if len(classes) == 0:
        raise RuntimeError('No class subfolders found under ' + args.dataset_root)

    # determine per-class count
    if args.all_content:
        per_class = max(len(v) for v in class_images.values())
    else:
        per_class = min(args.max_per_class, max(len(v) for v in class_images.values()))

    cols = len(classes)
    rows = per_class
    thumb_w = args.image_size
    thumb_h = args.image_size
    padding = 8

    grid = make_grid(rows, cols, thumb_w, thumb_h, padding=padding)

    # paste images: for each row r, for each class c
    for r in range(rows):
        for ci, c in enumerate(classes):
            imgs = class_images[c]
            if r < len(imgs) and (args.all_content or r < args.max_per_class):
                img_path = imgs[r]
                try:
                    im = Image.open(img_path).convert('RGB')
                    im = im.resize((thumb_w, thumb_h), Image.BILINEAR)
                except Exception:
                    im = Image.new('RGB', (thumb_w, thumb_h), color=(240,240,240))
            else:
                im = Image.new('RGB', (thumb_w, thumb_h), color=(255,255,255))

            x = padding + ci * (thumb_w + padding)
            y = padding + r * (thumb_h + padding)
            grid.paste(im, (x, y))

    grid_path = os.path.join(args.out_dir, 'dataset_gallery_grid.png')
    grid.save(grid_path)

    # write simple HTML
    html_path = os.path.join(args.out_dir, 'gallery.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('<html><body>\n')
        f.write('<h2>Dataset gallery: {} ({} classes)</h2>\n'.format(args.dataset_root, len(classes)))
        f.write('<p>Showing up to {} samples per class. To include all, re-run with --all-content.</p>\n'.format(args.max_per_class))
        f.write('<img src="{}" style="max-width:100%"/>\n'.format(os.path.basename(grid_path)))
        f.write('<hr/>\n')
        f.write('<ul>\n')
        for c in classes:
            f.write('<li><b>{}</b>: {} images</li>\n'.format(c, len(class_images[c])))
        f.write('</ul>\n')
        f.write('</body></html>')

    print('Wrote gallery:', grid_path, 'and', html_path)


if __name__ == '__main__':
    main()
