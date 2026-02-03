import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T
from linear_probe import LabeledImageFolder, find_latest_checkpoint

# optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def parse_args():
    p = argparse.ArgumentParser()
    # Default paths within the repo; change here if your layout differs
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_fruit_root = os.path.join(os.path.dirname(__file__), 'train')
    default_ckpt_dir = os.path.join(repo_dir, 'checkpoints', 'simclr_medium_run_50ep')
    default_out_dir = os.path.join(os.path.dirname(__file__), 'embeddings_out')

    p.add_argument('--fruit-root', type=str, default=default_fruit_root,
                   help=f'Path to training images (default: {default_fruit_root})')
    p.add_argument('--ckpt-dir', type=str, default=default_ckpt_dir,
                   help=f'Checkpoint directory (default: {default_ckpt_dir})')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out-dir', type=str, default=default_out_dir,
                   help=f'Output directory for embeddings (default: {default_out_dir})')
    p.add_argument('--queries-per-class', type=int, default=2)
    return p.parse_args()


def load_encoder(ckpt_dir, device):
    ckpt_path = find_latest_checkpoint(ckpt_dir)
    if ckpt_path is None:
        raise RuntimeError(f"No encoder checkpoint found in {ckpt_dir}")
    print('Loading encoder from', ckpt_path)
    data = torch.load(ckpt_path, map_location='cpu')
    from models import get_backbone
    model = get_backbone(pretrained=False)
    model.load_state_dict(data['model_state'], strict=False)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = args.out_dir or args.ckpt_dir
    os.makedirs(out_dir, exist_ok=True)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    dataset = LabeledImageFolder(args.fruit_root, transform=transform)
    print(f"Found {len(dataset)} images, {len(dataset.class_to_idx)} classes")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    encoder = load_encoder(args.ckpt_dir, device)

    embeddings = []
    labels = []
    # extract paths in dataset order
    paths = [p for p, _ in dataset.samples]

    with torch.no_grad():
        iterator = tqdm(loader, desc="Extracting embeddings", unit="batch") if tqdm is not None else loader
        for imgs, labs in iterator:
            imgs = imgs.to(device)
            feats = encoder.encoder(imgs).view(imgs.size(0), -1)
            feats = feats.cpu().numpy()
            embeddings.append(feats)
            labels.extend(labs.numpy().tolist())

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    # save embeddings and metadata
    emb_path = os.path.join(out_dir, 'embeddings.npy')
    labels_path = os.path.join(out_dir, 'labels.npy')
    paths_path = os.path.join(out_dir, 'paths.txt')

    np.save(emb_path, embeddings)
    np.save(labels_path, labels)
    with open(paths_path, 'w', encoding='utf-8') as f:
        for p in paths:
            f.write(p + '\n')

    print(f"Saved embeddings ({embeddings.shape}) to {emb_path}")
    print(f"Saved labels to {labels_path}")
    print(f"Saved paths to {paths_path}")

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    norm_emb = embeddings / norms

    # choose query indices: sample queries_per_class per class
    idx_by_class = {}
    for idx, lbl in enumerate(labels):
        idx_by_class.setdefault(int(lbl), []).append(idx)

    queries = []
    for cls, idxs in idx_by_class.items():
        n = min(len(idxs), args.queries_per_class)
        # pick evenly spaced or random
        if n == len(idxs):
            sel = idxs
        else:
            sel = np.random.choice(idxs, size=n, replace=False).tolist()
        queries.extend(sel)

    # compute pairwise similarity matrix
    sim = np.dot(norm_emb, norm_emb.T)

    # for each query, print top-5 neighbors (exclude self)
    k = 6  # includes self, will drop
    out_lines = []
    for q in queries:
        sims = sim[q]
        # argsort descending
        nn_idx = np.argsort(-sims)[:k]
        # remove self
        nn_idx = [i for i in nn_idx if i != q][:5]
        qpath = paths[q]
        qlabel = labels[q]
        out_lines.append(f"Query idx={q} label={qlabel} path={qpath}")
        for rank, nid in enumerate(nn_idx, start=1):
            out_lines.append(f"  {rank}. idx={nid} label={labels[nid]} path={paths[nid]} score={sims[nid]:.4f}")
        out_lines.append('')

    out_txt = os.path.join(out_dir, 'nn_results.txt')
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines))

    print(f"Saved nearest-neighbour results to {out_txt}")
    print('\n'.join(out_lines[:40]))


if __name__ == '__main__':
    main()
