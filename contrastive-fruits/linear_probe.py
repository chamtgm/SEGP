import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import precision_score, recall_score, f1_score
from models import get_backbone
from utils import find_images

# optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def find_latest_checkpoint(ckpt_dir):
    """Find the latest training checkpoint file containing a 'model_state' key.
    Preference order:
    1. Any file whose name contains 'ckpt' (e.g., 'ckpt_epoch_*.pt'), newest first.
    2. Otherwise, scan .pt files (newest first) and return the first that contains 'model_state'.
    Returns full path or None if nothing suitable found.
    """
    # if user passed a specific file path
    if os.path.isfile(ckpt_dir) and ckpt_dir.endswith('.pt'):
        return ckpt_dir

    if not os.path.isdir(ckpt_dir):
        return None

    pts = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
    if not pts:
        return None

    # prefer files that look like training checkpoints
    ckpt_like = [p for p in pts if 'ckpt' in os.path.basename(p).lower()]
    if ckpt_like:
        pts_sorted = sorted(ckpt_like, key=lambda p: os.path.getmtime(p), reverse=True)
        return pts_sorted[0]

    # fallback: inspect files for the expected key
    pts_sorted = sorted(pts, key=lambda p: os.path.getmtime(p), reverse=True)
    for p in pts_sorted:
        try:
            data = torch.load(p, map_location='cpu')
            if isinstance(data, dict) and 'model_state' in data:
                return p
        except Exception:
            # ignore unreadable/corrupt files
            continue

    return None


class LabeledImageFolder(Dataset):
    """Simple dataset: scans immediate subfolders of root as classes."""
    def __init__(self, root, transform=None, class_to_idx=None):
        self.root = root
        self.transform = transform
        self.samples = []
        # allow passing an explicit class->index mapping so train/val stay aligned
        if class_to_idx is not None:
            self.class_to_idx = dict(class_to_idx)
            # only include classes that exist under this root
            for c, idx in sorted(self.class_to_idx.items(), key=lambda x: x[1]):
                class_dir = os.path.join(root, c)
                if not os.path.isdir(class_dir):
                    continue
                for p in find_images(class_dir):
                    self.samples.append((p, idx))
        else:
            self.class_to_idx = {}
            # classes are subfolders under root
            classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            classes = sorted(classes)
            for i, c in enumerate(classes):
                self.class_to_idx[c] = i
                class_dir = os.path.join(root, c)
                for p in find_images(class_dir):
                    self.samples.append((p, i))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {root}. Make sure the folder contains class subfolders.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


class SubsetWithTransform(Dataset):
    """Wrap a Subset (from random_split) and apply a transform per-split."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        # expose commonly used attributes to be compatible with existing code
        self.indices = getattr(subset, 'indices', None)
        self.dataset = getattr(subset, 'dataset', None)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fruit-root', type=str, required=True)
    p.add_argument('--train-root', type=str, default=None, help='Optional: path to a train folder with class subfolders')
    p.add_argument('--val-root', type=str, default=None, help='Optional: path to a val folder with class subfolders')
    p.add_argument('--ckpt-dir', type=str, required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=5e-3)
    p.add_argument('--val-split', type=float, default=0.2, help='Fraction of data to use for validation when no val folder provided')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--balance', type=str, default='weighted', choices=['none', 'weighted'], help='Use class-balanced sampling for training (weighted sampler)')
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (used for sampler)')
    p.add_argument('--num-workers', type=int, default=4, help='Number of subprocesses to use for data loading')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt_path = find_latest_checkpoint(args.ckpt_dir)
    if ckpt_path is None:
        raise RuntimeError(f"No checkpoint found in {args.ckpt_dir}")
    print("Using checkpoint:", ckpt_path)
    # determine directory to save linear-probe best checkpoint
    if os.path.isdir(args.ckpt_dir):
        save_dir = args.ckpt_dir
    else:
        # if args.ckpt_dir was a file path or a file-containing folder, use the ckpt file's directory
        save_dir = os.path.dirname(ckpt_path) or os.getcwd()
    os.makedirs(save_dir, exist_ok=True)

    # Training Transform (Matches SimCLR Protocol)
    train_transform = T.Compose([
        T.RandomResizedCrop(224),       # Randomly crop and resize
        T.RandomHorizontalFlip(),       # Randomly flip left/right
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # Validation Transform (Standard Evaluation)
    val_transform = T.Compose([
        T.Resize(256),                  # Resize slightly larger
        T.CenterCrop(224),              # Crop the center
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # Support either: separate train/val folders (auto-detected or provided), or single folder + random split
    train_ds = None
    val_ds = None
    dataset = None

    # explicit train/val args take precedence
    if args.train_root or args.val_root:
        if not args.train_root or not args.val_root:
            raise RuntimeError('Both --train-root and --val-root must be provided when using explicit split folders')
        train_ds = LabeledImageFolder(args.train_root, transform=train_transform)
        # align val labels to train mapping
        val_ds = LabeledImageFolder(args.val_root, transform=val_transform, class_to_idx=train_ds.class_to_idx)
        num_classes = len(train_ds.class_to_idx)
        print(f"Using explicit folders. Train images: {len(train_ds)}, Val images: {len(val_ds)}, {num_classes} classes")
    else:
        # auto-detect 'train'/'val' subfolders under fruit_root
        train_dir = os.path.join(args.fruit_root, 'train')
        val_dir = os.path.join(args.fruit_root, 'val')
        if os.path.isdir(train_dir) and os.path.isdir(val_dir):
            train_ds = LabeledImageFolder(train_dir, transform=train_transform)
            val_ds = LabeledImageFolder(val_dir, transform=val_transform, class_to_idx=train_ds.class_to_idx)
            num_classes = len(train_ds.class_to_idx)
            print(f"Auto-detected train/val. Train images: {len(train_ds)}, Val images: {len(val_ds)}, {num_classes} classes")
        else:
            # single folder: build dataset and random-split according to --val-split
            # create base dataset without transform so we can apply different transforms
            dataset = LabeledImageFolder(args.fruit_root, transform=None)
            num_classes = len(dataset.class_to_idx)
            print(f"Found {len(dataset)} images, {num_classes} classes")
            val_size = int(len(dataset) * args.val_split)
            train_size = len(dataset) - val_size
            train_subset, val_subset = random_split(dataset, [train_size, val_size])
            train_ds = SubsetWithTransform(train_subset, transform=train_transform)
            val_ds = SubsetWithTransform(val_subset, transform=val_transform)

    # Optionally use class-balanced sampling to oversample minority classes
    train_loader = None
    if args.balance == 'weighted':
        # compute sample weights from the training dataset
        import numpy as _np
        counts = _np.zeros(num_classes, dtype=int)
        # when train_ds is a Subset (random_split), use the original dataset.samples
        if dataset is not None:
            samples_for_counts = dataset.samples
        else:
            # train_ds is a Dataset instance with .samples
            samples_for_counts = getattr(train_ds, 'samples', [])
        for _, lbl in samples_for_counts:
            counts[int(lbl)] += 1
        # weight per class = inverse frequency
        class_weights = 1.0 / (counts + 1e-6)
        if dataset is not None:
            sample_weights = _np.array([class_weights[int(lbl)] for _, lbl in dataset.samples])
        else:
            # map train subset indices to original sample indices if Subset
            sample_weights = _np.array([class_weights[int(lbl)] for _, lbl in getattr(train_ds, 'samples', [])])

        # for the train subset, extract weights for its indices
        train_indices = train_ds.indices if hasattr(train_ds, 'indices') else list(range(len(train_ds)))
        # if train_ds is a Subset, indices are indices into original dataset
        # map train subset indices to original sample indices
        if hasattr(train_ds, 'indices'):
            weights_for_train = [sample_weights[i] for i in train_ds.indices]
        else:
            weights_for_train = sample_weights.tolist()

        from torch.utils.data import WeightedRandomSampler
        # use a seeded generator for reproducible sampling
        gen = torch.Generator()
        gen.manual_seed(int(args.seed))
        sampler = WeightedRandomSampler(weights_for_train, num_samples=len(weights_for_train), replacement=True, generator=gen)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
        print('Using weighted sampler for training. Class counts:', counts.tolist())
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        # print class counts for visibility when not using weighted sampler
        counts = [0] * num_classes
        if dataset is not None:
            for _, lbl in dataset.samples:
                counts[int(lbl)] += 1
        else:
            for _, lbl in getattr(train_ds, 'samples', []):
                counts[int(lbl)] += 1
        print('Class counts:', counts)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # load backbone and checkpoint
    model = get_backbone(pretrained=False)
    # load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # ckpt['model_state'] contains encoder+projector
    raw_state = ckpt.get('model_state', ckpt)
    
    # --- The Translator: Convert Triplet '0.' keys to 'encoder.' keys ---
    fixed_state = {}
    for k, v in raw_state.items():
        if k.startswith("0."):
            new_key = k.replace("0.", "encoder.", 1)
            fixed_state[new_key] = v
        else:
            fixed_state[k] = v

    # Load and verify!
    load_result = model.load_state_dict(fixed_state, strict=False)
    print("\n--- BACKBONE LOAD RESULT ---")
    print(f"Missing keys: {len(load_result.missing_keys)}")
    print(f"Unexpected keys: {len(load_result.unexpected_keys)}")
    print("----------------------------\n")

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # determine feature dim by sending a dummy through encoder
    with torch.no_grad():
        dummy = torch.zeros(1,3,224,224).to(device)
        feat = model.encoder(dummy)
        feat_dim = feat.view(1, -1).shape[1]
    print('Feature dim:', feat_dim)

    # classifier
    classifier = nn.Linear(feat_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-2)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_train_preds = []
        all_train_labels = []
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch} Train", unit="batch") if tqdm is not None else train_loader
        for imgs, labels in train_iter:
            imgs = imgs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                feats = model.encoder(imgs).view(imgs.size(0), -1)
            logits = classifier(feats)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        train_loss = running_loss / total
        train_acc = correct / total if total > 0 else 0.0
        train_prec = precision_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)
        train_rec = recall_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)
        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)

        # val
        classifier.eval()
        v_correct = 0
        v_total = 0
        all_val_preds = []
        all_val_labels = []
        val_iter = tqdm(val_loader, desc=f"Epoch {epoch} Val", unit="batch") if tqdm is not None else val_loader
        with torch.no_grad():
            for imgs, labels in val_iter:
                imgs = imgs.to(device)
                labels = labels.to(device)
                feats = model.encoder(imgs).view(imgs.size(0), -1)
                logits = classifier(feats)
                preds = logits.argmax(dim=1)
                v_correct += (preds == labels).sum().item()
                v_total += imgs.size(0)
                
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                
        val_acc = v_correct / v_total if v_total > 0 else 0.0
        val_prec = precision_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
        val_rec = recall_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_prec={val_prec:.4f} val_rec={val_rec:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'classifier_state': classifier.state_dict(), 'epoch': epoch, 'val_acc': val_acc}, os.path.join(save_dir, 'linear_probe_best.pt'))

    print('Done. Best val acc =', best_val_acc)


if __name__ == '__main__':
    main()
