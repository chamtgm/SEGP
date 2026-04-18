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

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

class LabeledImageFolder(Dataset):
    """Simple dataset: scans immediate subfolders of root as classes."""
    def __init__(self, root, transform=None, class_to_idx=None):
        self.root = root
        self.transform = transform
        self.samples = []
        if class_to_idx is not None:
            self.class_to_idx = dict(class_to_idx)
            for c, idx in sorted(self.class_to_idx.items(), key=lambda x: x[1]):
                class_dir = os.path.join(root, c)
                if not os.path.isdir(class_dir):
                    continue
                for p in find_images(class_dir):
                    self.samples.append((p, idx))
        else:
            self.class_to_idx = {}
            classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            classes = sorted(classes)
            for i, c in enumerate(classes):
                self.class_to_idx[c] = i
                class_dir = os.path.join(root, c)
                for p in find_images(class_dir):
                    self.samples.append((p, i))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {root}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fruit-root', type=str, required=True, help='Path to real fruit dataset with class folders')
    # NEW: We need paths to BOTH checkpoints
    p.add_argument('--simclr-ckpt', type=str, required=True, help='Path to your SimCLR backbone .pt file')
    p.add_argument('--probe-ckpt', type=str, required=True, help='Path to your best linear probe .pt file')
    
    p.add_argument('--save-dir', type=str, default='./finetune_checkpoints')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    
    # NEW: Differential Learning Rates
    p.add_argument('--lr-backbone', type=float, default=1e-4, help='Tiny LR for the pre-trained backbone')
    p.add_argument('--lr-classifier', type=float, default=1e-2, help='Larger LR for the classifier head')
    
    p.add_argument('--val-split', type=float, default=0.2)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # Standard Supervised Transforms (No SimCLR specific color distortions)
    train_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # Load Dataset (Real data only)
    dataset = LabeledImageFolder(args.fruit_root, transform=None)
    num_classes = len(dataset.class_to_idx)
    
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_ds = SubsetWithTransform(train_subset, transform=train_transform)
    val_ds = SubsetWithTransform(val_subset, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- 1. LOAD THE BACKBONE ---
    model = get_backbone(pretrained=False)
    simclr_ckpt = torch.load(args.simclr_ckpt, map_location='cpu')
    model.load_state_dict(simclr_ckpt['model_state'], strict=False)
    model = model.to(device)
    
    # CRITICAL: Do NOT freeze the backbone!
    for p in model.parameters():
        p.requires_grad = True

    # Get feature dimension
    with torch.no_grad():
        dummy = torch.zeros(1,3,224,224).to(device)
        feat_dim = model.encoder(dummy).view(1, -1).shape[1]

    # --- 2. LOAD THE LINEAR HEAD ---
    classifier = nn.Linear(feat_dim, num_classes).to(device)
    probe_ckpt = torch.load(args.probe_ckpt, map_location='cpu')
    classifier.load_state_dict(probe_ckpt['classifier_state'])
    
    for p in classifier.parameters():
        p.requires_grad = True

    # --- 3. DIFFERENTIAL OPTIMIZER ---
    # We pass both the backbone and the classifier to the optimizer, but with different LRs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': model.parameters(), 'lr': args.lr_backbone},
        {'params': classifier.parameters(), 'lr': args.lr_classifier}
    ], momentum=0.9, weight_decay=1e-4)

    # --- 4. TRAINING LOOP ---
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()      # Set backbone to train mode (updates BatchNorm layers)
        classifier.train() # Set classifier to train mode
        
        running_loss = 0.0
        correct, total = 0, 0
        all_train_preds = []
        all_train_labels = []
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch} Train") if tqdm is not None else train_loader
        
        for imgs, labels in train_iter:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Forward pass through the whole unfrozen model
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

        # --- 5. VALIDATION LOOP ---
        model.eval()
        classifier.eval()
        v_correct, v_total = 0, 0
        all_val_preds = []
        all_val_labels = []
        val_iter = tqdm(val_loader, desc=f"Epoch {epoch} Val") if tqdm is not None else val_loader
        
        with torch.no_grad():
            for imgs, labels in val_iter:
                imgs, labels = imgs.to(device), labels.to(device)
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
            # Save the fully fine-tuned model (both backbone and classifier)
            torch.save({
                'backbone_state': model.state_dict(),
                'classifier_state': classifier.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc
            }, os.path.join(args.save_dir, 'finetuned_best.pt'))

    print(f'Fine-tuning Complete! Best val acc = {best_val_acc:.4f}')

if __name__ == '__main__':
    main()