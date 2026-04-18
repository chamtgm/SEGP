import argparse
import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Imports your model architecture just like your training scripts
from models import get_backbone

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train-root', type=str, required=True, help='Path to your training dataset folder (e.g., Datasets/train)')
    p.add_argument('--ckpt', type=str, required=True, help='Path to your trained backbone checkpoint')
    p.add_argument('--out', type=str, default='fruit_centroids.pt', help='Where to save the generated centroids')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    print("1. Setting up clean image transforms (No ColorJitter)...")
    # We want the pure, unaltered mathematical representation of the fruits
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"2. Loading dataset from {args.train_root}...")
    dataset = datasets.ImageFolder(args.train_root, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Store the exact names of your folders (e.g., 'Apple', 'Banana')
    class_names = dataset.classes 
    num_classes = len(class_names)
    print(f"   Found {len(dataset)} images across {num_classes} classes.")

    print("3. Loading Backbone with Parameter Mapping...")
    model = get_backbone(pretrained=False)
    
    # Load the checkpoint
    ckpt = torch.load(args.ckpt, map_location='cpu')
    raw_state = ckpt.get('model_state', ckpt.get('backbone_state', ckpt))
    
    # --- PARAMETER MAPPING ---
    # Safely converts '0.conv1.weight' -> 'encoder.conv1.weight'
    fixed_state = {}
    for k, v in raw_state.items():
        if k.startswith("0."):
            fixed_state[k.replace("0.", "encoder.", 1)] = v
        else:
            fixed_state[k] = v

    load_result = model.load_state_dict(fixed_state, strict=False)
    print(f"   Missing keys: {len(load_result.missing_keys)} (Should be 0 or only projection head keys)")
    
    model = model.to(device)
    model.eval()

    print("4. Calculating Master Centroids...")
    # We will store the sum of the embeddings and the count of images for each class
    class_sums = None
    class_counts = torch.zeros(num_classes).to(device)

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting Features"):
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Pass through the encoder
            try:
                feats = model.encoder(imgs).view(imgs.size(0), -1)
            except Exception:
                feats = model(imgs)

            # Initialize class_sums dynamically based on the observed feature dimension
            if class_sums is None:
                class_sums = torch.zeros(num_classes, feats.shape[1]).to(device)

            # L2 Normalize the features so lighting differences are minimized
            feats = F.normalize(feats, p=2, dim=1)

            # Add the features to their respective class sum
            for i in range(len(labels)):
                lbl = labels[i]
                class_sums[lbl] += feats[i]
                class_counts[lbl] += 1

    print("5. Averaging and Normalizing...")
    # Divide the sums by the counts to get the mathematical average (the centroid)
    centroids = class_sums / class_counts.unsqueeze(1)
    
    # Re-normalize the final master centroids
    centroids = F.normalize(centroids, p=2, dim=1)

    print(f"6. Saving 10 Master Centroids to {args.out}...")
    torch.save({
        'centroids': centroids.cpu(),
        'classes': class_names
    }, args.out)
    
    print("Done! Centroids are ready for deployment.")

if __name__ == '__main__':
    main()