import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from fine_tune import LabeledImageFolder
from models import get_backbone

def parse_args():
    p = argparse.ArgumentParser(description="Generate a confusion matrix from a fine-tuned SimCLR model.")
    p.add_argument('--fruit-root', type=str, required=True, help='Path to test dataset (e.g., Datasets - Copy)')
    p.add_argument('--ckpt-path', type=str, required=True, help='Path to finetuned .pt model (e.g., ../fine_tuned_models/finetuned_best(CFSIMCLR).pt)')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output-path', type=str, default='confusion_matrix.png', help='Path to save the output plot')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Same validation transforms as fine_tune.py
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Loading dataset from: {args.fruit_root}")
    dataset = LabeledImageFolder(args.fruit_root, transform=transform)
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    print(f"Found {len(dataset)} images across {num_classes} classes.")
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0 if os.name == 'nt' else 4)
    
    # Initialize Backbone
    print("Initializing model...")
    model = get_backbone(pretrained=False).to(device)
    
    # Determine the feature dimension output by the encoder
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224).to(device)
        feat_dim = model.encoder(dummy).view(1, -1).shape[1]
        
    classifier = nn.Linear(feat_dim, num_classes).to(device)
    
    # Load Fine-tuned Checkpoint
    print(f"Loading weights from {args.ckpt_path}...")
    ckpt = torch.load(args.ckpt_path, map_location=device)
    
    if 'backbone_state' in ckpt and 'classifier_state' in ckpt:
        model.load_state_dict(ckpt['backbone_state'], strict=False)
        classifier.load_state_dict(ckpt['classifier_state'])
    else:
        raise ValueError("Checkpoint is missing 'backbone_state' or 'classifier_state'. Ensure you are using a model trained via fine_tune.py")
        
    model.eval()
    classifier.eval()
    
    # Confusion Matrix accumulator
    conf = np.zeros((num_classes, num_classes), dtype=int)
    
    print("Evaluating predictions... This may take a few moments.")
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feats = model.encoder(imgs).view(imgs.size(0), -1)
            logits = classifier(feats)
            preds = logits.argmax(dim=1)
            
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                conf[t, p] += 1

    # Plot Confusion Matrix using Matplotlib
    acc = np.trace(conf) / conf.sum() if conf.sum() > 0 else 0.0
    print(f"Overall Test Accuracy: {acc:.4f}")

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(conf, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, shrink=0.75)
    
    classes = [idx_to_class[i] for i in range(num_classes)]
    ax.set(xticks=np.arange(num_classes), yticks=np.arange(num_classes),
           xticklabels=classes, yticklabels=classes,
           ylabel='True Predict/Class', xlabel='Model Output', 
           title=f'Confusion Matrix (Acc: {acc:.1%})')
           
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Apply text annotations to the cells
    thresh = conf.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, format(conf[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if conf[i, j] > thresh else 'black')

    plt.tight_layout()
    fig.savefig(args.output_path, dpi=300)
    print(f"✅ Generated and saved high-res confusion matrix to {args.output_path}")

if __name__ == '__main__':
    main()
