import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from linear_probe import LabeledImageFolder, find_latest_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fruit-root', type=str, required=True)
    p.add_argument('--ckpt-dir', type=str, required=True)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output-dir', type=str, default=None)
    return p.parse_args()


def load_encoder_and_classifier(ckpt_dir, device, num_classes, feature_dim):
    # find encoder checkpoint (training ckpt)
    enc_ckpt = find_latest_checkpoint(ckpt_dir)
    if enc_ckpt is None:
        raise RuntimeError(f"No encoder checkpoint found in {ckpt_dir}")
    enc_data = torch.load(enc_ckpt, map_location='cpu')

    from models import get_backbone
    model = get_backbone(pretrained=False)
    model.load_state_dict(enc_data['model_state'], strict=False)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # load classifier best file
    classifier_path = os.path.join(ckpt_dir, 'linear_probe_best.pt')
    if not os.path.isfile(classifier_path):
        raise RuntimeError(f"No linear_probe_best.pt found in {ckpt_dir}; run linear_probe first.")
    cls_data = torch.load(classifier_path, map_location='cpu')

    classifier = nn.Linear(feature_dim, num_classes)
    classifier.load_state_dict(cls_data['classifier_state'])
    classifier = classifier.to(device)
    classifier.eval()

    return model, classifier


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = args.output_dir or args.ckpt_dir
    os.makedirs(out_dir, exist_ok=True)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    dataset = LabeledImageFolder(args.fruit_root, transform=transform)
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    print(f"Dataset: {len(dataset)} images, {num_classes} classes")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # load encoder and classifier
    # determine feature dim
    from models import get_backbone
    tmp_model = get_backbone(pretrained=False)
    with torch.no_grad():
        dummy = torch.zeros(1,3,224,224)
        feat_dim = tmp_model.encoder(dummy).view(1, -1).shape[1]

    encoder, classifier = load_encoder_and_classifier(args.ckpt_dir, device, num_classes, feat_dim)

    # compute predictions
    conf = np.zeros((num_classes, num_classes), dtype=int)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            feats = encoder.encoder(imgs).view(imgs.size(0), -1)
            logits = classifier(feats)
            preds = logits.argmax(dim=1)
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                conf[t, p] += 1
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # per-class metrics
    precisions = []
    recalls = []
    f1s = []
    supports = conf.sum(axis=1)
    for i in range(num_classes):
        tp = conf[i, i]
        fp = conf[:, i].sum() - tp
        fn = conf[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    # print table
    print('\nPer-class metrics:')
    print('class\tcount\tprecision\trecall\tf1')
    for i in range(num_classes):
        print(f"{idx_to_class[i]}\t{supports[i]}\t{precisions[i]:.4f}\t{recalls[i]:.4f}\t{f1s[i]:.4f}")

    # overall accuracy
    acc = np.trace(conf) / conf.sum() if conf.sum() > 0 else 0.0
    print(f"\nOverall accuracy: {acc:.4f}")

    # save confusion matrix plot
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(conf, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # tick labels
    classes = [idx_to_class[i] for i in range(num_classes)]
    ax.set(xticks=np.arange(num_classes), yticks=np.arange(num_classes), xticklabels=classes, yticklabels=classes,
           ylabel='True label', xlabel='Predicted label', title='Confusion matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # annotate
    thresh = conf.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, format(conf[i, j], 'd'), ha='center', va='center', color='white' if conf[i, j] > thresh else 'black')

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'confusion_matrix.png')
    fig.savefig(out_path)
    print(f"Saved confusion matrix to {out_path}")

    # save metrics CSV
    csv_path = os.path.join(out_dir, 'per_class_metrics.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('class,support,precision,recall,f1\n')
        for i in range(num_classes):
            f.write(f"{idx_to_class[i]},{supports[i]},{precisions[i]:.4f},{recalls[i]:.4f},{f1s[i]:.4f}\n")
    print(f"Saved per-class metrics to {csv_path}")


if __name__ == '__main__':
    main()
