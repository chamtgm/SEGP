import os
import sys
import argparse
from pathlib import Path

# ensure project root is on path so SimCLR_model imports correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms, datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from SimCLR_model import SimCLRNet

# In-code configuration (set hyperparameters and checkpoint path here)
CONFIG = {
    'checkpoint': r"D:/Study materials/Year 2/SEGP/Code/checkpoints/simclr_medium_run_50ep/simclr_epoch200.pt",
    'data_dir': r"D:/Study materials/Year 2/SEGP/Code/Dataset, Angle Variable/Dataset, Angle Variable",
    'device': 'cuda',
    'batch_size': 64,
    'epochs': 20,            # linear probe epochs
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'seed': 42,
}


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = None
    if isinstance(ckpt, dict):
        # common keys
        for key in ("state_dict", "model", "model_state_dict"):
            if key in ckpt:
                state = ckpt[key]
                break
        if state is None:
            state = ckpt
    else:
        state = ckpt

    # handle DataParallel prefixes
    new_state = {}
    for k, v in state.items():
        nk = k.replace('module.', '')
        new_state[nk] = v
    try:
        model.load_state_dict(new_state, strict=False)
    except Exception as e:
        print('Warning: load_state_dict(strict=False) failed:', e)
    return model


def make_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def extract_embeddings(model, loader, device):
    model.eval()
    embs = []
    labels = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc='embed'):
            x = x.to(device)
            h, _ = model(x, return_feat=True)
            embs.append(h.cpu().numpy())
            labels.append(y.numpy())
    embs = np.concatenate(embs, axis=0)
    labels = np.concatenate(labels, axis=0)
    return embs, labels


def train_linear_probe(encoder, train_loader, val_loader, dim, n_classes, device,
                       epochs=20, lr=1e-3, batch_size=64, weight_decay=1e-4, seed=42):
    torch.manual_seed(seed)
    classifier = nn.Linear(dim, n_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_acc": []}

    encoder.eval()
    for ep in range(epochs):
        classifier.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                h, _ = encoder(x, return_feat=True)
            logits = classifier(h)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        mean_loss = float(np.mean(losses))
        # validation accuracy
        classifier.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                h, _ = encoder(x, return_feat=True)
                logits = classifier(h)
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                trues.append(y.numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        acc = float(accuracy_score(trues, preds))
        history['train_loss'].append(mean_loss)
        history['val_acc'].append(acc)
        print(f'Epoch {ep+1}/{epochs} loss={mean_loss:.4f} val_acc={acc:.4f}')

    return classifier, history


def knn_classify(train_embs, train_labels, test_embs, k=20):
    # cosine similarity
    train_norm = train_embs / np.linalg.norm(train_embs, axis=1, keepdims=True)
    test_norm = test_embs / np.linalg.norm(test_embs, axis=1, keepdims=True)
    sims = test_norm.dot(train_norm.T)
    topk = np.argsort(-sims, axis=1)[:, :k]
    preds = []
    for inds in topk:
        vals = train_labels[inds]
        # majority vote
        uniq, counts = np.unique(vals, return_counts=True)
        preds.append(uniq[np.argmax(counts)])
    return np.array(preds)


def retrieval_precision_at_k(train_embs, train_labels, query_embs, query_labels, ks=(1,5,10)):
    train_norm = train_embs / np.linalg.norm(train_embs, axis=1, keepdims=True)
    query_norm = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)
    sims = query_norm.dot(train_norm.T)
    inds = np.argsort(-sims, axis=1)
    results = {}
    for k in ks:
        correct = 0
        for i in range(len(query_labels)):
            topk = inds[i, :k]
            if query_labels[i] in train_labels[topk]:
                correct += 1
        results[f'p@{k}'] = correct / len(query_labels)
    return results


def main():
    parser = argparse.ArgumentParser()
    # keep CLI optional: when omitted, use CONFIG values
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    # resolve configuration: CLI overrides CONFIG
    ckpt_path = args.checkpoint or CONFIG['checkpoint']
    data_root = args.data_dir or CONFIG['data_dir']
    device_str = args.device or CONFIG['device']
    batch_size = args.batch_size or CONFIG['batch_size']
    probe_epochs = args.epochs or CONFIG['epochs']
    lr = args.lr or CONFIG['lr']
    weight_decay = args.weight_decay or CONFIG['weight_decay']
    seed = args.seed or CONFIG['seed']

    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print('Using device', device)

    # dataset root: allow pointing to parent that contains class folders
    transform = make_transforms()
    full = datasets.ImageFolder(root=data_root, transform=transform)

    # labels and stratified split
    labels = np.array([y for _, y in full.samples])
    idx = np.arange(len(full))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=labels, random_state=seed)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, stratify=labels[test_idx], random_state=seed)

    train_ds = Subset(full, train_idx)
    val_ds = Subset(full, val_idx)
    test_ds = Subset(full, test_idx)

    # create a balanced sampler for the training split to mitigate class imbalance
    labels_train = labels[train_idx]
    class_counts = np.bincount(labels_train)
    # avoid division by zero
    class_weights = 1.0 / (class_counts + 1e-8)
    sample_weights = class_weights[labels_train]
    sampler = WeightedRandomSampler(weights=sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # load model
    model = SimCLRNet()
    model = load_checkpoint(model, ckpt_path, device)
    model.to(device)

    # freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False

    # get feature dim by forwarding a dummy batch
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224).to(device)
        h, z = model(dummy, return_feat=True)
        feat_dim = h.shape[1]

    n_classes = len(full.classes)

    # linear probe
    print('Training linear probe with settings: epochs=', probe_epochs, 'batch_size=', batch_size, 'lr=', lr)
    classifier, history = train_linear_probe(model, train_loader, val_loader, feat_dim, n_classes, device,
                                            epochs=probe_epochs, lr=lr, batch_size=batch_size,
                                            weight_decay=weight_decay, seed=seed)

    # save training curve
    out_dir = Path('eval_results')
    out_dir.mkdir(exist_ok=True)
    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.savefig(out_dir / 'training_curve.png')

    # evaluate classifier on test set
    classifier.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            h, _ = model(x, return_feat=True)
            logits = classifier(h)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    top1 = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average='macro')
    cm = confusion_matrix(trues, preds)
    per_class = cm.diagonal() / cm.sum(axis=1)
    print('Linear probe test top1=', top1, 'f1_macro=', f1)
    np.save(out_dir / 'confusion_matrix.npy', cm)
    np.save(out_dir / 'per_class_accuracy.npy', per_class)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=full.classes, yticklabels=full.classes)
    plt.ylabel('true')
    plt.xlabel('pred')
    plt.title('Confusion matrix (linear probe)')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / 'confusion_matrix.png')

    # embeddings for train and test for k-NN and retrieval
    train_loader_embed = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader_embed = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    train_embs, train_labels = extract_embeddings(model, train_loader_embed, device)
    test_embs, test_labels = extract_embeddings(model, test_loader_embed, device)

    # k-NN
    knn_preds = knn_classify(train_embs, train_labels, test_embs, k=20)
    knn_acc = accuracy_score(test_labels, knn_preds)
    knn_f1 = f1_score(test_labels, knn_preds, average='macro')
    print('k-NN acc=', knn_acc, 'f1=', knn_f1)

    # retrieval precision@k
    p_at = retrieval_precision_at_k(train_embs, train_labels, test_embs, test_labels, ks=(1,5,10))
    print('Retrieval p@k', p_at)

    # clustering
    kmeans = KMeans(n_clusters=n_classes, random_state=args.seed).fit(test_embs)
    ari = adjusted_rand_score(test_labels, kmeans.labels_)
    nmi = normalized_mutual_info_score(test_labels, kmeans.labels_)
    print('Clustering ARI=', ari, 'NMI=', nmi)

    # PCA plot of test embeddings
    pca = PCA(n_components=2)
    proj = pca.fit_transform(test_embs)
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=proj[:,0], y=proj[:,1], hue=[full.classes[i] for i in test_labels], legend='full', s=10)
    plt.title('PCA of test embeddings')
    plt.tight_layout()
    plt.savefig(out_dir / 'pca_embeddings.png')

    # Save summary
    summary = {
        'linear_top1': float(top1),
        'linear_f1_macro': float(f1),
        'knn_top1': float(knn_acc),
        'knn_f1_macro': float(knn_f1),
        'retrieval': p_at,
        'clustering_ari': float(ari),
        'clustering_nmi': float(nmi),
        'per_class_accuracy': per_class.tolist(),
        'classes': full.classes,
    }
    import json
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print('Evaluation complete. Results in', out_dir)


if __name__ == '__main__':
    main()
