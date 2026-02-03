import numpy as np
import os
import argparse


def knn_accuracy(embeddings, labels, k=5):
    # cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    norm_emb = embeddings / norms
    sim = norm_emb.dot(norm_emb.T)
    n = embeddings.shape[0]
    correct = 0
    for i in range(n):
        sims = sim[i]
        sims[i] = -1.0  # exclude self
        idx = np.argsort(-sims)[:k]
        neigh_labels = labels[idx]
        # majority vote
        vals, counts = np.unique(neigh_labels, return_counts=True)
        pred = vals[np.argmax(counts)]
        if pred == labels[i]:
            correct += 1
    return correct / n


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--emb', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--k', type=int, default=5)
    args = p.parse_args()

    emb = np.load(args.emb)
    labels = np.load(args.labels)
    acc = knn_accuracy(emb, labels, k=args.k)
    print(f'KNN (k={args.k}) accuracy: {acc:.4f}')
