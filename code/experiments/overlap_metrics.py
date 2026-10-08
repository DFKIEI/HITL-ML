"""Which pair-overlap measure tracks the classifier's real confusion? No LLM
calls: for each checkpoint in real_latent_check.CHECKPOINTS, ranks all class
pairs by three overlap measures on the real latent features and compares each
ranking with the confusion matrix (Spearman rho, and how many of the 5 most
confused pairs are also its top 5).

- current r90:      llm/latent_state.py - share of each class's points inside
                    the other's 90% radius (what the prompt uses);
- nearest-centroid: share of each class's points closer to the other center;
- knn-10:           share of the 10 nearest neighbours from the other class.

    python experiments/overlap_metrics.py
"""

import os
import sys
from itertools import combinations

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import load_dataset  # noqa: E402
from experiments.real_latent_check import (CHECKPOINTS, PROJECT_DIR, TOP_K, balanced_sample,  # noqa: E402
                                           confused_pairs, device, latent_features)
from llm.latent_state import _compute_class_stats, _compute_pair_overlap, compute_centroids  # noqa: E402
from model import get_model  # noqa: E402


def nearest_centroid(points, labels, centroids, i, j):
    def share(a, b):
        p = points[labels == a]
        return float((np.linalg.norm(p - centroids[b], axis=1) < np.linalg.norm(p - centroids[a], axis=1)).mean())
    return 0.5 * (share(i, j) + share(j, i))


def knn_mix(points, labels, i, j, k=10):
    mask = (labels == i) | (labels == j)
    p, l = points[mask], labels[mask]
    neighbours = NearestNeighbors(n_neighbors=k + 1).fit(p).kneighbors(p, return_distance=False)[:, 1:]
    return float((l[neighbours] != l[:, None]).mean())


def main():
    os.chdir(PROJECT_DIR)
    dev = device()
    rng = np.random.default_rng(0)
    loaded = {}
    for dataset, checkpoint_path in CHECKPOINTS:
        if dataset not in loaded:
            loaded[dataset] = load_dataset(dataset, 512)
        _, valloader, _, num_classes, input_shape = loaded[dataset]
        model = get_model(f'CNN_{dataset}', input_shape, num_classes)
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=False)['model_state_dict'])
        features, labels, predictions = latent_features(model.to(dev), valloader, dev)
        classes = sorted(int(c) for c in np.unique(labels))
        confusion = dict(confused_pairs(labels, predictions, classes))
        top = set(sorted(confusion, key=lambda p: -confusion[p])[:TOP_K])

        for per_class in (10, 100):
            sample = balanced_sample(labels, per_class, rng)
            points, sample_labels = features[sample], labels[sample]
            centroids = compute_centroids(points, sample_labels, classes)
            vectors, stats = _compute_class_stats(points, sample_labels, centroids)
            pairs = list(combinations(classes, 2))
            truth = [confusion[frozenset(p)] for p in pairs]
            cells = []
            for name, measure in (
                    ('current r90', lambda i, j: _compute_pair_overlap(i, j, centroids, vectors, stats)),
                    ('nearest-centroid', lambda i, j: nearest_centroid(points, sample_labels, centroids, i, j)),
                    ('knn-10', lambda i, j: knn_mix(points, sample_labels, i, j))):
                scores = [measure(i, j) for i, j in pairs]
                best = {frozenset(pairs[k]) for k in np.argsort(scores)[::-1][:TOP_K]}
                cells.append(f"{name}: rho={spearmanr(scores, truth).correlation:.2f} "
                             f"top{TOP_K} {len(best & top)}/{TOP_K}")
            print(f"{checkpoint_path} {per_class:3d}/class | " + " | ".join(cells), flush=True)


if __name__ == '__main__':
    main()
