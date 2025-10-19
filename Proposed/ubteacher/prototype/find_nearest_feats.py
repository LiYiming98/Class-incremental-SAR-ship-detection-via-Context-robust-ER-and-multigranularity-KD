import numpy as np


def find_nearest_feats(prototype_key, prototypes, N):
    distances = []

    for i, element in enumerate(prototypes):
        distance = np.linalg.norm(prototype_key - element)
        distances.append((distance, i))

    distances.sort()
    nearest_indices = [index for _, index in distances[:N]]
    return nearest_indices