import numpy as np
import matplotlib.pyplot as plt


def sample_uneven_distribution(n_samples):
    mean1, mean2 = np.random.uniform(-1, 1, 2)
    ratio = np.random.uniform(0.5, 0.8)
    n1 = int(n_samples * ratio)
    n2 = n_samples - n1
    X1 = np.random.normal(mean1, 0.5, n1)
    X2 = np.random.normal(mean2, 0.5, n2)
    X = np.concatenate((X1, X2))
    return X.reshape(-1, 1)


def generate_flat_data(n_samples):
    X = sample_uneven_distribution(n_samples)
    Y = np.sin(X * np.pi) + np.random.normal(0, 0.1, X.shape)
    return X.astype(np.float32), Y.astype(np.float32)


def generate_hierarchical_data(n_samples_per_group, n_groups):
    X = np.empty((0, 1))
    Y = np.empty((0, 1))
    group = np.array([])

    for i in range(n_groups):
        X_group, Y_group = generate_flat_data(n_samples_per_group)
        X_group += np.random.uniform(0, 1)

        X = np.concatenate((X, X_group), axis=0)
        Y = np.concatenate((Y, Y_group), axis=0)
        group = np.append(group, np.full(n_samples_per_group, i))

    return X.astype(np.float32), group.astype(int), Y.astype(np.float32)
