# core/perturbation.py

import numpy as np

def perturb_dataset(X, y, noise_std=0.05):
    X_perturbed = X + np.random.normal(0, noise_std, size=X.shape)
    return X_perturbed, y