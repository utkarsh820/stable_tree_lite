# core/tree_distance.py

import numpy as np

def extract_structure(tree):
    """Extract feature indices and thresholds from a decision tree."""
    try:
        return tree.tree_.feature, tree.tree_.threshold
    except AttributeError:
        raise ValueError("Tree must be a fitted sklearn DecisionTreeClassifier")

def tree_distance(tree_a, tree_b):
    """Calculate structural distance between two decision trees."""
    try:
        features_a, _ = extract_structure(tree_a)
        features_b, _ = extract_structure(tree_b)

        min_len = min(len(features_a), len(features_b))
        mismatches = np.sum(features_a[:min_len] != features_b[:min_len])

        # Normalize mismatch over length
        return mismatches / max(len(features_a), len(features_b), 1)
    except Exception as e:
        raise ValueError(f"Error calculating tree distance: {e}")
