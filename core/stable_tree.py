# core/stable_tree.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from core.distance_tree import tree_distance
import copy

class StableDecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, lambda_stability=0.1, max_depth=None, random_state=None):
        self.lambda_stability = lambda_stability
        self.max_depth = max_depth
        self.random_state = random_state
        self.base_tree = None
        self.stable_tree = None

    def fit(self, X, y, reference_tree=None):
        # Input validation
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        if len(X) == 0 or len(y) == 0:
            raise ValueError("X and y cannot be empty")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
            
        self.base_tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.base_tree.fit(X, y)

        if reference_tree is None:
            self.stable_tree = self.base_tree
            return self

        # Generate candidate trees
        best_tree = None
        best_score = float("inf")

        for seed in range(5):
            candidate = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=self.random_state + seed if self.random_state is not None else seed
            )
            candidate.fit(X, y)

            try:
                distance = tree_distance(candidate, reference_tree)
                loss = 1 - candidate.score(X, y)  # classification error
                score = loss + self.lambda_stability * distance

                if score < best_score:
                    best_score = score
                    best_tree = candidate
            except Exception as e:
                print(f"Warning: Error evaluating candidate tree {seed}: {e}")
                continue

        # Ensure we have a valid tree
        if best_tree is None:
            print("Warning: No valid candidate tree found, using base tree")
            self.stable_tree = self.base_tree
        else:
            self.stable_tree = best_tree
            
        return self

    def predict(self, X):
        if self.stable_tree is None:
            raise ValueError("Model must be fitted before making predictions")
        return self.stable_tree.predict(X)
