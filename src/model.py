import joblib
from lightgbm import LGBMRegressor


class RULModel:

    def __init__(self):

        self.model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=8,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

    def train(self, X, y):

        self.model.fit(
            X,
            y
        )

    def predict(self, X):

        return self.model.predict(X)

    def save(self, path):

        joblib.dump(
            self.model,
            path
        )

from lightgbm import LGBMClassifier


class FailureClassifier:

    def __init__(self):

        self.model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=8,
            num_leaves=31,
            class_weight="balanced",
            random_state=42
        )

    def train(self, X, y):

        self.model.fit(
            X,
            y
        )

    def predict_probability(self, X):

        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):

        probabilities = self.predict_probability(X)

        return (
            probabilities >= threshold
        ).astype(int)

    def save(self, path):

        joblib.dump(
            self.model,
            path
        )