from sklearn.ensemble import RandomForestClassifier

from src.features.build_features import FEATURE_COLUMNS
from src.generator.config import ALL_LABELS


class TabularClassifier:
    """Random forest over the tabular session features. Slower-moving and
    fully SHAP-explainable, used alongside the GRU for attribution and as a
    sanity check on the sequence model's calls."""

    def __init__(self, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        self.classes_ = None

    def fit(self, features_df):
        x = features_df[FEATURE_COLUMNS].values
        y = features_df["label"].values
        self.model.fit(x, y)
        self.classes_ = list(self.model.classes_)
        return self

    def predict_proba(self, features_df):
        x = features_df[FEATURE_COLUMNS].values
        return self.model.predict_proba(x)

    def predict(self, features_df):
        x = features_df[FEATURE_COLUMNS].values
        return self.model.predict(x)
