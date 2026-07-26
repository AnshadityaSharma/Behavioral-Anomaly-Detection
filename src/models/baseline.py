import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.features.build_features import FEATURE_COLUMNS

COLD_START_THRESHOLD = 5


class BaselineProfiler:
    """Population-level isolation forest per entity_type, plus a per-entity
    statistical profile used for cold-start fallback and dashboard display."""

    def __init__(self, contamination=0.02, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.scalers = {}
        self.forests = {}
        self.entity_profiles = None
        self.type_profiles = None

    def fit(self, features_df):
        for entity_type, group in features_df.groupby("entity_type"):
            x = group[FEATURE_COLUMNS].values
            scaler = StandardScaler().fit(x)
            forest = IsolationForest(
                n_estimators=200,
                contamination=self.contamination,
                random_state=self.random_state,
            ).fit(scaler.transform(x))
            self.scalers[entity_type] = scaler
            self.forests[entity_type] = forest

        self.entity_profiles = self._build_entity_profiles(features_df)
        self.type_profiles = self._build_type_profiles(features_df)
        return self

    def _build_entity_profiles(self, features_df):
        agg = features_df.groupby("entity_id")[FEATURE_COLUMNS].agg(["mean", "std", "count"])
        agg.columns = ["_".join(c) for c in agg.columns]
        return agg

    def _build_type_profiles(self, features_df):
        agg = features_df.groupby("entity_type")[FEATURE_COLUMNS].agg(["mean", "std"])
        agg.columns = ["_".join(c) for c in agg.columns]
        return agg

    def score(self, features_df):
        scores = np.zeros(len(features_df))
        for entity_type, group in features_df.groupby("entity_type"):
            if entity_type not in self.forests:
                continue
            x = self.scalers[entity_type].transform(group[FEATURE_COLUMNS].values)
            raw = self.forests[entity_type].decision_function(x)
            scores[group.index] = -raw

        lo, hi = scores.min(), scores.max()
        if hi - lo < 1e-9:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)

    def entity_profile_summary(self, entity_id, entity_type):
        if self.entity_profiles is not None and entity_id in self.entity_profiles.index:
            row = self.entity_profiles.loc[entity_id]
            if row.get("hour_of_day_count", 0) >= COLD_START_THRESHOLD:
                return row, "entity"
        if self.type_profiles is not None and entity_type in self.type_profiles.index:
            return self.type_profiles.loc[entity_type], "population"
        return None, "none"
