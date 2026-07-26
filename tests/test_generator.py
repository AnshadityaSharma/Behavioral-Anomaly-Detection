import pandas as pd

from src.generator.config import ALL_LABELS
from src.generator.entities import build_entities
from src.generator.generate_dataset import COLUMN_ORDER, build_dataset


def test_build_entities_counts(small_config):
    profiles = build_entities(seed=1)
    n_users = sum(1 for p in profiles.values() if p["entity_type"] == "user")
    n_svc = sum(1 for p in profiles.values() if p["entity_type"] == "service_account")
    n_dev = sum(1 for p in profiles.values() if p["entity_type"] == "edge_device")
    assert n_users == small_config.N_USERS
    assert n_svc == small_config.N_SERVICE_ACCOUNTS
    assert n_dev == small_config.N_EDGE_DEVICES


def test_build_dataset_schema_and_labels(small_config):
    df, entities_df = build_dataset(seed=1)

    for col in COLUMN_ORDER + ["label"]:
        assert col in df.columns

    assert set(df["label"].unique()) <= set(ALL_LABELS)
    assert df["entity_id"].isin(entities_df["entity_id"]).all()
    assert not df["timestamp"].isna().any()
    assert (df["session_duration"] > 0).all()


def test_attack_rate_is_bounded(small_config):
    df, _ = build_dataset(seed=2)
    anomaly_rate = (df["label"] != "normal").mean()
    assert 0 < anomaly_rate < 0.25


def test_low_and_slow_exfil_present_and_off_hours(small_config):
    df, _ = build_dataset(seed=5)
    exfil = df[df["label"] == "low_and_slow_exfil"]
    assert len(exfil) > 0

    hours = pd.to_datetime(exfil["timestamp"]).dt.hour
    business_hours = hours.between(8, 17).mean()
    assert business_hours < 0.15  # concentrated outside the normal working day


def test_low_and_slow_exfil_spans_multiple_days(small_config):
    df, _ = build_dataset(seed=5)
    exfil = df[df["label"] == "low_and_slow_exfil"].copy()
    exfil["timestamp"] = pd.to_datetime(exfil["timestamp"])
    span = exfil.groupby("entity_id")["timestamp"].agg(lambda s: (s.max() - s.min()).days)
    # a genuine low-and-slow pattern builds up over multiple days, not a spike
    assert (span >= 3).any()
