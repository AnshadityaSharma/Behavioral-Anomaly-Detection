import pandas as pd

from src.features.build_features import FEATURE_COLUMNS, build_features
from src.generator.generate_dataset import COLUMN_ORDER, build_dataset


def _access_log(small_config):
    df, _ = build_dataset(seed=3)
    return df[COLUMN_ORDER].copy()


def test_no_nans_in_feature_columns(small_config):
    features = build_features(_access_log(small_config))
    assert features[FEATURE_COLUMNS].isna().sum().sum() == 0


def test_first_session_is_cold_start(small_config):
    features = build_features(_access_log(small_config))
    first_rows = features.sort_values(["entity_id", "timestamp"]).groupby("entity_id").first()
    assert (first_rows["entity_history_length"] == 0).all()
    assert (first_rows["is_cold_start"] == 1).all()
    assert (first_rows["hours_since_last_session"] == 9999.0).all()


def test_geo_velocity_known_pair():
    rows = pd.DataFrame([
        {
            "session_id": "s1", "entity_id": "e1", "entity_type": "user",
            "timestamp": "2026-01-01 00:00:00", "source_ip": "1.1.1.1",
            "geo_lat": 0.0, "geo_lon": 0.0, "geo_city": "a",
            "resource_accessed": "r1", "auth_method": "password",
            "auth_success": True, "session_duration": 100, "command_sequence": "",
            "device_os": "os1", "device_mac": "AA",
        },
        {
            "session_id": "s2", "entity_id": "e1", "entity_type": "user",
            "timestamp": "2026-01-01 01:00:00", "source_ip": "1.1.1.1",
            "geo_lat": 1.0, "geo_lon": 0.0, "geo_city": "b",
            "resource_accessed": "r1", "auth_method": "password",
            "auth_success": True, "session_duration": 100, "command_sequence": "",
            "device_os": "os1", "device_mac": "AA",
        },
    ])
    features = build_features(rows)
    second = features[features["session_id"] == "s2"].iloc[0]
    assert second["geo_distance_from_prev_km"] > 100
    assert second["geo_velocity_kmh"] > 100
    assert second["is_new_resource"] == 0
    assert second["is_device_mismatch"] == 0


def test_device_mismatch_flagged():
    rows = pd.DataFrame([
        {
            "session_id": "s1", "entity_id": "e1", "entity_type": "edge_device",
            "timestamp": "2026-01-01 00:00:00", "source_ip": "1.1.1.1",
            "geo_lat": 0.0, "geo_lon": 0.0, "geo_city": "a",
            "resource_accessed": "r1", "auth_method": "certificate",
            "auth_success": True, "session_duration": 10, "command_sequence": "",
            "device_os": "EdgeOS-3.2", "device_mac": "AA:AA:AA:AA:AA:AA",
        },
        {
            "session_id": "s2", "entity_id": "e1", "entity_type": "edge_device",
            "timestamp": "2026-01-01 00:10:00", "source_ip": "1.1.1.1",
            "geo_lat": 0.0, "geo_lon": 0.0, "geo_city": "a",
            "resource_accessed": "r1", "auth_method": "certificate",
            "auth_success": True, "session_duration": 10, "command_sequence": "",
            "device_os": "RTOS-1.8", "device_mac": "BB:BB:BB:BB:BB:BB",
        },
    ])
    features = build_features(rows)
    second = features[features["session_id"] == "s2"].iloc[0]
    assert second["is_device_mismatch"] == 1
