import numpy as np
import pandas as pd

from src.common.geo import haversine_km
from src.features.drift import rolling_zscore

COLD_START_THRESHOLD = 5
PRIVILEGED_MARKERS = {"escalate_privilege", "delete_file", "modify_permissions", "create_user"}


def build_features(access_log, cold_start_threshold=COLD_START_THRESHOLD):
    df = access_log.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    entity_home = (
        df.sort_values(["entity_id", "timestamp"])
        .groupby("entity_id")[["geo_lat", "geo_lon"]]
        .transform("first")
    )
    df["home_lat"] = entity_home["geo_lat"]
    df["home_lon"] = entity_home["geo_lon"]

    df["hour_of_day"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    grp = df.groupby("entity_id", sort=False)

    prev_ts = grp["timestamp"].shift(1)
    df["hours_since_last_session"] = (df["timestamp"] - prev_ts).dt.total_seconds() / 3600
    df["hours_since_last_session"] = df["hours_since_last_session"].fillna(9999.0)

    prev_lat = grp["geo_lat"].shift(1)
    prev_lon = grp["geo_lon"].shift(1)
    dist_from_prev = haversine_km(prev_lat, prev_lon, df["geo_lat"], df["geo_lon"])
    df["geo_distance_from_prev_km"] = dist_from_prev.fillna(0.0)

    safe_hours = df["hours_since_last_session"].clip(lower=1 / 60)
    df["geo_velocity_kmh"] = (df["geo_distance_from_prev_km"] / safe_hours).where(
        df["hours_since_last_session"] < 9999.0, 0.0
    )

    df["geo_distance_from_home_km"] = haversine_km(
        df["home_lat"], df["home_lon"], df["geo_lat"], df["geo_lon"]
    )

    df["entity_history_length"] = grp.cumcount()
    df["is_cold_start"] = (df["entity_history_length"] < cold_start_threshold).astype(int)

    is_first_resource = ~df.duplicated(subset=["entity_id", "resource_accessed"], keep="first")
    df["is_new_resource"] = is_first_resource.astype(int)
    df["is_new_resource"] = df["is_new_resource"].where(df["entity_history_length"] > 0, 0)

    fingerprint = df["device_os"] + "::" + df["device_mac"]
    baseline_fp = fingerprint.groupby(df["entity_id"], sort=False).transform("first")
    df["is_device_mismatch"] = (fingerprint != baseline_fp).astype(int)

    df["auth_failed"] = (~df["auth_success"].astype(bool)).astype(int)

    df["session_duration_z"] = rolling_zscore(df, "session_duration")
    df["hour_of_day_z"] = rolling_zscore(df, "hour_of_day")

    df["command_sequence_length"] = df["command_sequence"].fillna("").apply(
        lambda s: 0 if s == "" else len(s.split("|"))
    )
    df["has_privileged_command"] = df["command_sequence"].fillna("").apply(
        lambda s: int(any(tok in PRIVILEGED_MARKERS for tok in s.split("|")))
    )

    # rolling().apply() needs numeric input, so id/resource columns are
    # factorized to codes before the nunique-in-window computation.
    df["_entity_code"] = pd.factorize(df["entity_id"])[0].astype(float)
    df["_resource_code"] = pd.factorize(df["resource_accessed"])[0].astype(float)

    df = df.sort_values(["source_ip", "timestamp"]).reset_index(drop=True)
    ip_grp = df.groupby("source_ip", sort=False)
    df["ip_session_count_10min"] = (
        ip_grp.rolling("10min", on="timestamp")["session_id"].count().reset_index(drop=True) - 1
    )
    df["ip_distinct_entities_10min"] = (
        ip_grp.rolling("10min", on="timestamp")["_entity_code"]
        .apply(lambda s: len(set(s)), raw=True)
        .reset_index(drop=True)
        - 1
    ).clip(lower=0)
    df["ip_failure_rate_10min"] = (
        ip_grp.rolling("10min", on="timestamp")["auth_failed"].mean().reset_index(drop=True)
    )

    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    grp = df.groupby("entity_id", sort=False)
    df["entity_resource_breadth_24h"] = (
        grp.rolling("24h", on="timestamp")["_resource_code"]
        .apply(lambda s: len(set(s)), raw=True)
        .reset_index(drop=True)
    )
    # a longer trailing window than the 24h one above: a burst of lateral
    # movement spikes the 24h count, while a slow accumulation of new
    # resources over days/weeks only shows up once the window is wide enough
    # to hold the whole build-up.
    df["entity_resource_breadth_7d"] = (
        grp.rolling("7D", on="timestamp")["_resource_code"]
        .apply(lambda s: len(set(s)), raw=True)
        .reset_index(drop=True)
    )
    df["entity_failed_count_10min"] = (
        grp.rolling("10min", on="timestamp")["auth_failed"].sum().reset_index(drop=True)
    )
    df = df.drop(columns=["_entity_code", "_resource_code"])

    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    return df


FEATURE_COLUMNS = [
    "hour_of_day", "day_of_week", "is_weekend", "hours_since_last_session",
    "geo_distance_from_prev_km", "geo_velocity_kmh", "geo_distance_from_home_km",
    "entity_history_length", "is_cold_start", "is_new_resource", "is_device_mismatch",
    "auth_failed", "session_duration_z", "hour_of_day_z", "command_sequence_length",
    "has_privileged_command", "ip_session_count_10min", "ip_distinct_entities_10min",
    "ip_failure_rate_10min", "entity_resource_breadth_24h", "entity_resource_breadth_7d",
    "entity_failed_count_10min",
]
