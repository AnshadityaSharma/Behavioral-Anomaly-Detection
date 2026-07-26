import uuid
from datetime import timedelta

import numpy as np

from src.generator import config


def _sample_hour(rng, mean, std):
    hour = rng.normal(mean, std) % 24
    return hour


def _sample_command_sequence(rng, profile):
    if not profile["privileged"]:
        return []
    length = rng.integers(2, 7)
    return list(rng.choice(config.PRIVILEGED_ACTIONS, size=length, replace=True))


def _jitter_geo(rng, lat, lon, scale=0.15):
    return lat + rng.normal(0, scale), lon + rng.normal(0, scale)


def generate_normal_sessions(profiles, seed=config.SEED):
    rng = np.random.default_rng(seed + 1)
    rows = []

    for eid, profile in profiles.items():
        day = profile["joined_day"]
        while day < config.SIM_DAYS:
            weekday = (config.SIM_START + timedelta(days=day)).weekday()
            if weekday not in profile["active_weekdays"]:
                day += 1
                continue
            if rng.random() > 0.9 and profile["entity_type"] == "user":
                day += 1
                continue

            n_sessions = max(0, int(rng.poisson(profile["sessions_per_active_day"])))
            for _ in range(n_sessions):
                hour = _sample_hour(rng, profile["hour_mean"], profile["hour_std"])
                ts = (
                    config.SIM_START
                    + timedelta(days=day, hours=hour)
                    + timedelta(seconds=int(rng.integers(0, 60)))
                )

                traveling = profile["entity_type"] == "user" and rng.random() < 0.02
                if traveling:
                    lat, lon = _jitter_geo(rng, profile["home_lat"], profile["home_lon"], scale=8.0)
                    source_ip = f"{rng.integers(1, 223)}.{rng.integers(0, 256)}.{rng.integers(0, 256)}.{rng.integers(1, 255)}"
                else:
                    lat, lon = _jitter_geo(rng, profile["home_lat"], profile["home_lon"])
                    source_ip = profile["home_ip"]

                if rng.random() < 0.85:
                    resource = rng.choice(profile["typical_resources"])
                else:
                    resource = rng.choice(profile["all_resources"])

                duration = max(1.0, rng.normal(profile["duration_mean"], profile["duration_std"]))
                auth_success = rng.random() > 0.02
                auth_method = profile["auth_method"] if rng.random() > 0.05 else rng.choice(config.AUTH_METHODS)

                rows.append({
                    "session_id": str(uuid.uuid4()),
                    "entity_id": eid,
                    "entity_type": profile["entity_type"],
                    "timestamp": ts,
                    "source_ip": source_ip,
                    "geo_lat": lat,
                    "geo_lon": lon,
                    "geo_city": profile["home_city"] if not traveling else "unknown-transit",
                    "resource_accessed": resource,
                    "auth_method": auth_method,
                    "auth_success": auth_success,
                    "session_duration": duration,
                    "command_sequence": "|".join(_sample_command_sequence(rng, profile)),
                    "device_os": profile["os_name"],
                    "device_mac": profile["mac"],
                    "label": "normal",
                })
            day += 1

    return rows
