import random

import numpy as np
from faker import Faker

from src.generator import config


def _random_mac(rng):
    return ":".join(f"{rng.integers(0, 256):02X}" for _ in range(6))


def _make_profile(entity_id, entity_type, fake, rng):
    resources = config.RESOURCES_BY_TYPE[entity_type]
    n_typical = {"user": 4, "service_account": 3, "edge_device": 2}[entity_type]
    typical_resources = list(rng.choice(resources, size=min(n_typical, len(resources)), replace=False))

    city = fake.city()
    country = fake.country()
    lat, lon = float(fake.latitude()), float(fake.longitude())
    home_ip = fake.ipv4_public()

    if entity_type == "user":
        hour_mean = rng.uniform(7.5, 10.5)
        hour_std = rng.uniform(0.7, 1.8)
        active_weekdays = list(range(5))
        sessions_per_active_day = rng.uniform(2, 6)
        duration_mean = rng.uniform(300, 2400)
        auth_method = rng.choice(config.AUTH_METHODS, p=[0.35, 0.35, 0.1, 0.2])
        privileged = rng.random() < 0.12
        os_name = rng.choice(config.DEVICE_OS[:5])
        mac = _random_mac(rng)
    elif entity_type == "service_account":
        hour_mean = 12.0
        hour_std = 8.0
        active_weekdays = list(range(7))
        sessions_per_active_day = rng.uniform(15, 45)
        duration_mean = rng.uniform(5, 120)
        auth_method = "certificate" if rng.random() < 0.7 else "token"
        privileged = True
        os_name = "linux-container"
        mac = _random_mac(rng)
    else:
        hour_mean = 12.0
        hour_std = 7.0
        active_weekdays = list(range(7))
        sessions_per_active_day = rng.uniform(4, 12)
        duration_mean = rng.uniform(2, 30)
        auth_method = "certificate"
        privileged = False
        os_name = rng.choice(config.DEVICE_OS[5:])
        mac = _random_mac(rng)

    return {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "home_city": city,
        "home_country": country,
        "home_lat": lat,
        "home_lon": lon,
        "home_ip": home_ip,
        "hour_mean": hour_mean,
        "hour_std": hour_std,
        "active_weekdays": active_weekdays,
        "sessions_per_active_day": sessions_per_active_day,
        "duration_mean": duration_mean,
        "duration_std": duration_mean * 0.35,
        "auth_method": auth_method,
        "typical_resources": typical_resources,
        "all_resources": resources,
        "privileged": privileged,
        "os_name": os_name,
        "mac": mac,
        "joined_day": 0,
    }


def build_entities(seed=config.SEED):
    Faker.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    fake = Faker()

    profiles = {}
    for i in range(config.N_USERS):
        eid = f"user_{i:04d}"
        profiles[eid] = _make_profile(eid, "user", fake, rng)
    for i in range(config.N_SERVICE_ACCOUNTS):
        eid = f"svc_{i:04d}"
        profiles[eid] = _make_profile(eid, "service_account", fake, rng)
    for i in range(config.N_EDGE_DEVICES):
        eid = f"dev_{i:04d}"
        profiles[eid] = _make_profile(eid, "edge_device", fake, rng)

    # a handful of entities "join" partway through the simulation window to
    # exercise the cold-start path downstream.
    cold_start_ids = rng.choice(list(profiles.keys()), size=25, replace=False)
    for eid in cold_start_ids:
        profiles[eid]["joined_day"] = int(rng.integers(config.SIM_DAYS - 10, config.SIM_DAYS - 1))

    return profiles
