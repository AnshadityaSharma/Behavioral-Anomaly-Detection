import uuid
from datetime import timedelta

import numpy as np
import pandas as pd

from src.common.geo import haversine_km
from src.generator import config


def _random_ip(rng):
    return f"{rng.integers(1, 223)}.{rng.integers(0, 256)}.{rng.integers(0, 256)}.{rng.integers(1, 255)}"


def _random_far_geo(rng, lat, lon, min_km=3000):
    while True:
        cand_lat = rng.uniform(-60, 70)
        cand_lon = rng.uniform(-180, 180)
        if haversine_km(lat, lon, cand_lat, cand_lon) >= min_km:
            return cand_lat, cand_lon


def _row(entity_id, entity_type, ts, ip, lat, lon, city, resource, auth_method,
         success, duration, commands, os_name, mac, label):
    return {
        "session_id": str(uuid.uuid4()),
        "entity_id": entity_id,
        "entity_type": entity_type,
        "timestamp": ts,
        "source_ip": ip,
        "geo_lat": lat,
        "geo_lon": lon,
        "geo_city": city,
        "resource_accessed": resource,
        "auth_method": auth_method,
        "auth_success": success,
        "session_duration": duration,
        "command_sequence": "|".join(commands),
        "device_os": os_name,
        "device_mac": mac,
        "label": label,
    }


def inject_brute_force(profiles, rng, n_incidents):
    rows = []
    entity_ids = list(profiles.keys())
    for _ in range(n_incidents):
        target = profiles[rng.choice(entity_ids)]
        attacker_ip = _random_ip(rng)
        day = int(rng.integers(0, config.SIM_DAYS))
        start = config.SIM_START + timedelta(days=day, hours=float(rng.uniform(0, 24)))
        n_attempts = int(rng.integers(12, 60))
        gap_seconds = rng.uniform(1, 8, size=n_attempts)
        t = start
        for i in range(n_attempts):
            t = t + timedelta(seconds=float(gap_seconds[i]))
            success = i == n_attempts - 1 and rng.random() < 0.3
            rows.append(_row(
                target["entity_id"], target["entity_type"], t, attacker_ip,
                target["home_lat"], target["home_lon"], "unknown-transit",
                rng.choice(target["all_resources"]), "password", success,
                max(1.0, rng.normal(5, 2)), [], target["os_name"], _random_mac_attacker(rng),
                "brute_force",
            ))
    return rows


def _random_mac_attacker(rng):
    return ":".join(f"{rng.integers(0, 256):02X}" for _ in range(6))


def inject_impossible_travel(profiles, normal_rows_df, rng, n_incidents):
    rows = []
    candidates = normal_rows_df[normal_rows_df["entity_type"] == "user"]
    if candidates.empty:
        return rows
    for _ in range(n_incidents):
        base = candidates.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        profile = profiles[base["entity_id"]]
        gap_hours = rng.uniform(0.5, 3.0)
        ts = base["timestamp"] + timedelta(hours=float(gap_hours))
        if ts >= config.SIM_END:
            continue
        far_lat, far_lon = _random_far_geo(rng, base["geo_lat"], base["geo_lon"])
        rows.append(_row(
            base["entity_id"], profile["entity_type"], ts, _random_ip(rng),
            far_lat, far_lon, "unknown-transit",
            rng.choice(profile["typical_resources"]), profile["auth_method"], True,
            max(1.0, rng.normal(profile["duration_mean"], profile["duration_std"])),
            [], profile["os_name"], profile["mac"], "impossible_travel",
        ))
    return rows


def inject_credential_stuffing(profiles, rng, n_incidents):
    rows = []
    user_ids = [eid for eid, p in profiles.items() if p["entity_type"] == "user"]
    for _ in range(n_incidents):
        n_attacker_ips = int(rng.integers(1, 4))
        attacker_ips = [_random_ip(rng) for _ in range(n_attacker_ips)]
        n_targets = int(rng.integers(20, 70))
        targets = rng.choice(user_ids, size=min(n_targets, len(user_ids)), replace=False)
        day = int(rng.integers(0, config.SIM_DAYS))
        start = config.SIM_START + timedelta(days=day, hours=float(rng.uniform(0, 24)))
        t = start
        for eid in targets:
            profile = profiles[eid]
            t = t + timedelta(seconds=float(rng.uniform(0.5, 6)))
            success = rng.random() < 0.03
            rows.append(_row(
                eid, "user", t, rng.choice(attacker_ips),
                profile["home_lat"], profile["home_lon"], "unknown-transit",
                "vpn-gateway", "password", success,
                max(1.0, rng.normal(4, 1.5)), [], profile["os_name"], _random_mac_attacker(rng),
                "credential_stuffing",
            ))
    return rows


def inject_lateral_movement(profiles, rng, n_incidents):
    rows = []
    all_resources = sorted(set(sum(config.RESOURCES_BY_TYPE.values(), [])))
    candidates = [eid for eid, p in profiles.items() if p["entity_type"] in ("user", "service_account")]
    for _ in range(n_incidents):
        eid = rng.choice(candidates)
        profile = profiles[eid]
        day = int(rng.integers(profile["joined_day"], config.SIM_DAYS - 1))
        start = config.SIM_START + timedelta(days=day, hours=float(rng.uniform(0, 24)))
        unseen = [r for r in all_resources if r not in profile["typical_resources"]]
        n_hops = int(rng.integers(6, 16))
        targets = rng.choice(unseen, size=min(n_hops, len(unseen)), replace=False)
        t = start
        for resource in targets:
            t = t + timedelta(minutes=float(rng.uniform(1, 10)))
            commands = list(rng.choice(config.PRIVILEGED_ACTIONS, size=int(rng.integers(2, 5)), replace=True))
            rows.append(_row(
                eid, profile["entity_type"], t, profile["home_ip"],
                profile["home_lat"], profile["home_lon"], profile["home_city"],
                resource, profile["auth_method"], True,
                max(1.0, rng.normal(profile["duration_mean"] * 1.5, profile["duration_std"])),
                commands, profile["os_name"], profile["mac"], "lateral_movement",
            ))
    return rows


def inject_device_spoofing(profiles, rng, n_incidents):
    rows = []
    candidates = [eid for eid, p in profiles.items() if p["entity_type"] in ("edge_device", "user")]
    for _ in range(n_incidents):
        eid = rng.choice(candidates)
        profile = profiles[eid]
        day = int(rng.integers(profile["joined_day"], config.SIM_DAYS - 1))
        ts = config.SIM_START + timedelta(days=day, hours=float(rng.uniform(0, 24)))
        fake_os = rng.choice([o for o in config.DEVICE_OS if o != profile["os_name"]])
        rows.append(_row(
            eid, profile["entity_type"], ts, _random_ip(rng),
            profile["home_lat"], profile["home_lon"], profile["home_city"],
            rng.choice(profile["typical_resources"]), profile["auth_method"], True,
            max(1.0, rng.normal(profile["duration_mean"], profile["duration_std"])),
            [], fake_os, _random_mac_attacker(rng), "device_spoofing",
        ))
    return rows


def inject_insider_drift(profiles, rng, n_incidents):
    rows = []
    candidates = [eid for eid, p in profiles.items() if p["entity_type"] == "user"]
    for _ in range(n_incidents):
        eid = rng.choice(candidates)
        profile = profiles[eid]
        unseen = [r for r in profile["all_resources"] if r not in profile["typical_resources"]]
        if not unseen:
            continue
        start_day = int(rng.integers(profile["joined_day"], max(profile["joined_day"] + 1, config.SIM_DAYS - 20)))
        n_steps = int(rng.integers(8, 20))
        for step in range(n_steps):
            day = min(config.SIM_DAYS - 1, start_day + step * int(rng.integers(1, 3)))
            hour = _sample_hour_like(profile, rng)
            ts = config.SIM_START + timedelta(days=day, hours=hour)
            resource = unseen[min(step // 2, len(unseen) - 1)]
            rows.append(_row(
                eid, "user", ts, profile["home_ip"],
                profile["home_lat"], profile["home_lon"], profile["home_city"],
                resource, profile["auth_method"], True,
                max(1.0, rng.normal(profile["duration_mean"], profile["duration_std"])),
                [], profile["os_name"], profile["mac"], "insider_drift",
            ))
    return rows


def _sample_hour_like(profile, rng):
    return float(rng.normal(profile["hour_mean"], profile["hour_std"]) % 24)


def inject_low_and_slow_exfil(profiles, rng, n_incidents):
    """Small resource-access bursts concentrated in off-hours, accumulating
    gradually over days to weeks rather than in one spike. Distinct from
    lateral_movement (many resources hopped in a single short burst) and from
    insider_drift (which samples hours from the entity's own normal
    distribution, not an off-hours window - drift looks routine, this
    doesn't)."""
    rows = []
    candidates = [eid for eid, p in profiles.items() if p["entity_type"] in ("user", "service_account")]
    for _ in range(n_incidents):
        eid = rng.choice(candidates)
        profile = profiles[eid]
        unseen = [r for r in profile["all_resources"] if r not in profile["typical_resources"]]
        if not unseen:
            continue
        span_days = int(rng.integers(10, 25))
        start_day = int(rng.integers(
            profile["joined_day"], max(profile["joined_day"] + 1, config.SIM_DAYS - span_days)
        ))
        off_hour_center = (profile["hour_mean"] + 12) % 24

        touched = []
        for day_offset in range(span_days):
            day = start_day + day_offset
            if day >= config.SIM_DAYS:
                break
            if rng.random() < 0.6:
                continue  # gradual: most days see nothing at all
            n_sessions_today = int(rng.integers(1, 3))
            for _ in range(n_sessions_today):
                hour = float(rng.normal(off_hour_center, 1.5) % 24)
                ts = config.SIM_START + timedelta(days=day, hours=hour)
                if len(touched) < len(unseen) and (not touched or rng.random() < 0.5):
                    resource = unseen[len(touched)]
                    touched.append(resource)
                else:
                    resource = rng.choice(touched) if touched else unseen[0]
                commands = ["export_data"] if rng.random() < 0.3 else []
                rows.append(_row(
                    eid, profile["entity_type"], ts, profile["home_ip"],
                    profile["home_lat"], profile["home_lon"], profile["home_city"],
                    resource, profile["auth_method"], True,
                    max(1.0, rng.normal(profile["duration_mean"] * 1.3, profile["duration_std"])),
                    commands, profile["os_name"], profile["mac"], "low_and_slow_exfil",
                ))
    return rows


def inject_all(profiles, normal_rows, rng):
    normal_df = pd.DataFrame(normal_rows)
    total_normal = len(normal_rows)

    rows = []
    rows += inject_brute_force(
        profiles, rng, max(1, int(total_normal * config.ATTACK_RATES["brute_force"] / 30)))
    rows += inject_impossible_travel(
        profiles, normal_df, rng, max(1, int(total_normal * config.ATTACK_RATES["impossible_travel"])))
    rows += inject_credential_stuffing(
        profiles, rng, max(1, int(total_normal * config.ATTACK_RATES["credential_stuffing"] / 40)))
    rows += inject_lateral_movement(
        profiles, rng, max(1, int(total_normal * config.ATTACK_RATES["lateral_movement"] / 10)))
    rows += inject_device_spoofing(
        profiles, rng, max(1, int(total_normal * config.ATTACK_RATES["device_spoofing"])))
    rows += inject_insider_drift(
        profiles, rng, max(1, int(total_normal * config.ATTACK_RATES["insider_drift"] / 12)))
    rows += inject_low_and_slow_exfil(
        profiles, rng, max(1, int(total_normal * config.ATTACK_RATES["low_and_slow_exfil"] / 10)))
    return rows
