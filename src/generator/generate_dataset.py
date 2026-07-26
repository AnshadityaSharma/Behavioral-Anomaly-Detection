import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.generator import config
from src.generator.attacks import inject_all
from src.generator.entities import build_entities
from src.generator.sessions import generate_normal_sessions

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

COLUMN_ORDER = [
    "session_id", "entity_id", "entity_type", "timestamp", "source_ip",
    "geo_lat", "geo_lon", "geo_city", "resource_accessed", "auth_method",
    "auth_success", "session_duration", "command_sequence", "device_os",
    "device_mac",
]


def build_dataset(seed=config.SEED):
    profiles = build_entities(seed)
    normal_rows = generate_normal_sessions(profiles, seed)
    attack_rows = inject_all(profiles, normal_rows, np.random.default_rng(seed + 2))

    all_rows = normal_rows + attack_rows
    df = pd.DataFrame(all_rows).sort_values("timestamp").reset_index(drop=True)

    entities_df = pd.DataFrame([
        {
            "entity_id": eid,
            "entity_type": p["entity_type"],
            "home_city": p["home_city"],
            "home_country": p["home_country"],
            "home_ip": p["home_ip"],
            "joined_day": p["joined_day"],
            "typical_resources": "|".join(p["typical_resources"]),
        }
        for eid, p in profiles.items()
    ])

    return df, entities_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--out", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, entities_df = build_dataset(args.seed)

    labels = df[["session_id", "label"]].copy()
    access_log = df[COLUMN_ORDER].copy()

    access_log.to_csv(out_dir / "access_log.csv", index=False)
    labels.to_csv(out_dir / "labels.csv", index=False)
    entities_df.to_csv(out_dir / "entities.csv", index=False)

    n_anomaly = (labels["label"] != "normal").sum()
    print(f"sessions: {len(access_log)}")
    print(f"anomalous sessions: {n_anomaly} ({n_anomaly / len(access_log):.3%})")
    print(labels["label"].value_counts())


if __name__ == "__main__":
    main()
