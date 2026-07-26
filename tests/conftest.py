from datetime import timedelta

import pytest

from src.generator import config


@pytest.fixture
def small_config(monkeypatch):
    monkeypatch.setattr(config, "N_USERS", 10)
    monkeypatch.setattr(config, "N_SERVICE_ACCOUNTS", 3)
    monkeypatch.setattr(config, "N_EDGE_DEVICES", 4)
    monkeypatch.setattr(config, "SIM_DAYS", 12)
    monkeypatch.setattr(config, "SIM_END", config.SIM_START + timedelta(days=12))
    return config
