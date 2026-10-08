import pandas as pd
import pytest

from src.dashboard.detection import MAX_UPLOAD_ROWS, risk_levels, validate_access_log
from src.generator.generate_dataset import COLUMN_ORDER


def access_row():
    row = dict.fromkeys(COLUMN_ORDER, "example")
    row.update(session_id="one", entity_id="user_1", entity_type="user",
               timestamp="2026-05-01 09:00:00", source_ip="10.0.0.1",
               geo_lat=28.6, geo_lon=77.2, auth_success="False",
               session_duration=12, resource_accessed="vpn", device_os="Linux",
               device_mac="aa:bb", command_sequence="")
    return row


def test_validate_access_log_coerces_schema():
    result = validate_access_log(pd.DataFrame([access_row()]))
    assert result.loc[0, "auth_success"] == False
    assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])


def test_validate_access_log_rejects_invalid_rows():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_access_log(pd.DataFrame({"session_id": ["one"]}))
    duplicate = pd.DataFrame([access_row(), access_row()])
    with pytest.raises(ValueError, match="unique"):
        validate_access_log(duplicate)
    invalid = access_row()
    invalid["timestamp"] = "not a date"
    with pytest.raises(ValueError, match="timestamp"):
        validate_access_log(pd.DataFrame([invalid]))


def test_upload_limit_and_relative_risk_bands():
    frame = pd.DataFrame([access_row()] * (MAX_UPLOAD_ROWS + 1))
    with pytest.raises(ValueError, match="at most"):
        validate_access_log(frame)
    levels = risk_levels(pd.Series(range(1000)))
    assert levels.iloc[0] == "LOW"
    assert levels.iloc[-1] == "CRITICAL"
    assert set(risk_levels(pd.Series([0.3, 0.3, 0.3]))) == {"LOW"}
