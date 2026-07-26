import numpy as np

DRIFT_WINDOW = "30D"
MIN_PERIODS = 3


def rolling_zscore(df, value_col, window=DRIFT_WINDOW, min_periods=MIN_PERIODS):
    """Trailing-window z-score per entity, excluding the current row, so a
    baseline that shifted a month ago stops being penalized today instead of
    staying anchored to all-time history."""
    tmp = df[["entity_id", "timestamp", value_col]].copy()
    tmp["_sq"] = tmp[value_col] ** 2

    # groupby.rolling returns a (group, timestamp) multiindex; the group
    # order matches tmp's row order since tmp is pre-sorted by entity/time,
    # so a full positional reset lines these back up with tmp by row.
    g = tmp.groupby("entity_id", sort=False)
    roll_count = g.rolling(window, on="timestamp")[value_col].count().reset_index(drop=True)
    roll_sum = g.rolling(window, on="timestamp")[value_col].sum().reset_index(drop=True)
    roll_sumsq = g.rolling(window, on="timestamp")["_sq"].sum().reset_index(drop=True)

    value = tmp[value_col].reset_index(drop=True)
    n_excl = (roll_count - 1).clip(lower=0)
    safe_n = n_excl.replace(0, np.nan)
    mean_excl = (roll_sum - value) / safe_n
    var_excl = (roll_sumsq - value ** 2) / safe_n - mean_excl ** 2
    std_excl = np.sqrt(var_excl.clip(lower=0))

    z = (value - mean_excl) / std_excl.replace(0, np.nan)
    z = z.fillna(0.0)
    z[n_excl < min_periods] = 0.0
    return z.reset_index(drop=True)
