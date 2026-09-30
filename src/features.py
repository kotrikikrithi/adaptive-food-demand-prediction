"""
Turn raw rows into model inputs.

- The target is demand divided by the outlet's average training demand, so every
  outlet is on the same scale (1.0 = a normal day for that outlet).
- Lag and rolling features use ONLY past days (shifted by at least one day).
- Calendar, weather and event features are known in advance (from forecasts and schedules).
"""
import numpy as np
import pandas as pd

import config


def fit_scalers(base_df):
    """Statistics learned from the TRAINING period only (no peeking at the future)."""
    train = base_df[base_df.day_index < config.TRAIN_END]
    return {
        "outlet_mean": train.groupby("outlet").demand.mean().to_dict(),
        "temp_mean": float(train.temp.mean()),
        "temp_std": float(train.temp.std()),
    }


def build_features(df, lags, scalers, autoregressive=True):
    """Return (feature DataFrame, list of feature column names).

    autoregressive=False leaves out recent-demand features (lags, rolling stats), so the
    model can forecast months ahead without feeding its own forecasts back in.
    """
    df = df.sort_values(["outlet", "day_index"]).copy()
    df["scale"] = df.outlet.map(scalers["outlet_mean"])
    df["y"] = df.demand / df.scale
    g = df.groupby("outlet")["y"]
    cols = []
    for L in (lags if autoregressive else []):
        df[f"lag_{L}"] = g.shift(L)
        cols.append(f"lag_{L}")
    if autoregressive:
        df["roll_mean_7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
        df["roll_mean_28"] = g.transform(lambda s: s.shift(1).rolling(28).mean())
        df["roll_std_7"] = g.transform(lambda s: s.shift(1).rolling(7).std())
        cols += ["roll_mean_7", "roll_mean_28", "roll_std_7"]

    df["dow_sin"] = np.sin(2 * np.pi * df.dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.dow / 7)
    df["temp_z"] = (df.temp - scalers["temp_mean"]) / scalers["temp_std"]
    cols += ["dow_sin", "dow_cos", "weekend", "holiday", "temp_z", "rain", "event", "semester_break"]
    for grp in config.GROUPS:
        df[f"is_{grp}"] = (df.group == grp).astype(int)
        cols.append(f"is_{grp}")
    if not autoregressive:
        # Without recent demand to lean on, give each outlet type its OWN calendar signals
        # (e.g. "semester break AND hostel") so types don't borrow each other's patterns.
        for grp in config.GROUPS:
            for c in ["weekend", "holiday", "semester_break", "rain", "event"]:
                df[f"{c}_x_{grp}"] = df[c] * df[f"is_{grp}"]
                cols.append(f"{c}_x_{grp}")
        if "regime" in df:
            # regime = 1 after the drift detector found a change in customer behaviour.
            # Lets the model tell "new behaviour" apart from "season" (e.g. semester break).
            cols.append("regime")
            for grp in config.GROUPS:
                for c, name in [(df[f"is_{grp}"], "regime_x_" + grp),
                                (df[f"is_{grp}"] * df.weekend, "regime_weekend_x_" + grp),
                                (df[f"is_{grp}"] * df.holiday, "regime_holiday_x_" + grp)]:
                    df[name] = df.regime * c
                    cols.append(name)

    df = df.dropna(subset=cols).reset_index(drop=True)
    df[cols] = df[cols].astype("float32")
    return df, cols


# Features grouped into concepts, for the explainability report
def feature_concepts(cols):
    concepts = {
        "recent demand (lags)": [c for c in cols if c.startswith("lag_")],
        "rolling averages": [c for c in cols if c.startswith("roll_")],
        "day of week": ["dow_sin", "dow_cos"],
        "weekend flag": ["weekend"],
        "holiday": ["holiday"],
        "weather": ["temp_z", "rain"],
        "special event": ["event"],
        "semester break": ["semester_break"],
        "outlet type": [c for c in cols if c.startswith("is_") or ("_x_" in c and not c.startswith("regime"))],
        "behaviour change (regime)": [c for c in cols if c.startswith("regime")],
    }
    return {k: v for k, v in concepts.items() if v}
