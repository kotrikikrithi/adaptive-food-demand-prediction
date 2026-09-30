"""
Synthetic food-demand data with controllable drift.

Demand for each outlet on each day is built in log space:
    log(demand) = log(base) + trend + weekend + holiday + rain + heat
                  + event + semester_break + noise
Each outlet type (restaurant / cafeteria / hostel) reacts differently to each driver.

A drift scenario changes these rules from `drift_start_day` onward. With the same
seed, the days before the drift are identical in every scenario, so the only thing
that differs between scenarios is the drift itself.

Run `python -m src.data_generator` to save a sample CSV and plot to results/.
"""
import numpy as np
import pandas as pd

import config

# Fixed-date public holidays and festival days (month, day)
HOLIDAYS = {(1, 1), (1, 14), (1, 26), (3, 14), (4, 14), (8, 15), (10, 2),
            (10, 20), (10, 21), (11, 1), (12, 25), (12, 31)}


def generate(scenario="none", drift_start_day=None, strength=1.0, seed=config.SEED):
    """Return a DataFrame with one row per (outlet, day)."""
    if scenario not in config.DRIFT_SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {config.DRIFT_SCENARIOS}")
    rng = np.random.default_rng(seed)
    dates = pd.date_range(config.START_DATE, periods=config.N_DAYS, freq="D")
    n = len(dates)
    t = np.arange(n)
    doy = dates.dayofyear.values
    dow = dates.dayofweek.values
    month = dates.month.values

    # Shared drivers. All base randomness is drawn first, in a fixed order.
    weekend = dow >= 5
    holiday = np.array([(d.month, d.day) in HOLIDAYS for d in dates])
    temp = 27 + 6 * np.sin(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 2, n)
    rain = rng.random(n) < np.where(np.isin(month, [6, 7, 8, 9]), 0.55, 0.08)  # monsoon
    event = rng.random(n) < config.EVENT_PROB
    semester_break = np.isin(month, [5, 6])
    noise = rng.normal(0, config.NOISE_STD, (len(config.OUTLETS), n))

    # Drift masks
    start = n if drift_start_day is None or scenario == "none" else drift_start_day
    after = t >= start
    ramp = np.clip((t - start) / max(1, n - start), 0, 1)  # 0 -> 1 across the drift period
    s = strength
    event_boost = np.ones(n)
    if scenario == "sudden_event_spike":
        # Scenario-only randomness uses its own generator so base data stays identical
        extra = np.random.default_rng(seed + 1).random(n) < 0.20 * s
        event = event | (after & extra)
        event_boost = np.where(after, 1 + 0.5 * s, 1.0)

    rows = []
    for k, (name, otype, base) in enumerate(config.OUTLETS):
        p = config.GROUP_EFFECTS[otype]
        weekend_eff = np.where(weekend, p["weekend"], 0.0)
        rain_eff = np.where(rain, p["rain"], 0.0)
        heat = p["heat"] * (temp - 27) / 6
        drift_eff = np.zeros(n)

        if scenario == "weekend_pattern_flip":
            # e.g. the office cafeteria starts opening at weekends and restaurants
            # lose their weekend rush: the weekend effect shrinks and then reverses
            weekend_eff = np.where(after, weekend_eff * (1 - 1.5 * s), weekend_eff)
        elif scenario == "weather_sensitivity_change":
            rain_eff = np.where(after, rain_eff * (1 + 3 * s), rain_eff)
            heat = np.where(after, heat * (1 + 2 * s), heat)
        elif scenario == "level_shift":
            drift_eff = after * s * config.LEVEL_SHIFT[otype]
        elif scenario == "gradual_trend":
            drift_eff = ramp * s * config.GRADUAL_TREND[otype]

        log_d = (np.log(base)
                 + config.TREND_PER_YEAR * t / 365
                 + weekend_eff
                 + np.where(holiday, p["holiday"], 0.0)
                 + rain_eff + heat
                 + np.where(event, p["event"] * event_boost, 0.0)
                 + np.where(semester_break, p["semester_break"], 0.0)
                 + drift_eff
                 + noise[k])
        demand = np.round(np.exp(log_d)).astype(int)
        rows.append(pd.DataFrame({
            "date": dates, "day_index": t, "outlet": name, "group": otype,
            "dow": dow, "weekend": weekend.astype(int), "holiday": holiday.astype(int),
            "temp": temp.round(1), "rain": rain.astype(int), "event": event.astype(int),
            "semester_break": semester_break.astype(int), "demand": demand,
        }))
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("results", exist_ok=True)
    df = generate(config.TEST_SCENARIO, drift_start_day=config.VAL_END)
    df.to_csv("results/sample_data.csv", index=False)
    fig, axes = plt.subplots(len(config.GROUPS), 1, figsize=(12, 8), sharex=True)
    for ax, g in zip(axes, config.GROUPS):
        for outlet, d in df[df.group == g].groupby("outlet"):
            ax.plot(d.date, d.demand, lw=0.7, label=outlet)
        ax.axvline(df.date.iloc[config.VAL_END], color="red", ls="--", label="drift starts")
        ax.set_title(g)
        ax.legend(fontsize=7)
    fig.suptitle(f"Synthetic demand, drift scenario: {config.TEST_SCENARIO}")
    fig.tight_layout()
    fig.savefig("results/sample_data.png", dpi=120)
    print(df.head(10).to_string())
    print(f"\nSaved {len(df)} rows to results/sample_data.csv and a plot to results/sample_data.png")
    print("\nMean demand by type, weekday vs weekend:")
    print(df[df.day_index < config.VAL_END].groupby(["group", "weekend"]).demand.mean().round(1).unstack())
