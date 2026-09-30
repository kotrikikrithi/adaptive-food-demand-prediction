"""
6-month forecast: learn from ALL 2 years of data, then predict meals for every outlet
for every day until config.FORECAST_END.

Method ("direct" forecasting):
  The model predicts each future day straight from that day's conditions. It does not
  feed its own forecasts back in, so errors can't pile up over 6 months.
    - weekends, holidays, semester break : known from the calendar
    - weather                            : typical for that date (average of past years)
    - special events                     : only the dates in config.FUTURE_EVENT_DATES
    - outlet type                        : each type gets its own calendar signals
    - behaviour change ("regime")        : the Page-Hinkley drift detector scans the history
                                           and finds when customer behaviour changed. Days
                                           after that are marked, so the model learns the new
                                           behaviour separately from the season.
  Recent days also count more in training (half-life config.RECENCY_HALF_LIFE_DAYS).

How much to trust it (backtest): pretend it is 30 June 2025 and forecast Jul-Dec 2025, then
compare with what actually happened:
    - "one-shot"        : forecast all 6 months at once
    - "updated monthly" : retrain with the newest data at the start of each month (adaptive)
    - "naive"           : baseline, same weekday average of the last 4 weeks

Run: python -m src.forecast        (saves results to results/forecast/)
"""
import json
import os

import numpy as np
import pandas as pd

import config
from src import metrics as M
from src.adaptive import PageHinkley
from src.data_generator import HOLIDAYS, generate
from src.features import build_features
from src.model import count_params, predict, train_model

OUT_DIR = "results/forecast"
DEFAULT_HP = {"n_layers": 1, "width": 16, "lr": 0.00031, "dropout": 0.07, "weight_decay": 0.00019,
              "lag_set": 1, "activation": "relu", "huber_delta": 0.53}


def load_history():
    """Our 2 years of data (includes the Aug-2025 change in customer behaviour)."""
    return generate(config.TEST_SCENARIO, drift_start_day=config.VAL_END, strength=config.TEST_STRENGTH)


def best_settings():
    """Network settings found by evolution: the best full (non-quick) attempt, else a default."""
    path = "results/attempts.csv"
    if os.path.exists(path):
        a = pd.read_csv(path)
        full = a[~a.quick.astype(bool)] if "quick" in a else a
        a = full if len(full) else a
        return json.loads(a.sort_values("final_fitness").iloc[0].settings)
    return DEFAULT_HP


def future_rows(hist, start, end):
    """Rows for future days, with conditions filled in (demand unknown)."""
    one = hist[hist.outlet == hist.outlet.iloc[0]]
    temp_by_doy = one.groupby(one.date.dt.dayofyear).temp.mean().reindex(range(1, 367)).interpolate()
    temp_by_doy = pd.concat([temp_by_doy.iloc[-15:], temp_by_doy, temp_by_doy.iloc[:15]]) \
        .rolling(31, center=True, min_periods=1).mean().iloc[15:-15]   # smooth, wrapping around the year
    rain_by_month = one.groupby(one.date.dt.month).rain.mean()
    events = set(pd.to_datetime(config.FUTURE_EVENT_DATES))
    dates = pd.date_range(start, end, freq="D")
    first = int(hist.day_index.max()) + 1
    base = pd.DataFrame({
        "date": dates, "day_index": np.arange(first, first + len(dates)),
        "dow": dates.dayofweek, "weekend": (dates.dayofweek >= 5).astype(int),
        "holiday": [int((d.month, d.day) in HOLIDAYS) for d in dates],
        "temp": temp_by_doy.reindex(dates.dayofyear).values.round(1),
        "rain": rain_by_month.reindex(dates.month).values,     # chance of rain (0-1)
        "event": [int(d in events) for d in dates],
        "semester_break": dates.month.isin([5, 6]).astype(int),
    })
    rows = [base.assign(outlet=o, group=g) for o, g, _ in config.OUTLETS]
    return pd.concat(rows, ignore_index=True).assign(demand=np.nan)


def _scalers(hist):
    return {"outlet_mean": hist.groupby("outlet").demand.mean().to_dict(),
            "temp_mean": float(hist.temp.mean()), "temp_std": float(hist.temp.std())}


def fit(hist, hp, max_epochs=config.MAX_EPOCHS):
    """Train on all rows of `hist` (recent days weighted more)."""
    scalers = _scalers(hist)
    f, cols = build_features(hist, [], scalers, autoregressive=False)
    last = f.day_index.max()
    w = 0.5 ** ((last - f.day_index.values) / config.RECENCY_HALF_LIFE_DAYS)
    w = (w / w.mean()).astype("float32")
    model, _, _ = train_model(f[cols].values, f.y.values, None, None, hp, max_epochs=max_epochs,
                              patience=None, sample_weight=w)
    return model, scalers, cols


def forecast_rows(model, scalers, cols, rows):
    f, _ = build_features(rows.fillna({"demand": 0.0}), [], scalers, autoregressive=False)
    return f.assign(pred=np.clip(predict(model, f[cols].values), 0, None) * f.scale.values)


def detect_change_day(hist, hp):
    """Find the first day customer behaviour changed, using the Page-Hinkley drift detector.

    A model learns the first year, then we watch its daily errors on the rest of the history.
    Returns the day_index of the first alarm (or None when no change is found).
    """
    first_year = hist[hist.day_index < 365]
    rest = hist[hist.day_index >= 365]
    if rest.empty:
        return None
    scalers = _scalers(first_year)
    f, cols = build_features(first_year, [], scalers, autoregressive=False)
    model, _, _ = train_model(f[cols].values, f.y.values, None, None, hp, patience=None)
    r = forecast_rows(model, scalers, cols, rest)
    r["err"] = (r.demand - r.pred) / r.scale
    detectors = {g: PageHinkley(config.CHANGE_SCAN_DELTA, config.CHANGE_SCAN_THRESHOLD) for g in config.GROUPS}
    for day, d in r.groupby("day_index"):
        for g, dg in d.groupby("group"):
            if detectors[g].update(float(dg.err.mean())):
                return int(day)
    return None


def with_regime(hist, change_day):
    return hist.assign(regime=(hist.day_index >= change_day).astype(int) if change_day is not None else 0)


def fit_and_forecast(hist, hp, start, end):
    change = detect_change_day(hist, hp)
    h = with_regime(hist, change)
    model, scalers, cols = fit(h, hp)
    fut = future_rows(hist, start, end).assign(regime=int(change is not None))
    return forecast_rows(model, scalers, cols, fut), model, change


def seasonal_naive(hist, rows):
    """Baseline: same weekday, average of the last 4 weeks before the forecast starts."""
    recent = hist[hist.day_index > hist.day_index.max() - 28]
    avg = recent.groupby(["outlet", "dow"]).demand.mean()
    return avg.reindex(list(zip(rows.outlet, rows.date.dt.dayofweek))).values


def run(log=print):
    os.makedirs(OUT_DIR, exist_ok=True)
    hist = load_history()
    hp = best_settings()
    last_date = hist.date.max()
    log(f"Settings from evolution: {hp}")

    # ---------- backtest: pretend it is 30 June 2025 ----------
    bt_start = pd.Timestamp(config.BACKTEST_START)
    past, actual = hist[hist.date < bt_start], hist[hist.date >= bt_start]
    log("Backtest 1/2: one 6-month forecast made on 30 June 2025 ...")
    one_shot, _, _ = fit_and_forecast(past, hp, bt_start, last_date)
    log("Backtest 2/2: forecast updated at the start of every month ...")
    monthly = []
    for month_start in pd.date_range(bt_start, last_date, freq="MS"):
        month_end = min(month_start + pd.offsets.MonthEnd(0), last_date)
        seen = hist[hist.date < month_start]
        fc_m, _, ch = fit_and_forecast(seen, hp, month_start, month_end)
        monthly.append(fc_m)
        log(f"  {month_start:%b %Y}: learned from data up to {(month_start - pd.Timedelta(days=1)).date()}"
            + (f", behaviour change detected on day {ch}" if ch is not None else ""))
    monthly = pd.concat(monthly, ignore_index=True)

    bt = actual[["date", "outlet", "group", "demand"]].rename(columns={"demand": "actual"})
    bt = bt.merge(one_shot[["date", "outlet", "pred"]].rename(columns={"pred": "one_shot"}), on=["date", "outlet"])
    bt = bt.merge(monthly[["date", "outlet", "pred"]].rename(columns={"pred": "monthly"}), on=["date", "outlet"])
    bt["naive"] = seasonal_naive(past, bt)
    bt[["one_shot", "monthly", "naive"]] = bt[["one_shot", "monthly", "naive"]].round()
    bt.to_csv(f"{OUT_DIR}/backtest.csv", index=False)

    def scores(col):
        return {"overall": M.wape(bt.actual.values, bt[col].values),
                **{g: M.wape(bt.actual[bt.group == g].values, bt[col][bt.group == g].values) for g in config.GROUPS}}
    bt_scores = {c: scores(c) for c in ["naive", "one_shot", "monthly"]}
    # 90% safe range per outlet type, from the monthly-updated backtest errors (relative to forecast)
    rel = (bt.actual - bt.monthly).abs() / bt.monthly.clip(lower=1)
    band = {g: float(np.quantile(rel[bt.group == g], config.CONFORMAL_LEVEL)) for g in config.GROUPS}

    # ---------- final: learn from all 2 years, forecast the next 6 months ----------
    log("Final: learning from all 2 years and forecasting the next 6 months ...")
    start = last_date + pd.Timedelta(days=1)
    fc, model, change = fit_and_forecast(hist, hp, start, config.FORECAST_END)
    fc["low"] = fc.pred * (1 - fc.group.map(band))
    fc["high"] = fc.pred * (1 + fc.group.map(band))
    fc[["pred", "low", "high"]] = fc[["pred", "low", "high"]].round().clip(lower=0).astype(int)
    fc = fc[["date", "outlet", "group", "pred", "low", "high", "weekend", "holiday", "semester_break",
             "temp", "rain", "event"]].rename(columns={"rain": "rain_chance"})
    fc.to_csv(f"{OUT_DIR}/forecast.csv", index=False)

    change_date = str(hist.date[hist.day_index == change].iloc[0].date()) if change is not None else None
    summary = {"settings": hp, "n_params": count_params(model),
               "history_start": str(hist.date.min().date()), "history_end": str(last_date.date()),
               "forecast_start": str(start.date()), "forecast_end": config.FORECAST_END,
               "change_detected_on": change_date, "backtest_start": config.BACKTEST_START,
               "backtest_wape": bt_scores, "band": band, "half_life_days": config.RECENCY_HALF_LIFE_DAYS}
    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Behaviour change detected on: {change_date}")
    log(f"Backtest error (WAPE): naive {bt_scores['naive']['overall']:.1%} | one-shot "
        f"{bt_scores['one_shot']['overall']:.1%} | updated monthly {bt_scores['monthly']['overall']:.1%}")
    log(f"Saved to {OUT_DIR}/")
    return summary


if __name__ == "__main__":
    run()
