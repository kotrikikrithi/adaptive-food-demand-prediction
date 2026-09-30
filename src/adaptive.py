"""
Adaptation to drift, applied while the model is in use ("online").

The model forecasts one day at a time. When the true demand for that day arrives:
  1. Bias correction  : each outlet type keeps a moving average (EWMA) of its recent
                        errors, and that average is added to the next forecast.
  2. Drift detection  : a Page-Hinkley test per outlet type watches the errors. A
                        persistent shift in the errors raises a drift alarm.
  3. Fine-tuning      : after an alarm, the network is retrained on the most recent days,
                        once a week for ADAPT_DAYS days. The bias is then reset.
  4. Fair intervals   : a separate (Mondrian) conformal interval per outlet type, built from
                        a rolling window of that type's recent errors. Every type then gets
                        about CONFORMAL_LEVEL coverage, even after drift.
"""
from collections import deque

import numpy as np
import pandas as pd

import config
from src.model import predict, train_model


class PageHinkley:
    """Two-sided Page-Hinkley change detector on a stream of errors."""

    def __init__(self, delta=config.PH_DELTA, threshold=config.PH_THRESHOLD):
        self.delta, self.threshold = delta, threshold
        self.reset()

    def reset(self):
        self.n, self.mean, self.up, self.down, self.min_up, self.max_down = 0, 0.0, 0.0, 0.0, 0.0, 0.0

    def update(self, x):
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.up += x - self.mean - self.delta
        self.down += x - self.mean + self.delta
        self.min_up, self.max_down = min(self.min_up, self.up), max(self.max_down, self.down)
        return (self.up - self.min_up > self.threshold) or (self.max_down - self.down > self.threshold)


def run_online(model, hp, cols, history, stream, init_residuals, log=print):
    """Forecast `stream` day by day, adapting as the true values arrive.

    history        : feature rows before the stream (used for fine-tuning windows)
    stream         : feature rows to forecast (the drifted test period)
    init_residuals : {group: array of recent errors} used to start the conformal intervals
    Returns a DataFrame with static and adaptive predictions, intervals, and drift alarms.
    """
    static_model, current = model, model
    bias = {g: 0.0 for g in config.GROUPS}
    detectors = {g: PageHinkley() for g in config.GROUPS}
    windows = {g: deque(np.abs(init_residuals[g])[-config.CONFORMAL_WINDOW:], maxlen=config.CONFORMAL_WINDOW)
               for g in config.GROUPS}
    static_q = {g: float(np.quantile(np.abs(init_residuals[g]), config.CONFORMAL_LEVEL)) for g in config.GROUPS}
    adapt_until, last_ft, alarms, finetunes = -1, -10 ** 9, [], []
    seen = [history]
    out = []

    for day in sorted(stream.day_index.unique()):
        rows = stream[stream.day_index == day]
        X, y, grp = rows[cols].values, rows.y.values, rows.group.values
        p_static = predict(static_model, X)
        p_model = predict(current, X)
        p_adapt = p_model + np.array([bias[g] for g in grp])
        q = np.array([np.quantile(windows[g], config.CONFORMAL_LEVEL) for g in grp])
        sq = np.array([static_q[g] for g in grp])
        out.append(pd.DataFrame({
            "day_index": day, "date": rows.date.values, "outlet": rows.outlet.values, "group": grp,
            "y": y, "scale": rows.scale.values, "pred_static": p_static, "pred_adaptive": p_adapt,
            "lo_static": p_static - sq, "hi_static": p_static + sq,
            "lo_adaptive": p_adapt - q, "hi_adaptive": p_adapt + q}))

        # ---- the true demand for today arrives: learn from it ----
        seen.append(rows)
        for g in np.unique(grp):
            m = grp == g
            err = float(np.mean(y[m] - p_model[m]))
            bias[g] = (1 - config.BIAS_ALPHA) * bias[g] + config.BIAS_ALPHA * err
            windows[g].extend(np.abs(y[m] - p_adapt[m]))
            if detectors[g].update(float(np.mean(y[m] - p_adapt[m]))):
                alarms.append({"day_index": int(day), "group": g})
                log(f"  drift alarm on day {day} for {g}")
                detectors[g].reset()
                adapt_until = day + config.ADAPT_DAYS

        if day <= adapt_until and day - last_ft >= config.FINETUNE_EVERY:
            recent = pd.concat(seen[-(config.FINETUNE_WINDOW + 1):])  # history + recent days
            recent = recent[recent.day_index > day - config.FINETUNE_WINDOW]
            current, _, _ = train_model(recent[cols].values, recent.y.values, None, None, hp,
                                        max_epochs=config.FINETUNE_EPOCHS, patience=None,
                                        init_model=current, lr=hp["lr"] * config.FINETUNE_LR_MULT)
            bias = {g: 0.0 for g in config.GROUPS}
            last_ft = day
            finetunes.append(int(day))
            log(f"  fine-tuned on days {int(recent.day_index.min())}-{int(day)}")

    return pd.concat(out, ignore_index=True), alarms, finetunes
