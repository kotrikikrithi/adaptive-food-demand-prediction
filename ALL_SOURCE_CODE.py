"""
ALL SOURCE CODE of adaptive-food-demand-prediction, combined into one file for reading and submission.
Each section below is one project file (the file name is shown in the banner).
To RUN the project, use the separate files:  python run_attempt.py   /   streamlit run app.py
Repo: https://github.com/kotrikikrithi/adaptive-food-demand-prediction
"""


# ============================================================================
# FILE: config.py
# ============================================================================

"""
All parameters for the project, in ONE place.

Hackathon checklist item: "Parameter configuration used, clearly stated in code comments".
Change a value here, run `python run_attempt.py --note "what changed and why"`, and the
new score is logged in results/attempts.csv.
"""

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42                      # every random choice (data, weights, evolution) comes from this seed

# ---------------------------------------------------------------------------
# Synthetic data (src/data_generator.py)
# ---------------------------------------------------------------------------
START_DATE = "2024-01-01"      # first simulated day
N_DAYS = 730                   # 2 years of daily data

# (outlet name, outlet type, average meals per day). The outlet TYPE is the
# "sub-population" used for the fairness / calibration checks.
OUTLETS = [
    ("restaurant_A", "restaurant", 220),
    ("restaurant_B", "restaurant", 140),
    ("cafeteria_A", "cafeteria", 400),
    ("cafeteria_B", "cafeteria", 260),
    ("hostel_A", "hostel", 600),
    ("hostel_B", "hostel", 350),
]
GROUPS = ["restaurant", "cafeteria", "hostel"]

# How each outlet type reacts to each driver. Values are in LOG space:
# +0.30 means about +35% demand, -1.0 means about -63% demand.
GROUP_EFFECTS = {
    #               weekend holiday  rain  heat(per 6C) event semester_break
    "restaurant": dict(weekend=0.35, holiday=0.30, rain=-0.15, heat=-0.05, event=0.45, semester_break=0.00),
    "cafeteria":  dict(weekend=-1.00, holiday=-1.10, rain=0.05, heat=-0.03, event=0.25, semester_break=-0.05),
    "hostel":     dict(weekend=-0.10, holiday=-0.35, rain=0.08, heat=-0.04, event=0.15, semester_break=-0.60),
}
TREND_PER_YEAR = 0.05          # slow growth in demand (+5% per year)
NOISE_STD = 0.08               # random day-to-day noise (about 8%)
EVENT_PROB = 0.03              # chance of a special event on any day (events are known in advance)

# ---------------------------------------------------------------------------
# Time splits (by day index). Time order is never shuffled, so the model
# never sees the future.
# ---------------------------------------------------------------------------
TRAIN_END = 438                # days [0, 438)   -> training           (60%)
VAL_END = 584                  # days [438, 584) -> OOD validation     (20%)
                               # days [584, 730) -> final test / drift (20%)
EARLY_STOP_DAYS = 60           # last 60 days before a split are used for early stopping

# ---------------------------------------------------------------------------
# Drift scenarios (non-stationary behaviour). Each scenario starts at a given
# day with a strength from 0 (no change) to 1 (full change).
# ---------------------------------------------------------------------------
DRIFT_SCENARIOS = ["none", "level_shift", "weekend_pattern_flip",
                   "weather_sensitivity_change", "sudden_event_spike", "gradual_trend"]
# Per-type size of the level shift / trend drift (log space). Different types
# drift differently, which is what makes fair calibration hard.
LEVEL_SHIFT = {"restaurant": 0.30, "cafeteria": -0.35, "hostel": 0.20}
GRADUAL_TREND = {"restaurant": 0.45, "cafeteria": 0.25, "hostel": -0.30}

# Validation = several DIFFERENT mildly-drifted copies of the validation period.
# Scoring a model on all of them rewards models that generalise out of distribution.
VAL_SCENARIOS = ["none", "level_shift", "gradual_trend", "weather_sensitivity_change"]
VAL_STRENGTH = 0.5             # validation drifts are milder than the test drift
TEST_SCENARIO = "weekend_pattern_flip"   # default Round-2 style drift (change with --drift)
TEST_STRENGTH = 1.0

# ---------------------------------------------------------------------------
# Features (src/features.py)
# ---------------------------------------------------------------------------
# Lag sets the evolution may choose from ("how many past days to look at").
LAG_SETS = [[1, 7], [1, 2, 7, 14], [1, 2, 3, 7, 14, 21, 28]]

# ---------------------------------------------------------------------------
# Neural network training (src/model.py)
# ---------------------------------------------------------------------------
MAX_EPOCHS = 80                # upper limit on training passes over the data
PATIENCE = 10                  # stop early after 10 epochs with no validation improvement
BATCH_SIZE = 64
GRAD_CLIP_NORM = 1.0           # gradient clipping -> no gradient explosion

# ---------------------------------------------------------------------------
# Evolutionary search, NSGA-II (src/evolution.py)
# ---------------------------------------------------------------------------
POP_SIZE = 10                  # candidate networks per generation
N_GENERATIONS = 6              # generations of evolution
CROSSOVER_PROB = 0.9           # chance that two parents are mixed
MUTATION_PROB = 0.25           # per-gene chance of a random change
MUTATION_SIGMA = 0.15          # size of that change (genes live in [0, 1])
# Search space (decoded from genes in [0, 1])
LAYER_CHOICES = [1, 2, 3]
WIDTH_CHOICES = [8, 16, 32, 64]
LR_RANGE_LOG10 = (-4.0, -2.0)            # learning rate 1e-4 .. 1e-2
DROPOUT_RANGE = (0.0, 0.4)
WEIGHT_DECAY_RANGE_LOG10 = (-6.0, -2.0)  # L2 regularisation 1e-6 .. 1e-2
HUBER_DELTA_RANGE = (0.05, 1.0)          # robust loss: small delta = less sensitive to outliers
ACTIVATIONS = ["relu", "gelu", "tanh"]

# ---------------------------------------------------------------------------
# Deployment budget (hard constraint)
# ---------------------------------------------------------------------------
PARAM_BUDGET = 5000            # max trainable parameters allowed
LATENCY_BUDGET_MS = 2.0        # max milliseconds for one prediction (checked and reported)

# ---------------------------------------------------------------------------
# Fitness = weighted sum of the objectives (LOWER is better)
# ---------------------------------------------------------------------------
W_ACCURACY = 1.0               # weight on error (WAPE)
W_VARIANCE = 1.0               # weight on error variance + overfitting gap
W_COST = 0.1                   # weight on model size vs budget
W_FAIRNESS = 0.5               # weight on the calibration gap between outlet types

# ---------------------------------------------------------------------------
# Adaptation to drift (src/adaptive.py)
# ---------------------------------------------------------------------------
BIAS_ALPHA = 0.15              # how fast the per-type bias correction follows recent errors
PH_DELTA = 0.02                # Page-Hinkley: tolerated error drift before it counts
PH_THRESHOLD = 0.6             # Page-Hinkley: alarm level (lower = detects drift sooner)
ADAPT_DAYS = 35                # after a drift alarm, keep adapting for 35 days
FINETUNE_EVERY = 7             # ...fine-tuning every 7 days in that period
FINETUNE_WINDOW = 42           # fine-tune on the most recent 42 days
FINETUNE_EPOCHS = 30
FINETUNE_LR_MULT = 0.5         # fine-tune learning rate = best lr * 0.5
CONFORMAL_LEVEL = 0.90         # prediction intervals should contain 90% of true values
CONFORMAL_WINDOW = 60          # rolling window of recent errors per type, for the intervals

# ---------------------------------------------------------------------------
# 6-month forecast (src/forecast.py): train on ALL 2 years, predict the future
# ---------------------------------------------------------------------------
FORECAST_END = "2026-06-30"    # forecast every day from the day after our data ends until this date
BACKTEST_START = "2025-07-01"  # honesty check: pretend it is 30 June 2025 and forecast Jul-Dec 2025
RECENCY_HALF_LIFE_DAYS = 120   # a day 120 days old counts half as much as yesterday in training,
                               # so the forecast follows the NEWEST customer behaviour
FUTURE_EVENT_DATES = []        # planned special events, e.g. ["2026-02-14", "2026-03-20"]
# Future weather is unknown, so we use the TYPICAL weather for that time of year
# (average temperature for the date, and the chance of rain for the month).
# Drift detector settings for scanning the 2-year HISTORY once (less sensitive than the daily
# online detector, so the slow +5%/year growth is not mistaken for a behaviour change).
CHANGE_SCAN_DELTA = 0.08
CHANGE_SCAN_THRESHOLD = 1.5


# ============================================================================
# FILE: src/data_generator.py
# ============================================================================

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


# ============================================================================
# FILE: src/features.py
# ============================================================================

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


# ============================================================================
# FILE: src/model.py
# ============================================================================

"""
A small neural network (MLP) and a training loop built for stability.

Stability tools used:
  - Huber loss            : robust to outlier days (festivals, event spikes)
  - AdamW weight decay    : L2 regularisation against overfitting
  - Dropout               : more regularisation
  - Gradient clipping     : gradients can never explode (max norm = config.GRAD_CLIP_NORM)
  - Early stopping        : stop when validation loss stops improving
  - Fixed seeds + deterministic algorithms : same input -> same result, every run
"""
import copy
import random
import time

import numpy as np
import torch
from torch import nn

import config

ACT = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class MLP(nn.Module):
    def __init__(self, in_dim, n_layers, width, dropout, activation):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, width), ACT[activation](), nn.Dropout(dropout)]
            d = width
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(X_tr, y_tr, X_va, y_va, hp, seed=config.SEED, max_epochs=config.MAX_EPOCHS,
                patience=config.PATIENCE, init_model=None, lr=None, sample_weight=None):
    """Train (or fine-tune, if init_model is given). Returns (model, history, diverged).

    sample_weight (optional): per-row importance, e.g. to make recent days count more.
    """
    set_seed(seed)
    X_tr, y_tr = torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.float32)
    has_val = X_va is not None and len(X_va) > 0
    if has_val:
        X_va, y_va = torch.tensor(X_va), torch.tensor(y_va, dtype=torch.float32)

    model = copy.deepcopy(init_model) if init_model is not None else \
        MLP(X_tr.shape[1], hp["n_layers"], hp["width"], hp["dropout"], hp["activation"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr or hp["lr"], weight_decay=hp["weight_decay"])
    loss_fn = nn.HuberLoss(delta=hp["huber_delta"])
    w = None if sample_weight is None else torch.tensor(sample_weight, dtype=torch.float32)
    weighted_fn = nn.HuberLoss(delta=hp["huber_delta"], reduction="none")
    gen = torch.Generator().manual_seed(seed)

    history, best_state, best_val, bad = [], None, float("inf"), 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr), generator=gen)
        total, max_gn = 0.0, 0.0
        for i in range(0, len(X_tr), config.BATCH_SIZE):
            idx = perm[i:i + config.BATCH_SIZE]
            opt.zero_grad()
            if w is None:
                loss = loss_fn(model(X_tr[idx]), y_tr[idx])
            else:
                loss = (weighted_fn(model(X_tr[idx]), y_tr[idx]) * w[idx]).mean()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            max_gn = max(max_gn, float(gn))
            opt.step()
            total += loss.item() * len(idx)
        train_loss = total / len(X_tr)
        if not np.isfinite(train_loss):
            return model, history, True  # diverged (should never happen thanks to clipping)

        val_loss = float("nan")
        if has_val:
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(X_va), y_va))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "max_grad_norm_before_clip": max_gn})

        if has_val and patience:
            if val_loss < best_val - 1e-6:
                best_val, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history, False


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X)).numpy()


def measure_latency_ms(model, in_dim, n=300):
    """Average time for ONE prediction (batch of 1), in milliseconds."""
    x = torch.zeros(1, in_dim)
    model.eval()
    with torch.no_grad():
        for _ in range(20):  # warm-up
            model(x)
        t0 = time.perf_counter()
        for _ in range(n):
            model(x)
    return (time.perf_counter() - t0) / n * 1000


# ============================================================================
# FILE: src/evolution.py
# ============================================================================

"""
NSGA-II: a multi-objective genetic algorithm that evolves the network's hyperparameters.

Representation: a genome is 8 numbers in [0, 1]. `decode` turns them into settings:
    [n_layers, width, learning rate, dropout, weight decay, lag set, activation, huber delta]
Operators:
    - Binary tournament selection (lower Pareto rank wins; ties go to the larger crowding distance)
    - Blend crossover (each child gene is a random mix of the two parent genes)
    - Gaussian mutation (small random nudge, clipped to [0, 1])
Survival: parents + children are sorted into Pareto fronts. The best fronts are kept, and
    crowding distance keeps the front diverse.
Constraints: constrained domination. A feasible solution (within the parameter budget, did
    not diverge) always beats an infeasible one.
"""
import json

import numpy as np

import config

GENE_NAMES = ["n_layers", "width", "lr", "dropout", "weight_decay", "lag_set", "activation", "huber_delta"]


def _pick(choices, u):
    return choices[min(int(u * len(choices)), len(choices) - 1)]


def decode(g):
    lo, hi = config.LR_RANGE_LOG10
    wlo, whi = config.WEIGHT_DECAY_RANGE_LOG10
    dlo, dhi = config.DROPOUT_RANGE
    hlo, hhi = config.HUBER_DELTA_RANGE
    return {
        "n_layers": _pick(config.LAYER_CHOICES, g[0]),
        "width": _pick(config.WIDTH_CHOICES, g[1]),
        "lr": float(f"{10 ** (lo + (hi - lo) * g[2]):.2g}"),
        "dropout": float(round(dlo + (dhi - dlo) * g[3], 2)),
        "weight_decay": float(f"{10 ** (wlo + (whi - wlo) * g[4]):.2g}"),
        "lag_set": _pick(list(range(len(config.LAG_SETS))), g[5]),
        "activation": _pick(config.ACTIVATIONS, g[6]),
        "huber_delta": float(round(hlo + (hhi - hlo) * g[7], 2)),
    }


def dominates(fa, va, fb, vb):
    """Constrained domination: does solution a beat solution b?"""
    if va == 0 and vb > 0:
        return True
    if va > 0 and vb > 0:
        return va < vb
    if va > 0 and vb == 0:
        return False
    return bool(np.all(fa <= fb) and np.any(fa < fb))


def nondominated_sort(F, V):
    n = len(F)
    S = [[] for _ in range(n)]
    dom_count = np.zeros(n, int)
    fronts = [[]]
    for i in range(n):
        for j in range(n):
            if i != j and dominates(F[i], V[i], F[j], V[j]):
                S[i].append(j)
            elif i != j and dominates(F[j], V[j], F[i], V[i]):
                dom_count[i] += 1
        if dom_count[i] == 0:
            fronts[0].append(i)
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in S[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding_distance(F):
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for k in range(m):
        order = np.argsort(F[:, k], kind="stable")
        d[order[0]] = d[order[-1]] = np.inf
        span = F[order[-1], k] - F[order[0], k]
        if span > 0:
            d[order[1:-1]] += (F[order[2:], k] - F[order[:-2], k]) / span
    return d


def _rank_and_crowd(F, V):
    rank, crowd = np.zeros(len(F), int), np.zeros(len(F))
    fronts = nondominated_sort(F, V)
    for r, fr in enumerate(fronts):
        rank[fr] = r
        crowd[fr] = crowding_distance(F[fr])
    return rank, crowd, fronts


def run_nsga2(evaluate, rng, pop_size=config.POP_SIZE, n_gen=config.N_GENERATIONS, log=print):
    """evaluate(hp) -> dict with 'objectives' (np.array, lower is better), 'violation', 'fitness'.

    Returns (final population as a list of result dicts, per-generation log rows).
    """
    cache = {}

    def eval_genome(g):
        hp = decode(g)
        key = json.dumps(hp, sort_keys=True)
        if key not in cache:  # identical settings -> identical (deterministic) result
            cache[key] = {**evaluate(hp), "hp": hp}
        return cache[key]

    n_genes = len(GENE_NAMES)
    pop = [rng.random(n_genes) for _ in range(pop_size)]
    res = [eval_genome(g) for g in pop]
    gen_log = []

    for gen in range(n_gen + 1):
        F = np.array([r["objectives"] for r in res])
        V = np.array([r["violation"] for r in res])
        rank, crowd, fronts = _rank_and_crowd(F, V)

        feasible = [r for r in res if r["violation"] == 0]
        best = min(feasible or res, key=lambda r: r["fitness"])
        seen = [r for r in cache.values() if r["violation"] == 0] or list(cache.values())
        row = {"generation": gen, "evaluations": len(cache), "pareto_front_size": len(fronts[0]),
               "best_fitness": best["fitness"],
               "best_fitness_so_far": min(r["fitness"] for r in seen),
               "mean_pop_fitness": float(np.mean([r["fitness"] for r in res])),
               "best_val_wape": float(F[:, 0].min()),
               "min_variance": float(F[:, 1].min()), "min_cost": float(F[:, 2].min()),
               "min_calibration_gap": float(F[:, 3].min()), "feasible": len(feasible)}
        gen_log.append(row)
        log(f"  gen {gen:2d} | evals {row['evaluations']:3d} | front {row['pareto_front_size']:2d} | "
            f"best fitness {row['best_fitness']:.4f} | pop mean {row['mean_pop_fitness']:.4f} | best val WAPE {row['best_val_wape']:.4f} | "
            f"min calib gap {row['min_calibration_gap']:.4f} | best = {best['hp']}")
        if gen == n_gen:
            break

        # --- make children ---
        def tournament():
            a, b = rng.integers(0, pop_size, 2)
            if rank[a] != rank[b]:
                return a if rank[a] < rank[b] else b
            return a if crowd[a] >= crowd[b] else b

        children = []
        while len(children) < pop_size:
            p1, p2 = pop[tournament()], pop[tournament()]
            if rng.random() < config.CROSSOVER_PROB:
                w = rng.random(n_genes)
                c1, c2 = w * p1 + (1 - w) * p2, w * p2 + (1 - w) * p1
            else:
                c1, c2 = p1.copy(), p2.copy()
            for c in (c1, c2):
                mask = rng.random(n_genes) < config.MUTATION_PROB
                c[mask] += rng.normal(0, config.MUTATION_SIGMA, mask.sum())
                children.append(np.clip(c, 0, 1))
        children = children[:pop_size]
        child_res = [eval_genome(g) for g in children]

        # --- survival: best fronts of parents + children ---
        all_pop, all_res = pop + children, res + child_res
        F = np.array([r["objectives"] for r in all_res])
        V = np.array([r["violation"] for r in all_res])
        _, _, fronts = _rank_and_crowd(F, V)
        chosen = []
        for fr in fronts:
            if len(chosen) + len(fr) <= pop_size:
                chosen += fr
            else:
                cd = crowding_distance(F[fr])
                chosen += [fr[i] for i in np.argsort(-cd, kind="stable")[:pop_size - len(chosen)]]
                break
        pop = [all_pop[i] for i in chosen]
        res = [all_res[i] for i in chosen]

    return res, gen_log


# ============================================================================
# FILE: src/adaptive.py
# ============================================================================

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


# ============================================================================
# FILE: src/metrics.py
# ============================================================================

"""
Evaluation metrics. y and pred are in normalised units (1.0 = the outlet's average day)
unless stated otherwise.
"""
import numpy as np

import config


def wape(y, pred):
    """Weighted absolute percentage error: total |error| / total demand. 0.10 = 10% off."""
    return float(np.abs(y - pred).sum() / max(np.abs(y).sum(), 1e-9))


def rmse(y, pred):
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def group_bias(y, pred, groups):
    """Relative bias per outlet type: +0.05 means that type is over-predicted by 5%."""
    return {g: float((pred[groups == g] - y[groups == g]).sum() / max(y[groups == g].sum(), 1e-9))
            for g in config.GROUPS if (groups == g).any()}


def bias_gap(y, pred, groups):
    """Calibration gap between types: max bias - min bias (0 = equally calibrated)."""
    b = group_bias(y, pred, groups)
    return float(max(b.values()) - min(b.values()))


def coverage_by_group(y, lo, hi, groups):
    """Share of true values that fall inside the prediction interval, per type."""
    inside = (y >= lo) & (y <= hi)
    return {g: float(inside[groups == g].mean()) for g in config.GROUPS if (groups == g).any()}


def waste_and_shortage(demand_meals, pred_meals):
    """Meals over-prepared (waste) and meals short (shortage)."""
    diff = pred_meals - demand_meals
    return float(diff.clip(min=0).sum()), float((-diff).clip(min=0).sum())


def fitness(acc, var, cost, fair):
    """Single score (LOWER is better) = weighted sum of the four objectives."""
    return float(config.W_ACCURACY * acc + config.W_VARIANCE * var
                 + config.W_COST * cost + config.W_FAIRNESS * fair)


# ============================================================================
# FILE: src/forecast.py
# ============================================================================

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


# ============================================================================
# FILE: run_attempt.py
# ============================================================================

"""
Run one complete hackathon attempt:

  1. generate data (base + several drifted validation sets + a drifted test set)
  2. evolve network settings with NSGA-II (multi-objective)
  3. retrain the best network on all data before the test period
  4. forecast the drifted test period: STATIC model vs ADAPTIVE model
  5. print and save the fitness score, logs, plots, and a row in results/attempts.csv

Usage:
  python run_attempt.py --note "baseline"
  python run_attempt.py --drift level_shift --note "tested a level-shift drift to check adaptation"
  python run_attempt.py --quick --note "smoke test"      (small, fast search)
"""
import argparse
import datetime as dt
import json
import os
import time

import numpy as np
import pandas as pd

import config
from src import metrics as M
from src.adaptive import run_online
from src.data_generator import generate
from src.evolution import run_nsga2
from src.features import build_features, feature_concepts, fit_scalers
from src.model import count_params, measure_latency_ms, predict, set_seed, train_model

ATTEMPTS_CSV = "results/attempts.csv"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift", default=config.TEST_SCENARIO, choices=config.DRIFT_SCENARIOS,
                    help="drift scenario for the test period")
    ap.add_argument("--note", default="", help="one-line 'what changed and why' (required after attempt 1)")
    ap.add_argument("--quick", action="store_true", help="small search for a fast check")
    return ap.parse_args()


def next_attempt_number():
    if not os.path.exists(ATTEMPTS_CSV):
        return 1
    return len(pd.read_csv(ATTEMPTS_CSV)) + 1


def main():
    args = parse_args()
    attempt = next_attempt_number()
    if attempt > 1 and not args.note.strip():
        raise SystemExit("Every attempt after the first needs a note: --note \"what changed and why\"")
    out_dir = f"results/attempt_{attempt:02d}"
    os.makedirs(out_dir, exist_ok=True)
    set_seed(config.SEED)
    t_start = time.time()
    pop, gens = (6, 2) if args.quick else (config.POP_SIZE, config.N_GENERATIONS)
    print(f"=== Attempt {attempt} | test drift: {args.drift} | population {pop} x {gens} generations ===")

    # ---------------- 1. data ----------------
    base = generate("none")
    scalers = fit_scalers(base)
    val_raw = {s: generate(s, drift_start_day=config.TRAIN_END, strength=config.VAL_STRENGTH)
               for s in config.VAL_SCENARIOS if s != args.drift}  # validation drifts never include the test drift
    test_raw = generate(args.drift, drift_start_day=config.VAL_END, strength=config.TEST_STRENGTH)
    print(f"Validation scenarios (out-of-distribution): {list(val_raw)}")

    feat_cache = {}

    def feats(lag_set):
        if lag_set not in feat_cache:
            lags = config.LAG_SETS[lag_set]
            b, cols = build_features(base, lags, scalers)
            v = {s: build_features(d, lags, scalers)[0] for s, d in val_raw.items()}
            t = build_features(test_raw, lags, scalers)[0]
            feat_cache[lag_set] = (b, v, t, cols)
        return feat_cache[lag_set]

    # ---------------- 2. evolutionary search ----------------
    def evaluate(hp):
        b, v, _, cols = feats(hp["lag_set"])
        tr = b[b.day_index < config.TRAIN_END - config.EARLY_STOP_DAYS]
        es = b[(b.day_index >= config.TRAIN_END - config.EARLY_STOP_DAYS) & (b.day_index < config.TRAIN_END)]
        model, hist, diverged = train_model(tr[cols].values, tr.y.values, es[cols].values, es.y.values, hp)
        n_params = count_params(model)
        train_wape = M.wape(tr.y.values, predict(model, tr[cols].values))
        wapes, gaps = [], []
        for d in v.values():
            vb = d[(d.day_index >= config.TRAIN_END) & (d.day_index < config.VAL_END)]
            p = predict(model, vb[cols].values)
            wapes.append(M.wape(vb.y.values, p))
            gaps.append(M.bias_gap(vb.y.values, p, vb.group.values))
        acc = float(np.mean(wapes))
        var = float(np.std(wapes) + max(0.0, acc - train_wape))   # variance + overfitting penalty
        cost = n_params / config.PARAM_BUDGET                      # deterministic size proxy
        fair = float(np.mean(gaps))
        violation = max(0.0, n_params - config.PARAM_BUDGET) / config.PARAM_BUDGET + (1.0 if diverged else 0.0)
        if diverged:
            acc = var = fair = 10.0
        return {"objectives": np.array([acc, var, cost, fair]), "violation": violation,
                "fitness": M.fitness(acc, var, cost, fair), "n_params": n_params,
                "epochs": len(hist), "val_wapes": wapes}

    print("\n[Evolution] NSGA-II over network settings (lower is better for every objective)")
    rng = np.random.default_rng(config.SEED)
    final_pop, gen_log = run_nsga2(evaluate, rng, pop_size=pop, n_gen=gens)
    pd.DataFrame(gen_log).to_csv(f"{out_dir}/evolution_log.csv", index=False)
    pareto = pd.DataFrame([{**r["hp"], "val_wape": r["objectives"][0], "variance": r["objectives"][1],
                            "cost": r["objectives"][2], "calib_gap": r["objectives"][3],
                            "fitness": r["fitness"], "n_params": r["n_params"], "feasible": r["violation"] == 0}
                           for r in final_pop]).sort_values("fitness")
    pareto.to_csv(f"{out_dir}/final_population.csv", index=False)
    feasible = [r for r in final_pop if r["violation"] == 0] or final_pop
    best = min(feasible, key=lambda r: r["fitness"])
    hp = best["hp"]
    print(f"\nChosen settings: {hp}  (validation fitness {best['fitness']:.4f})")

    # ---------------- 3. retrain on everything before the test period ----------------
    b, _, t, cols = feats(hp["lag_set"])
    tr = b[b.day_index < config.VAL_END - config.EARLY_STOP_DAYS]
    es = b[(b.day_index >= config.VAL_END - config.EARLY_STOP_DAYS) & (b.day_index < config.VAL_END)]
    model, hist, diverged = train_model(tr[cols].values, tr.y.values, es[cols].values, es.y.values, hp)
    pd.DataFrame(hist).to_csv(f"{out_dir}/training_log.csv", index=False)
    print("\n[Training] final model, loss per epoch:")
    for h in hist[:: max(1, len(hist) // 10)] + [hist[-1]]:
        print(f"  epoch {h['epoch']:3d} | train {h['train_loss']:.5f} | val {h['val_loss']:.5f} | "
              f"max grad norm (clipped to {config.GRAD_CLIP_NORM}) {h['max_grad_norm_before_clip']:.3f}")
    n_params = count_params(model)
    latency = measure_latency_ms(model, len(cols))

    # ---------------- 4. drifted test period: static vs adaptive ----------------
    es_pred = predict(model, es[cols].values)
    init_res = {g: (es.y.values - es_pred)[es.group.values == g] for g in config.GROUPS}
    history = t[t.day_index < config.VAL_END]
    stream = t[t.day_index >= config.VAL_END]
    print(f"\n[Test] forecasting {stream.day_index.nunique()} drifted days one at a time")
    res, alarms, finetunes = run_online(model, hp, cols, history, stream, init_res)
    res.to_csv(f"{out_dir}/test_predictions.csv", index=False)

    y, g = res.y.values, res.group.values
    summary = {}
    for kind in ["static", "adaptive"]:
        p = res[f"pred_{kind}"].values
        wk = pd.DataFrame({"week": res.day_index // 7, "e": np.abs(y - p), "y": y}).groupby("week").sum()
        weekly = wk.e / wk.y
        cov = M.coverage_by_group(y, res[f"lo_{kind}"].values, res[f"hi_{kind}"].values, g)
        waste, short = M.waste_and_shortage(y * res.scale.values, p * res.scale.values)
        summary[kind] = {
            "wape": M.wape(y, p), "rmse": M.rmse(y, p), "weekly_wape_std": float(weekly.std()),
            "group_bias": M.group_bias(y, p, g), "bias_gap": M.bias_gap(y, p, g),
            "coverage": cov, "coverage_gap": max(cov.values()) - min(cov.values()),
            "waste_meals": waste, "shortage_meals": short,
            "group_wape": {k: M.wape(y[g == k], p[g == k]) for k in config.GROUPS},
        }
    a = summary["adaptive"]
    final_fitness = M.fitness(a["wape"], a["weekly_wape_std"], n_params / config.PARAM_BUDGET,
                              a["bias_gap"] + a["coverage_gap"])

    # ---------------- 5. explainability: permutation importance ----------------
    perm_rng = np.random.default_rng(config.SEED)
    Xs, ys = stream[cols].values.copy(), stream.y.values
    base_err = M.wape(ys, predict(model, Xs))
    importance = {}
    for concept, ccols in feature_concepts(cols).items():
        Xp = Xs.copy()
        idx = [cols.index(c) for c in ccols]
        Xp[:, idx] = Xp[perm_rng.permutation(len(Xp))][:, idx]
        importance[concept] = M.wape(ys, predict(model, Xp)) - base_err
    importance = dict(sorted(importance.items(), key=lambda kv: -kv[1]))

    # ---------------- report ----------------
    runtime = time.time() - t_start
    print("\n================ RESULTS ================")
    print(f"Parameters: {n_params} (budget {config.PARAM_BUDGET}) | latency {latency:.3f} ms "
          f"(budget {config.LATENCY_BUDGET_MS} ms) | runtime {runtime:.0f}s")
    print(f"Drift alarms: {alarms}\nFine-tunes on days: {finetunes}")
    for kind in ["static", "adaptive"]:
        s = summary[kind]
        print(f"{kind:>9}: WAPE {s['wape']:.4f} | weekly std {s['weekly_wape_std']:.4f} | bias gap "
              f"{s['bias_gap']:.4f} | coverage {({k: round(v, 3) for k, v in s['coverage'].items()})} | "
              f"waste {s['waste_meals']:.0f} meals | shortage {s['shortage_meals']:.0f} meals")
    print("Feature importance (WAPE increase when shuffled):")
    for k, v in importance.items():
        print(f"  {k:<22} {v:+.4f}")
    print(f"\nFINAL FITNESS (lower is better): {final_fitness:.4f}")

    record = {
        "attempt": attempt, "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "test_drift": args.drift, "note": args.note, "quick": args.quick,
        "settings": json.dumps(hp), "n_params": n_params, "latency_ms": round(latency, 4),
        "val_fitness": round(best["fitness"], 4),
        "test_wape_static": round(summary["static"]["wape"], 4),
        "test_wape_adaptive": round(a["wape"], 4),
        "bias_gap_adaptive": round(a["bias_gap"], 4), "coverage_gap_adaptive": round(a["coverage_gap"], 4),
        "final_fitness": round(final_fitness, 4), "runtime_s": round(runtime, 1),
    }
    pd.DataFrame([record]).to_csv(ATTEMPTS_CSV, mode="a", header=not os.path.exists(ATTEMPTS_CSV), index=False)
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump({"record": record, "config_hp": hp, "summary": summary, "alarms": alarms,
                   "finetunes": finetunes, "importance": importance, "evolution": gen_log}, f, indent=2)
    make_plots(out_dir, gen_log, hist, res, importance, alarms, args.drift)
    print(f"Saved logs and plots to {out_dir}/ and appended to {ATTEMPTS_CSV}")


def make_plots(out_dir, gen_log, hist, res, importance, alarms, drift):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gl = pd.DataFrame(gen_log)
    h = pd.DataFrame(hist)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(gl.generation, gl.best_fitness, marker="o", label="best fitness")
    ax[0].plot(gl.generation, gl.mean_pop_fitness, marker="^", label="population mean fitness")
    ax[0].plot(gl.generation, gl.best_val_wape, marker="s", label="best validation WAPE")
    ax[0].set(title="Evolution convergence", xlabel="generation", ylabel="lower is better")
    ax[0].legend()
    ax[1].plot(h.epoch, h.train_loss, label="train")
    ax[1].plot(h.epoch, h.val_loss, label="validation")
    ax[1].set(title="Final model training loss", xlabel="epoch", ylabel="Huber loss")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/convergence.png", dpi=120)

    outlets = res.drop_duplicates("group").outlet.tolist()
    fig, axes = plt.subplots(len(outlets), 1, figsize=(12, 9), sharex=True)
    for ax, o in zip(axes, outlets):
        d = res[res.outlet == o]
        ax.plot(d.date, d.y * d.scale, "k.-", lw=0.8, ms=3, label="actual")
        ax.plot(d.date, d.pred_static * d.scale, lw=1, label="static model")
        ax.plot(d.date, d.pred_adaptive * d.scale, lw=1, label="adaptive model")
        ax.fill_between(d.date, d.lo_adaptive * d.scale, d.hi_adaptive * d.scale, alpha=0.2,
                        label=f"{int(config.CONFORMAL_LEVEL * 100)}% interval (adaptive)")
        for al in alarms:
            if al["group"] == d.group.iloc[0]:
                ax.axvline(d.date[d.day_index == al["day_index"]].iloc[0], color="red", ls="--", lw=0.8)
        ax.set_title(f"{o} (red dashed = drift alarm)")
        ax.set_ylabel("meals")
    axes[0].legend(fontsize=7, ncol=4)
    fig.suptitle(f"Test period with drift: {drift}")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/test_predictions.png", dpi=120)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(list(importance)[::-1], list(importance.values())[::-1])
    ax.set(title="What the model relies on (permutation importance)", xlabel="WAPE increase when shuffled")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/feature_importance.png", dpi=120)
    plt.close("all")


if __name__ == "__main__":
    main()


# ============================================================================
# FILE: app.py
# ============================================================================

"""
Website: one simple page in three parts.
  1. The data   - what we used
  2. The model  - how it works
  3. The result - meals forecast for the next 6 months, and how accurate it is

Start it with:   streamlit run app.py      (opens http://localhost:8501)
"""
import json
import os

import altair as alt
import pandas as pd
import streamlit as st

import config
from src import forecast as F

FC_DIR = F.OUT_DIR
COLORS = alt.Scale(domain=["actual", "forecast"], range=["#333333", "#e4572e"])

st.set_page_config(page_title="Food Demand Forecast", page_icon="🍱", layout="centered")
st.markdown("<style>.block-container{max-width:900px} h2{margin-top:2.2rem}</style>", unsafe_allow_html=True)


@st.cache_data
def history():
    return F.load_history()


def load_forecast():
    if not os.path.exists(f"{FC_DIR}/summary.json"):
        return None
    with open(f"{FC_DIR}/summary.json") as f:
        s = json.load(f)
    fc = pd.read_csv(f"{FC_DIR}/forecast.csv", parse_dates=["date"])
    bt = pd.read_csv(f"{FC_DIR}/backtest.csv", parse_dates=["date"])
    return s, fc, bt


hist = history()
res = load_forecast()

st.title("🍱 Food Demand Forecast")
st.write("How many meals should each kitchen prepare over the next 6 months? We learn from 2 years of "
         "data and adapt when customer behaviour changes.")

# ===========================================================================
st.header("1. The data")
st.write(f"Daily meal counts for **6 food outlets** from **{hist.date.min():%d %b %Y}** to "
         f"**{hist.date.max():%d %b %Y}** ({hist.day_index.nunique()} days, {len(hist):,} rows). "
         "The data is synthetic: generated with realistic rules, as the hackathon benchmark asks.")

c = st.columns(3)
for col, g in zip(c, config.GROUPS):
    outlets = [o for o, t, _ in config.OUTLETS if t == g]
    col.metric(g.capitalize() + "s", f"{hist[hist.group == g].demand.mean():.0f} meals/day",
               ", ".join(outlets), delta_color="off")

st.write("**What changes demand**")
st.table(pd.DataFrame({
    "Condition": ["Weekend", "Holiday", "Rain", "Hot day", "Special event", "Semester break (May–Jun)"],
    "Restaurant": ["more", "more", "less", "slightly less", "much more", "no change"],
    "Cafeteria": ["much less", "much less", "slightly more", "slightly less", "more", "no change"],
    "Hostel": ["slightly less", "less", "slightly more", "slightly less", "slightly more", "about half"],
}).set_index("Condition"))

st.write("**⚠ Customer behaviour changed in August 2025** (red line): cafeterias started getting busy at "
         "weekends and restaurants lost their weekend rush. A good model must notice this.")
group = st.radio("Show", config.GROUPS, horizontal=True, format_func=str.capitalize, key="g1")
d = hist[hist.group == group]
change_date = hist.date[hist.day_index == config.VAL_END].iloc[0]
chart = alt.Chart(d).mark_line(strokeWidth=0.8).encode(
    x=alt.X("date:T", title=None), y=alt.Y("demand:Q", title="meals per day"), color=alt.Color("outlet:N", title=None),
    tooltip=["date:T", "outlet", "demand", "weekend", "holiday", "rain", "event"])
rule = alt.Chart(pd.DataFrame({"date": [change_date]})).mark_rule(color="red", strokeDash=[5, 3]).encode(x="date:T")
st.altair_chart((chart + rule).properties(height=280), width="stretch")
with st.expander("See the raw data"):
    st.dataframe(hist.drop(columns=["day_index"]), height=260, width="stretch", hide_index=True)
    st.download_button("Download data (CSV)", hist.to_csv(index=False), "food_demand_2years.csv", "text/csv")

# ===========================================================================
st.header("2. The model")
st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f3f4f6", color="#9ca3af", fontname="Helvetica", fontsize=11];
  edge [color="#6b7280"];
  data [label="2 years of\\nmeal data"];
  detect [label="Drift detector\\nfinds when behaviour\\nchanged", fillcolor="#fde2d8"];
  evo [label="Evolution (NSGA-II)\\npicks network settings", fillcolor="#e0ecff"];
  nn [label="Small neural\\nnetwork", fillcolor="#e0ecff"];
  future [label="Future conditions\\ncalendar, typical weather,\\nplanned events"];
  out [label="Meals per day\\nfor 6 months\\n+ safe range", fillcolor="#dcfce7"];
  data -> detect -> nn; evo -> nn; future -> nn -> out;
}""")
st.markdown("""
1. **What it looks at**: day of week, weekend, holiday, semester break, typical weather for that date,
   planned events, and the outlet type (each type gets its own signals).
2. **Drift detector (Page-Hinkley test)**: scans the history for a lasting change in the model's errors,
   and marks the days after it as *"new behaviour"*. The model then learns the new pattern without
   forgetting seasonal effects such as the semester break.
3. **Recent days count more**: a day 120 days old counts half as much as yesterday.
4. **Evolution chose the network**: a genetic algorithm (NSGA-II) tried many designs and kept the best
   balance of accuracy, stability, size and fairness across outlet types.
5. **Direct forecast**: each future day is predicted from its own conditions, so errors don't pile up over 6 months.
6. **Safe range**: from past forecast errors, a range that should contain the real value 90% of the time.
""")
if res:
    s = res[0]
    hp = s["settings"]
    c = st.columns(4)
    c[0].metric("Network", f"{hp['n_layers']} layer × {hp['width']}")
    c[1].metric("Parameters", s["n_params"], f"budget {config.PARAM_BUDGET}", delta_color="off")
    c[2].metric("Activation", hp["activation"])
    c[3].metric("Change detected", pd.Timestamp(s["change_detected_on"]).strftime("%d %b %Y")
                if s["change_detected_on"] else "none")

# ===========================================================================
st.header("3. The result")
if res is None:
    st.info("No forecast yet.")
    if st.button("▶ Build the 6-month forecast (about 2 minutes)", type="primary"):
        with st.spinner("Learning from 2 years of data and forecasting..."):
            F.run(log=lambda *_: None)
        st.rerun()
    st.stop()

s, fc, bt = res
fc = fc[fc.date >= pd.Timestamp(s["forecast_start"]).replace(day=1) + pd.offsets.MonthBegin(1)] \
    if pd.Timestamp(s["forecast_start"]).day != 1 else fc     # show whole months only
start, end = fc.date.min(), fc.date.max()
st.write(f"Forecast for **{start:%d %b %Y} – {end:%d %b %Y}**.")

st.subheader("Meals to prepare per month")
monthly = fc.assign(month=fc.date.dt.strftime("%b %Y")).pivot_table(
    index="outlet", columns="month", values="pred", aggfunc="sum", sort=False)
monthly["Total"] = monthly.sum(axis=1)
monthly.loc["All outlets"] = monthly.sum()
st.dataframe(monthly.style.format("{:,.0f}"), width="stretch")

st.subheader("Daily forecast")
outlet = st.selectbox("Outlet", [o for o, _, _ in config.OUTLETS])
recent = hist[(hist.outlet == outlet) & (hist.date >= start - pd.Timedelta(days=90))]
f_o = fc[fc.outlet == outlet]
lines = pd.concat([recent[["date", "demand"]].assign(series="actual"),
                   f_o[["date", "pred"]].rename(columns={"pred": "demand"}).assign(series="forecast")])
band = alt.Chart(f_o).mark_area(opacity=0.18, color="#e4572e").encode(
    x="date:T", y=alt.Y("low:Q", title="meals per day"), y2="high:Q")
line = alt.Chart(lines).mark_line(strokeWidth=1).encode(
    x=alt.X("date:T", title=None), y="demand:Q", color=alt.Color("series:N", scale=COLORS, title=None),
    tooltip=["date:T", "series", "demand"])
st.altair_chart((band + line).properties(height=300).interactive(bind_y=False), width="stretch")
st.caption("Shaded = 90% safe range. The drop in May–June for hostels is the semester break.")
with st.expander("Daily numbers"):
    show = f_o.drop(columns=["outlet", "group"]).rename(columns={
        "pred": "meals", "low": "safe low", "high": "safe high", "rain_chance": "rain chance"})
    st.dataframe(show, hide_index=True, width="stretch", height=260)
    st.download_button("Download forecast (CSV)", fc.to_csv(index=False), "forecast_6_months.csv", "text/csv")

st.subheader("How accurate is it?")
st.write(f"We pretended it was **{pd.Timestamp(s['backtest_start']) - pd.Timedelta(days=1):%d %b %Y}**, "
         "forecast the next 6 months, and compared the forecast with what really happened. Lower error is better.")
w = s["backtest_wape"]
c = st.columns(3)
c[0].metric("Simple average", f"{w['naive']['overall']:.1%} error", "last 4 weeks, same weekday", delta_color="off")
c[1].metric("Our model, one forecast", f"{w['one_shot']['overall']:.1%} error", "made once for 6 months",
            delta_color="off")
c[2].metric("Our model, updated monthly", f"{w['monthly']['overall']:.1%} error", "adapts to new data ✅",
            delta_color="off")
st.write("The one-time forecast could not know that behaviour would change in August. **Updated monthly**, "
         "the drift detector notices the change and the model re-learns, so the error drops a lot. "
         "That's why we recommend updating the forecast at the start of every month.")
b = bt[bt.outlet == outlet].melt("date", ["actual", "one_shot", "monthly"], "series", "meals")
b.series = b.series.map({"actual": "actual", "one_shot": "one forecast", "monthly": "updated monthly"})
st.altair_chart(alt.Chart(b).mark_line(strokeWidth=1).encode(
    x=alt.X("date:T", title=None), y=alt.Y("meals:Q", title="meals per day"),
    color=alt.Color("series:N", title=None, scale=alt.Scale(
        domain=["actual", "one forecast", "updated monthly"], range=["#333333", "#9ca3af", "#e4572e"])),
    tooltip=["date:T", "series", "meals"]).properties(height=240, title=f"Backtest: {outlet}"), width="stretch")
st.table(pd.DataFrame({t: [f"{w[k][t]:.1%}" for k in ["naive", "one_shot", "monthly"]] for t in config.GROUPS},
                      index=["Simple average", "One forecast", "Updated monthly"]).rename(columns=str.capitalize))

st.divider()
st.caption("Assumptions: weather = typical for the time of year; no special events unless listed in "
           "config.FUTURE_EVENT_DATES; holidays and semester break from the calendar.")
if st.button("↻ Re-run the forecast"):
    with st.spinner("Re-running (about 2 minutes)..."):
        F.run(log=lambda *_: None)
    st.rerun()
