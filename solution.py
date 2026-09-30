#!/usr/bin/env python3
"""
Adaptive Food Demand Prediction: single-file solution (needs only numpy).

Problem : predict tomorrow's meals for restaurants, cafeterias and hostels, and stay
          reliable when customer behaviour shifts (non-stationary drift).
Method  : 1. synthetic multi-scenario benchmark (weekends, holidays, weather, events,
             semester break, and 5 drift scenarios)
          2. small MLP trained with Huber loss, AdamW weight decay, dropout,
             gradient-norm clipping and early stopping (deterministic, seeded)
          3. NSGA-II multi-objective evolution of the MLP hyperparameters
             objectives: OOD error, error variance + overfit gap, size, calibration gap
          4. online adaptation: per-group EWMA bias correction, Page-Hinkley drift
             detection, windowed fine-tuning, per-group (Mondrian) conformal intervals

Run     : python solution.py                        (full run, about 1-2 minutes)
          python solution.py --fast                 (smaller search, about 30 seconds)
          python solution.py --note "what changed and why"
Output  : parameter configuration, per-generation convergence log, per-epoch training
          log, per-scenario results, FINAL FITNESS, and one row per attempt in
          results/solution_attempts.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np


# =============================================================================
# 1. PARAMETER CONFIGURATION (all tunable values, each explained)
# =============================================================================
@dataclass(frozen=True)
class Config:
    seed: int = 42                    # every random choice derives from this seed
    n_days: int = 730                 # 2 years of daily data
    train_end: int = 438              # days [0, 438) train
    val_end: int = 584                # days [438, 584) OOD validation; [584, 730) test
    early_stop_days: int = 60         # last days before a split used for early stopping
    noise_std: float = 0.08           # day-to-day noise (log scale, about 8%)
    trend_per_year: float = 0.05      # slow growth
    event_prob: float = 0.03          # chance of a special event per day
    val_strength: float = 0.5         # validation drifts are milder than test drifts
    test_strength: float = 1.0
    # training
    max_epochs: int = 60
    patience: int = 8
    batch_size: int = 64
    grad_clip: float = 1.0            # gradient-norm clipping -> no gradient explosion
    # NSGA-II
    pop_size: int = 12
    generations: int = 8
    crossover_prob: float = 0.9
    mutation_prob: float = 0.25
    mutation_sigma: float = 0.15
    # deployment budget (hard constraints)
    param_budget: int = 3000          # max trainable parameters
    latency_budget_ms: float = 1.0    # max ms per single prediction
    # fitness weights (lower fitness is better)
    w_accuracy: float = 1.0
    w_variance: float = 1.0
    w_cost: float = 0.05
    w_fairness: float = 0.5
    # online adaptation
    bias_alpha: float = 0.15          # EWMA speed of per-group bias correction
    ph_delta: float = 0.02            # Page-Hinkley tolerance
    ph_threshold: float = 0.6         # Page-Hinkley alarm level
    adapt_days: int = 35              # keep adapting this long after an alarm
    finetune_every: int = 7
    finetune_window: int = 42
    finetune_epochs: int = 20
    finetune_lr_mult: float = 0.5
    conformal_level: float = 0.90     # target interval coverage for every group
    conformal_window: int = 60


CFG = Config()
GROUPS = ("restaurant", "cafeteria", "hostel")
# (name, group index, average meals per day)
OUTLETS = (("restaurant_A", 0, 220), ("restaurant_B", 0, 140), ("cafeteria_A", 1, 400),
           ("cafeteria_B", 1, 260), ("hostel_A", 2, 600), ("hostel_B", 2, 350))
# log-scale effects per group: weekend, holiday, rain, heat(per 6C), event, semester break
EFFECTS = np.array([[0.35, 0.30, -0.15, -0.05, 0.45, 0.00],
                    [-1.00, -1.10, 0.05, -0.03, 0.25, -0.05],
                    [-0.10, -0.35, 0.08, -0.04, 0.15, -0.60]])
LEVEL_SHIFT = np.array([0.30, -0.35, 0.20])
GRADUAL_TREND = np.array([0.45, 0.25, -0.30])
SCENARIOS = ("none", "level_shift", "weekend_pattern_flip", "weather_sensitivity_change",
             "sudden_event_spike", "gradual_trend")
VAL_SCENARIOS = ("none", "level_shift", "gradual_trend")
HOLIDAYS = {(1, 1), (1, 14), (1, 26), (3, 14), (4, 14), (8, 15), (10, 2), (10, 20),
            (10, 21), (11, 1), (12, 25), (12, 31)}
LAG_SETS = ((1, 7), (1, 2, 7, 14), (1, 2, 3, 7, 14, 21, 28))
WIDTHS = (8, 16, 32)


# =============================================================================
# 2. SYNTHETIC MULTI-SCENARIO DATA
# =============================================================================
def calendar(n_days: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Shared daily drivers: day of week, holidays, weather, events, semester break."""
    dates = np.datetime64("2024-01-01") + np.arange(n_days)
    days = dates.astype("datetime64[D]").astype(np.int64)
    month = dates.astype("datetime64[M]").astype(np.int64) % 12 + 1
    dom = (dates - dates.astype("datetime64[M]")).astype(np.int64) + 1
    doy = (dates - dates.astype("datetime64[Y]")).astype(np.int64) + 1
    dow = (days + 3) % 7                                   # Monday = 0
    temp = 27 + 6 * np.sin(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 2, n_days)
    rain = rng.random(n_days) < np.where(np.isin(month, [6, 7, 8, 9]), 0.55, 0.08)
    return {"dow": dow, "weekend": (dow >= 5).astype(float),
            "holiday": np.array([(m, d) in HOLIDAYS for m, d in zip(month, dom)], float),
            "temp": temp, "rain": rain.astype(float),
            "event": (rng.random(n_days) < CFG.event_prob).astype(float),
            "brk": np.isin(month, [5, 6]).astype(float)}


def generate(scenario: str, drift_start: int, strength: float) -> tuple[np.ndarray, dict]:
    """Return demand (outlets x days) and calendar. Pre-drift days match across scenarios."""
    rng = np.random.default_rng(CFG.seed)
    cal = calendar(CFG.n_days, rng)
    noise = rng.normal(0, CFG.noise_std, (len(OUTLETS), CFG.n_days))
    t = np.arange(CFG.n_days)
    after = (t >= drift_start) & (scenario != "none")
    ramp = np.clip((t - drift_start) / max(1, CFG.n_days - drift_start), 0, 1) * after
    cal = dict(cal)
    event_boost = np.ones(CFG.n_days)
    if scenario == "sudden_event_spike":
        extra = np.random.default_rng(CFG.seed + 1).random(CFG.n_days) < 0.2 * strength
        cal["event"] = np.maximum(cal["event"], (after & extra).astype(float))
        event_boost = np.where(after, 1 + 0.5 * strength, 1.0)
    demand = np.empty((len(OUTLETS), CFG.n_days))
    for k, (_, g, base) in enumerate(OUTLETS):
        demand[k] = base * np.exp(_log_effects(scenario, g, cal, after, ramp, strength, event_boost)
                                  + CFG.trend_per_year * t / 365 + noise[k])
    return np.round(demand), cal


def _log_effects(scenario: str, g: int, cal: dict, after: np.ndarray, ramp: np.ndarray,
                 s: float, event_boost: np.ndarray) -> np.ndarray:
    """Sum of log-scale effects for one outlet group, including the drift."""
    wk, hol, rain, heat, ev, brk = EFFECTS[g]
    weekend = cal["weekend"] * wk
    rain_e = cal["rain"] * rain
    heat_e = heat * (cal["temp"] - 27) / 6
    shift = np.zeros_like(ramp)
    if scenario == "weekend_pattern_flip":
        weekend = np.where(after, weekend * (1 - 1.5 * s), weekend)
    elif scenario == "weather_sensitivity_change":
        rain_e = np.where(after, rain_e * (1 + 3 * s), rain_e)
        heat_e = np.where(after, heat_e * (1 + 2 * s), heat_e)
    elif scenario == "level_shift":
        shift = after * s * LEVEL_SHIFT[g]
    elif scenario == "gradual_trend":
        shift = ramp * s * GRADUAL_TREND[g]
    return (weekend + cal["holiday"] * hol + rain_e + heat_e + cal["event"] * ev * event_boost
            + cal["brk"] * brk + shift)


# =============================================================================
# 3. FEATURES (only past demand + known-in-advance calendar/weather/events)
# =============================================================================
@dataclass
class Rows:
    X: np.ndarray       # features
    y: np.ndarray       # target: demand / outlet's training mean
    t: np.ndarray       # day index
    g: np.ndarray       # group index
    scale: np.ndarray   # outlet's training mean (to convert back to meals)


def build_rows(demand: np.ndarray, cal: dict, lags: tuple[int, ...], scale: np.ndarray,
               temp_stats: tuple[float, float]) -> Rows:
    """One row per (day, outlet), ordered day-major, starting at day 28."""
    Y = demand / scale[:, None]
    ts = np.arange(28, demand.shape[1])
    csum = np.concatenate([np.zeros((len(Y), 1)), np.cumsum(Y, axis=1)], axis=1)
    groups = np.array([g for _, g, _ in OUTLETS])
    onehot = np.eye(len(GROUPS))[groups]                                    # outlets x 3
    per_day = lambda v: np.repeat(v[ts][:, None], len(OUTLETS), axis=1)    # noqa: E731
    feats = [Y[:, ts - L].T for L in lags]
    feats += [((csum[:, ts] - csum[:, ts - w]) / w).T for w in (7, 28)]
    feats += [per_day(np.sin(2 * np.pi * cal["dow"] / 7)), per_day(np.cos(2 * np.pi * cal["dow"] / 7)),
              per_day(cal["weekend"]), per_day(cal["holiday"]), per_day(cal["rain"]),
              per_day(cal["event"]), per_day(cal["brk"]),
              per_day((cal["temp"] - temp_stats[0]) / temp_stats[1])]
    feats += [np.broadcast_to(onehot[:, j], (len(ts), len(OUTLETS))) for j in range(len(GROUPS))]
    feats += [per_day(cal["weekend"]) * onehot[:, j] for j in range(len(GROUPS))]
    feats += [per_day(cal["holiday"]) * onehot[:, j] for j in range(len(GROUPS))]
    X = np.stack(feats, axis=-1).reshape(-1, len(feats))
    return Rows(X=X, y=Y[:, ts].T.reshape(-1), t=np.repeat(ts, len(OUTLETS)),
                g=np.tile(groups, len(ts)), scale=np.tile(scale, len(ts)))


def select(rows: Rows, mask: np.ndarray) -> Rows:
    return Rows(rows.X[mask], rows.y[mask], rows.t[mask], rows.g[mask], rows.scale[mask])


# =============================================================================
# 4. NEURAL NETWORK (numpy MLP with manual backprop, AdamW, clipping)
# =============================================================================
def init_net(n_in: int, n_layers: int, width: int, rng: np.random.Generator) -> list[np.ndarray]:
    sizes = [n_in] + [width] * n_layers + [1]
    params = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        params += [rng.normal(0, np.sqrt(2 / a), (a, b)), np.zeros(b)]
    return params


def forward(params: list[np.ndarray], X: np.ndarray, act: str, dropout: float = 0.0,
            rng: np.random.Generator | None = None) -> tuple[np.ndarray, list]:
    h, cache = X, []
    for i in range(0, len(params) - 2, 2):
        z = h @ params[i] + params[i + 1]
        a = np.maximum(z, 0) if act == "relu" else np.tanh(z)
        mask = None
        if rng is not None and dropout > 0:
            mask = (rng.random(a.shape) >= dropout) / (1 - dropout)
            a = a * mask
        cache.append((h, z, a, mask))
        h = a
    cache.append(h)
    return (h @ params[-2] + params[-1]).ravel(), cache


def backward(params: list[np.ndarray], cache: list, grad_out: np.ndarray, act: str) -> list[np.ndarray]:
    grads = [None] * len(params)
    g = grad_out[:, None]
    grads[-2], grads[-1] = cache[-1].T @ g, g.sum(0)
    g = g @ params[-2].T
    for layer in range(len(cache) - 2, -1, -1):
        h, z, a, mask = cache[layer]
        if mask is not None:
            g = g * mask
        g = g * ((z > 0) if act == "relu" else (1 - np.tanh(z) ** 2))
        grads[2 * layer], grads[2 * layer + 1] = h.T @ g, g.sum(0)
        g = g @ params[2 * layer].T
    return grads


def huber_grad(pred: np.ndarray, y: np.ndarray, delta: float) -> tuple[float, np.ndarray]:
    r = pred - y
    loss = np.where(np.abs(r) <= delta, 0.5 * r ** 2, delta * (np.abs(r) - 0.5 * delta)).mean()
    return float(loss), np.clip(r, -delta, delta) / len(r)


def clip_grads(grads: list[np.ndarray], max_norm: float) -> float:
    norm = float(np.sqrt(sum((g ** 2).sum() for g in grads)))
    if norm > max_norm:
        for g in grads:
            g *= max_norm / norm
    return norm


def train(rows: Rows, val: Rows | None, hp: dict, epochs: int = CFG.max_epochs,
          init: list[np.ndarray] | None = None, lr: float | None = None) -> tuple[list, list[dict]]:
    """AdamW + Huber + dropout + clipping + early stopping. Deterministic for a given seed."""
    rng = np.random.default_rng(CFG.seed)
    params = [p.copy() for p in init] if init else init_net(rows.X.shape[1], hp["n_layers"], hp["width"], rng)
    m, v = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
    lr, step, history = lr or hp["lr"], 0, []
    best, best_loss, bad = [p.copy() for p in params], np.inf, 0
    for epoch in range(1, epochs + 1):
        perm, total, max_norm = rng.permutation(len(rows.y)), 0.0, 0.0
        for i in range(0, len(perm), CFG.batch_size):
            idx = perm[i:i + CFG.batch_size]
            pred, cache = forward(params, rows.X[idx], hp["act"], hp["dropout"], rng)
            loss, g_out = huber_grad(pred, rows.y[idx], hp["huber"])
            grads = backward(params, cache, g_out, hp["act"])
            max_norm = max(max_norm, clip_grads(grads, CFG.grad_clip))
            step += 1
            _adamw_step(params, grads, m, v, step, lr, hp["wd"])
            total += loss * len(idx)
        val_loss = huber_grad(predict(params, val.X, hp["act"]), val.y, hp["huber"])[0] if val else np.nan
        history.append({"epoch": epoch, "train_loss": total / len(rows.y), "val_loss": val_loss,
                        "grad_norm": max_norm})
        if val is None:
            continue
        if val_loss < best_loss - 1e-6:
            best, best_loss, bad = [p.copy() for p in params], val_loss, 0
        elif (bad := bad + 1) >= CFG.patience:
            break
    return (best if val is not None else params), history


def _adamw_step(params, grads, m, v, step, lr, wd, b1=0.9, b2=0.999, eps=1e-8) -> None:
    for i, (p, g) in enumerate(zip(params, grads)):
        m[i] = b1 * m[i] + (1 - b1) * g
        v[i] = b2 * v[i] + (1 - b2) * g ** 2
        update = (m[i] / (1 - b1 ** step)) / (np.sqrt(v[i] / (1 - b2 ** step)) + eps)
        p -= lr * (update + (wd * p if p.ndim == 2 else 0))


def predict(params: list[np.ndarray], X: np.ndarray, act: str) -> np.ndarray:
    return forward(params, X, act)[0]


def n_params(params: list[np.ndarray]) -> int:
    return int(sum(p.size for p in params))


def latency_ms(params: list[np.ndarray], n_in: int, act: str, reps: int = 500) -> float:
    x = np.zeros((1, n_in))
    t0 = time.perf_counter()
    for _ in range(reps):
        predict(params, x, act)
    return (time.perf_counter() - t0) / reps * 1000


# =============================================================================
# 5. METRICS
# =============================================================================
def wape(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.abs(y - p).sum() / max(np.abs(y).sum(), 1e-9))


def group_bias(y: np.ndarray, p: np.ndarray, g: np.ndarray) -> np.ndarray:
    return np.array([(p[g == k] - y[g == k]).sum() / y[g == k].sum() for k in range(len(GROUPS))])


def bias_gap(y: np.ndarray, p: np.ndarray, g: np.ndarray) -> float:
    b = group_bias(y, p, g)
    return float(b.max() - b.min())


def fitness(acc: float, var: float, cost: float, fair: float) -> float:
    return CFG.w_accuracy * acc + CFG.w_variance * var + CFG.w_cost * cost + CFG.w_fairness * fair


# =============================================================================
# 6. NSGA-II MULTI-OBJECTIVE EVOLUTION OF HYPERPARAMETERS
# =============================================================================
def decode(gene: np.ndarray) -> dict:
    """Genome (7 genes in [0,1]) -> hyperparameters."""
    pick = lambda choices, u: choices[min(int(u * len(choices)), len(choices) - 1)]  # noqa: E731
    return {"n_layers": pick((1, 2), gene[0]), "width": pick(WIDTHS, gene[1]),
            "lr": float(f"{10 ** (-3.5 + 1.5 * gene[2]):.2g}"), "dropout": float(round(0.3 * gene[3], 2)),
            "wd": float(f"{10 ** (-6 + 3 * gene[4]):.2g}"), "lags": pick(range(len(LAG_SETS)), gene[5]),
            "huber": float(round(0.05 + 0.95 * gene[6], 2)), "act": "relu"}


def dominates(a: dict, b: dict) -> bool:
    """Constrained domination: feasible beats infeasible, then Pareto dominance."""
    if a["violation"] != b["violation"]:
        return a["violation"] < b["violation"]
    return bool(np.all(a["obj"] <= b["obj"]) and np.any(a["obj"] < b["obj"]))


def pareto_fronts(pop: list[dict]) -> list[list[int]]:
    n = len(pop)
    beats = [[j for j in range(n) if j != i and dominates(pop[i], pop[j])] for i in range(n)]
    count = [sum(dominates(pop[j], pop[i]) for j in range(n) if j != i) for i in range(n)]
    fronts, current = [], [i for i in range(n) if count[i] == 0]
    while current:
        fronts.append(current)
        nxt = []
        for i in current:
            for j in beats[i]:
                count[j] -= 1
                if count[j] == 0:
                    nxt.append(j)
        current = nxt
    return fronts


def crowding(pop: list[dict], front: list[int]) -> dict[int, float]:
    dist = {i: 0.0 for i in front}
    objs = np.array([pop[i]["obj"] for i in front])
    for k in range(objs.shape[1]):
        order = np.argsort(objs[:, k], kind="stable")
        dist[front[order[0]]] = dist[front[order[-1]]] = np.inf
        span = objs[order[-1], k] - objs[order[0], k] or 1.0
        for a in range(1, len(front) - 1):
            dist[front[order[a]]] += (objs[order[a + 1], k] - objs[order[a - 1], k]) / span
    return dist


def survive(pop: list[dict], size: int) -> list[dict]:
    chosen = []
    for front in pareto_fronts(pop):
        if len(chosen) + len(front) <= size:
            chosen += front
        else:
            dist = crowding(pop, front)
            chosen += sorted(front, key=lambda i: -dist[i])[:size - len(chosen)]
            break
    return [pop[i] for i in chosen]


def make_children(pop: list[dict], rng: np.random.Generator) -> list[np.ndarray]:
    rank = {id(p): r for r, front in enumerate(pareto_fronts(pop)) for p in (pop[i] for i in front)}

    def tournament() -> np.ndarray:
        a, b = rng.integers(0, len(pop), 2)
        return (pop[a] if rank[id(pop[a])] <= rank[id(pop[b])] else pop[b])["gene"]

    children = []
    while len(children) < len(pop):
        p1, p2 = tournament(), tournament()
        w = rng.random(p1.size) if rng.random() < CFG.crossover_prob else np.ones(p1.size)
        for c in (w * p1 + (1 - w) * p2, w * p2 + (1 - w) * p1):
            mutate = rng.random(c.size) < CFG.mutation_prob
            children.append(np.clip(c + mutate * rng.normal(0, CFG.mutation_sigma, c.size), 0, 1))
    return children[:len(pop)]


def evolve(evaluate, pop_size: int, generations: int) -> tuple[dict, list[dict]]:
    """Run NSGA-II; return the best feasible solution and the per-generation log."""
    rng, cache, log = np.random.default_rng(CFG.seed), {}, []

    def score(gene: np.ndarray) -> dict:
        hp = decode(gene)
        key = json.dumps(hp, sort_keys=True)
        if key not in cache:
            cache[key] = {**evaluate(hp), "hp": hp}
        return {**cache[key], "gene": gene}

    pop = [score(rng.random(7)) for _ in range(pop_size)]
    for gen in range(generations + 1):
        best = min(pop, key=lambda p: (p["violation"], p["fitness"]))
        log.append({"generation": gen, "evaluations": len(cache), "best_fitness": round(best["fitness"], 5),
                    "mean_fitness": round(float(np.mean([p["fitness"] for p in pop])), 5),
                    "best_val_wape": round(float(min(p["obj"][0] for p in pop)), 5),
                    "pareto_front": len(pareto_fronts(pop)[0])})
        print("  gen {generation:2d} | evals {evaluations:3d} | best fitness {best_fitness:.4f} | "
              "mean {mean_fitness:.4f} | best val WAPE {best_val_wape:.4f} | front {pareto_front}".format(**log[-1]))
        if gen < generations:
            pop = survive(pop + [score(c) for c in make_children(pop, rng)], pop_size)
    return min(pop, key=lambda p: (p["violation"], p["fitness"])), log


# =============================================================================
# 7. ONLINE ADAPTATION TO DRIFT
# =============================================================================
class PageHinkley:
    """Two-sided Page-Hinkley change detector on a stream of errors."""

    def __init__(self) -> None:
        self.n = self.mean = self.up = self.down = self.min_up = self.max_down = 0.0

    def update(self, x: float) -> bool:
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.up += x - self.mean - CFG.ph_delta
        self.down += x - self.mean + CFG.ph_delta
        self.min_up, self.max_down = min(self.min_up, self.up), max(self.max_down, self.down)
        alarm = self.up - self.min_up > CFG.ph_threshold or self.max_down - self.down > CFG.ph_threshold
        if alarm:
            self.__init__()
        return alarm


def run_online(params: list, hp: dict, rows: Rows, init_resid: np.ndarray, init_groups: np.ndarray) -> dict:
    """Forecast the test period day by day; adapt after each day's true demand arrives."""
    net, bias = params, np.zeros(len(GROUPS))
    detectors = [PageHinkley() for _ in GROUPS]
    windows = [deque(np.abs(init_resid[init_groups == k]), maxlen=CFG.conformal_window) for k in range(len(GROUPS))]
    adapt_until, last_ft, alarms = -1, -10 ** 9, 0
    out = {k: [] for k in ("y", "g", "static", "adaptive", "lo", "hi", "scale")}
    for day in np.unique(rows.t[rows.t >= CFG.val_end]):
        today = select(rows, rows.t == day)
        p_model = predict(net, today.X, hp["act"])
        p_adapt = p_model + bias[today.g]
        q = np.array([np.quantile(windows[k], CFG.conformal_level) for k in today.g])
        for key, val in (("y", today.y), ("g", today.g), ("static", predict(params, today.X, hp["act"])),
                         ("adaptive", p_adapt), ("lo", p_adapt - q), ("hi", p_adapt + q), ("scale", today.scale)):
            out[key].append(val)
        for k in np.unique(today.g):
            mk = today.g == k
            bias[k] = (1 - CFG.bias_alpha) * bias[k] + CFG.bias_alpha * float(np.mean(today.y[mk] - p_model[mk]))
            windows[k].extend(np.abs(today.y[mk] - p_adapt[mk]))
            if detectors[k].update(float(np.mean(today.y[mk] - p_adapt[mk]))):
                alarms, adapt_until = alarms + 1, day + CFG.adapt_days
        if day <= adapt_until and day - last_ft >= CFG.finetune_every:
            recent = select(rows, (rows.t > day - CFG.finetune_window) & (rows.t <= day))
            net, _ = train(recent, None, hp, CFG.finetune_epochs, init=net, lr=hp["lr"] * CFG.finetune_lr_mult)
            bias[:], last_ft = 0.0, day
    return {**{k: np.concatenate(v) for k, v in out.items()}, "alarms": alarms}


# =============================================================================
# 8. EXPERIMENT PIPELINE
# =============================================================================
class Benchmark:
    """Holds the base data, OOD validation sets and drifted test sets, with feature caching."""

    def __init__(self) -> None:
        self.base, self.cal = generate("none", CFG.n_days, 0.0)
        self.scale = self.base[:, :CFG.train_end].mean(axis=1)
        temp = self.cal["temp"][:CFG.train_end]
        self.temp_stats = (float(temp.mean()), float(temp.std()))
        self.val = {s: generate(s, CFG.train_end, CFG.val_strength) for s in VAL_SCENARIOS}
        self.test = {s: generate(s, CFG.val_end, CFG.test_strength) for s in SCENARIOS}
        self._cache: dict = {}

    def rows(self, kind: str, scenario: str, lag_idx: int) -> Rows:
        key = (kind, scenario, lag_idx)
        if key not in self._cache:
            demand, cal = {"base": (self.base, self.cal), "val": self.val.get(scenario),
                           "test": self.test.get(scenario)}[kind]
            self._cache[key] = build_rows(demand, cal, LAG_SETS[lag_idx], self.scale, self.temp_stats)
        return self._cache[key]

    def split(self, lag_idx: int, end: int) -> tuple[Rows, Rows]:
        base = self.rows("base", "none", lag_idx)
        cut = end - CFG.early_stop_days
        return select(base, base.t < cut), select(base, (base.t >= cut) & (base.t < end))


def evaluate_hp(bench: Benchmark, hp: dict) -> dict:
    """Train on the train split; score on every OOD validation scenario (4 objectives)."""
    tr, es = bench.split(hp["lags"], CFG.train_end)
    params, hist = train(tr, es, hp)
    diverged = not np.isfinite(hist[-1]["train_loss"])
    wapes, gaps = [], []
    for s in VAL_SCENARIOS:
        v = bench.rows("val", s, hp["lags"])
        v = select(v, (v.t >= CFG.train_end) & (v.t < CFG.val_end))
        p = predict(params, v.X, hp["act"])
        wapes.append(wape(v.y, p))
        gaps.append(bias_gap(v.y, p, v.g))
    train_wape = wape(tr.y, predict(params, tr.X, hp["act"]))
    acc, cost = float(np.mean(wapes)), n_params(params) / CFG.param_budget
    var = float(np.std(wapes) + max(0.0, acc - train_wape))
    obj = np.array([acc, var, cost, float(np.mean(gaps))])
    violation = max(0.0, cost - 1) + float(diverged)
    return {"obj": obj, "violation": violation, "fitness": fitness(*obj)}


def test_all_scenarios(bench: Benchmark, hp: dict, params: list, es: Rows) -> list[dict]:
    """Stability under dynamic perturbations: static vs adaptive on every drift scenario."""
    resid = es.y - predict(params, es.X, hp["act"])
    results = []
    for s in SCENARIOS:
        r = run_online(params, hp, bench.rows("test", s, hp["lags"]), resid, es.g)
        inside = (r["y"] >= r["lo"]) & (r["y"] <= r["hi"])
        cov = np.array([inside[r["g"] == k].mean() for k in range(len(GROUPS))])
        meals_err = (r["adaptive"] - r["y"]) * r["scale"]
        results.append({"scenario": s, "static_wape": wape(r["y"], r["static"]),
                        "adaptive_wape": wape(r["y"], r["adaptive"]),
                        "bias_gap": bias_gap(r["y"], r["adaptive"], r["g"]),
                        "coverage": cov, "coverage_gap": float(cov.max() - cov.min()), "alarms": r["alarms"],
                        "waste_meals": float(meals_err.clip(min=0).sum()),
                        "shortage_meals": float((-meals_err).clip(min=0).sum())})
    return results


def report(results: list[dict], hp: dict, params: list, n_in: int, log: list, runtime: float) -> dict:
    print("\n  scenario                    static  adaptive  bias_gap  coverage(rest/caf/hos)  alarms")
    for r in results:
        cov = "/".join(f"{c:.2f}" for c in r["coverage"])
        print(f"  {r['scenario']:<26} {r['static_wape']:.4f}  {r['adaptive_wape']:.4f}    "
              f"{r['bias_gap']:.4f}    {cov:<22}  {r['alarms']}")
    ad = np.array([r["adaptive_wape"] for r in results])
    acc, var = float(ad.mean()), float(ad.std())
    fair = float(np.mean([r["bias_gap"] + r["coverage_gap"] for r in results]))
    cost = n_params(params) / CFG.param_budget
    lat = latency_ms(params, n_in, hp["act"])
    summary = {"final_fitness": round(fitness(acc, var, cost, fair), 5),
               "accuracy_pct": round(100 * (1 - acc), 2),
               "mean_static_wape": round(float(np.mean([r["static_wape"] for r in results])), 5),
               "mean_adaptive_wape": round(acc, 5), "wape_std_across_scenarios": round(var, 5),
               "mean_fairness_gap": round(fair, 5), "n_params": n_params(params),
               "latency_ms": round(lat, 4), "runtime_s": round(runtime, 1),
               "best_hp": hp, "convergence": log}
    print(f"\n  parameters {summary['n_params']} (budget {CFG.param_budget}) | latency {lat:.3f} ms "
          f"(budget {CFG.latency_budget_ms}) | runtime {runtime:.1f}s")
    print(f"  mean WAPE static {summary['mean_static_wape']:.4f} -> adaptive {acc:.4f} | "
          f"std across scenarios {var:.4f} | fairness gap {fair:.4f}")
    print(f"  ACCURACY (100 x (1 - adaptive WAPE)): {summary['accuracy_pct']:.2f}%")
    print(f"  FINAL FITNESS (lower is better): {summary['final_fitness']:.4f}")
    return summary


def log_attempt(summary: dict, note: str, path: str = "results/solution_attempts.csv") -> None:
    """Append one row per attempt; after the first, a 'what changed and why' note is mandatory."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    attempt = sum(1 for _ in open(path)) if os.path.exists(path) else 1
    if attempt > 1 and not note:
        raise SystemExit('Attempt after the first needs --note "what changed and why"')
    row = {"attempt": attempt, "note": note or "baseline", **{k: v for k, v in summary.items() if k != "convergence"}}
    row["best_hp"] = json.dumps(row["best_hp"])
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if attempt == 1:
            w.writeheader()
        w.writerow(row)
    print(f"  logged attempt {attempt} to {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="smaller evolutionary search")
    ap.add_argument("--note", default="", help="one-line 'what changed and why' for this attempt")
    args = ap.parse_args()
    t0 = time.time()
    pop, gens = (8, 4) if args.fast else (CFG.pop_size, CFG.generations)
    print("PARAMETER CONFIGURATION:", json.dumps(asdict(CFG)))
    bench = Benchmark()

    print(f"\n[1] NSGA-II evolution: population {pop}, {gens} generations")
    best, log = evolve(lambda hp: evaluate_hp(bench, hp), pop, gens)
    hp = best["hp"]
    print(f"  best settings: {hp}")

    print("\n[2] Final training on all data before the test period")
    tr, es = bench.split(hp["lags"], CFG.val_end)
    params, hist = train(tr, es, hp)
    for h in hist[::max(1, len(hist) // 8)] + [hist[-1]]:
        print(f"  epoch {h['epoch']:3d} | train {h['train_loss']:.5f} | val {h['val_loss']:.5f} | "
              f"grad norm {h['grad_norm']:.3f} (clip {CFG.grad_clip})")

    print("\n[3] Drifted test on every scenario (static vs adaptive)")
    results = test_all_scenarios(bench, hp, params, es)
    summary = report(results, hp, params, tr.X.shape[1], log, time.time() - t0)
    log_attempt(summary, args.note)


if __name__ == "__main__":
    main()
