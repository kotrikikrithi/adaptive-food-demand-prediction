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
