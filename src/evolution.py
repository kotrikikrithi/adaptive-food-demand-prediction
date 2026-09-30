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
