# Written Explanation: Adaptive Food Demand Prediction

## 1. Problem
Predict tomorrow's meals for each outlet (2 restaurants, 2 cafeterias, 2 hostels). Two costs matter: over-prediction becomes **waste** and under-prediction becomes a **shortage**. Demand depends on weekends, holidays, weather, events and semester breaks, and these rules **change over time** (drift).

## 2. Data and benchmark scenarios
`src/data_generator.py` simulates 730 days. Each driver has a different effect on each outlet type. For example, weekends raise restaurant demand but empty the office cafeterias, and hostels drop during the semester break. Six scenarios can be switched on from any day and at any strength: `none`, `level_shift`, `weekend_pattern_flip`, `weather_sensitivity_change`, `sudden_event_spike` and `gradual_trend`. With the same seed, the days before the drift are identical in every scenario, so the drift is the only thing that differs.

Splits in time order: train (days 0-437), **out-of-distribution validation** (days 438-583, under 3-4 *different* mild drifts), and test (days 584-729, under the chosen strong drift, which is never one of the validation drifts).

## 3. Representation
- **Model input**: lagged demand (the lag set is chosen by evolution), rolling mean/std, day of week (sin/cos), weekend, holiday, temperature, rain, event, semester break, and a one-hot outlet type. The target is demand divided by the outlet's training mean, which puts every outlet on the same scale.
- **Model**: an MLP with 1-3 layers and 8-64 units.
- **Genome** (evolution): 8 real genes in [0, 1] that decode to `n_layers, width, learning rate (log), dropout, weight decay (log), lag set, activation, Huber delta`.

## 4. Operators and rules (NSGA-II, `src/evolution.py`)
| Step | Rule |
|---|---|
| Selection | binary tournament: lower Pareto rank wins, then larger crowding distance |
| Crossover | blend: child = w·p1 + (1-w)·p2, with a random w per gene (prob 0.9) |
| Mutation | Gaussian, σ = 0.15, per gene with prob 0.25, clipped to [0, 1] |
| Survival | (μ+λ): keep the best Pareto fronts of parents + children, trimmed by crowding distance |
| Constraints | constrained domination: within the parameter budget and not diverged always beats infeasible |
| Cache | identical decoded settings are not retrained (evaluation is deterministic) |

**Four objectives (all minimised)**
1. Mean WAPE over the OOD validation scenarios (accuracy and generalisation)
2. Std of WAPE across scenarios + the train-to-validation gap (loss variance and overfitting penalty)
3. Parameters / budget (runtime cost; deterministic, unlike measured latency)
4. Calibration gap: max minus min relative bias across outlet types (fairness)

The reported fitness is `1.0·f1 + 1.0·f2 + 0.1·f3 + 0.5·f4` (lower is better). The weights are in `config.py`. The deployed model is the lowest-fitness feasible member of the final population.

## 5. How each hard constraint is met
| Constraint | Mechanism |
|---|---|
| Generalisation on OOD sets | selection is scored on several *differently drifted* validation sets; the test drift is held out |
| Deterministic convergence, no gradient explosion | fixed seeds, `torch.use_deterministic_algorithms`, gradient-norm clipping (1.0), Huber loss, early stopping; max gradient norm is logged every epoch; reruns give identical scores |
| Parameter/runtime budget | hard constraint `PARAM_BUDGET = 5000`; latency is measured and reported against `LATENCY_BUDGET_MS` |
| Fair calibration across sub-populations | objective 4 in the search, a per-type bias correction online, and **Mondrian (per-type) conformal intervals**, so each outlet type gets about 90% coverage |

## 6. Adaptation to drift (`src/adaptive.py`), the CI rationale
A model trained offline cannot anticipate unseen drift, so it must adapt while running:
1. **Per-type EWMA bias correction**: cheap, and responds within days.
2. **Page-Hinkley test** on each type's errors: raises an alarm when the errors shift persistently.
3. **Fine-tuning** on the last 42 days, weekly for 35 days after an alarm. It uses a reduced learning rate, and gradient clipping still applies.
4. **Rolling conformal intervals** per type: the intervals widen automatically when errors grow.

**Why evolutionary multi-objective search?** The objectives conflict: bigger models are more accurate but more costly, and the model that fits best on average is not the fairest. NSGA-II returns a whole **Pareto set** of trade-offs in one run, needs no gradients through the hyperparameters, and handles discrete choices (layers, activation, lag set) naturally.

## 7. Explainability and stability
- Permutation importance by concept (`feature_importance.png`): the WAPE increase when that feature group is shuffled.
- A small MLP, a reproducible seed, and a per-epoch log of loss and gradient norm.
- `test_predictions.png` shows actual demand, the static and adaptive forecasts, the interval, and the drift alarms.

## 8. Attempt log
Every run appends to `results/attempts.csv`, together with its mandatory "what changed and why" note.

## 9. 6-month forecast (`src/forecast.py`)
- **Direct forecasting**: each future day is predicted from its own known conditions (calendar, typical weather for the date, planned events, outlet type with per-type interaction features). Nothing is fed back recursively. We tried a recursive version first, and its errors piled up into a false downward trend.
- **Regime detection**: a Page-Hinkley scan of a first-year model's errors finds when behaviour changed (detected 9 Aug 2025; true change 7 Aug). A `regime` flag and its per-type interactions keep the "new behaviour" separate from seasonal effects. Without it, cafeterias wrongly dropped in May–June, because the only May–June examples were from the old regime.
- **Recency weighting**: sample weight halves every 120 days.
- **Backtest** (forecast made on 30 Jun 2025 for Jul–Dec 2025), WAPE: seasonal-naive 38.7%, one-shot 20.2%, **updated monthly 10.1%**. We therefore recommend re-forecasting monthly.
- **Safe range**: the 90% quantile of the relative backtest errors, computed per outlet type.
