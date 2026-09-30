# adaptive-food-demand-prediction

Predicts daily meal demand for restaurants, cafeterias and hostels, and **keeps working when
demand patterns change** (drift). The network's settings are found with a multi-objective
evolutionary algorithm (NSGA-II). A drift detector plus online fine-tuning keeps the model
reliable after demand patterns change.

## Setup (once per laptop)
1. Install Python 3.12 (python.org, tick "Add python.exe to PATH").
2. In this folder, open a terminal (PowerShell) and run:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
   Each time you open a new terminal, run `venv\Scripts\activate` again. You'll see `(venv)` at the start of the line.

## Run
| Command | What it does |
|---|---|
| `python -m src.data_generator` | makes sample data + plot in `results/` |
| `python run_attempt.py --note "baseline"` | full attempt: evolve → train → drifted test → score |
| `python run_attempt.py --drift level_shift --note "why"` | same, with a different drift |
| `python run_attempt.py --quick --note "why"` | small, fast search (~2 min) |

Drift options: `none, level_shift, weekend_pattern_flip, weather_sensitivity_change, sudden_event_spike, gradual_trend`.

Every attempt after the first **requires** `--note "what changed and why"`.

## Outputs
- `results/attempts.csv`: one row per attempt (settings, fitness, static vs adaptive error, note)
- `results/attempt_XX/`: `evolution_log.csv` (convergence), `training_log.csv` (loss per epoch),
  `final_population.csv` (Pareto set), `summary.json`, and plots (`convergence.png`,
  `test_predictions.png`, `feature_importance.png`)

## Where things are
| File | Purpose |
|---|---|
| `config.py` | **all parameters**, with comments |
| `src/data_generator.py` | synthetic data + drift scenarios |
| `src/features.py` | model inputs (lags, calendar, weather, outlet type) |
| `src/model.py` | small neural network + stable training |
| `src/evolution.py` | NSGA-II hyperparameter search |
| `src/adaptive.py` | drift detection, bias correction, fine-tuning, fair intervals |
| `src/metrics.py` | WAPE, calibration, coverage, waste/shortage, fitness |
| `docs/EXPLANATION.md` | written explanation for the jury |
