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
