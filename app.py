"""
Web interface for the project.

Start it with:   streamlit run app.py
It opens in the browser at http://localhost:8501

Pages:
  1. Data      - see and download the synthetic data, try different drift scenarios
  2. Run model - start a new attempt and watch the live log
  3. Results   - scores, charts and meal predictions for every attempt
"""
import json
import os
import subprocess
import sys

import altair as alt
import pandas as pd
import streamlit as st

import config
from src.data_generator import generate

ATTEMPTS_CSV = "results/attempts.csv"

st.set_page_config(page_title="Adaptive Food Demand", page_icon="🍱", layout="wide")


@st.cache_data
def load_data(scenario, strength):
    return generate(scenario, drift_start_day=config.VAL_END, strength=strength)


def load_attempts():
    return pd.read_csv(ATTEMPTS_CSV) if os.path.exists(ATTEMPTS_CSV) else pd.DataFrame()


def split_name(day):
    if day < config.TRAIN_END:
        return "1 train"
    return "2 validation" if day < config.VAL_END else "3 test (drift)"


# ---------------------------------------------------------------------------
st.sidebar.title("🍱 Adaptive Food Demand")
page = st.sidebar.radio("Go to", ["1. Data", "2. Run model", "3. Results"])
st.sidebar.caption("Predicts daily meals for restaurants, cafeterias and hostels, "
                   "and adapts when customer behaviour changes.")

# ===========================================================================
if page == "1. Data":
    st.title("1. The data used")
    st.write("Synthetic daily meal counts for 6 outlets over 2 years. Demand depends on weekends, "
             "holidays, weather, special events and semester breaks. From the red line onward, "
             "a **drift** changes customer behaviour. This is the period the model is tested on.")
    c1, c2 = st.columns(2)
    scenario = c1.selectbox("Drift scenario", config.DRIFT_SCENARIOS,
                            index=config.DRIFT_SCENARIOS.index(config.TEST_SCENARIO))
    strength = c2.slider("Drift strength", 0.0, 1.0, float(config.TEST_STRENGTH), 0.1)
    df = load_data(scenario, strength)
    df["split"] = df.day_index.map(split_name)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{len(df):,}")
    m2.metric("Outlets", df.outlet.nunique())
    m3.metric("Days", df.day_index.nunique())
    m4.metric("Drift starts on", str(df.date[df.day_index == config.VAL_END].iloc[0].date()))

    group = st.radio("Outlet type", config.GROUPS, horizontal=True)
    d = df[df.group == group]
    line = alt.Chart(d).mark_line(strokeWidth=1).encode(
        x=alt.X("date:T", title="date"), y=alt.Y("demand:Q", title="meals"),
        color="outlet:N", tooltip=["date:T", "outlet", "demand", "weekend", "holiday", "rain", "event"])
    rule = alt.Chart(pd.DataFrame({"date": [df.date[df.day_index == config.VAL_END].iloc[0]]})) \
        .mark_rule(color="red", strokeDash=[5, 3]).encode(x="date:T")
    st.altair_chart((line + rule).properties(height=350).interactive(), width="stretch")

    st.subheader("Average meals: weekday vs weekend, before and after drift")
    d2 = df.assign(period=(df.day_index >= config.VAL_END).map({False: "before drift", True: "after drift"}),
                   day=df.weekend.map({0: "weekday", 1: "weekend"}))
    st.dataframe(d2.pivot_table(index="group", columns=["period", "day"], values="demand", aggfunc="mean")
                 .round(0), width="stretch")

    st.subheader("Raw data")
    st.caption("train = model learns · validation = used to choose the best settings · "
               "test = drifted period for the final score")
    st.dataframe(df, width="stretch", height=300)
    st.download_button("Download CSV", df.to_csv(index=False), f"food_demand_{scenario}.csv", "text/csv")

# ===========================================================================
elif page == "2. Run model":
    st.title("2. Run the model")
    attempts = load_attempts()
    n_next = len(attempts) + 1
    st.write(f"This will be **Attempt {n_next}**. The run generates data, evolves network settings "
             "(NSGA-II), trains the best network, then tests it on the drifted period with and without "
             "adaptation.")
    c1, c2 = st.columns(2)
    drift = c1.selectbox("Drift scenario for the test", config.DRIFT_SCENARIOS,
                         index=config.DRIFT_SCENARIOS.index(config.TEST_SCENARIO))
    quick = c2.checkbox("Quick mode (about 2 min instead of about 8 min)", value=True)
    note = st.text_input("What changed and why? (required after Attempt 1)",
                         placeholder="e.g. tested level_shift drift to check adaptation")
    st.caption(f"Search settings from config.py: population {config.POP_SIZE}, "
               f"generations {config.N_GENERATIONS}, parameter budget {config.PARAM_BUDGET}.")

    if st.button("▶ Run attempt", type="primary"):
        if n_next > 1 and not note.strip():
            st.error("Please write a note: what changed and why.")
        else:
            cmd = [sys.executable, "-u", "run_attempt.py", "--drift", drift, "--note", note]
            if quick:
                cmd.append("--quick")
            status = st.status(f"Running Attempt {n_next}...", expanded=True)
            box = status.empty()
            lines = []
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace")
            for line in proc.stdout:
                if "Warning" in line or "warnings.warn" in line:
                    continue
                lines.append(line.rstrip())
                box.code("\n".join(lines[-40:]))
            proc.wait()
            os.makedirs("results", exist_ok=True)
            with open(f"results/attempt{n_next}_console.log", "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            if proc.returncode == 0:
                status.update(label=f"Attempt {n_next} finished ✅", state="complete")
                final = [l for l in lines if "FINAL FITNESS" in l]
                st.success((final[-1] if final else "Done") + ". Open the **3. Results** page.")
            else:
                status.update(label="Run failed ❌", state="error")

# ===========================================================================
else:
    st.title("3. Results")
    attempts = load_attempts()
    if attempts.empty:
        st.info("No attempts yet. Go to **2. Run model**.")
        st.stop()

    st.subheader("All attempts (lower fitness is better)")
    show = ["attempt", "timestamp", "test_drift", "note", "quick", "n_params", "latency_ms",
            "test_wape_static", "test_wape_adaptive", "final_fitness", "runtime_s"]
    st.dataframe(attempts[[c for c in show if c in attempts]], width="stretch", hide_index=True)
    if len(attempts) > 1:
        st.line_chart(attempts.set_index("attempt")[["final_fitness"]], height=180)

    n = st.selectbox("Look at attempt", attempts.attempt.tolist()[::-1])
    folder = f"results/attempt_{int(n):02d}"
    with open(f"{folder}/summary.json") as f:
        S = json.load(f)
    rec, st_, ad = S["record"], S["summary"]["static"], S["summary"]["adaptive"]

    k = st.columns(5)
    k[0].metric("Final fitness", f"{rec['final_fitness']:.4f}")
    k[1].metric("Error: normal model", f"{st_['wape']:.1%}")
    k[2].metric("Error: adaptive model", f"{ad['wape']:.1%}", f"{ad['wape'] - st_['wape']:+.1%}",
                delta_color="inverse")
    k[3].metric("Parameters", f"{rec['n_params']}", f"budget {config.PARAM_BUDGET}", delta_color="off")
    k[4].metric("Latency", f"{rec['latency_ms']:.3f} ms", f"budget {config.LATENCY_BUDGET_MS} ms",
                delta_color="off")
    st.caption(f"Drift: **{rec['test_drift']}** · Note: {rec['note'] or '-'} · Settings: `{rec['settings']}`")

    tab1, tab2, tab3, tab4 = st.tabs(["Predictions", "Fairness & waste", "Convergence", "Explainability"])

    with tab1:
        pred = pd.read_csv(f"{folder}/test_predictions.csv", parse_dates=["date"])
        for c in ["y", "pred_static", "pred_adaptive", "lo_adaptive", "hi_adaptive"]:
            pred[c + "_meals"] = (pred[c] * pred.scale).round()
        outlet = st.selectbox("Outlet", sorted(pred.outlet.unique()))
        p = pred[pred.outlet == outlet]
        long = p.melt("date", ["y_meals", "pred_static_meals", "pred_adaptive_meals"], "series", "meals")
        long.series = long.series.map({"y_meals": "actual", "pred_static_meals": "normal model",
                                       "pred_adaptive_meals": "adaptive model"})
        band = alt.Chart(p).mark_area(opacity=0.2).encode(
            x="date:T", y=alt.Y("lo_adaptive_meals:Q", title="meals"), y2="hi_adaptive_meals:Q")
        lines = alt.Chart(long).mark_line(strokeWidth=1.5).encode(
            x=alt.X("date:T", title="date"), y="meals:Q",
            color=alt.Color("series:N", scale=alt.Scale(domain=["actual", "normal model", "adaptive model"],
                                                        range=["black", "#4c78a8", "#f58518"])),
            tooltip=["date:T", "series", "meals"])
        alarms = [a for a in S["alarms"] if a["group"] == p.group.iloc[0]]
        chart = band + lines
        if alarms:
            ad_dates = p[p.day_index.isin([a["day_index"] for a in alarms])][["date"]]
            chart += alt.Chart(ad_dates).mark_rule(color="red", strokeDash=[4, 3]).encode(x="date:T")
        st.altair_chart(chart.properties(height=380).interactive(), width="stretch")
        st.caption("Shaded band = 90% safe range · red dashed lines = drift alarms (the model retrained itself)")
        st.dataframe(p[["date", "y_meals", "pred_static_meals", "pred_adaptive_meals", "lo_adaptive_meals",
                        "hi_adaptive_meals"]].rename(columns={
                            "y_meals": "actual", "pred_static_meals": "normal model",
                            "pred_adaptive_meals": "adaptive model", "lo_adaptive_meals": "safe range low",
                            "hi_adaptive_meals": "safe range high"}),
                     width="stretch", hide_index=True, height=250)

    with tab2:
        rows = []
        for g in config.GROUPS:
            rows.append({"outlet type": g,
                         "error normal": f"{st_['group_wape'][g]:.1%}", "error adaptive": f"{ad['group_wape'][g]:.1%}",
                         "bias normal": f"{st_['group_bias'][g]:+.1%}", "bias adaptive": f"{ad['group_bias'][g]:+.1%}",
                         "90% range coverage normal": f"{st_['coverage'][g]:.0%}",
                         "90% range coverage adaptive": f"{ad['coverage'][g]:.0%}"})
        st.write("**Fair calibration:** every outlet type should have a bias near 0% and a coverage near 90%.")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        w = st.columns(4)
        w[0].metric("Waste, normal (meals)", f"{st_['waste_meals']:,.0f}")
        w[1].metric("Waste, adaptive", f"{ad['waste_meals']:,.0f}")
        w[2].metric("Shortage, normal (meals)", f"{st_['shortage_meals']:,.0f}")
        w[3].metric("Shortage, adaptive", f"{ad['shortage_meals']:,.0f}")

    with tab3:
        ev = pd.DataFrame(S["evolution"])
        st.write("**Evolution (NSGA-II):** best and average fitness of the population per generation.")
        st.line_chart(ev.set_index("generation")[["best_fitness", "mean_pop_fitness"]], height=250)
        st.dataframe(ev, width="stretch", hide_index=True)
        tr = pd.read_csv(f"{folder}/training_log.csv")
        st.write("**Final model training:** loss per epoch (it should go down smoothly, with no explosion).")
        st.line_chart(tr.set_index("epoch")[["train_loss", "val_loss"]], height=250)
        st.write("**Final population (Pareto trade-offs):**")
        st.dataframe(pd.read_csv(f"{folder}/final_population.csv"), width="stretch", hide_index=True)

    with tab4:
        imp = pd.DataFrame({"feature": list(S["importance"]), "importance": list(S["importance"].values())})
        st.write("How much the error grows when a feature is scrambled. Bigger = the model relies on it more.")
        st.altair_chart(alt.Chart(imp).mark_bar().encode(
            x=alt.X("importance:Q", title="error increase"), y=alt.Y("feature:N", sort="-x", title=None)),
            width="stretch")
