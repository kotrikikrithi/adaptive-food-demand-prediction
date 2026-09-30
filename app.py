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
