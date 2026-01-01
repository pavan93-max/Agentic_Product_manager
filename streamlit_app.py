import sys
from pathlib import Path

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import matplotlib.pyplot as plt
from engine.memory import get_experiments

st.set_page_config(
    page_title="Autonomous Experimentation Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    memory = get_experiments()
except Exception as e:
    st.error(f"Failed to load experiment data: {e}")
    memory = []

st.title("🚀 Autonomous Product Experimentation Agent")
st.markdown("---")

with st.sidebar:
    st.header("📊 Dashboard Info")
    st.metric("Total Experiments", len(memory))
    if memory:
        decisions = [e.get("decision", "UNKNOWN") for e in memory]
        st.metric("SHIP Decisions", decisions.count("SHIP"))
        st.metric("ITERATE Decisions", decisions.count("ITERATE"))
        st.metric("ROLLBACK Decisions", decisions.count("ROLLBACK"))

if not memory:
    st.warning("⚠️ No experiments run yet. Run `python main.py` to start an experiment.")
    st.info("💡 This dashboard will display experiment results, Bayesian analysis, and decision history.")
    st.stop()

latest = memory[-1]

st.subheader("📈 Latest Experiment Results")
col1, col2, col3, col4 = st.columns(4)

decision = latest.get("decision", "UNKNOWN")
decision_color = {
    "SHIP": "🟢",
    "ROLLBACK": "🔴",
    "ITERATE": "🟡"
}.get(decision, "⚪")

col1.metric("Decision", f"{decision_color} {decision}", delta=None)

bayesian_result = latest.get("bayesian_result", {})
lift_mean = bayesian_result.get("lift_mean", 0.0)
col2.metric("Posterior Lift", f"{lift_mean*100:.2f}%", delta=f"{lift_mean*100:.2f}%" if lift_mean > 0 else None)

prob_better = bayesian_result.get("prob_treatment_better", 0.0)
col3.metric("P(Treatment > Control)", f"{prob_better:.2%}", delta=f"{prob_better:.1%}" if prob_better > 0.5 else None)

experiment = latest.get("experiment", {})
sample_size = experiment.get("sample_size", 0)
col4.metric("Sample Size", f"{sample_size:,}", delta=None)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Posterior Lift Distribution")
    ci_95 = bayesian_result.get("ci_95", [0.0, 0.0])
    ci_low, ci_high = ci_95[0], ci_95[1]
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.hlines(1, ci_low, ci_high, linewidth=6, color='steelblue', alpha=0.7, label='95% CI')
    ax.plot(lift_mean, 1, "o", markersize=12, color='darkblue', label='Mean')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='No effect')
    ax.set_yticks([])
    ax.set_xlabel("Lift (Treatment - Control)", fontsize=11)
    ax.set_title("95% Credible Interval", fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    st.caption(f"95% Credible Interval: [{ci_low:.4f}, {ci_high:.4f}]")

with col2:
    st.subheader("📋 Experiment Details")
    if experiment:
        st.json({
            "Metric": experiment.get("metric", "N/A"),
            "Control": experiment.get("control", {}),
            "Treatment": experiment.get("treatment", {}),
            "Sample Size": experiment.get("sample_size", 0)
        })
    else:
        st.info("No experiment details available")

st.markdown("---")

st.subheader("📊 Decision History")
decisions = [e.get("decision", "UNKNOWN") for e in memory]
decision_counts = {d: decisions.count(d) for d in set(decisions)}
fig, ax = plt.subplots(figsize=(10, 4))
colors = {"SHIP": "#2ecc71", "ITERATE": "#f39c12", "ROLLBACK": "#e74c3c", "UNKNOWN": "#95a5a6"}
bars = ax.bar(decision_counts.keys(), decision_counts.values(), 
               color=[colors.get(d, "#95a5a6") for d in decision_counts.keys()])
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Decision Distribution", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontweight='bold')
st.pyplot(fig, use_container_width=True)

st.subheader("📈 Experiment Trends")
tab1, tab2 = st.tabs(["Cumulative Regret", "Lift Over Time"])

with tab1:
    regret = []
    total = 0
    for e in memory:
        if e.get("decision") != "SHIP":
            total += 1
        regret.append(total)
    st.line_chart(regret, use_container_width=True)
    st.caption("Cumulative count of non-SHIP decisions (proxy for regret)")

with tab2:
    lifts = [e.get("bayesian_result", {}).get("lift_mean", 0.0) for e in memory]
    if lifts:
        st.line_chart(lifts, use_container_width=True)
        st.caption("Posterior lift mean across experiments")
    else:
        st.info("No lift data available")

st.markdown("---")
st.caption("💡 Dashboard updates automatically when new experiments are logged.")

