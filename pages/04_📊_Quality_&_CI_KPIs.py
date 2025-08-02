# pages/04_📊_Quality_&_CI_KPIs.py
import streamlit as st
from utils import render_metric_summary, plot_rft_gauge, plot_capa_funnel

st.set_page_config(layout="wide", page_title="Quality & CI KPIs")
st.title("📊 IV. Quality & Continuous Improvement KPIs")
st.markdown("---")

st.header("1. Right-First-Time (RFT) Metrics")
render_metric_summary(
    "Right-First-Time Protocol Execution",
    "Measures the quality of planning and execution. A low RFT rate indicates underlying issues that lead to rework, delays, and increased costs.",
    plot_rft_gauge,
    "An RFT rate of 82% is a good starting point, but there is room for improvement. Action: Set a quarterly goal to increase RFT to 90% by focusing on the root causes of failure identified in the re-test Pareto chart."
)

st.header("2. CAPA & NCR Metrics")
render_metric_summary(
    "CAPA Closure Effectiveness",
    "Tracks the efficiency of the Corrective and Preventive Action (CAPA) process, from identification to confirmed effectiveness.",
    plot_capa_funnel,
    "There is a significant drop-off between 'Implementation' and 'Effectiveness Check'. Action: Reinforce the importance of scheduling and executing effectiveness checks to ensure CAPAs are truly solving problems.",
    "FDA 21 CFR 820.100 (Corrective and preventive action)"
)
