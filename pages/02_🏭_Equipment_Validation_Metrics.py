# pages/02_🏭_Equipment_Validation_Metrics.py
import streamlit as st
from utils import render_metric_summary, plot_validation_gantt_baseline, plot_sat_to_pq_violin

st.set_page_config(layout="wide", page_title="Equipment Validation")
st.title("🏭 II. Equipment Validation Metrics (FAT/SAT/IQ/OQ/PQ)")
st.markdown("---")

st.header("1. Validation Execution Health")
render_metric_summary(
    "Validation On-Time Rate",
    "Compares planned validation timelines against actual execution, highlighting delays in specific phases (FAT, SAT, IQ, etc.).",
    plot_validation_gantt_baseline,
    "The 'SAT' phase experienced a significant delay. Action: Investigate the root cause of the Site Acceptance Testing delay to prevent recurrence on the next equipment installation."
)

st.header("2. Readiness & Qualification")
render_metric_summary(
    "Time from SAT to PQ Approval",
    "Measures the efficiency of the entire on-site qualification process. Long cycles can indicate issues with facility readiness, documentation, or equipment performance.",
    plot_sat_to_pq_violin,
    "Sample Prep' equipment shows a much wider and longer qualification cycle time than 'Analyzers'. Action: Launch a process improvement (Kaizen) event to streamline the sample prep qualification workflow.",
    "GAMP 5 - A Risk-Based Approach to Compliant GxP Computerized Systems"
)
