# pages/01_Assay_V_and_V_Metrics.py
import streamlit as st
from utils import render_metric_summary, plot_protocol_completion_burndown, plot_pass_rate_heatmap, plot_retest_pareto, plot_trace_coverage_sankey, plot_rpn_waterfall

st.set_page_config(layout="wide", page_title="Assay V&V Metrics")
st.title("🧪 Assay V&V-Specific Metrics & KPIs")
st.markdown("---")

st.header("1. Test Execution Metrics")
render_metric_summary("Protocol Completion (per project)", "Tracks progress against the plan, indicating project velocity and potential delays.", plot_protocol_completion_burndown, "The team is slightly behind the ideal burndown. Action: Investigate the root cause for the slowdown.")
render_metric_summary("Pass Rate by Test Type", "Identifies problematic test areas or assay weaknesses across projects.", plot_pass_rate_heatmap, "The low pass rate for 'Robustness' (88%) is a red flag. Action: Convene with R&D to review the robustness study design.")
render_metric_summary("Re-test Rate & Root Cause", "Highlights sources of inefficiency and quality issues in the lab. A high re-test rate impacts timelines and cost.", plot_retest_pareto, "'Operator Error' is the primary cause of re-tests. Action: Schedule refresher training on the specific SOPs where errors are most common.")

st.header("2. Design Control & Traceability")
render_metric_summary("Requirements Trace Coverage", "The most critical metric for audit readiness, ensuring no gaps exist.", plot_trace_coverage_sankey, "The Sankey diagram clearly shows an uncovered requirement. This is a critical gap that must be closed before the design review.", "FDA 21 CFR 820.30(j) - Design History File (DHF)")

st.header("3. Regulatory Compliance & Risk")
render_metric_summary("FMEA Risk Reduction", "Demonstrates the effectiveness of V&V activities as risk mitigations, a key requirement of ISO 14971.", plot_rpn_waterfall, "V&V activities successfully reduced the total risk portfolio by mitigating high-RPN items. This provides objective evidence of building a safer product.", "ISO 14971 - Risk management for medical devices")
