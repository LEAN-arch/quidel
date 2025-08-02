# pages/03_👥_Team_&_Project_KPIs.py
import streamlit as st
from utils import render_metric_summary, plot_protocol_review_cycle_histogram, plot_training_donut

st.set_page_config(layout="wide", page_title="Team & Project KPIs")
st.title("👥 III. Team & Project Management KPIs")
st.markdown("---")

st.header("1. Productivity & Load")
render_metric_summary(
    "Protocol Review Cycle Time",
    "Measures the efficiency of the documentation workflow. Long cycle times can be a bottleneck, delaying the start of execution.",
    plot_protocol_review_cycle_histogram,
    "The distribution shows a long tail, with some protocols taking over 8 days to approve. Action: Implement a daily review board meeting to address and resolve comments on aging documents."
)

st.header("2. Training & Competency")
render_metric_summary(
    "Training Completion Rate",
    "A critical quality system metric ensuring the team is competent and compliant with all required procedures and regulations.",
    plot_training_donut,
    "The team is 100% compliant on the critical ISO 13485 training but lagging in GAMP5. Action: Schedule a GAMP5 training session for the team within the next quarter.",
    "ISO 13485:2016 Sec 6.2 (Human Resources)"
)
