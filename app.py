# app.py

import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="V&V Executive Briefing | Portfolio",
    page_icon="🎯"
)

# --- Sidebar ---
with st.sidebar:
    # FIX: Replaced deprecated `use_column_width` with `use_container_width`
    st.image("https://i.imgur.com/3X6L24F.png", use_container_width=True)
    st.title("AssayVantage Executive Briefing")
    st.markdown("---")
    st.success("Select a competency area below.")
    st.markdown("---")
    st.info("""
    **Objective:** This interactive portfolio is a comprehensive demonstration of the strategic, technical, and leadership capabilities required for the **Associate Director, Assay V&V** role.
    """)

# --- Main Page ---
st.title("🎯 The V&V Executive Command Center")
st.markdown("A definitive showcase of data-driven leadership in a regulated GxP environment.")
st.markdown("---")
st.markdown("""
Welcome. This application translates the core responsibilities of V&V leadership into a suite of interactive, high-density dashboards. Each section, accessible via the sidebar, addresses a critical aspect of the role, from assay-specific metrics to advanced statistical modeling.

This is not just a report; it's a live demonstration of how to lead a V&V function with strategic foresight and technical excellence.
""")

st.subheader("Core Competency Dashboards")
st.info("Please use the sidebar on the left to navigate to each detailed dashboard.")

row1_cols = st.columns(3)
row2_cols = st.columns(3)

# FIX: Replaced all st.page_link calls with st.markdown to prevent KeyError
with row1_cols[0]:
    with st.container(border=True):
        st.subheader("🧪 Assay V&V")
        st.markdown("Execution, Traceability, Risk & Compliance Metrics.")
        st.markdown("`Select from sidebar →`")
with row1_cols[1]:
    with st.container(border=True):
        st.subheader("🏭 Equipment")
        st.markdown("FAT, SAT, IQ, OQ, PQ Validation & Readiness KPIs.")
        st.markdown("`Select from sidebar →`")
with row1_cols[2]:
    with st.container(border=True):
        st.subheader("👥 Team & Project")
        st.markdown("Productivity, Load, Cycle Time & Competency KPIs.")
        st.markdown("`Select from sidebar →`")
with row2_cols[0]:
    with st.container(border=True):
        st.subheader("📊 Quality & CI")
        st.markdown("Right-First-Time, CAPA Effectiveness & Recurrence.")
        st.markdown("`Select from sidebar →`")
with row2_cols[1]:
    with st.container(border=True):
        st.subheader("💻 Software V&V")
        st.markdown("IEC 62304, Risk-Based Testing & Part 11 Compliance.")
        st.markdown("`Select from sidebar →`")
with row2_cols[2]:
    with st.container(border=True):
        st.subheader("📐 Statistical Methods")
        st.markdown("Interactive workbench for ANOVA, SPC, Regression, etc.")
        st.markdown("`Select from sidebar →`")

st.markdown("---")
st.success("Built with Python, Streamlit, and Plotly to demonstrate expertise in creating bespoke digital tools for operational excellence in Life Sciences.")
