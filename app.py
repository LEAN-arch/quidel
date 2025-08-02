# app.py

import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="V&V Executive Briefing | Portfolio",
    page_icon="🎯"
)

# --- Sidebar ---
with st.sidebar:
    st.image("https://i.imgur.com/3X6L24F.png", use_column_width=True)
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
st.info("Please use the sidebar to navigate to each detailed dashboard.")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    with st.container(border=True):
        st.subheader("🧪 Assay V&V")
        st.markdown("Execution, Traceability, Risk & Compliance Metrics.")
        st.page_link("pages/01_🧪_Assay_V&V_Metrics.py", label="Explore Assay Metrics →")
with col2:
    with st.container(border=True):
        st.subheader("🏭 Equipment")
        st.markdown("FAT, SAT, IQ, OQ, PQ Validation & Readiness KPIs.")
        st.page_link("pages/02_🏭_Equipment_Validation_Metrics.py", label="Explore Equipment KPIs →")
with col3:
    with st.container(border=True):
        st.subheader("👥 Team & Project")
        st.markdown("Productivity, Load, Cycle Time & Competency KPIs.")
        st.page_link("pages/03_👥_Team_&_Project_KPIs.py", label="Explore Management KPIs →")
with col4:
    with st.container(border=True):
        st.subheader("📊 Quality & CI")
        st.markdown("Right-First-Time, CAPA Effectiveness & Recurrence.")
        st.page_link("pages/04_📊_Quality_&_CI_KPIs.py", label="Explore Quality Metrics →")
with col5:
    with st.container(border=True):
        st.subheader("📐 Statistical Methods")
        st.markdown("Interactive workbench for ANOVA, SPC, Regression, etc.")
        st.page_link("pages/05_📐_Advanced_Statistical_Methods.py", label="Explore Statistical Tools →")

st.markdown("---")
st.success("Built with Python, Streamlit, and Plotly to demonstrate expertise in creating bespoke digital tools for operational excellence in Life Sciences.")
