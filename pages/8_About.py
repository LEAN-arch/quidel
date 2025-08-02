# pages/8_About.py

import streamlit as st

st.set_page_config(
    page_title="About | V&V Command Center",
    layout="centered",
    page_icon="⚕️"
)

st.image("https://images.wsj.net/im-509536/social", use_container_width=True)

st.title("Assay V&V Command Center: A Strategic Management Tool")
st.markdown("---")
st.markdown("""
This application is a high-fidelity simulation of a **V&V Command Center**, a mission-critical tool designed for an **Associate Director of Assay Verification & Validation** at a world-class medical device company like QuidelOrtho.

Its purpose is to translate complex V&V operations, quality system data, and regulatory requirements into a clear, actionable, and auditable management dashboard. It moves beyond simple data tracking to provide strategic, data-scientist-grade insights, enabling a senior leader to manage risk, allocate resources, and ensure unwavering compliance with **FDA 21 CFR 820, ISO 13485, ISO 14971,** and other global regulations.
""")

st.header("Core Modules & Strategic Value")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("🏠 Main Dashboard")
        st.write("**Purpose:** Executive oversight of the entire V&V portfolio.")
        st.markdown("- **Value:** Enables management-by-exception through high-level KPIs and provides a holistic view of project timelines and resource allocation to proactively identify bottlenecks.")

    with st.container(border=True):
        st.subheader("🗺️ V&V Planning & Strategy")
        st.write("**Purpose:** Define V&V strategy and ensure traceability.")
        st.markdown("- **Value:** Ensures V&V plans are aligned with risk (via Risk Burndown charts) and regulatory strategy from the outset, establishing the RTM as the auditable backbone of the project.")

    with st.container(border=True):
        st.subheader("🚀 V&V Execution & Leadership")
        st.write("**Purpose:** Tactical management of V&V execution and post-launch changes.")
        st.markdown("- **Value:** Provides oversight of protocol execution, deviation management, and the crucial V&V role in the post-market change control (ECO) process via insightful treemaps.")

    with st.container(border=True):
        st.subheader("🖥️ System & Software V&V")
        st.write("**Purpose:** Manage V&V for software and integrated systems.")
        st.markdown("- **Value:** Ensures compliance with **IEC 62304** and **21 CFR Part 11**, managing software-specific risks, testing, and defect resolution with tools like Defect Burndown charts.")

with col2:
    with st.container(border=True):
        st.subheader("📈 Analytical Studies Dashboard")
        st.write("**Purpose:** Deep-dive review of key analytical V&V data.")
        st.markdown("- **Value:** Guarantees data integrity by facilitating rigorous review of study results (e.g., Bland-Altman plots, ANOVA, TOST) against acceptance criteria.")

    with st.container(border=True):
        st.subheader("📦 Regulatory Submission Dashboard")
        st.write("**Purpose:** Final gate-check for V&V submission packages.")
        st.markdown("- **Value:** De-risks regulatory filings by providing a quantitative readiness score and a visual traceability flow (Sankey diagram) of all required V&V evidence.")

    with st.container(border=True):
        st.subheader("🛡️ QMS & Audit Readiness")
        st.write("**Purpose:** Oversight of V&V-related quality events.")
        st.markdown("- **Value:** Demonstrates a state of control to auditors by providing a real-time, visual analysis of CAPA aging, sources, and cycle times.")

    with st.container(border=True):
        st.subheader("🔬 V&V Lab Operations Hub")
        st.write("**Purpose:** Management of laboratory resources.")
        st.markdown("- **Value:** Ensures execution readiness by tracking instrument availability, reagent integrity, and personnel competency, with quantitative utilization analysis.")

st.info(
    """
    **Disclaimer:** This is a demonstrative application. All data is synthetically generated to reflect
    realistic scenarios in the medical device industry and does not represent actual QuidelOrtho products,
    data, or internal processes.
    """,
    icon="ℹ️"
)
