# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import generate_vv_project_data, generate_risk_management_data
from datetime import date

# --- Page Configuration ---
st.set_page_config(
    page_title="Assay V&V Command Center | QuidelOrtho",
    page_icon="⚕️",
    layout="wide"
)

# --- Data Loading ---
# These functions in utils.py now generate data specific to QuidelOrtho's platforms and V&V lifecycle.
projects_df = generate_vv_project_data()
risks_df = generate_risk_management_data()

# --- Page Title and Header ---
st.title("⚕️ Assay V&V Command Center | QuidelOrtho")
st.markdown("### Strategic Oversight of Assay & Consumable Verification and Validation Activities")

# --- KPIs: Director-Level Oversight Metrics ---
st.header("Executive Summary: V&V Portfolio & Resource Posture")
total_projects = len(projects_df)
execution_phase_projects = projects_df[projects_df['V&V Phase'].isin(['Execution', 'Data Analysis'])].shape[0]
at_risk_projects = projects_df[projects_df['Overall Status'] == 'At Risk'].shape[0]
high_impact_risks = risks_df[risks_df['Risk_Score'] >= 15].shape[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total V&V Projects in Pipeline", f"{total_projects}")
col2.metric("Projects in Execution/Analysis", f"{execution_phase_projects}")
col3.metric("Projects At Risk (Timeline/Budget)", f"{at_risk_projects}", delta=f"{at_risk_projects} At Risk", delta_color="inverse")
col4.metric("High-Impact Risks (Score > 15)", f"{high_impact_risks}")

st.divider()

# --- Main Content Area ---
col1, col2 = st.columns((2, 1.2))

with col1:
    st.header("V&V Portfolio Timeline & Resource Allocation")
    st.caption("Tracking the lifecycle of all V&V projects from planning to final report and DHF compilation.")
    fig = px.timeline(
        projects_df,
        x_start="Start Date",
        x_end="Due Date",
        y="Project/Assay",
        color="V&V Phase",
        title="V&V Project Timelines by Phase",
        hover_name="Project/Assay",
        hover_data={
            "V&V Lead": True,
            "Overall Status": True,
            "Platform": True,
            "Regulatory Pathway": True,
            "Start Date": "|%B %d, %Y",
            "Due Date": "|%B %d, %Y",
        },
        color_discrete_map={
            'Planning': '#17A2B8',          # Info Blue
            'Protocol Development': '#007BFF', # Primary Blue
            'Execution': '#FFC107',         # Warning Yellow
            'Data Analysis': '#FD7E14',      # Orange
            'Reporting': '#28A745',         # Success Green
            'On Hold': '#DC3545',           # Danger Red
            'Complete': '#6C757D'           # Secondary Gray
        }
    )
    fig.update_yaxes(categoryorder="total ascending", title=None)
    fig.update_layout(legend_title_text='V&V Phase')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("Assay & Project Risk Matrix (ISO 14971)")
    st.caption("Prioritizing risks to product quality, patient safety, and regulatory submission timelines.")
    fig_risk = px.scatter(
        risks_df, x="Probability", y="Severity", size="Risk_Score", color="Risk_Score",
        color_continuous_scale=px.colors.sequential.OrRd, hover_name="Risk Description",
        hover_data=["Project", "Owner", "Mitigation"], size_max=40, title="Risk Severity vs. Probability"
    )
    fig_risk.update_layout(
        xaxis=dict(tickvals=[1, 2, 3, 4, 5], ticktext=['Improbable', 'Remote', 'Occasional', 'Probable', 'Frequent'], title='Probability of Occurrence'),
        yaxis=dict(tickvals=[1, 2, 3, 4, 5], ticktext=['Negligible', 'Minor', 'Serious', 'Critical', 'Catastrophic'], title='Severity of Harm'),
        coloraxis_showscale=False
    )
    # Highlight the "Unacceptable Risk" region per ISO 14971
    fig_risk.add_shape(type="rect", xref="x", yref="y", x0=3.5, y0=3.5, x1=5.5, y1=5.5, fillcolor="rgba(220, 53, 69, 0.2)", layer="below", line_width=0)
    fig_risk.add_annotation(x=4.5, y=4.5, text="Action Required", showarrow=False, font=dict(color="#DC3545", size=14, family="Arial, bold"))
    st.plotly_chart(fig_risk, use_container_width=True)

st.header("V&V Project Portfolio: Detailed View")
st.dataframe(projects_df, use_container_width=True, hide_index=True)

# --- REGULATORY CONTEXT ---
st.divider()
with st.expander("🌐 Regulatory Context & Dashboard Purpose (Associate Director's View)"):
    st.markdown("""
    As the Associate Director of Assay V&V, my primary responsibility is to provide strategic leadership and oversight to ensure that all verification and validation activities are executed efficiently, effectively, and in full compliance with global regulatory standards. This Command Center is my central tool for managing the V&V function and demonstrating control to executive leadership and regulatory auditors.

    #### **How This Dashboard Fulfills My Core Responsibilities:**

    - **Management, Planning, & Resourcing:**
        - The **V&V Portfolio Timeline** provides a comprehensive view of all active projects, their current phase, and assigned leads. This is essential for strategic planning, resource allocation, and identifying potential bottlenecks before they impact critical project timelines.
        - The **Executive KPIs** provide an at-a-glance health check of the entire portfolio, allowing me to focus my attention on at-risk projects and manage by exception.

    - **Oversight of Design Controls (21 CFR 820.30 & ISO 13485):**
        - This dashboard serves as a master index for our Design Control process. The **Project Portfolio Details** table tracks key V&V deliverables that form the backbone of the Design History File (DHF).
        - The **Project Execution Hub** provides tactical oversight of task completion and document readiness, ensuring a smooth and auditable transfer of design outputs to manufacturing.

    - **Risk Management Leadership (ISO 14971):**
        - The **Risk Matrix** is a direct implementation of our risk management procedure. It allows me and my team to continuously evaluate risks to product quality and project success, ensuring mitigations are prioritized and effective. This is a key input for our regulatory submissions.

    - **Regulatory Submission & Audit Support (510(k)/PMA):**
        - The entire dashboard is designed to be "audit-ready." It provides clear, traceable, objective evidence of our V&V process.
        - The **Regulatory Submission Package Dashboard** consolidates all final V&V evidence, ensuring our 510(k) and PMA submissions are complete, robust, and defensible.

    - **Team Development & Coaching:**
        - By monitoring project progress and challenges, I can identify areas where the team needs support, coaching, or additional training. The **V&V Lab Hub** provides specific tools to track and manage team competency and development plans.
    """)
