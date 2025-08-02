# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import generate_vv_project_data, generate_risk_management_data
from datetime import date

# --- Page Configuration ---
st.set_page_config(
    page_title="Validation Command Center | QuidelOrtho",
    page_icon="⚕️",
    layout="wide"
)

# --- Data Loading ---
projects_df = generate_vv_project_data()
risks_df = generate_risk_management_data()

# --- Page Title and Header ---
st.title("⚕️ Validation Command Center | QuidelOrtho")
st.markdown("### An Integrated Hub for Assay V&V and Equipment & Process Validation")

# --- KPIs: SME-Grade Actionable Metrics ---
st.header("Executive Validation Portfolio Health")
total_budget = projects_df['Budget (USD)'].sum()
total_spent = projects_df['Spent (USD)'].sum()
burn_rate = (total_spent / total_budget) * 100 if total_budget > 0 else 0
on_schedule_projects = projects_df[projects_df['Schedule Performance Index (SPI)'] >= 1.0].shape[0]
schedule_performance = (on_schedule_projects / len(projects_df)) * 100 if len(projects_df) > 0 else 0
ftr_rate = projects_df['First Time Right %'].mean()
at_risk_projects = projects_df[projects_df['Overall Status'] != 'On Track'].shape[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Portfolio Budget Burn Rate", f"{burn_rate:.1f}%", help=f"Total Spent: ${total_spent:,.0f} of ${total_budget:,.0f} total budget.")
col2.metric("Schedule Performance Index (SPI)", f"{projects_df['Schedule Performance Index (SPI)'].mean():.2f}", help="SPI > 1.0 is ahead of schedule. SPI < 1.0 is behind schedule. Calculated as EV/PV.")
col3.metric("PQ First Time Right (FTR) Rate", f"{ftr_rate:.1f}%", help="Average percentage of PQ batches that pass without deviation across all projects.")
col4.metric("Projects Requiring Intervention", f"{at_risk_projects}", delta=f"{at_risk_projects} At Risk", delta_color="inverse", help="Projects flagged with Amber/Red status for timeline, budget, or quality risks.")

st.divider()

# --- Main Content Area (ENHANCED VISUALIZATIONS) ---
st.header("Integrated Validation Portfolio Dashboard")
tab1, tab2, tab3 = st.tabs(["**Portfolio Timeline & Critical Path**", "**Resource Utilization & Allocation**", "**Integrated Risk Posture (ISO 14971)**"])

with tab1:
    st.caption("Visualize project velocity, identify critical path dependencies, and track progress against key manufacturing readiness dates.")
    fig = px.timeline(
        projects_df, x_start="Start Date", x_end="Due Date", y="Project/Assay", color="Type",
        title="Validation Project Timelines by Type", hover_name="Project/Assay",
        hover_data={"V&V Lead": True, "V&V Phase": True, "Key Milestone": True, "Overall Status": True}
    )
    # Add Critical Path overlay
    critical_path_df = projects_df[projects_df['On Critical Path']]
    for i, row in critical_path_df.iterrows():
        fig.add_shape(type="line", x0=row['Start Date'], y0=row['Project/Assay'], x1=row['Due Date'], y1=row['Project/Assay'],
                      line=dict(color="red", width=6, dash="solid"), name="Critical Path")
    fig.update_yaxes(categoryorder="total ascending", title=None)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.caption("Analyze project workload and utilization across the Validation team to proactively manage burnout risk and resource contention.")
    fig_resource = px.timeline(
        projects_df, x_start="Start Date", x_end="Due Date", y="V&V Lead", color="Utilization %",
        color_continuous_scale="RdYlGn_r", range_color=[50, 120],
        title="Project Allocation & Utilization by Validation Lead", hover_name="Project/Assay",
        hover_data={"V&V Phase": True, "Platform": True}
    )
    fig_resource.update_yaxes(categoryorder="total ascending", title="Validation Lead")
    st.plotly_chart(fig_resource, use_container_width=True)

with tab3:
    st.caption("Prioritize actions by visualizing product and project risks based on severity and probability, including risk distribution analysis.")
    fig_risk = px.scatter(
        risks_df, x="Probability", y="Severity", size="Risk_Score", color="Risk_Score",
        color_continuous_scale=px.colors.sequential.OrRd, hover_name="Risk Description",
        hover_data={"Project": True, "Owner": True, "Mitigation": True, "Risk_Score": True},
        marginal_x="histogram", marginal_y="histogram", title="Risk Heatmap with Distribution Analysis"
    )
    fig_risk.update_layout(
        xaxis=dict(tickvals=[1, 2, 3, 4, 5], ticktext=['Improbable', 'Remote', 'Occasional', 'Probable', 'Frequent'], title='Probability of Occurrence'),
        yaxis=dict(tickvals=[1, 2, 3, 4, 5], ticktext=['Negligible', 'Minor', 'Serious', 'Critical', 'Catastrophic'], title='Severity of Harm'),
        coloraxis_colorbar_title_text='Risk Score'
    )
    fig_risk.add_shape(type="rect", xref="x", yref="y", x0=3.5, y0=3.5, x1=5.5, y1=5.5, fillcolor="rgba(220, 53, 69, 0.2)", layer="below", line_width=0)
    fig_risk.add_annotation(x=4.5, y=4.5, text="Unacceptable Region", showarrow=False, font=dict(color="#DC3545", size=12, family="Arial, bold"))
    st.plotly_chart(fig_risk, use_container_width=True)

st.header("Validation Portfolio: Detailed Financial and Performance View")
st.dataframe(projects_df[[
    'Project/Assay', 'Type', 'V&V Lead', 'Overall Status', 'V&V Phase', 
    'Budget (USD)', 'Spent (USD)', 'Schedule Performance Index (SPI)', 'First Time Right %'
]], use_container_width=True, hide_index=True)

# --- REGULATORY CONTEXT (FULL VERSION) ---
st.divider()
with st.expander("🌐 Manager's Mandate: The 'Why' Behind This Dashboard", expanded=True):
    st.markdown("""
    As a Manager of Validation Engineering, my mandate is to ensure every piece of manufacturing equipment and its associated processes are robust, compliant, and ready to support production. This integrated Command Center provides the objective evidence needed to manage my team, collaborate with engineering partners, mitigate risks, and confidently attest to the validated state of our facility during regulatory audits.

    #### **How This Dashboard Demonstrates a State of Control:**

    - **Comprehensive Validation Planning (21 CFR 820.70, 820.75):**
        - The **Portfolio Gantt Chart** provides a unified view of all validation activities—from assays to the equipment that makes them. This allows for strategic planning, resource allocation, and ensures equipment validation timelines (FAT, SAT, IQ, OQ, PQ) align with production launch needs.

    - **Risk-Based Validation (ASTM E2500, GAMP 5):**
        - The **Integrated Risk Management Dashboard** visualizes risks from both product design (assays) and manufacturing processes (equipment). This allows me to ensure our validation effort is focused on the highest-risk areas, justifying the level of testing from IQ to PQ based on sound risk principles and focusing on Critical Process Parameters (CPPs).

    - **Cross-Functional Leadership & Financial Oversight:**
        - This dashboard is a critical tool for interfacing with Automation, Process Engineering, QA, and Manufacturing. The shared timeline and risk posture create a common language for discussing project dependencies and ensuring equipment URS and critical process parameters are defined with validation in mind from day one.
        - The **actionable financial and performance KPIs** (Burn Rate, SPI) provide the data needed to report on departmental targets and make informed decisions about resource allocation and project prioritization.
    """)
