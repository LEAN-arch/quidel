# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import generate_vv_project_data, generate_risk_management_data
from datetime import date

# --- Page Configuration ---
st.set_page_config(
    page_title="V&V Command Center | QuidelOrtho",
    page_icon="⚕️",
    layout="wide"
)

# --- Data Loading ---
projects_df = generate_vv_project_data()
risks_df = generate_risk_management_data()

# --- Page Title and Header ---
st.title("⚕️ Assay V&V Command Center | QuidelOrtho")
st.markdown("### A Centralized Hub for V&V Portfolio Management, Risk Mitigation, and Regulatory Compliance")

# --- KPIs: Director-Level Oversight Metrics ---
st.header("Executive V&V Portfolio Health")
total_projects = len(projects_df)
active_vv_projects = projects_df[projects_df['V&V Phase'].isin(['Execution', 'Data Analysis'])].shape[0]
at_risk_projects = projects_df[projects_df['Overall Status'] != 'On Track'].shape[0]
high_impact_risks = risks_df[risks_df['Risk_Score'] >= 15].shape[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Active V&V Portfolio", f"{total_projects}", help="Total number of new and on-market assay projects with active V&V workstreams.")
col2.metric("Projects in Active V&V", f"{active_vv_projects}", help="Number of projects currently undergoing protocol execution or data analysis, consuming lab resources.")
col3.metric("Projects with Amber/Red Status", f"{at_risk_projects}", delta=f"{at_risk_projects} Require Attention", delta_color="inverse", help="Projects flagged for potential timeline, budget, or technical risks requiring management intervention.")
col4.metric("Residual Risks in 'Action Required' Zone", f"{high_impact_risks}", help="Count of risks in the unacceptable region of the risk matrix, requiring immediate mitigation or risk-benefit analysis per ISO 14971.")

st.divider()

# --- Main Content Area (ENHANCED VISUALIZATIONS) ---
st.header("V&V Portfolio Dashboard: Timelines, Resources, & Risk Posture")
tab1, tab2, tab3 = st.tabs(["**Portfolio Timeline by Phase**", "**Resource Allocation by V&V Lead**", "**Integrated Risk Posture (ISO 14971)**"])

with tab1:
    st.caption("Visualize project velocity, identify potential phase-gate bottlenecks, and track progress against key regulatory submission dates.")
    fig = px.timeline(
        projects_df,
        x_start="Start Date",
        x_end="Due Date",
        y="Project/Assay",
        color="V&V Phase",
        title="V&V Project Timelines by Phase",
        hover_name="Project/Assay",
        hover_data={"V&V Lead": True, "Overall Status": True, "Platform": True, "Key Milestone": True}
    )
    for index, row in projects_df.iterrows():
        if pd.notna(row['Milestone Date']):
            fig.add_trace(go.Scatter(
                x=[row['Milestone Date']], y=[row['Project/Assay']],
                mode='markers', marker_symbol='diamond', marker_size=12, marker_color='red',
                name='Key Milestone', hovertemplate=f"Milestone: {row['Key Milestone']}<br>Date: {row['Milestone Date'].strftime('%Y-%m-%d')}<extra></extra>",
                showlegend=False
            ))
    fig.update_yaxes(categoryorder="total ascending", title=None)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.caption("Analyze project workload distribution across the V&V team to identify and mitigate resource contention risks proactively.")
    fig_resource = px.timeline(
        projects_df,
        x_start="Start Date",
        x_end="Due Date",
        y="V&V Lead",
        color="Project/Assay",
        title="Project Allocation by V&V Lead",
        hover_name="Project/Assay",
        hover_data={"V&V Phase": True, "Overall Status": True, "Platform": True}
    )
    fig_resource.update_yaxes(categoryorder="total ascending", title="V&V Lead")
    st.plotly_chart(fig_resource, use_container_width=True)
    with st.expander("**Director's Analysis**"):
        st.markdown("""
        This view is critical for my resource management responsibilities. It immediately highlights potential over-allocation. For example, if **M. Rodriguez** has multiple large projects running concurrently, it represents a significant risk to all of those timelines. This visualization provides the data needed to justify re-assigning projects, delaying non-critical start dates, or hiring additional staff.
        """)

with tab3:
    st.caption("Prioritize actions by visualizing product and project risks based on severity and probability, including risk distribution analysis.")
    fig_risk = px.scatter(
        risks_df, x="Probability", y="Severity", size="Risk_Score", color="Risk_Score",
        color_continuous_scale=px.colors.sequential.OrRd,
        hover_name="Risk Description",
        hover_data={"Project": True, "Owner": True, "Mitigation": True, "Risk_Score": True},
        marginal_x="histogram", marginal_y="histogram",
        title="Risk Heatmap with Distribution Analysis"
    )
    fig_risk.update_layout(
        xaxis=dict(tickvals=[1, 2, 3, 4, 5], ticktext=['Improbable', 'Remote', 'Occasional', 'Probable', 'Frequent'], title='Probability of Occurrence'),
        yaxis=dict(tickvals=[1, 2, 3, 4, 5], ticktext=['Negligible', 'Minor', 'Serious', 'Critical', 'Catastrophic'], title='Severity of Harm'),
        coloraxis_colorbar_title_text='Risk Score'
    )
    fig_risk.add_shape(type="rect", xref="x", yref="y", x0=3.5, y0=3.5, x1=5.5, y1=5.5, fillcolor="rgba(220, 53, 69, 0.2)", layer="below", line_width=0)
    fig_risk.add_annotation(x=4.5, y=4.5, text="Unacceptable Region", showarrow=False, font=dict(color="#DC3545", size=12, family="Arial, bold"))
    st.plotly_chart(fig_risk, use_container_width=True)
    with st.expander("**Director's Analysis**"):
        st.markdown("""
        This enhanced risk plot provides two layers of insight. The central heatmap identifies the highest-scoring individual risks that require immediate action. The **marginal histograms** on the axes reveal systemic trends. For example, a large peak in the top histogram at 'Probable' (4) indicates a systemic issue with recurring, high-probability risks across the portfolio, even if their severity varies. This might signal a need to improve our upstream R&D processes to design out risk earlier.
        """)

st.header("V&V Portfolio: Detailed View & Status")
st.dataframe(projects_df, use_container_width=True, hide_index=True)

# --- REGULATORY CONTEXT (EXTENDED) ---
st.divider()
with st.expander("🌐 Director's Mandate: The 'Why' Behind This Dashboard", expanded=True):
    st.markdown("""
    As the Associate Director of Assay V&V, my mandate is to ensure every assay and consumable we release is safe, effective, and compliant with global regulations. This Command Center is the embodiment of that mandate, providing the objective evidence needed to manage my team, mitigate risks, and confidently attest to the integrity of our V&V processes during regulatory audits. Each element on this page serves a distinct compliance and management purpose.

    #### **How This Dashboard Demonstrates a State of Control:**

    - **V&V Planning & Strategy (ISO 13485:2016, 7.3.5 & 7.3.6):**
        - The **Portfolio Gantt Chart** is my primary tool for Design and Development Planning. It provides a dynamic, forward-looking view of all V&V workstreams, enabling me to allocate personnel effectively, anticipate resource constraints, and ensure that our activities align with critical business and regulatory submission deadlines. This directly supports the creation of the overarching **V&V Master Plan (VVMP)** for each project.

    - **Risk-Based V&V (ISO 14971:2019 & 21 CFR 820.30(g)):**
        - The **Integrated Risk Management Dashboard** is our live Risk Management File summary. It is a direct output of our risk analysis process (e.g., FMEAs), where we identify potential hazards, estimate and evaluate the associated risks, and control these risks. The 'Unacceptable Region' immediately flags where my team's mitigation efforts and V&V testing depth must be focused to reduce risk "As Low As Reasonably Practicable" (ALARP).

    - **Executive Oversight & Management by Exception:**
        - The **Executive KPIs** distill the complex status of the entire portfolio into actionable insights. They allow me to immediately identify projects that are off-track or risks that have breached our action thresholds, enabling me to manage by exception and focus my attention where it is most needed.

    - **Regulatory Strategy Alignment (FDA 510(k), PMA & EU IVDR):**
        - This dashboard is designed from the ground up to be "audit-ready." It serves as a master index for our V&V-related Design History File (DHF) contributions and ensures alignment of V&V evidence with the specific requirements of different regulatory pathways, including the General Safety and Performance Requirements (GSPRs) of IVDR.
    """)
