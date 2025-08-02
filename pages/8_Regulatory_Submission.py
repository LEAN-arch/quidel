# pages/8_Regulatory_Submission.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
from utils import generate_vv_project_data, generate_submission_package_data, generate_traceability_matrix_data

st.set_page_config(
    page_title="Regulatory Submission | QuidelOrtho",
    layout="wide"
)

st.title("📦 Regulatory Submission Dashboard")
st.markdown("### Final Gate Check of V&V Deliverables for 510(k) and PMA Submissions")

with st.expander("🌐 Manager's View: Ensuring Submission Integrity and Success", expanded=True):
    st.markdown("""
    A regulatory submission is the culmination of years of work. As the Manager, I am responsible for ensuring the V&V portion of that submission is irrefutable. A weak or incomplete V&V package is the primary cause of costly Additional Information (AI) requests from the FDA, significant project delays, and potential rejection.

    This dashboard is my final "go/no-go" checkpoint. It provides a consolidated, evidence-based view of all required V&V deliverables. I use this tool to formally attest that the V&V package is complete, all data has been verified, and all reports are approved before they are transmitted to Regulatory Affairs for inclusion in the final eCopy submission.

    **Key Responsibilities & Regulatory Imperatives:**
    - **Submission Integrity (21 CFR 807.87 for 510(k), 21 CFR 814.20 for PMA):** This dashboard is a direct control mechanism to ensure all required V&V data and summaries are present and accounted for, as mandated by these regulations.
    - **Design History File (DHF) as the Source of Truth:** Every item on this checklist must be a final, approved document within the DHF. This page serves as the auditable index of the V&V contributions to the DHF for this specific submission.
    - **Risk Mitigation for Timelines:** The "Submission Readiness" score is a critical leading indicator of potential delays. A score below 100% provides a clear, data-driven justification for not proceeding with a submission, preventing a premature filing.
    """)

# --- Project Selection ---
projects_df = generate_vv_project_data()
assay_project_list = projects_df[projects_df['Type'].isin(['Assay', 'Software'])]['Project/Assay'].tolist()
project_info = projects_df.set_index('Project/Assay')

selected_project = st.selectbox(
    "**Select an Assay or Software Project to Review its Submission Package:**",
    options=assay_project_list,
    index=0,
    help="Choose a project to see the status of its V&V deliverables for regulatory submission."
)

pathway = project_info.loc[selected_project, 'Regulatory Pathway']
st.info(f"**Project:** {selected_project} | **Target Submission:** {pathway}")
st.divider()

# --- Submission Readiness Section ---
tab1, tab2, tab3 = st.tabs(["**Deliverables Checklist & Readiness Score**", "**Traceability Flow Analysis**", "**Predictive Regulatory Strategy**"])

with tab1:
    st.header("V&V Deliverables Checklist & Readiness Score")
    submission_df = generate_submission_package_data(selected_project, pathway)
    
    col1, col2 = st.columns([1, 1.8])
    with col1:
        st.subheader("V&V Package Readiness")
        progress_values = submission_df['Progress']
        weights = [1.5 if "Master" in x or "Summary" in x else 1 for x in submission_df['Deliverable']]
        readiness_score = np.average(progress_values, weights=weights)

        if readiness_score == 100: gauge_color, status_text = "#28A745", "Ready for Submission"
        elif readiness_score >= 80: gauge_color, status_text = "#007BFF", "Final Review Stage"
        elif readiness_score >= 50: gauge_color, status_text = "#FFC107", "In Progress"
        else: gauge_color, status_text = "#DC3545", "Early Stage"

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=readiness_score,
            number={'suffix': '%', 'font': {'size': 40}},
            title={'text': f"Submission Readiness<br><span style='font-size:0.9em;color:{gauge_color}'>{status_text}</span>"},
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': gauge_color}}
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        remaining_items_df = submission_df[submission_df['Status'] != 'Approved'].sort_values('Progress')
        if not remaining_items_df.empty:
            st.warning("Critical Path Blockers:")
            st.dataframe(remaining_items_df[['Deliverable', 'Status']], hide_index=True, use_container_width=True)
        else:
            st.success("All V&V deliverables are approved!")

    with col2:
        st.subheader(f"V&V Deliverables Checklist for {pathway}")
        def get_status_icon(status):
            if status == "Approved": return "✅"
            if status == "In Review": return "🔄"
            if status == "Data Analysis": return "📊"
            if status == "Execution": return "🔬"
            return "📝"
        
        st.write('<div style="height: 450px; overflow-y: auto; padding-right: 10px;">', unsafe_allow_html=True)
        for index, row in submission_df.iterrows():
            icon = get_status_icon(row['Status'])
            st.markdown(f"""
            <div style="margin-bottom: 10px; border: 1px solid #DEE2E6; border-radius: 5px; padding: 15px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="font-size: 1.1em;">{row['Deliverable']}</strong><br>
                        <span style="font-size: 0.9em; color: #6C757D;">Doc ID: {row['Document ID']}</span>
                    </div>
                    <div style="text-align: right;"><span style="font-size: 1.1em; font-weight: bold;">{icon} {row['Status']}</span></div>
                </div>
                <div style="background-color: #E9ECEF; border-radius: 10px; margin-top: 10px;">
                    <div style="background-color: {'#28A745' if row['Progress'] == 100 else '#007BFF'}; width: {row['Progress']}%; height: 10px; border-radius: 10px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.write('</div>', unsafe_allow_html=True)

with tab2:
    st.header("V&V Traceability Flow Analysis")
    st.caption("Visualize the end-to-end flow from requirements to final V&V status, highlighting gaps and failures.")
    rtm_df = generate_traceability_matrix_data()

    all_nodes = list(pd.concat([rtm_df['Requirement Type'], rtm_df['Test Result']]).unique())
    source_indices = [all_nodes.index(req_type) for req_type in rtm_df['Requirement Type']]
    target_indices = [all_nodes.index(test_result) for test_result in rtm_df['Test Result']]
    
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=all_nodes,
                  color=["#0039A6", "#00AEEF", "#FFC72C", "#F47321", "#28A745", "#DC3545"]),
        link=dict(source=source_indices, target=target_indices, value=[1] * len(rtm_df),
                  color=[ 'rgba(220, 53, 69, 0.5)' if res == 'Fail' else 'rgba(40, 167, 69, 0.5)' for res in rtm_df['Test Result']])
    )])
    fig_sankey.update_layout(title_text="Requirement Traceability to V&V Outcome", font_size=12)
    st.plotly_chart(fig_sankey, use_container_width=True)
    with st.expander("**Manager's Analysis**"):
        st.markdown("""
        The Sankey diagram provides a powerful, intuitive visualization of our V&V coverage and outcomes. The **red link(s)** immediately draw attention to verification failures. This single chart elegantly demonstrates that every requirement type has a flow path to a V&V outcome, visually confirming the core principle of traceability for an audit.
        """)

with tab3:
    st.header("Predictive Regulatory Strategy: AI Query Forecaster")
    st.caption("Proactively identify areas in the V&V package that are most likely to receive questions from regulatory reviewers.")
    
    submission_df['Regulatory_Risk'] = submission_df['Regulatory_Impact'].map({'High': 3, 'Medium': 2, 'Low': 1}) * (10 - submission_df['Statistical_Robustness'])
    
    fig_heatmap = px.treemap(
        submission_df,
        path=[px.Constant("All Deliverables"), 'Regulatory_Impact', 'Deliverable'],
        values='Regulatory_Risk',
        color='Statistical_Robustness',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=np.average(submission_df['Statistical_Robustness'], weights=submission_df['Regulatory_Risk']),
        title='Regulatory Submission Risk Heatmap'
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    with st.expander("**Manager's Analysis**"):
        st.markdown("""
        This treemap is a proactive tool for audit and submission preparation. It helps me focus my team's attention on areas of potential weakness *before* we submit.
        - **Size of the Box:** Represents the overall "Regulatory Risk Score" (a combination of the deliverable's importance and the robustness of its data). Larger boxes demand more of my attention.
        - **Color of the Box:** Represents the "Statistical Robustness" of the underlying data (1-10 scale). Bright green means the data is overwhelmingly conclusive and unlikely to be questioned. **Yellow or red boxes** indicate results that, while passing, may be statistically borderline (e.g., p-value of 0.04, %CV just under the limit).
        - **Actionable Insight:** The large, yellowish box for the **Specificity Report** indicates it has a high regulatory impact and only moderately robust data. I will direct the V&V lead to prepare a formal memo preemptively addressing any potential reviewer questions about the borderline results in that study, strengthening our submission package.
        """)
