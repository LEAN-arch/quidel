# pages/4_Regulatory_Submission_Dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import generate_vv_project_data, generate_submission_package_data

st.set_page_config(
    page_title="Submission Readiness | QuidelOrtho",
    layout="wide"
)

st.title("📦 V&V Regulatory Submission Dashboard")
st.markdown("### Final Gate Check of V&V Deliverables for 510(k) and PMA Submissions")

with st.expander("🌐 Director's View: Ensuring Submission Integrity and Success", expanded=True):
    st.markdown("""
    A regulatory submission is the culmination of years of work. My role as Associate Director is to ensure the V&V portion of that submission is irrefutable. A weak or incomplete V&V package is the primary cause of costly Additional Information (AI) requests from the FDA, significant project delays, and potential rejection.

    This dashboard is my final "go/no-go" checkpoint. It provides a consolidated, evidence-based view of all required V&V deliverables. I use this tool to formally attest that the V&V package is complete, all data has been verified, and all reports are approved before they are transmitted to Regulatory Affairs for inclusion in the final eCopy submission.

    **Key Responsibilities & Regulatory Imperatives:**
    - **Submission Integrity (21 CFR 807.87 for 510(k), 21 CFR 814.20 for PMA):** This dashboard is a direct control mechanism to ensure all required V&V data and summaries are present and accounted for, as mandated by these regulations.
    - **Design History File (DHF) as the Source of Truth:** Every item on this checklist must be a final, approved document within the DHF. This page serves as the auditable index of the V&V contributions to the DHF for this specific submission.
    - **Risk Mitigation for Timelines:** The "Submission Readiness" score is a critical leading indicator of potential delays. A score below 100% provides a clear, data-driven justification for not proceeding with a submission, preventing a premature filing.
    """)

# --- Project Selection ---
projects_df = generate_vv_project_data()
project_list = projects_df['Project/Assay'].tolist()
project_info = projects_df.set_index('Project/Assay')

selected_project = st.selectbox(
    "**Select a Project to Review its Submission Package:**",
    options=project_list,
    index=0,
    help="Choose a project to see the status of its V&V deliverables for regulatory submission."
)

pathway = project_info.loc[selected_project, 'Regulatory Pathway']
st.info(f"**Project:** {selected_project} | **Target Submission:** {pathway}")

# --- Submission Package Data ---
submission_df = generate_submission_package_data(selected_project, pathway)

st.divider()

# --- Submission Readiness KPIs & Checklist (ENHANCED) ---
col1, col2 = st.columns([1, 1.8])

with col1:
    st.header("V&V Package Readiness")
    # Calculate Readiness Score
    progress_values = submission_df['Progress']
    weights = [1.5 if "Master" in x or "Summary" in x else 1 for x in submission_df['Deliverable']] # Weight summary docs higher
    readiness_score = np.average(progress_values, weights=weights)

    # Determine Gauge Color and Text
    if readiness_score == 100:
        gauge_color = "#28A745"
        status_text = "Ready for Submission"
    elif readiness_score >= 80:
        gauge_color = "#007BFF"
        status_text = "Final Review Stage"
    elif readiness_score >= 50:
        gauge_color = "#FFC107"
        status_text = "In Progress"
    else:
        gauge_color = "#DC3545"
        status_text = "Early Stage"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=readiness_score,
        number={'suffix': '%', 'font': {'size': 40}},
        title={'text': f"Submission Readiness<br><span style='font-size:0.9em;color:{gauge_color}'>{status_text}</span>"},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': gauge_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#E9ECEF",
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Summary of remaining items
    st.subheader("Critical Path Blockers")
    remaining_items_df = submission_df[submission_df['Status'] != 'Approved'].sort_values('Progress')
    if not remaining_items_df.empty:
        st.warning("The following deliverables must be 'Approved' before submission can proceed.")
        st.dataframe(
            remaining_items_df[['Deliverable', 'Status']],
            hide_index=True,
            use_container_width=True
        )
    else:
        st.success("All V&V deliverables are approved! Package is ready for submission.")


with col2:
    st.header(f"V&V Deliverables Checklist for {pathway} Submission")
    st.caption("Status of key V&V documents from the DHF required for the submission.")

    def get_status_icon(status):
        if status == "Approved": return "✅"
        if status == "In Review": return "🔄"
        if status == "Data Analysis": return "📊"
        if status == "Execution": return "🔬"
        return "📝" # Drafting

    # Custom styled checklist
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
                <div style="text-align: right;">
                    <span style="font-size: 1.1em; font-weight: bold;">{icon} {row['Status']}</span>
                </div>
            </div>
            <div style="background-color: #E9ECEF; border-radius: 10px; margin-top: 10px;">
                <div style="background-color: {gauge_color if row['Progress'] == 100 else '#007BFF'}; width: {row['Progress']}%; height: 10px; border-radius: 10px; text-align: center; color: white; font-size: 0.8em; line-height: 10px;">
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.write('</div>', unsafe_allow_html=True)
