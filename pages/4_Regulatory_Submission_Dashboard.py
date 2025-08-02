# pages/4_Regulatory_Submission_Dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import generate_vv_project_data, generate_submission_package_data

st.set_page_config(
    page_title="Regulatory Submission Dashboard | QuidelOrtho",
    layout="wide"
)

st.title("📦 Regulatory Submission Package Dashboard")
st.markdown("### Managing V&V Deliverables for 510(k) and PMA Submissions")

with st.expander("🌐 Director's View: Ensuring Submission Readiness"):
    st.markdown("""
    As the Associate Director, my ultimate responsibility is to ensure that when we submit a new or modified assay to a regulatory body, the V&V section of that submission is complete, robust, and defensible. A weak or incomplete V&V package can lead to costly delays, additional questions (AIs), or even rejection.

    This dashboard is my final "gate check." It provides a consolidated view of all required V&V evidence for a specific project, allowing me to confirm with my V&V Leads and cross-functional partners (Regulatory, QA) that every deliverable is complete and approved before the submission is finalized.

    **Key Responsibilities & Regulatory Imperatives:**
    - **510(k) and PMA Submissions:** This tool directly supports the compilation of these critical regulatory filings by tracking the completion status of all required V&V reports and DHF documents.
    - **FDA Design Control Requirements (21 CFR 820.30):** The checklist below represents the culmination of the entire design control process. Each "Approved" item signifies that the verification and validation activities for that part of the design have been successfully completed and documented in the DHF.
    - **Traceability:** This dashboard ensures that all planned V&V activities, as defined in the V&V Master Plan, have been executed and reported, providing a clear, traceable path for auditors.
    """)

# --- Project Selection ---
projects_df = generate_vv_project_data()
project_list = projects_df['Project/Assay'].tolist()
project_info = projects_df.set_index('Project/Assay')

selected_project = st.selectbox(
    "**Select a Project to View its Submission Package:**",
    options=project_list,
    index=0,
    help="Choose a project to see the status of its V&V deliverables for regulatory submission."
)

pathway = project_info.loc[selected_project, 'Regulatory Pathway']
st.info(f"**Project:** {selected_project} | **Regulatory Pathway:** {pathway}")

# --- Submission Package Data ---
submission_df = generate_submission_package_data(selected_project, pathway)

st.divider()

# --- Submission Readiness KPIs & Checklist ---
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Submission Readiness")
    # Calculate Readiness Score
    approved_progress = submission_df[submission_df['Status'] == 'Approved']['Progress'].sum()
    total_possible_progress = len(submission_df) * 100
    readiness_score = (approved_progress / total_possible_progress) * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=readiness_score,
        title={'text': f"V&V Package Readiness (%)<br><span style='font-size:0.8em;color:gray'>Based on 'Approved' status</span>"},
        gauge={
            'axis': {'range': [None, 100]},
            'steps': [
                {'range': [0, 50], 'color': "#DC3545"},
                {'range': [50, 90], 'color': "#FFC107"},
                {'range': [90, 100], 'color': "#28A745"}],
            'bar': {'color': "#0039A6"}
        }
    ))
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Summary of remaining items
    remaining_items_df = submission_df[submission_df['Status'] != 'Approved']
    st.subheader("Action Items (Not Approved)")
    if not remaining_items_df.empty:
        st.dataframe(
            remaining_items_df[['Deliverable', 'Status']],
            hide_index=True,
            use_container_width=True
        )
    else:
        st.success("All V&V deliverables are approved!")


with col2:
    st.header(f"V&V Deliverables Checklist for {pathway}")
    st.caption("Status of key V&V documents required for the submission's Design History File (DHF).")

    # Using st.data_editor to show progress bars
    st.data_editor(
        submission_df,
        column_config={
            "Progress": st.column_config.ProgressColumn(
                "Progress",
                format="%d%%",
                min_value=0,
                max_value=100,
            ),
            "Status": st.column_config.SelectboxColumn(
                "Status",
                options=["Drafting", "Execution", "In Review", "Data Analysis", "Approved"],
                required=True,
            )
        },
        hide_index=True,
        use_container_width=True,
        height=500
    )
