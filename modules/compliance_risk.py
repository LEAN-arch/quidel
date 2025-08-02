# modules/compliance_risk.py
import streamlit as st
import pandas as pd
from utils import helpers

def render_page():
    st.title("Compliance & Risk Hub")
    st.markdown("Centralized management of product risk, audit trails, and regulatory submission packages.")
    tab1, tab2, tab3 = st.tabs(["🛡️ Product Risk Management (FMEA)", "🔎 Audit Trail & Change Control", "🗂️ Regulatory Package Builder"])
    
    with tab1:
        st.subheader("Failure Modes and Effects Analysis (FMEA)")
        st.info("Manage product risks according to ISO 14971. High RPNs should have mitigation actions linked to a V&V protocol.")
        projects_list = ["All Projects"] + st.session_state.projects_df['Project'].tolist(); risk_project_filter = st.selectbox("Filter by Project", projects_list)
        risk_df = st.session_state.risk_df
        if risk_project_filter != "All Projects": risk_df = risk_df[risk_df['Project'] == risk_project_filter]
        edited_risk_df = st.data_editor(risk_df, column_config={"Risk_ID": st.column_config.TextColumn("Risk ID", disabled=True),"Failure_Mode": st.column_config.TextColumn("Failure Mode", width="large"),"Severity": st.column_config.NumberColumn("S", min_value=1, max_value=10),"Occurrence": st.column_config.NumberColumn("O", min_value=1, max_value=10),"Detection": st.column_config.NumberColumn("D", min_value=1, max_value=10),"RPN": st.column_config.ProgressColumn("RPN", help="Risk Priority Number (S x O x D)",format="%f",min_value=0,max_value=1000,disabled=True),"Mitigation_Action": st.column_config.TextColumn("Mitigation / V&V Link", width="medium"),"Linked_Protocol_ID": st.column_config.TextColumn("Protocol ID")}, hide_index=True, use_container_width=True, num_rows="dynamic")
        if st.button("Save Risk Assessment Changes"):
            edited_risk_df['RPN'] = edited_risk_df['Severity'] * edited_risk_df['Occurrence'] * edited_risk_df['Detection']; st.session_state.risk_df.update(edited_risk_df); helpers.log_action("director", "Updated FMEA Risk Assessment", f"Project Filter: {risk_project_filter}"); st.success("Risk assessment updated successfully!"); st.experimental_rerun()

    with tab2:
        st.subheader("System Audit Trail (21 CFR Part 11)")
        st.markdown("A secure, timestamped log of all critical actions performed within the system.")
        audit_log_df = pd.DataFrame(st.session_state.audit_log); col1, col2 = st.columns(2)
        with col1: user_filter = st.multiselect("Filter by User", options=audit_log_df['User'].unique(), default=audit_log_df['User'].unique())
        with col2: action_filter = st.text_input("Filter by Action contains...")
        filtered_log = audit_log_df[audit_log_df['User'].isin(user_filter)]
        if action_filter: filtered_log = filtered_log[filtered_log['Action'].str.contains(action_filter, case=False, na=False)]
        st.dataframe(filtered_log, use_container_width=True, hide_index=True)
        st.markdown("---"); st.subheader("ECO V&V Impact Assessment (Mock-up)"); st.info("This section would integrate with a Change Control system to track V&V activities required for post-launch design changes (ECOs).")
        eco_df = pd.DataFrame({'ECO Number': ['ECO-2024-034', 'ECO-2024-038'],'Change Description': ['Update consumable plastic material', 'Software patch for UI bug'],'Required V&V': ['Material Biocompatibility (re-verify)', 'Regression Testing Suite'],'V&V Status': ['In Progress', 'Completed']}); st.dataframe(eco_df, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Regulatory Submission Package Builder"); st.warning("DISCLAIMER: This is a tool to assist in gathering documentation and is not a substitute for a formal regulatory review.")
        col1, col2 = st.columns(2)
        with col1: submission_project = st.selectbox("Select Project for Submission", options=st.session_state.projects_df['Project'].unique())
        with col2: submission_type = st.selectbox("Select Submission Type", ["510(k)", "PMA", "CE Mark (IVDR)"])
        if submission_project:
            st.markdown("---"); st.subheader(f"Document Checklist for {submission_project} ({submission_type})")
            project_reqs = st.session_state.requirements_df[st.session_state.requirements_df['Project'] == submission_project]; project_protocols = st.session_state.protocols_df[st.session_state.protocols_df['Project'] == submission_project]; project_risks = st.session_state.risk_df[st.session_state.risk_df['Project'] == submission_project]
            st.checkbox("1. Final V&V Plan", value=True, disabled=True); st.checkbox(f"2. Final Risk Management File (FMEA)", value=not project_risks.empty, disabled=True); st.checkbox(f"3. Final Traceability Matrix", value=not project_reqs.empty, disabled=True)
            st.markdown("**V&V Summary Reports:**"); signed_off_protocols = project_protocols[project_protocols['Status'].str.contains("Executed", na=False)]
            if not signed_off_protocols.empty:
                for idx, row in signed_off_protocols.iterrows(): st.checkbox(f"  - {row['Protocol_ID']}: {row['Title']}", value=True, disabled=True, key=row['Protocol_ID'])
            else: st.markdown("  - *No signed-off reports found for this project.*")
            if st.button("Bundle & Download Documentation (Simulation)", type="primary"):
                st.success("Simulation Complete! In a real application, this would zip and download all relevant generated reports and data files."); helpers.log_action("director", "Simulated regulatory package bundle", f"Project: {submission_project}")
