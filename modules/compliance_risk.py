# modules/compliance_risk.py (Complete Enterprise Version)

import streamlit as st
import pandas as pd
from utils import helpers
from database import SessionLocal, AuditLog, User, Project, Protocol

@helpers.role_required('director') # This page is locked down to directors
def render_page():
    st.title("Compliance & Risk Hub")
    st.markdown("A secure, director-level portal for governance, risk, and compliance oversight.")
    
    db = SessionLocal()
    try:
        tab1, tab2, tab3 = st.tabs(["🔎 System Audit Trail", "🛡️ Risk Management (Placeholder)", "🗂️ Regulatory Package Builder"])

        with tab1:
            st.subheader("System Audit Trail (21 CFR Part 11 Compliant)")
            st.markdown("A secure, timestamped log of all critical actions performed within the system.")
            
            # Query the audit log and join with user table to get names
            audit_trail_query = db.query(AuditLog, User.full_name).join(User, AuditLog.user_id == User.id).order_by(AuditLog.timestamp.desc()).limit(100).all()
            
            if audit_trail_query:
                audit_data = [
                    {
                        "Timestamp (UTC)": log.timestamp.strftime('%Y-%m-%d %H:%M:%S'), 
                        "User": name, 
                        "Action": log.action, 
                        "Record Type": log.record_type,
                        "Record ID": log.record_id,
                        "Details": log.details
                    }
                    for log, name in audit_trail_query
                ]
                audit_df = pd.DataFrame(audit_data)
                st.dataframe(audit_df, use_container_width=True, hide_index=True)
            else:
                st.info("No audit records found.")
        
        with tab2:
            st.subheader("Product Risk Management (FMEA)")
            st.info("This section would be built out with a dedicated 'Risk' table in the database, similar to the Protocol Management module.")
            # Placeholder for FMEA table and logic

        with tab3:
            st.subheader("Regulatory Submission Package Builder")
            st.info("Select a project to generate a downloadable zip archive of its submission documents.")
            projects = db.query(Project).order_by(Project.name).all()
            submission_project_name = st.selectbox("Select Project for Submission", [p.name for p in projects])
            
            if st.button(f"🚀 Generate & Download Package for {submission_project_name}", type="primary"):
                with st.spinner("Bundling submission package... Please wait."):
                    project = db.query(Project).filter(Project.name == submission_project_name).first()
                    
                    # In a real app, you would query all necessary tables (Requirements, Risks, etc.)
                    # Here we just pass what we have
                    protocols_df = pd.read_sql(db.query(Protocol).filter(Protocol.project_id == project.id).statement, db.connection())
                    
                    zip_buffer = helpers.create_submission_zip(
                        project.name,
                        pd.DataFrame(), # Placeholder for requirements df
                        protocols_df,
                        pd.DataFrame()  # Placeholder for risk df
                    )
                    
                    user = helpers.get_current_user()
                    helpers.log_action(user.id, "Generated Regulatory Package", f"Project: {project.name}")
                    
                    st.download_button(
                        label="✅ Download Ready! Click Here.",
                        data=zip_buffer,
                        file_name=f"{project.name}_Regulatory_Package.zip",
                        mime="application/zip",
                    )
    finally:
        db.close()
