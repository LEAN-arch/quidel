# modules/planning_execution.py (Final Guaranteed Version)

import streamlit as st
import pandas as pd
from utils import helpers
from database import SessionLocal, Project, Protocol
from datetime import datetime

@helpers.role_required('engineer')
def render_page():
    st.title("V&V Lifecycle Management")
    st.markdown("A secure, database-driven workspace for V&V records.")
    
    db = SessionLocal()
    user = helpers.get_current_user()
    config = helpers.load_config()

    try:
        projects = db.query(Project).order_by(Project.name).all()
        selected_project_name = st.selectbox("Select a Project to Manage", [p.name for p in projects])
        if not selected_project_name: st.info("Please select a project."); st.stop()

        project = db.query(Project).filter(Project.name == selected_project_name).first()
        tab1, tab2, tab3 = st.tabs(["📝 Protocol Management", "📈 Data Execution & Analysis", "📄 Reporting & e-Signature"])

        with tab1:
            st.subheader(f"Protocols for {project.name}")
            protocols_df = pd.read_sql(db.query(Protocol).filter(Protocol.project_id == project.id).statement, db.connection())
            st.dataframe(protocols_df[['protocol_id_str', 'title', 'status', 'creation_date']], use_container_width=True)
            
            with st.expander("➕ Create New V&V Protocol"):
                with st.form("new_protocol_form"):
                    new_title, new_type, new_criteria = st.text_input("Protocol Title"), st.selectbox("Protocol Type", config['protocol_types']), st.text_area("Acceptance Criteria")
                    if st.form_submit_button("Create Protocol Draft"):
                        if new_title and new_criteria:
                            new_proto = Protocol(protocol_id_str=f"{project.name.split('-')[0]}-{new_type[:4].upper()}-{db.query(Protocol).count()+1}", title=new_title, project_id=project.id, author_id=user.id, status="Draft", acceptance_criteria=new_criteria)
                            db.add(new_proto); db.commit()
                            helpers.log_action(user.id, "Created Protocol Draft", details=f"Title: {new_proto.title}", record_type="Protocol", record_id=new_proto.id)
                            st.success(f"Protocol '{new_proto.protocol_id_str}' created."); st.rerun()
                        else: st.error("Please fill in all fields.")

        with tab2:
            st.subheader("Execute Protocol & Analyze Data")
            # In a real app, this would be fully built out.
            st.info("This section is a placeholder for instrument data upload and analysis against a protocol record.")

        with tab3:
            st.subheader("Reporting & Electronic Signature")
            
            # Use a dictionary for easier lookup
            protocol_options = {p.protocol_id_str: p for p in db.query(Protocol).filter(Protocol.project_id == project.id).all()}
            protocol_to_sign_id = st.selectbox("Select Protocol", protocol_options.keys())

            if protocol_to_sign_id:
                protocol_record = protocol_options[protocol_to_sign_id]
                
                # Logic for Approval
                if user.role == 'director' and protocol_record.status == 'Awaiting Approval':
                    if st.button(f"Approve & Sign Protocol {protocol_record.protocol_id_str}", type="primary"):
                        protocol_record.status = "Approved"; protocol_record.approver_id = user.id; protocol_record.approval_date = datetime.utcnow()
                        db.commit(); helpers.log_action(user.id, "Approved Protocol", record_type="Protocol", record_id=protocol_record.id)
                        st.success("Protocol approved and electronically signed."); st.balloons(); st.rerun()

                # Logic for generating a text report after execution
                if protocol_record.status in ['Executed - Passed', 'Executed - Failed']:
                    st.info(f"Protocol **{protocol_record.protocol_id_str}** has been executed. You can download its summary report.")
                    
                    protocol_dict = protocol_record.__dict__
                    protocol_dict['signed_by'] = user.full_name
                    
                    # Mock results for the text report
                    mock_results = {'Status': protocol_dict['status'], 'Note': 'Analysis data would be populated here.'}

                    report_bytes = helpers.generate_text_report(protocol_dict, mock_results)

                    st.download_button(
                        label=f"Download Summary Report (.txt) for {protocol_record.protocol_id_str}",
                        data=report_bytes,
                        file_name=f"{protocol_record.protocol_id_str}_Summary_Report.txt",
                        mime="text/plain"
                    )

    finally: db.close()
