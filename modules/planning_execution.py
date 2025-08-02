# modules/planning_execution.py (Complete Enterprise Version)

import streamlit as st
import pandas as pd
from utils import helpers
from database import SessionLocal, Project, Protocol, User
from datetime import datetime

@helpers.role_required('engineer') # Minimum role 'engineer' to access this page
def render_page():
    st.title("V&V Lifecycle Management")
    st.markdown("A secure, database-driven workspace for V&V records.")
    
    db = SessionLocal()
    user = helpers.get_current_user()
    config = helpers.load_config()

    try:
        projects = db.query(Project).order_by(Project.name).all()
        project_names = [p.name for p in projects]
        
        selected_project_name = st.selectbox("Select a Project to Manage", project_names)

        if not selected_project_name:
            st.info("Please select a project.")
            st.stop()

        project = db.query(Project).filter(Project.name == selected_project_name).first()
        
        tab1, tab2, tab3 = st.tabs(["📝 Protocol Management", "📈 Data Execution & Analysis", "📄 Reporting & e-Signature"])

        with tab1:
            st.subheader(f"Protocols for {project.name}")
            
            # Fetch and display protocols
            protocols_query = db.query(Protocol).filter(Protocol.project_id == project.id).statement
            protocols_df = pd.read_sql(protocols_query, db.connection())
            st.dataframe(protocols_df[['protocol_id_str', 'title', 'status', 'creation_date', 'approval_date']], use_container_width=True)
            
            with st.expander("➕ Create New V&V Protocol"):
                with st.form("new_protocol_form"):
                    st.write("Define a new test protocol.")
                    new_protocol_title = st.text_input("Protocol Title")
                    new_protocol_type = st.selectbox("Protocol Type", config['protocol_types'])
                    new_acceptance_criteria = st.text_area("Acceptance Criteria")
                    
                    submitted = st.form_submit_button("Create Protocol Draft")
                    if submitted:
                        if new_protocol_title and new_acceptance_criteria:
                            new_protocol = Protocol(
                                protocol_id_str=f"{project.name.split('-')[0]}-{new_protocol_type[:4].upper()}-{db.query(Protocol).count() + 1}",
                                title=new_protocol_title,
                                project_id=project.id,
                                author_id=user.id,
                                status="Draft",
                                acceptance_criteria=new_acceptance_criteria
                            )
                            db.add(new_protocol); db.commit()
                            helpers.log_action(user.id, "Created Protocol Draft", details=f"Title: {new_protocol.title}", record_type="Protocol", record_id=new_protocol.id)
                            st.success(f"Protocol '{new_protocol.protocol_id_str}' created successfully in the database."); st.rerun()
                        else:
                            st.error("Please fill in all fields.")

        with tab2:
            st.subheader("Execute Protocol & Analyze Data")
            st.info("This section would contain the UI for uploading instrument data against a specific protocol record.")
            # Placeholder for data upload and analysis functionality
            
        with tab3:
            st.subheader("Reporting & Electronic Signature")
            protocol_list = db.query(Protocol).filter(Protocol.project_id == project.id, Protocol.status.in_(['Awaiting Approval', 'Executed - Passed'])).all()
            protocol_to_sign = st.selectbox("Select Protocol to Approve/Sign", [p.protocol_id_str for p in protocol_list])
            
            if protocol_to_sign:
                if st.button(f"Approve & Sign Protocol {protocol_to_sign}", type="primary"):
                    if user.role != 'director':
                        st.error("Only users with the 'Director' role can approve protocols.")
                    else:
                        protocol_record = db.query(Protocol).filter(Protocol.protocol_id_str == protocol_to_sign).first()
                        protocol_record.status = "Approved"
                        protocol_record.approver_id = user.id
                        protocol_record.approval_date = datetime.utcnow()
                        db.commit()
                        helpers.log_action(user.id, "Approved Protocol", details=f"Protocol: {protocol_to_sign}", record_type="Protocol", record_id=protocol_record.id)
                        st.success("Protocol approved and electronically signed.")
                        st.balloons(); st.rerun()
    finally:
        db.close()
