# modules/planning_execution.py

import streamlit as st
import pandas as pd
from utils import helpers
from datetime import datetime

def render_page():
    """Renders the V&V lifecycle management page."""
    st.title("V&V Lifecycle Management")
    st.markdown("Manage the end-to-end process for a V&V project, from requirements to final report.")

    # --- Project Selector ---
    projects_list = ["<Select a Project>"] + st.session_state.projects_df['Project'].tolist()
    selected_project = st.selectbox("Select a Project to Manage", projects_list)

    if selected_project == "<Select a Project>":
        st.info("Please select a project from the dropdown above to begin.")
        st.stop()
    
    # Filter all data related to the selected project
    project_reqs_df = st.session_state.requirements_df[st.session_state.requirements_df['Project'] == selected_project]
    project_protocols_df = st.session_state.protocols_df[st.session_state.protocols_df['Project'] == selected_project]

    # --- Main Tabbed Interface ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Requirements & Traceability", 
        "📝 V&V Planning & Protocols", 
        "📈 Data Execution & Analysis", 
        "📄 Reporting & e-Signature"
    ])

    # --- TAB 1: REQUIREMENTS & TRACEABILITY ---
    with tab1:
        st.subheader(f"Traceability Matrix for {selected_project}")
        
        st.info("Link requirements to specific test protocols. Gaps indicate untested requirements.")
        
        # Display an editable traceability matrix
        edited_reqs_df = st.data_editor(
            project_reqs_df,
            column_config={
                "Req_ID": st.column_config.TextColumn("Req ID", disabled=True),
                "Requirement_Text": st.column_config.TextColumn("Requirement", width="large"),
                "Linked_Protocol_ID": st.column_config.TextColumn("Linked Protocol(s)"),
                "Status": st.column_config.SelectboxColumn("Status", options=['Covered', 'Gap', 'In Progress'])
            },
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic"
        )
        
        if st.button("Save Traceability Changes"):
            # In a real app, this would write back to the database. Here we update session state.
            # This is a simplified update logic. A real app would need to merge based on Req_ID.
            st.session_state.requirements_df.update(edited_reqs_df)
            helpers.log_action("director", "Updated traceability matrix", f"Project: {selected_project}")
            st.success("Traceability matrix updated successfully!")
            st.experimental_rerun()

    # --- TAB 2: V&V PLANNING & PROTOCOLS ---
    with tab2:
        st.subheader(f"Protocols for {selected_project}")
        st.dataframe(project_protocols_df, use_container_width=True, hide_index=True)

        with st.expander("➕ Create New V&V Protocol"):
            with st.form("new_protocol_form"):
                st.write("Define a new test protocol.")
                new_protocol_id = st.text_input("Protocol ID (e.g., IP-NEW-01)")
                new_protocol_title = st.text_input("Protocol Title")
                new_protocol_type = st.selectbox("Protocol Type", ['Precision', 'Linearity', 'Sensitivity', 'Specificity', 'Performance', 'Other'])
                new_acceptance_criteria = st.text_area("Acceptance Criteria")

                submitted = st.form_submit_button("Create Protocol Draft")
                if submitted:
                    if new_protocol_id and new_protocol_title and new_acceptance_criteria:
                        # Add new protocol to the main dataframe
                        new_row = pd.DataFrame([{
                            'Protocol_ID': new_protocol_id,
                            'Project': selected_project,
                            'Title': new_protocol_title,
                            'Type': new_protocol_type,
                            'Status': 'Draft',
                            'Creation_Date': datetime.now(),
                            'Approval_Date': pd.NaT,
                            'Failure_Reason': None,
                            'Acceptance_Criteria': new_acceptance_criteria
                        }])
                        st.session_state.protocols_df = pd.concat([st.session_state.protocols_df, new_row], ignore_index=True)
                        helpers.log_action("director", "Created new protocol draft", f"ID: {new_protocol_id}")
                        st.success(f"Protocol '{new_protocol_id}' created as a draft.")
                        st.experimental_rerun()
                    else:
                        st.error("Please fill in all fields.")

    # --- TAB 3: DATA EXECUTION & ANALYSIS ---
    with tab3:
        st.subheader("Execute Protocol & Analyze Data")
        
        # Select a protocol to execute
        protocol_list = project_protocols_df['Protocol_ID'].tolist()
        selected_protocol_id = st.selectbox("Select Protocol to Execute", protocol_list)
        
        if selected_protocol_id:
            protocol_details = project_protocols_df[project_protocols_df['Protocol_ID'] == selected_protocol_id].iloc[0]
            st.info(f"**Executing:** {protocol_details['Title']}\n\n**Acceptance Criteria:** {protocol_details['Acceptance_Criteria']}")

            uploaded_file = st.file_uploader("Upload Raw Data File (CSV)", type=['csv'])

            if uploaded_file is not None:
                try:
                    data_df = pd.read_csv(uploaded_file)
                    st.write("Uploaded Data Preview:")
                    st.dataframe(data_df.head())

                    analysis_type = protocol_details['Type']
                    results = None
                    fig = None

                    if analysis_type == 'Precision':
                        results, fig = helpers.analyze_precision(data_df)
                    elif analysis_type == 'Linearity':
                        results, fig = helpers.analyze_linearity(data_df)
                    else:
                        st.warning(f"Automated analysis for '{analysis_type}' is not yet implemented. Please perform manual analysis.")

                    if results and fig:
                        st.subheader("Analysis Results")
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.dataframe(pd.DataFrame([results]))
                        with col2:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Store results in session for reporting tab
                        st.session_state.last_analysis = {
                            'protocol_data': protocol_details.to_dict(),
                            'analysis_results': results,
                            'analysis_fig': fig
                        }
                        
                        st.success("Analysis complete. Proceed to the Reporting tab to generate a summary.")

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                    st.error("Please ensure the CSV format is correct for the selected analysis type (e.g., has 'Value' column for Precision).")

    # --- TAB 4: REPORTING & e-SIGNATURE ---
    with tab4:
        st.subheader("Generate Report & Finalize")
        
        if 'last_analysis' not in st.session_state or st.session_state.last_analysis['protocol_data']['Project'] != selected_project:
            st.warning("Please run an analysis in the 'Data Execution & Analysis' tab first.")
        else:
            last_analysis = st.session_state.last_analysis
            protocol_id = last_analysis['protocol_data']['Protocol_ID']
            st.info(f"Ready to generate a report for **{protocol_id}**.")

            col1, col2 = st.columns(2)
            with col1:
                final_status = st.radio("Set Final Protocol Status", ('Executed - Passed', 'Executed - Failed', 'Deviation'), horizontal=True)
            
            with col2:
                # e-Signature (21 CFR Part 11 mock)
                sign_off_name = st.text_input("Enter Full Name to Sign Off", value="Assay Director")
            
            if st.button("Generate & Sign Report", type="primary"):
                with st.spinner("Generating report..."):
                    # Generate PPT
                    ppt_buffer = helpers.generate_ppt_report(
                        last_analysis['protocol_data'],
                        last_analysis['analysis_results'],
                        last_analysis['analysis_fig']
                    )
                    
                    # Log the action
                    helpers.log_action("director", f"Generated & Signed Report: {protocol_id}", f"Status: {final_status}")
                    
                    # Update protocol status in session state
                    idx = st.session_state.protocols_df.index[st.session_state.protocols_df['Protocol_ID'] == protocol_id].tolist()[0]
                    st.session_state.protocols_df.at[idx, 'Status'] = final_status
                    st.session_state.protocols_df.at[idx, 'Signed_Off_By'] = sign_off_name
                    # In a real app, this date would be the execution date. We'll use now().
                    st.session_state.protocols_df.at[idx, 'Approval_Date'] = datetime.now()
                    
                    st.success(f"Report for {protocol_id} has been generated and signed!")
                    
                    # Provide download link
                    st.download_button(
                        label="Download PowerPoint Report",
                        data=ppt_buffer,
                        file_name=f"{protocol_id}_Summary_Report.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    )
                    # Clear the temp analysis data
                    del st.session_state.last_analysis
