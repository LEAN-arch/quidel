# modules/dashboard.py

import streamlit as st
import pandas as pd
from utils import helpers

def render_page():
    """Renders the executive dashboard page."""
    
    st.title("Executive Dashboard")
    st.markdown("A real-time overview of the V&V department's projects, resources, and compliance status.")
    st.markdown("---")

    # --- TOP-LEVEL KPI METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate KPIs from session state data
    projects_df = st.session_state.projects_df
    protocols_df = st.session_state.protocols_df
    risk_df = st.session_state.risk_df
    
    total_projects = len(projects_df)
    at_risk_projects = len(projects_df[projects_df['Status'] != 'On Track'])
    awaiting_approval = len(protocols_df[protocols_df['Status'] == 'Approved']) # "Approved" means ready for execution
    high_rpn_risks = len(risk_df[risk_df['RPN'] > 100])

    with col1:
        st.metric(label="Total Active Projects", value=total_projects)
    with col2:
        st.metric(label="Projects At Risk / Delayed", value=at_risk_projects)
    with col3:
        st.metric(label="Protocols Awaiting Execution", value=awaiting_approval)
    with col4:
        st.metric(label="Open High-Impact Risks", value=high_rpn_risks)

    st.markdown("---")

    # --- MAIN VISUALIZATION AREA ---
    col1, col2 = st.columns([3, 2]) # Gantt chart is wider

    with col1:
        st.subheader("Project Portfolio Timeline")
        # Rename columns for the Gantt chart function's specific needs
        gantt_data = projects_df.copy()
        gantt_data.rename(columns={'Project': 'Task'}, inplace=True)
        fig_gantt = helpers.create_gantt_chart(gantt_data)
        st.plotly_chart(fig_gantt, use_container_width=True)

    with col2:
        st.subheader("Live Alert & Activity Feed")
        
        # Create a dynamic feed of important items
        alerts = []
        # Alert for delayed projects
        delayed = projects_df[projects_df['Status'] == 'Delayed']['Project'].tolist()
        if delayed:
            alerts.append(f"🔴 **Delayed Project(s):** {', '.join(delayed)}")
        
        # Alert for overdue training
        overdue_training = st.session_state.team_df[st.session_state.team_df['Training_Status'] == 'Overdue']['Member'].tolist()
        if overdue_training:
            alerts.append(f"🟡 **Training Overdue:** {', '.join(overdue_training)}")

        # Alert for failed protocols
        failed_protocols = protocols_df[protocols_df['Status'] == 'Executed - Failed']['Protocol_ID'].tolist()
        if failed_protocols:
            alerts.append(f"🔴 **Execution Failed:** {', '.join(failed_protocols)}")
            
        # Alert for unmitigated high risks
        unmitigated_risks = risk_df[(risk_df['RPN'] > 100) & (risk_df['Linked_Protocol_ID'] == 'N/A')]['Risk_ID'].tolist()
        if unmitigated_risks:
            alerts.append(f"🟠 **Unmitigated High Risk:** {', '.join(unmitigated_risks)}")

        # Display alerts in an expander
        with st.expander("View All Alerts", expanded=True):
            if alerts:
                for alert in alerts:
                    st.markdown(f"- {alert}")
            else:
                st.success("✅ All systems normal. No critical alerts.")
        
        st.subheader("Recent Audit Log")
        audit_log_df = pd.DataFrame(st.session_state.audit_log)
        st.dataframe(audit_log_df.head(5), use_container_width=True, hide_index=True)


    st.markdown("---")
    
    # --- RESOURCE & COMPLIANCE SECTION ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Team Workload Distribution")
        team_df = st.session_state.team_df
        fig_workload = helpers.create_bar_chart(team_df, 'Member', 'Active_Protocols', 'Active Protocols per Team Member')
        st.plotly_chart(fig_workload, use_container_width=True)

    with col2:
        st.subheader("Compliance Health Status")
        
        # Calculate compliance metrics
        total_team = len(st.session_state.team_df)
        compliant_team = len(st.session_state.team_df[st.session_state.team_df['Training_Status'] == 'Compliant'])
        training_compliance_percent = int((compliant_team / total_team) * 100)
        
        total_reqs = len(st.session_state.requirements_df)
        covered_reqs = len(st.session_state.requirements_df[st.session_state.requirements_df['Status'] == 'Covered'])
        traceability_percent = int((covered_reqs / total_reqs) * 100)
        
        c1, c2 = st.columns(2)
        with c1:
            fig_donut_train = helpers.create_donut_chart([training_compliance_percent, 100-training_compliance_percent], 
                                                         ['Compliant', 'Overdue'], 'Training Compliance')
            st.plotly_chart(fig_donut_train, use_container_width=True)
        with c2:
            fig_donut_trace = helpers.create_donut_chart([traceability_percent, 100-traceability_percent], 
                                                         ['Covered', 'Gap'], 'Requirements Traceability')
            st.plotly_chart(fig_donut_trace, use_container_width=True)
