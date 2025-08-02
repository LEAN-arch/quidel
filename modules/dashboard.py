# modules/dashboard.py (Simplified Call)

import streamlit as st
from utils import helpers
import pandas as pd
from datetime import datetime
from prophet.plot import plot_plotly
from database import SessionLocal, Project, Protocol, User
import numpy as np

def render_page():
    st.title("V&V Command Center Dashboard")
    st.markdown("Diagnostic overview of portfolio health, process efficiency, quality, and risk. *Designed for action.*")

    db = SessionLocal()
    try:
        projects_query = db.query(Project).statement
        projects_df = pd.read_sql(projects_query, db.connection())
        if not projects_df.empty:
            projects_df['Pct_Complete'] = np.random.randint(20, 95, projects_df.shape[0])
        else:
            projects_df['Pct_Complete'] = []

        protocols_query = db.query(Protocol).statement
        protocols_df = pd.read_sql(protocols_query, db.connection())

        users_query = db.query(User).statement
        team_df = pd.read_sql(users_query, db.connection())
        if not team_df.empty:
            team_df['Capacity (hrs/wk)'] = 40
            team_df['Assigned_Hrs'] = np.random.randint(30, 50, team_df.shape[0])
        else:
            team_df['Capacity (hrs/wk)'] = []
            team_df['Assigned_Hrs'] = []

        risk_df = pd.DataFrame()
        if not projects_df.empty:
            risk_df = pd.DataFrame({
                'Project': projects_df['name'].tolist() * 2,
                'Severity': np.random.randint(3, 10, len(projects_df)*2),
                'Occurrence': np.random.randint(1, 8, len(projects_df)*2),
                'RPN': np.random.randint(50, 200, len(projects_df)*2),
                'Failure_Mode': [f'Mode {i}' for i in range(len(projects_df)*2)]
            })

        tab1, tab2 = st.tabs(["📊 Main Dashboard", "🔮 Project Timeline Forecasting"])

        with tab1:
            st.subheader("Department Vital Signs")
            st.metric("Total Active Projects", len(projects_df))
            st.metric("Total Protocols Logged", len(protocols_df))
            st.metric("Team Headcount", len(team_df))
            
            st.markdown("---")
            st.subheader("Portfolio & Risk Landscape")
            p_col1, p_col2 = st.columns([3, 2])
            with p_col1:
                st.info("What's the status and progress of our key projects?")
                # THE FIX IS HERE: We no longer need to do any renaming.
                # The helper function now handles it.
                fig_gantt = helpers.create_enhanced_gantt(projects_df)
                st.plotly_chart(fig_gantt, use_container_width=True)
            with p_col2:
                st.info("Where are our biggest product risks hiding?")
                fig_risk = helpers.create_risk_bubble_chart(risk_df)
                st.plotly_chart(fig_risk, use_container_width=True)
        
        # ... (rest of the file is unchanged) ...
        with tab2:
            st.subheader("Predictive Project Completion Forecasting")
            st.info("This tool uses historical progress to forecast a realistic completion date, identifying potential delays before they become critical.")
            
            if not projects_df.empty:
                project_to_forecast = st.selectbox("Select a Project to Forecast", options=projects_df['name'], help="Choose a project to see its forecasted completion date.")

                if project_to_forecast:
                    with st.spinner(f"Generating forecast for {project_to_forecast}..."):
                        project_data = projects_df[projects_df['name'] == project_to_forecast].iloc[0]
                        project_dict = {'Start': project_data['start_date'], 'Pct_Complete': project_data['Pct_Complete']}
                        history_df = helpers.generate_prophet_history(project_dict)

                        if history_df.empty:
                            st.warning("Cannot forecast a project that has not started yet.")
                        else:
                            model, forecast = helpers.run_prophet_forecast(history_df)
                            if model and forecast is not None:
                                st.success("Forecast generated successfully.")
                                fig = plot_plotly(model, forecast)
                                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No projects available to forecast.")

    finally:
        db.close()
