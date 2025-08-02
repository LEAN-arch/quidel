# modules/dashboard.py (UPDATED with Prophet Forecasting Tab)

import streamlit as st
from utils import helpers
import pandas as pd
from datetime import datetime

def render_page():
    """Renders the enhanced dashboard with a new forecasting tab."""
    
    st.title("V&V Command Center Dashboard")
    st.markdown("Diagnostic overview of portfolio health, process efficiency, quality, and risk. *Designed for action.*")

    tab1, tab2 = st.tabs(["📊 Main Dashboard", "🔮 Project Timeline Forecasting"])

    # --- TAB 1: Main KPI and Diagnostic Dashboard ---
    with tab1:
        # --- Section 1: Top-Level Health Metrics (The "Vital Signs") ---
        st.subheader("Department Vital Signs")
        kpis = helpers.calculate_kpis(st.session_state.protocols_df, st.session_state.team_df)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Avg. Protocol Cycle Time", value=kpis["avg_cycle_time"], delta="-0.5 days vs last month", delta_color="inverse", help="Average time from protocol creation to approval. Lower is better.")
        with col2:
            st.metric(label="Protocol Rejection Rate", value=kpis["rejection_rate"], delta="+2% vs target", delta_color="normal", help="Percentage of submitted protocols rejected during review.")
        with col3:
            st.metric(label="Executed Test Failure Rate", value=kpis["failure_rate"], delta="-5% vs target", delta_color="inverse", help="Percentage of executed tests that failed.")
        with col4:
            st.metric(label="Avg. Team Utilization", value=kpis["avg_utilization"], delta="Target: 85-95%", delta_color="off", help="Average workload vs. capacity.")

        st.markdown("---")

        # --- Section 2: Portfolio Status & High-Impact Risk ---
        st.subheader("Portfolio & Risk Landscape")
        p_col1, p_col2 = st.columns([3, 2])
        with p_col1:
            st.info("What's the status and progress of our key projects?")
            fig_gantt = helpers.create_enhanced_gantt(st.session_state.projects_df)
            st.plotly_chart(fig_gantt, use_container_width=True)
        with p_col2:
            st.info("Where are our biggest product risks hiding?")
            fig_risk = helpers.create_risk_bubble_chart(st.session_state.risk_df)
            st.plotly_chart(fig_risk, use_container_width=True)

        st.markdown("---")

        # --- Section 3: Process Quality & Bottlenecks ---
        st.subheader("Process Performance & Quality Analysis")
        q_col1, q_col2 = st.columns(2)
        with q_col1:
            st.info("What are the primary drivers of our test failures?")
            fig_pareto = helpers.create_failure_pareto_chart(st.session_state.protocols_df)
            st.plotly_chart(fig_pareto, use_container_width=True)
            st.caption("Action: Focus improvement efforts on the top 1-2 reasons (the 'vital few').")
        with q_col2:
            st.info("How is our team's capacity and workload distributed?")
            fig_util = helpers.create_team_utilization_chart(st.session_state.team_df)
            st.plotly_chart(fig_util, use_container_width=True)
            st.caption("Action: Address over-utilized members to prevent delays and burnout.")

    # --- TAB 2: Prophet Forecasting ---
    with tab2:
        st.subheader("Predictive Project Completion Forecasting")
        st.info("This tool uses historical progress to forecast a realistic completion date, identifying potential delays before they become critical.")

        projects_df = st.session_state.projects_df
        project_to_forecast = st.selectbox(
            "Select a Project to Forecast", 
            options=projects_df['Project'],
            help="Choose a project to see its forecasted completion date based on current velocity."
        )

        if project_to_forecast:
            with st.spinner(f"Generating forecast for {project_to_forecast}..."):
                project_data = projects_df[projects_df['Project'] == project_to_forecast].iloc[0]
                
                # Generate synthetic progress data
                history_df = helpers.generate_prophet_history(project_data)

                if history_df.empty:
                    st.warning("Cannot forecast a project that has not started yet.")
                else:
                    # Run the forecast
                    model, forecast = helpers.run_prophet_forecast(history_df)
                    
                    # Find the forecasted completion date (where yhat crosses 100)
                    completion_forecast = forecast[forecast['yhat'] >= 100]
                    forecasted_date = completion_forecast['ds'].iloc[0] if not completion_forecast.empty else None
                    
                    # Display results
                    planned_date = pd.to_datetime(project_data['Finish'])
                    
                    st.markdown("---")
                    f_col1, f_col2 = st.columns(2)
                    with f_col1:
                        st.metric(label="Planned Completion Date", value=planned_date.strftime('%Y-%m-%d'))
                    with f_col2:
                        if forecasted_date:
                            delta_days = (forecasted_date - planned_date).days
                            st.metric(
                                label="Forecasted Completion Date", 
                                value=forecasted_date.strftime('%Y-%m-%d'),
                                delta=f"{delta_days} days from plan",
                                delta_color="inverse" if delta_days <= 0 else "normal"
                            )
                        else:
                            st.metric(label="Forecasted Completion Date", value="> 90 days")

                    # Plot the forecast
                    fig = plot_plotly(model, forecast)
                    fig.update_layout(title=f"Forecast for '{project_to_forecast}' Progress",
                                      xaxis_title="Date", yaxis_title="Completion (%)")
                    fig.add_hline(y=100, line_dash="dot", line_color="green", annotation_text="100% Complete")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("The black dots represent simulated historical progress. The dark blue line is the forecast, and the light blue area is the uncertainty interval.")
