# modules/dashboard.py (SME ENHANCED VERSION)

import streamlit as st
from utils import helpers

def render_page():
    """Renders the significantly enhanced, SME-driven executive dashboard page."""
    
    st.title("V&V Command Center Dashboard")
    st.markdown("Diagnostic overview of portfolio health, process efficiency, quality, and risk. *Designed for action.*")
    
    # --- Calculate All KPIs at the top ---
    kpis = helpers.calculate_kpis(st.session_state.protocols_df, st.session_state.team_df)
    projects_df = st.session_state.projects_df

    # --- Section 1: Top-Level Health Metrics (The "Vital Signs") ---
    st.subheader("Department Vital Signs")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Avg. Protocol Cycle Time",
            value=kpis["avg_cycle_time"],
            delta="-0.5 days vs last month",
            delta_color="inverse",
            help="Average time from protocol creation to approval. Lower is better."
        )
    with col2:
        st.metric(
            label="Protocol Rejection Rate",
            value=kpis["rejection_rate"],
            delta="+2% vs target",
            delta_color="normal",
            help="Percentage of submitted protocols rejected during review. Indicates quality of initial planning."
        )
    with col3:
        st.metric(
            label="Executed Test Failure Rate",
            value=kpis["failure_rate"],
            delta="-5% vs target",
            delta_color="inverse",
            help="Percentage of executed tests that failed to meet acceptance criteria."
        )
    with col4:
        st.metric(
            label="Avg. Team Utilization",
            value=kpis["avg_utilization"],
            delta="Target: 85-95%",
            delta_color="off",
            help="Average workload vs. capacity. Over 100% indicates burnout risk."
        )

    st.markdown("---")

    # --- Section 2: Portfolio Status & High-Impact Risk (The "Where to Focus") ---
    st.subheader("Portfolio & Risk Landscape")
    col1, col2 = st.columns([3, 2]) # Gantt is wider

    with col1:
        st.info("What's the status and progress of our key projects?")
        fig_gantt = helpers.create_enhanced_gantt(projects_df)
        st.plotly_chart(fig_gantt, use_container_width=True)

    with col2:
        st.info("Where are our biggest product risks hiding?")
        fig_risk = helpers.create_risk_bubble_chart(st.session_state.risk_df)
        st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("---")

    # --- Section 3: Process Quality & Bottlenecks (The "Why Are We Having Problems?") ---
    st.subheader("Process Performance & Quality Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.info("What are the primary drivers of our test failures?")
        fig_pareto = helpers.create_failure_pareto_chart(st.session_state.protocols_df)
        st.plotly_chart(fig_pareto, use_container_width=True)
        st.caption("Action: Focus improvement efforts on the top 1-2 reasons (the 'vital few').")
        
    with col2:
        st.info("How is our team's capacity and workload distributed?")
        fig_util = helpers.create_team_utilization_chart(st.session_state.team_df)
        st.plotly_chart(fig_util, use_container_width=True)
        st.caption("Action: Address over-utilized members (e.g., Bob, Edward) to prevent delays and burnout.")
        
