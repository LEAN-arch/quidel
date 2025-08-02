# pages/11_Validation_Process_Excellence.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
from utils import generate_process_excellence_data, generate_workload_forecast_data

st.set_page_config(
    page_title="Validation Process Excellence | QuidelOrtho",
    layout="wide"
)

st.title("🏆 Validation Process Excellence Dashboard")
st.markdown("### Measuring, Improving, and Forecasting the Performance of the Validation Engine")

with st.expander("🌐 Manager's View: From Managing Projects to Optimizing the Process", expanded=True):
    st.markdown("""
    As a leader, my role is not just to ensure individual projects succeed, but to continuously improve the underlying *process* by which we achieve that success. This dashboard moves beyond project-specific metrics to monitor the health, efficiency, and quality of our validation process itself. It provides the data to answer critical questions: Are we getting faster? Is the quality of our documentation improving? Can we accurately forecast future resource needs?

    **Key Responsibilities & Quality Imperatives:**
    - **Continuous Improvement (ISO 13485, 8.5.1):** This dashboard is the embodiment of the continuous improvement principle. By tracking key process indicators (KPIs), I can identify systemic trends, measure the impact of improvement initiatives, and demonstrate a commitment to operational excellence.
    - **Quality Objectives (ISO 13485, 5.4.1):** The metrics on this page are our department's Quality Objectives. They are measurable, data-driven goals that define what "good" looks like for the validation function.
    - **Data-Driven Management:** This dashboard provides objective evidence to justify investments in training, tools, or process changes. For example, a rising `Report_Rework_Rate` provides a clear justification for investing in better report templates or author training.
    - **Predictive Resource Planning:** The workload forecast allows me to move from reactive to proactive hiring and resource management, ensuring I have the right team in place to meet the business's future needs.
    """)

# --- Data Generation ---
process_df = generate_process_excellence_data()

# --- KPIs for Process Excellence ---
st.header("Validation Department Key Performance Indicators (KPIs)")
latest_cycle_time = process_df['Protocol_Approval_Cycle_Time_Days'].iloc[-1]
avg_cycle_time = process_df['Protocol_Approval_Cycle_Time_Days'].mean()
latest_rework_rate = process_df['Report_Rework_Rate_Percent'].iloc[-1]
avg_rework_rate = process_df['Report_Rework_Rate_Percent'].mean()
latest_deviation_rate = process_df['Deviations_per_100_Test_Hours'].iloc[-1]
avg_deviation_rate = process_df['Deviations_per_100_Test_Hours'].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Protocol Approval Cycle Time (Days)", f"{latest_cycle_time:.1f}", f"{latest_cycle_time-avg_cycle_time:.1f} vs. 18mo Avg", delta_color="inverse")
col2.metric("Validation Report Rework Rate (%)", f"{latest_rework_rate:.1f}%", f"{latest_rework_rate-avg_rework_rate:.1f} vs. 18mo Avg")
col3.metric("Deviations per 100 Test Hours", f"{latest_deviation_rate:.1f}", f"{latest_deviation_rate-avg_deviation_rate:.1f} vs. 18mo Avg", delta_color="inverse")

st.divider()

# --- Process Monitoring & Forecasting Tabs ---
tab1, tab2 = st.tabs(["**Process Control Charts (SPC)**", "**Predictive Workload Forecasting (Prophet)**"])

with tab1:
    st.header("Validation Process Control Charts (XmR)")
    st.caption("Monitoring key validation process metrics for statistically significant shifts over time.")

    def create_spc_chart(df, value_col, title):
        mean = df[value_col].mean()
        mr = df[value_col].diff().abs()
        mean_mr = mr.mean()
        
        ucl_x = mean + 2.66 * mean_mr
        lcl_x = mean - 2.66 * mean_mr
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Month'], y=df[value_col], name='Metric (X)', mode='lines+markers', marker_color='#0039A6'))
        fig.add_hline(y=mean, line_dash="solid", line_color="green", annotation_text="Mean")
        fig.add_hline(y=ucl_x, line_dash="dash", line_color="red", annotation_text="UCL")
        fig.add_hline(y=lcl_x, line_dash="dash", line_color="red", annotation_text="LCL")
        
        out_of_control = df[(df[value_col] > ucl_x) | (df[value_col] < lcl_x)]
        fig.add_trace(go.Scatter(x=out_of_control['Month'], y=out_of_control[value_col], mode='markers',
                                 marker=dict(color='red', size=12, symbol='x'), name='Out of Control'))
        
        fig.update_layout(title=title, yaxis_title=value_col.replace('_', ' '))
        return fig

    chart1, chart2, chart3 = st.columns(3)
    with chart1:
        st.subheader("Protocol Approval Cycle Time")
        fig1 = create_spc_chart(process_df, 'Protocol_Approval_Cycle_Time_Days', 'Cycle Time (X Chart)')
        st.plotly_chart(fig1, use_container_width=True)
    with chart2:
        st.subheader("Validation Report Rework Rate")
        fig2 = create_spc_chart(process_df, 'Report_Rework_Rate_Percent', 'Rework Rate (X Chart)')
        st.plotly_chart(fig2, use_container_width=True)
    with chart3:
        st.subheader("Execution Deviation Rate")
        fig3 = create_spc_chart(process_df, 'Deviations_per_100_Test_Hours', 'Deviation Rate (X Chart)')
        st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.header("Predictive Resource & Workload Forecasting")
    st.caption("Using historical data to forecast future departmental workload (measured in test hours) for budget and headcount planning.")

    workload_df = generate_workload_forecast_data()
    
    # Prophet model
    m = Prophet(yearly_seasonality=True, daily_seasonality=False)
    m.fit(workload_df)
    future = m.make_future_dataframe(periods=12, freq='MS')
    forecast = m.predict(future)
    
    fig_forecast = plot_plotly(m, forecast)
    fig_forecast.update_layout(
        title="12-Month Workload Forecast (Test Hours)",
        xaxis_title="Date",
        yaxis_title="Project Test Hours / Month"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    with st.expander("**Manager's Analysis & Action Plan**"):
        st.markdown("""
        This forecast is a strategic tool for my annual budget and resource planning.
        - **Black Dots:** Represent our actual historical workload.
        - **Dark Blue Line:** The model's prediction of future workload.
        - **Light Blue Shaded Area:** The uncertainty interval. We should plan resources to handle the upper end of this range.
        - **Actionable Insight:** The forecast predicts a significant increase in workload in the next 6-9 months, driven by both overall trend and seasonal project kickoffs. This provides me with the data-driven justification to open a new headcount requisition *now* to ensure the new hire is trained and ready before the workload peak, preventing team burnout and project delays.
        """)
