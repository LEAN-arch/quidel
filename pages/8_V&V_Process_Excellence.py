# pages/8_V&V_Process_Excellence.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import generate_process_excellence_data

st.set_page_config(
    page_title="V&V Process Excellence | QuidelOrtho",
    layout="wide"
)

st.title("🏆 V&V Process Excellence Dashboard")
st.markdown("### Measuring and Improving the Performance of the V&V Engine")

with st.expander("🌐 Director's View: From Managing Projects to Optimizing the Process", expanded=True):
    st.markdown("""
    As a leader, my role is not just to ensure individual projects succeed, but to continuously improve the underlying *process* by which we achieve that success. This dashboard moves beyond project-specific metrics to monitor the health, efficiency, and quality of our V&V process itself. It provides the data to answer critical questions: Are we getting faster? Is the quality of our documentation improving? Are we becoming more efficient in our testing?

    **Key Responsibilities & Quality Imperatives:**
    - **Continuous Improvement (ISO 13485, 8.5.1):** This dashboard is the embodiment of the continuous improvement principle. By tracking key process indicators (KPIs), I can identify systemic trends, measure the impact of improvement initiatives, and demonstrate a commitment to operational excellence.
    - **Quality Objectives (ISO 13485, 5.4.1):** The metrics on this page are our department's Quality Objectives. They are measurable, data-driven goals that define what "good" looks like for the V&V function.
    - **Data-Driven Management:** Instead of relying on gut feelings, this dashboard provides objective evidence to justify investments in training, tools, or process changes. For example, a rising `Report_Rework_Rate` provides a clear justification for investing in better report templates or author training.
    - **Statistical Process Control (SPC):** Applying SPC to our internal processes is a highly advanced management technique that allows us to distinguish between normal process variation ("common cause") and significant shifts that require investigation ("special cause").
    """)

# --- Data Generation ---
process_df = generate_process_excellence_data()

# --- KPIs for Process Excellence ---
st.header("V&V Department Key Performance Indicators (KPIs)")
latest_cycle_time = process_df['Protocol_Approval_Cycle_Time_Days'].iloc[-1]
avg_cycle_time = process_df['Protocol_Approval_Cycle_Time_Days'].mean()
latest_rework_rate = process_df['Report_Rework_Rate_Percent'].iloc[-1]
avg_rework_rate = process_df['Report_Rework_Rate_Percent'].mean()
latest_deviation_rate = process_df['Deviations_per_100_Test_Hours'].iloc[-1]
avg_deviation_rate = process_df['Deviations_per_100_Test_Hours'].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Protocol Approval Cycle Time (Days)", f"{latest_cycle_time:.1f}", f"{latest_cycle_time-avg_cycle_time:.1f} vs. 18mo Avg", delta_color="inverse")
col2.metric("V&V Report Rework Rate (%)", f"{latest_rework_rate:.1f}%", f"{latest_rework_rate-avg_rework_rate:.1f} vs. 18mo Avg")
col3.metric("Deviations per 100 Test Hours", f"{latest_deviation_rate:.1f}", f"{latest_deviation_rate-avg_deviation_rate:.1f} vs. 18mo Avg", delta_color="inverse")

st.divider()

# --- SPC Charts for Process Monitoring ---
st.header("V&V Process Control Charts (XmR)")
st.caption("Monitoring key V&V process metrics for statistically significant shifts over time.")

def create_spc_chart(df, value_col, title, lower_is_better=True):
    """Creates an XmR chart for a given metric."""
    mean = df[value_col].mean()
    mr = df[value_col].diff().abs()
    mean_mr = mr.mean()
    
    ucl_x = mean + 2.66 * mean_mr
    lcl_x = mean - 2.66 * mean_mr
    ucl_mr = 3.267 * mean_mr
    
    fig = go.Figure()
    # X Chart
    fig.add_trace(go.Scatter(x=df['Month'], y=df[value_col], name='Metric (X)', mode='lines+markers', marker_color='#0039A6'))
    fig.add_hline(y=mean, line_dash="solid", line_color="green", annotation_text="Mean")
    fig.add_hline(y=ucl_x, line_dash="dash", line_color="red", annotation_text="UCL")
    fig.add_hline(y=lcl_x, line_dash="dash", line_color="red", annotation_text="LCL")
    
    # Highlight out-of-control points
    out_of_control = df[(df[value_col] > ucl_x) | (df[value_col] < lcl_x)]
    fig.add_trace(go.Scatter(x=out_of_control['Month'], y=out_of_control[value_col], mode='markers',
                             marker=dict(color='red', size=12, symbol='x'), name='Out of Control'))
    
    fig.update_layout(title=title, yaxis_title=value_col.replace('_', ' '))
    return fig

chart1, chart2, chart3 = st.columns(3)

with chart1:
    st.subheader("Protocol Approval Cycle Time")
    fig1 = create_spc_chart(process_df, 'Protocol_Approval_Cycle_Time_Days', 'Cycle Time (X Chart)', lower_is_better=True)
    st.plotly_chart(fig1, use_container_width=True)

with chart2:
    st.subheader("V&V Report Rework Rate")
    fig2 = create_spc_chart(process_df, 'Report_Rework_Rate_Percent', 'Rework Rate (X Chart)', lower_is_better=True)
    st.plotly_chart(fig2, use_container_width=True)
    
with chart3:
    st.subheader("Execution Deviation Rate")
    fig3 = create_spc_chart(process_df, 'Deviations_per_100_Test_Hours', 'Deviation Rate (X Chart)', lower_is_better=True)
    st.plotly_chart(fig3, use_container_width=True)

with st.expander("**Director's Analysis & Action Plan**"):
    st.markdown("""
    These control charts provide a powerful, objective way to manage my department's performance.
    - **Protocol Approval Cycle Time:** We see a clear, sustained downward trend, indicating our process improvement initiatives (e.g., new templates, weekly review meetings) have been effective. The process is demonstrating a positive shift and is in a state of control.
    - **V&V Report Rework Rate:** The chart shows a stable process operating within predictable limits, but there is one **"Out of Control"** point. This represents a "special cause" variation. I would task the V&V Manager (M. Rodriguez) to investigate that specific month to understand the root cause (e.g., a new complex assay type, a new hire's first reports). The goal is to learn from this event to prevent recurrence.
    - **Execution Deviation Rate:** This process also shows a favorable downward trend. This metric is a key indicator of the quality of our planning and training. A lower deviation rate means our protocols are clearer, our team is better trained, and our projects are more likely to stay on schedule.
    """)
