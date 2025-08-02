# utils.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import statsmodels.api as sm

def render_metric_summary(metric_name, description, viz_function, insight_text, reg_context=""):
    st.subheader(metric_name)
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"**Description:** {description}")
            st.info(f"**Director's Insight:** {insight_text}")
            if reg_context:
                st.warning(f"**Regulatory Context:** {reg_context}")
        with col2:
            fig = viz_function()
            st.plotly_chart(fig, use_container_width=True)

# --- I. Assay V&V Metrics ---
def plot_protocol_completion_burndown():
    days = np.arange(1, 31); planned = np.linspace(100, 0, 30); actual = np.clip(planned + np.random.randn(30).cumsum() * 1.5, 0, 100)
    fig = go.Figure(); fig.add_trace(go.Scatter(x=days, y=planned, mode='lines', name='Planned Burndown', line=dict(dash='dash'))); fig.add_trace(go.Scatter(x=days, y=actual, mode='lines+markers', name='Actual Burndown'))
    fig.update_layout(title='Protocol Completion Burndown Chart (Project ImmunoPro-A)', xaxis_title='Sprint Day', yaxis_title='% Protocols Remaining'); return fig

def plot_pass_rate_heatmap():
    df = pd.DataFrame({'Test Type': ['LoD', 'Linearity', 'Specificity', 'Precision', 'Robustness'], 'Pass Rate (%)': [95, 98, 92, 100, 88]})
    fig = px.bar(df, x='Pass Rate (%)', y='Test Type', orientation='h', title='Pass Rate by Test Type', text='Pass Rate (%)'); fig.update_traces(texttemplate='%{text}%', textposition='inside'); return fig

def plot_retest_pareto():
    df = pd.DataFrame({'Reason': ['Operator Error', 'Reagent Lot Variability', 'Instrument Drift', 'Sample Prep Issue', 'Software Glitch'], 'Count': [15, 9, 5, 3, 1]}).sort_values(by='Count', ascending=False)
    df['Cumulative Pct'] = (df['Count'].cumsum() / df['Count'].sum()) * 100; fig = go.Figure(); fig.add_trace(go.Bar(x=df['Reason'], y=df['Count'], name='Re-test Count')); fig.add_trace(go.Scatter(x=df['Reason'], y=df['Cumulative Pct'], name='Cumulative %', yaxis='y2'))
    fig.update_layout(title="Pareto Chart of Re-test Causes", yaxis2=dict(overlaying='y', side='right', title='Cumulative %')); return fig

def plot_trace_coverage_sankey():
    fig = go.Figure(go.Sankey(node=dict(pad=15, thickness=20, label=["URS-01", "URS-02", "SRS-01", "DVP-01", "DVP-02", "UNCOVERED"]), link=dict(source=[0, 1, 2, 2], target=[3, 4, 3, 4], value=[8, 4, 2, 8])))
    fig.update_layout(title_text="Requirements Trace Coverage (Sankey Diagram)"); return fig

def plot_rpn_waterfall():
    fig = go.Figure(go.Waterfall(measure = ["relative", "relative", "total", "relative", "relative", "total"], x = ["Initial Risk (FP)", "Mitigation 1", "Subtotal", "Initial Risk (FN)", "Mitigation 2", "Final Risk Portfolio"], y = [120, -40, 0, 80, -60, 0]))
    fig.update_layout(title = "FMEA Risk Reduction (RPN Waterfall)"); return fig

# --- II. Equipment Validation ---
def plot_validation_gantt_baseline():
    df = pd.DataFrame([dict(Task="FAT", Start='2023-01-01', Finish='2023-01-10', Type="Planned"), dict(Task="FAT", Start='2023-01-01', Finish='2023-01-12', Type="Actual"), dict(Task="SAT", Start='2023-01-15', Finish='2023-01-20', Type="Planned"), dict(Task="SAT", Start='2023-01-18', Finish='2023-01-22', Type="Actual"), dict(Task="IQ", Start='2023-01-21', Finish='2023-01-25', Type="Planned"), dict(Task="IQ", Start='2023-01-23', Finish='2023-01-25', Type="Actual")])
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Type", title="Validation On-Time Rate (Gantt vs Baseline)"); return fig

def plot_sat_to_pq_violin():
    df = pd.DataFrame({'Days': np.concatenate([np.random.normal(20, 5, 20), np.random.normal(35, 8, 15)]), 'Equipment Type': ['Analyzer'] * 20 + ['Sample Prep'] * 15})
    fig = px.violin(df, y="Days", x="Equipment Type", color="Equipment Type", box=True, points="all", title="Time from SAT to PQ Approval (by Equipment Type)"); return fig

# --- III. Team & Project ---
def plot_protocol_review_cycle_histogram():
    fig = px.histogram(np.random.gamma(4, 2, 100), title="Protocol Review Cycle Time (Draft to Approved)", labels={'value': 'Days'}); return fig

def plot_training_donut():
    fig = go.Figure(data=[go.Pie(labels=['ISO 13485','GAMP5','21 CFR 820', 'GDP', 'Not Started'], values=[100, 95, 98, 90, 5], hole=.4, title="Training Completion Rate")]); return fig

# --- IV. Quality & CI ---
def plot_rft_gauge():
    fig = go.Figure(go.Indicator(mode = "gauge+number", value = 82, title = {'text': "Right-First-Time (RFT) Protocol Execution"}, gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "cornflowerblue"}})); return fig

def plot_capa_funnel():
    fig = go.Figure(go.Funnel(y = ["Identified", "Investigation", "Root Cause Analysis", "Implementation", "Effectiveness Check"], x = [100, 80, 65, 60, 55], textinfo = "value+percent initial"))
    fig.update_layout(title="CAPA Closure Effectiveness Funnel"); return fig

# --- V. Statistical Methods ---
def run_anova_ttest(add_shift):
    group_a = np.random.normal(10, 2, 30); group_b_mean = 10.5 if not add_shift else 12.5; group_b = np.random.normal(group_b_mean, 2, 30)
    fig = px.box(pd.DataFrame({'Group A': group_a, 'Group B': group_b}), title="Performance Comparison (Lot A vs Lot B)")
    t_stat, p_value = stats.ttest_ind(group_a, group_b); result = f"**T-test Result:** p-value = {p_value:.4f}. "
    result += "**Conclusion:** Difference is statistically significant." if p_value < 0.05 else "**Conclusion:** No significant difference detected."; return fig, result

def run_regression_analysis():
    rpn = np.random.randint(20, 150, 50); failure_prob = rpn / 200; failures = np.random.binomial(1, failure_prob)
    df = pd.DataFrame({'RPN': rpn, 'Failure Occurred': failures}); df['Failure Occurred'] = df['Failure Occurred'].astype('category')
    fig = px.scatter(df, x='RPN', y='Failure Occurred', title="Correlation of Risk (RPN) to Failure Rate", marginal_y="histogram")
    return fig, "**Insight:** Higher RPN values show a clear trend towards a higher likelihood of test failure, validating the risk assessment process."

def run_descriptive_stats():
    data = np.random.normal(50, 2, 100); df = pd.DataFrame(data, columns=["LoD Measurement (copies/mL)"])
    mean, std, cv = df.iloc[:,0].mean(), df.iloc[:,0].std(), (df.iloc[:,0].std() / df.iloc[:,0].mean()) * 100
    fig = px.histogram(df, x="LoD Measurement (copies/mL)", marginal="box", title="Descriptive Statistics for Limit of Detection (LoD) Study")
    return fig, f"**Mean:** {mean:.2f} | **Std Dev:** {std:.2f} | **%CV:** {cv:.2f}%"

def run_control_charts():
    data = [np.random.normal(10, 0.5, 5) for _ in range(20)]; data[15:] = [np.random.normal(10.8, 0.5, 5) for _ in range(5)]
    df = pd.DataFrame(data, columns=[f'm{i}' for i in range(1,6)]); df['mean'] = df.mean(axis=1); df['range'] = df.max(axis=1) - df.min(axis=1)
    x_bar_cl = df['mean'].mean(); x_bar_ucl = x_bar_cl + 3 * (df['range'].mean() / 2.326); x_bar_lcl = x_bar_cl - 3 * (df['range'].mean() / 2.326)
    fig = go.Figure(); fig.add_trace(go.Scatter(x=df.index, y=df['mean'], name='Subgroup Mean', mode='lines+markers')); fig.add_hline(y=x_bar_cl, line_dash="dash", line_color="green", annotation_text="CL")
    fig.add_hline(y=x_bar_ucl, line_dash="dot", line_color="red", annotation_text="UCL"); fig.add_hline(y=x_bar_lcl, line_dash="dot", line_color="red", annotation_text="LCL")
    fig.update_layout(title="X-bar Control Chart for Process Monitoring"); return fig

def run_kaplan_meier():
    time_to_failure = np.random.weibull(2, 50) * 24; observed = np.random.binomial(1, 0.8, 50); df = pd.DataFrame({'Months': time_to_failure, 'Observed': observed}).sort_values(by='Months')
    at_risk = len(df); survival_prob = []
    for i, row in df.iterrows():
        survival = (at_risk - 1) / at_risk if row['Observed'] == 1 else 1; at_risk -= 1; survival_prob.append(survival)
    df['Survival'] = np.cumprod(survival_prob)
    fig = px.line(df, x='Months', y='Survival', title="Kaplan-Meier Survival Plot for Shelf-Life Validation", markers=True); fig.update_yaxes(range=[0, 1.05]); return fig, "**Conclusion:** The estimated median shelf-life (time to 50% survival) is approximately 21 months."

def run_monte_carlo():
    n_sims = 5000; task1, task2, task3 = np.random.triangular(8,10,15,n_sims), np.random.triangular(15,20,30,n_sims), np.random.triangular(5,8,12,n_sims)
    total_times = task1 + task2 + task3; p90 = np.percentile(total_times, 90)
    fig = px.histogram(total_times, nbins=50, title="Monte Carlo Simulation of V&V Plan Duration (5000 runs)"); fig.add_vline(x=p90, line_dash="dash", line_color="red", annotation_text=f"P90 = {p90:.1f} days")
    return fig, f"**Conclusion:** While the 'most likely' duration is ~38 days, there is a 10% chance the project will take **{p90:.1f} days or longer**. This P90 value should be used for risk-adjusted planning."

# --- Software V&V ---
def create_v_model_figure():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1], mode='lines+markers+text', text=["User Needs", "System Req.", "Architecture", "Module Design"], textposition="top right", line=dict(color='royalblue', width=2), marker=dict(size=10)))
    fig.add_trace(go.Scatter(x=[5, 6, 7, 8], y=[1, 2, 3, 4], mode='lines+markers+text', text=["Unit Test", "Integration Test", "System V&V", "User Validation (UAT)"], textposition="top left", line=dict(color='green', width=2), marker=dict(size=10)))
    for i in range(4): fig.add_shape(type="line", x0=4-i, y0=1+i, x1=5+i, y1=1+i, line=dict(color="grey", width=1, dash="dot"))
    fig.update_layout(title_text="The V-Model for Software Validation (IEC 62304)", showlegend=False, xaxis=dict(showticklabels=False, zeroline=False, showgrid=False), yaxis=dict(showticklabels=False, zeroline=False, showgrid=False)); return fig

def get_software_risk_data():
    return pd.DataFrame([{"Software Item": "Patient Result Algorithm", "Description": "Calculates the final diagnostic result from raw signal.", "Potential Harm": "Misdiagnosis (death or serious injury)", "IEC 62304 Class": "Class C"},{"Software Item": "Database Middleware", "Description": "Transfers data between application and database.", "Potential Harm": "Data loss/corruption (indirect harm)", "IEC 62304 Class": "Class B"},{"Software Item": "UI Color Theme Module", "Description": "Controls the look and feel of the user interface.", "Potential Harm": "No potential for harm", "IEC 62304 Class": "Class A"}])

def get_part11_checklist_data():
    return {"Controls for closed systems": {"11.10(a) Validation": True, "11.10(b) Accurate Copies": True, "11.10(c) Protection of Records": True, "11.10(d) Limiting System Access": True, "11.10(e) Audit Trails": True, "11.10(f) Operational System Checks": False, "11.10(g) Authority Checks": True, "11.10(h) Device Checks": True}, "Controls for open systems": {"11.30 Data Encryption": True, "11.30 Digital Signature Standards": True}, "Electronic Signatures": {"11.50 General Requirements": True, "11.70 Signature/Record Linking": True, "11.200(a) Signature Components": False,}}
