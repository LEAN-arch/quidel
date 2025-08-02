# utils/helpers.py (ENHANCED VERSION)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy import stats
from python_pptx import Presentation
from python_pptx.util import Inches
from datetime import datetime, timedelta
import io

# --- ENHANCED MOCK DATA GENERATION ---

def get_mock_team_data():
    """Generates a mock team dataframe with more detail."""
    return pd.DataFrame({
        'Member': ['Alice', 'Bob', 'Charlie', 'Diana', 'Edward'],
        'Role': ['V&V Engineer', 'Sr. V&V Engineer', 'V&V Specialist', 'V&V Engineer', 'Sr. V&V Engineer'],
        'Capacity (hrs/wk)': [40, 40, 40, 40, 40],
        'Assigned_Hrs': [35, 45, 38, 25, 42],
        'Training_Status': ['Compliant', 'Compliant', 'Overdue', 'Compliant', 'Compliant']
    })

def get_mock_projects_data():
    """Generates mock projects with completion %."""
    today = datetime.now()
    return pd.DataFrame([
        dict(Project="ImmunoPro-A", Start=str(today-timedelta(days=60)), Finish=str(today+timedelta(days=30)), Status='On Track', Owner='Alice', Pct_Complete=75),
        dict(Project="MolecularDX-2", Start=str(today-timedelta(days=90)), Finish=str(today+timedelta(days=10)), Status='At Risk', Owner='Bob', Pct_Complete=90),
        dict(Project="CardioMarker-V", Start=str(today-timedelta(days=20)), Finish=str(today+timedelta(days=90)), Status='On Track', Owner='Charlie', Pct_Complete=20),
        dict(Project="Consumable-X", Start=str(today-timedelta(days=45)), Finish=str(today+timedelta(days=45)), Status='Delayed', Owner='Diana', Pct_Complete=50),
        dict(Project="OncologyPanel-1", Start=str(today-timedelta(days=120)), Finish=str(today+timedelta(days=60)), Status='On Track', Owner='Edward', Pct_Complete=60),
    ])

def get_mock_protocols_data():
    """Generates mock protocols with dates for cycle time and failure reasons."""
    today = datetime.now()
    return pd.DataFrame({
        'Protocol_ID': ['IP-PREC-01', 'IP-SENS-01', 'IP-SPEC-01', 'MDX-LOD-01', 'IP-TTR-01', 'IP-REJ-01', 'CM-PREC-01', 'CM-STAB-01'],
        'Project': ['ImmunoPro-A', 'ImmunoPro-A', 'ImmunoPro-A', 'MolecularDX-2', 'ImmunoPro-A', 'ImmunoPro-A', 'CardioMarker-V', 'CardioMarker-V'],
        'Title': ['Precision Study', 'Sensitivity Study', 'Specificity Study', 'Limit of Detection', 'Time to Result', 'Linearity Study', 'Precision Study', 'Stability Study'],
        'Status': ['Executed - Passed', 'Approved', 'Rejected', 'Executed - Passed', 'Executed - Failed', 'Draft', 'Approved', 'Executed - Failed'],
        'Creation_Date': [today-timedelta(days=40), today-timedelta(days=30), today-timedelta(days=25), today-timedelta(days=60), today-timedelta(days=20), today-timedelta(days=5), today-timedelta(days=15), today-timedelta(days=18)],
        'Approval_Date': [today-timedelta(days=35), today-timedelta(days=28), today-timedelta(days=22), today-timedelta(days=55), today-timedelta(days=18), pd.NaT, today-timedelta(days=10), today-timedelta(days=15)],
        'Failure_Reason': [np.nan, np.nan, 'Incorrect Acceptance Criteria', np.nan, 'Reagent Issue', np.nan, np.nan, 'Operator Error'],
        'Acceptance_Criteria': ['CV <= 5%', 'Detect 95/100', 'No cross-reactivity', 'LoD < 50 copies/mL', 'Result < 15min', 'R^2 > 0.99', 'CV <= 8%', '30-day stability']
    })

# --- (Keep other mock data functions as they are) ---
def get_mock_requirements_data():
    return pd.DataFrame({'Req_ID':['USR-001','USR-002','FNC-001','FNC-002','FNC-003','SYS-001'],'Project':['ImmunoPro-A','ImmunoPro-A','ImmunoPro-A','MolecularDX-2','MolecularDX-2','ImmunoPro-A'],'Requirement_Type':['User','User','Functional','Functional','Functional','System'],'Requirement_Text':['The assay shall detect Antigen A with >99% sensitivity.','The assay shall have a specificity of >99%.','The system must provide results within 15 minutes.','The assay shall amplify DNA target region B.','The assay must have a limit of detection of <50 copies/mL.','The software shall display results in a clear format.'],'Linked_Protocol_ID':['IP-PREC-01, IP-SENS-01','IP-SPEC-01','IP-TTR-01','MDX-LOD-01','MDX-LOD-01',np.nan],'Status':['Covered','Covered','Covered','Covered','Covered','Gap']})
def get_mock_risk_data():
    return pd.DataFrame({'Risk_ID':['R-001','R-002','R-003','R-004'],'Project':['ImmunoPro-A','ImmunoPro-A','MolecularDX-2','MolecularDX-2'],'Failure_Mode':['False Positive due to cross-reactivity','Incorrect reagent dispense volume','Sample contamination during prep','Reagent degradation at room temp'],'Severity':[8,9,7,8],'Occurrence':[3,2,4,3],'Detection':[4,7,3,5],'RPN':[96,126,84,120],'Mitigation_Action':['Test with specificity panel','Verify dispense volume in OQ','Implement new cleaning procedure','Conduct real-time stability study'],'Linked_Protocol_ID':['IP-SPEC-01','Consumable-OQ-01','N/A','MDX-STAB-01']})
# --- (Keep auth, logging, basic analysis, and PPT functions as they are) ---
def render_login():
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if not st.session_state.logged_in:
        st.title("AssayVantage Command Center Login")
        with st.form("login_form"):
            st.text_input("Username", value="director", disabled=True); password = st.text_input("Password", type="password"); submitted = st.form_submit_button("Login")
            if submitted:
                if password == "quidel": st.session_state.logged_in = True; log_action("director", "User login successful."); st.experimental_rerun()
                else: st.error("Incorrect password")
def log_action(user,action,details=""):
    if 'audit_log' not in st.session_state: st.session_state.audit_log = []
    st.session_state.audit_log.insert(0,{'Timestamp':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'User':user,'Action':action,'Details':details})
def load_data(source):
    if 'data_loaded' not in st.session_state:
        st.session_state.projects_df=get_mock_projects_data();st.session_state.requirements_df=get_mock_requirements_data();st.session_state.protocols_df=get_mock_protocols_data();st.session_state.risk_df=get_mock_risk_data();st.session_state.team_df=get_mock_team_data();st.session_state.audit_log=[];st.session_state.data_loaded=True;log_action("SYSTEM","Initialized mock data sets.")
def analyze_precision(data_df): return ({'N':len(data_df),'Mean':f"{data_df['Value'].mean():.2f}",'Std Dev':f"{data_df['Value'].std():.2f}",'CV (%)':f"{(data_df['Value'].std()/data_df['Value'].mean())*100:.2f}"}, px.box(data_df,y='Value',title='Precision Data Distribution',points='all'))
def analyze_linearity(data_df): slope,intercept,r,p,se=stats.linregress(data_df['Expected'],data_df['Observed']); return ({'N':len(data_df),'Slope':f"{slope:.4f}",'Intercept':f"{intercept:.4f}",'R-squared':f"{r**2:.4f}"}, px.scatter(data_df,x='Expected',y='Observed',title='Linearity Plot',trendline='ols',trendline_color_override='red'))
def generate_ppt_report(protocol_data,analysis_results,analysis_fig): prs=Presentation();slide=prs.slides.add_slide(prs.slide_layouts[0]);slide.shapes.title.text="V&V Summary Report";slide.placeholders[1].text=f"Protocol: {protocol_data['Protocol_ID']} - {protocol_data['Title']}";slide=prs.slides.add_slide(prs.slide_layouts[1]);slide.shapes.title.text="Protocol Summary";tf=slide.shapes.add_textbox(Inches(0.5),Inches(1.5),Inches(9.0),Inches(5.5)).text_frame;tf.clear();tf.paragraphs[0].text=f"Project: {protocol_data['Project']}";tf.add_paragraph().text=f"Acceptance Criteria: {protocol_data['Acceptance_Criteria']}";p=tf.add_paragraph();p.text=f"Status: {protocol_data['Status']}";p.space_before=Inches(0.2);p=tf.add_paragraph();p.text="Execution Results:";p.space_before=Inches(0.5);[p:=tf.add_paragraph(),p.text:=f"  • {k}: {v}",p.level:=1 for k,v in analysis_results.items()];slide=prs.slides.add_slide(prs.slide_layouts[5]);slide.shapes.title.text="Graphical Analysis";img_bytes=io.BytesIO();analysis_fig.write_image(img_bytes,format='png',scale=2);img_bytes.seek(0);slide.shapes.add_picture(img_bytes,Inches(1.0),Inches(1.5),width=Inches(8.0));ppt_io=io.BytesIO();prs.save(ppt_io);ppt_io.seek(0);return ppt_io

# --- NEW & IMPROVED VISUALIZATION/ANALYTICS FUNCTIONS ---

def calculate_kpis(protocols_df, team_df):
    """Calculate advanced KPIs for the dashboard."""
    # Convert to datetime, coercing errors
    protocols_df['Creation_Date'] = pd.to_datetime(protocols_df['Creation_Date'], errors='coerce')
    protocols_df['Approval_Date'] = pd.to_datetime(protocols_df['Approval_Date'], errors='coerce')

    # 1. Protocol Review Cycle Time
    valid_dates = protocols_df.dropna(subset=['Creation_Date', 'Approval_Date'])
    cycle_time = (valid_dates['Approval_Date'] - valid_dates['Creation_Date']).dt.days
    avg_cycle_time = cycle_time.mean()

    # 2. Protocol Rejection Rate
    total_reviewed = len(protocols_df[protocols_df['Status'].isin(['Executed - Passed', 'Executed - Failed', 'Approved', 'Rejected'])])
    rejected_count = len(protocols_df[protocols_df['Status'] == 'Rejected'])
    rejection_rate = (rejected_count / total_reviewed) * 100 if total_reviewed > 0 else 0

    # 3. Test Failure Rate (of those executed)
    executed = protocols_df[protocols_df['Status'].str.contains("Executed", na=False)]
    failed_tests = len(executed[executed['Status'] == 'Executed - Failed'])
    failure_rate = (failed_tests / len(executed)) * 100 if len(executed) > 0 else 0
    
    # 4. Team Capacity Utilization
    team_df['Utilization'] = (team_df['Assigned_Hrs'] / team_df['Capacity (hrs/wk)']) * 100
    avg_utilization = team_df['Utilization'].mean()

    return {
        "avg_cycle_time": f"{avg_cycle_time:.1f} days",
        "rejection_rate": f"{rejection_rate:.1f}%",
        "failure_rate": f"{failure_rate:.1f}%",
        "avg_utilization": f"{avg_utilization:.0f}%"
    }

def create_enhanced_gantt(df):
    """Gantt chart with % complete and more details."""
    gantt_df = df.copy()
    gantt_df.rename(columns={'Project': 'Task', 'Pct_Complete': 'Completion_pct'}, inplace=True)
    gantt_df['Completion_pct'] = gantt_df['Completion_pct'] / 100 # plotly expects 0-1
    fig = px.timeline(gantt_df, x_start="Start", x_end="Finish", y="Task", color="Status",
                      custom_data=['Owner', 'Completion_pct'],
                      title="Project Portfolio Status & Progress")
    fig.update_traces(text=gantt_df['Completion_pct'].apply(lambda x: f'{x:.0%}'), textposition='inside')
    fig.update_layout(xaxis_title="Timeline", yaxis_title="Project")
    return fig
    
def create_risk_bubble_chart(risk_df):
    """Creates a Severity vs. Occurrence bubble chart where bubble size is RPN."""
    fig = px.scatter(
        risk_df,
        x="Severity",
        y="Occurrence",
        size="RPN",
        color="Project",
        hover_name="Failure_Mode",
        size_max=60,
        title="Product Risk Landscape (FMEA)",
        labels={"Severity": "Severity (Impact)", "Occurrence": "Likelihood of Occurrence"}
    )
    fig.add_shape(type="rect", x0=6.5, y0=3.5, x1=10, y1=10, line=dict(color="Red", width=2, dash="dash"), fillcolor="rgba(255,0,0,0.1)")
    fig.add_annotation(x=9.5, y=9.5, text="High-Risk Zone", showarrow=False, xanchor='right', yanchor='top')
    return fig

def create_failure_pareto_chart(protocols_df):
    """Creates a Pareto chart for test failure reasons."""
    failed_df = protocols_df[protocols_df['Failure_Reason'].notna()].copy()
    if failed_df.empty:
        return go.Figure().update_layout(title="Failure Analysis (Pareto)", annotations=[dict(text="No Failures with Reasons Logged", showarrow=False)])
        
    counts = failed_df['Failure_Reason'].value_counts().reset_index()
    counts.columns = ['Reason', 'Count']
    counts = counts.sort_values(by='Count', ascending=False)
    counts['Cumulative Pct'] = (counts['Count'].cumsum() / counts['Count'].sum()) * 100

    fig = go.Figure()
    # Bar chart for counts
    fig.add_trace(go.Bar(x=counts['Reason'], y=counts['Count'], name='Failure Count', marker_color='cornflowerblue'))
    # Line chart for cumulative percentage
    fig.add_trace(go.Scatter(x=counts['Reason'], y=counts['Cumulative Pct'], name='Cumulative %', yaxis='y2',
                             line=dict(color='red', width=2)))
    
    fig.update_layout(
        title="Root Causes of Test Failure (Pareto Analysis)",
        xaxis_title="Failure Reason",
        yaxis_title="Number of Occurrences",
        yaxis2=dict(title="Cumulative Percentage (%)", overlaying='y', side='right', range=[0, 105], showgrid=False),
        legend=dict(x=0.01, y=0.98),
        barmode='group'
    )
    return fig

def create_team_utilization_chart(team_df):
    """Creates a bar chart showing team capacity utilization."""
    team_df['Utilization'] = (team_df['Assigned_Hrs'] / team_df['Capacity (hrs/wk)']) * 100
    
    def get_color(util):
        if util > 100: return 'crimson'
        if util > 90: return 'orange'
        return 'green'
        
    team_df['Color'] = team_df['Utilization'].apply(get_color)

    fig = px.bar(team_df, x='Member', y='Utilization', title='Team Capacity Utilization',
                 text=team_df['Utilization'].apply(lambda x: f'{x:.0f}%'))
    fig.update_traces(marker_color=team_df['Color'], textposition='outside')
    fig.add_hline(y=100, line_dash="dot", line_color="red", annotation_text="Max Capacity", annotation_position="bottom right")
    fig.update_layout(yaxis_title="Utilization (%)", yaxis_range=[0, team_df['Utilization'].max() * 1.2])
    return fig
