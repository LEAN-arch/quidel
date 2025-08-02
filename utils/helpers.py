# utils/helpers.py (Final Guaranteed Version)

import streamlit as st
import yaml
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from prophet import Prophet
from datetime import datetime
from functools import wraps
from database import SessionLocal, User, AuditLog
import io
import zipfile

@st.cache_data
def load_config():
    with open("config.yml", 'r') as f: return yaml.safe_load(f)

def get_current_user():
    if 'username' in st.session_state:
        db = SessionLocal(); user = db.query(User).filter(User.username == st.session_state.username).first(); db.close(); return user
    return None

def role_required(required_role: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            user = get_current_user(); roles = {"viewer": 1, "engineer": 2, "director": 3}
            if user and roles.get(user.role, 0) >= roles.get(required_role, 0): return func(*args, **kwargs)
            else: st.error("🚫 Access Denied: You do not have the required permissions for this section."); st.stop()
        return wrapper
    return decorator

def log_action(user_id: int, action: str, details: str = "", record_type: str = None, record_id: int = None):
    db = SessionLocal(); new_log = AuditLog(user_id=user_id, action=action, details=details, record_type=record_type, record_id=record_id)
    db.add(new_log); db.commit(); db.close()

def render_login():
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if not st.session_state.logged_in:
        config = load_config(); st.title(f"{config['app_name']} Login")
        username = st.text_input("Username (e.g., director, alice, charlie)")
        if st.button("Login", type="primary"):
            if not username: st.warning("Please enter a username."); return
            db = SessionLocal(); user = db.query(User).filter(User.username == username.lower()).first(); db.close()
            if user:
                st.session_state.logged_in = True; st.session_state.user_id = user.id; st.session_state.username = user.username
                st.session_state.full_name = user.full_name; st.session_state.role = user.role
                log_action(user.id, "User login successful."); st.rerun()
            else: st.error("Invalid username. Please try again.")

def analyze_precision(data_df):
    if 'Value' not in data_df.columns: return None, "Error: 'Value' column not found in uploaded data."
    mean_val=data_df['Value'].mean(); std_val=data_df['Value'].std(); cv_val=(std_val/mean_val)*100 if mean_val!=0 else 0
    results={'N':len(data_df),'Mean':f"{mean_val:.2f}",'Std Dev':f"{std_val:.2f}",'CV (%)':f"{cv_val:.2f}"}
    fig=px.box(data_df,y='Value',title='Precision Data Distribution',points='all'); return results,fig

def analyze_linearity(data_df):
    if 'Expected' not in data_df.columns or 'Observed' not in data_df.columns: return None, "Error: 'Expected' and 'Observed' columns not found."
    model = smf.ols('Observed ~ Expected', data=data_df).fit(); results = {'N':len(data_df),'Slope':f"{model.params['Expected']:.4f}",'Intercept':f"{model.params['Intercept']:.4f}",'R-squared':f"{model.rsquared:.4f}"}
    fig = px.scatter(data_df, x='Expected', y='Observed', title='Linearity Plot', trendline='ols', trendline_color_override='red'); return results, fig

def create_enhanced_gantt(df):
    gantt_df=df.copy();gantt_df.rename(columns={'name':'Task','pct_complete':'Completion_pct', 'start_date':'Start', 'finish_date':'Finish'},inplace=True);gantt_df['Completion_pct']=gantt_df['Completion_pct']/100;fig=px.timeline(gantt_df,x_start="Start",x_end="Finish",y="Task",color="status",custom_data=['owner_id','Completion_pct'],title="Project Portfolio Status & Progress");fig.update_traces(text=gantt_df['Completion_pct'].apply(lambda x:f'{x:.0%}'),textposition='inside');fig.update_layout(xaxis_title="Timeline",yaxis_title="Project");return fig

def create_risk_bubble_chart(risk_df):
    fig=px.scatter(risk_df,x="Severity",y="Occurrence",size="RPN",color="Project",hover_name="Failure_Mode",size_max=60,title="Product Risk Landscape (FMEA)",labels={"Severity":"Severity (Impact)","Occurrence":"Likelihood of Occurrence"});fig.add_shape(type="rect",x0=6.5,y0=3.5,x1=10,y1=10,line=dict(color="Red",width=2,dash="dash"),fillcolor="rgba(255,0,0,0.1)");fig.add_annotation(x=9.5,y=9.5,text="High-Risk Zone",showarrow=False,xanchor='right',yanchor='top');return fig

def generate_text_report(protocol_data, analysis_results):
    """Generates a downloadable .txt summary report."""
    report_content = f"""
=================================================
VERIFICATION & VALIDATION SUMMARY REPORT
=================================================

Protocol ID:   {protocol_data.get('protocol_id_str', 'N/A')}
Title:         {protocol_data.get('title', 'N/A')}
Project:       {protocol_data.get('project_name', 'N/A')}
Status:        {protocol_data.get('status', 'N/A')}

-------------------------------------------------
Acceptance Criteria:
{protocol_data.get('acceptance_criteria', 'N/A')}
-------------------------------------------------

Execution Results:
"""
    for key, value in analysis_results.items():
        report_content += f"- {key}: {value}\n"
    
    report_content += f"""
-------------------------------------------------
Electronically Signed By: {protocol_data.get('signed_by', 'N/A')}
Signature Date (UTC):   {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}
=================================================
"""
    return report_content.encode('utf-8')

def create_submission_zip(project_name, project_reqs, project_protocols, project_risks):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        plan_content=f"Verification & Validation Plan\nProject: {project_name}\n\n(Auto-generated placeholder document)."; zip_file.writestr(f"{project_name}_V&V_Plan.txt", plan_content)
        if not project_risks.empty: zip_file.writestr(f"{project_name}_Risk_Management_File.csv", project_risks.to_csv(index=False))
        if not project_reqs.empty: zip_file.writestr(f"{project_name}_Traceability_Matrix.csv", project_reqs.to_csv(index=False))
        
        executed_protocols = project_protocols[project_protocols['status'].str.contains("Executed", na=False)]
        if not executed_protocols.empty:
            reports_folder = "V&V_Summary_Reports/"
            for _, protocol_row in executed_protocols.iterrows():
                protocol_dict = protocol_row.to_dict()
                mock_results = {'Status': protocol_dict['status'], 'Signed Off By': protocol_dict.get('approver_id', 'N/A')}
                report_content = generate_text_report(protocol_dict, mock_results)
                report_filename = f"{reports_folder}{protocol_dict['protocol_id_str']}_Summary_Report.txt"
                zip_file.writestr(report_filename, report_content)
                
    zip_buffer.seek(0); return zip_buffer
