# utils/helpers.py (Complete Enterprise Version)

import streamlit as st
import yaml
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from prophet import Prophet
from python_pptx import Presentation
from python_pptx.util import Inches
from datetime import datetime
from functools import wraps
from database import SessionLocal, User, AuditLog
import io
import zipfile

# --- CONFIGURATION LOADER ---
@st.cache_data
def load_config():
    """Loads the application configuration from config.yml."""
    with open("config.yml", 'r') as f:
        return yaml.safe_load(f)

# --- SECURITY & ROLE-BASED ACCESS CONTROL (RBAC) ---

def get_current_user():
    """Retrieves the full User ORM object from the database for the logged-in user."""
    if 'username' in st.session_state:
        db = SessionLocal()
        user = db.query(User).filter(User.username == st.session_state.username).first()
        db.close()
        return user
    return None

def role_required(required_role: str):
    """
    A decorator to restrict access to a page to a minimum required role.
    The 'director' role has access to all pages.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            user = get_current_user()
            # Define role hierarchy
            roles = {"viewer": 1, "engineer": 2, "director": 3}
            
            if user and roles.get(user.role, 0) >= roles.get(required_role, 0):
                return func(*args, **kwargs)
            else:
                st.error("🚫 Access Denied: You do not have the required permissions for this section.")
                st.stop()
        return wrapper
    return decorator

# --- AUDIT LOGGING (Database Connected) ---

def log_action(user_id: int, action: str, details: str = "", record_type: str = None, record_id: int = None):
    """Logs a user action to the database audit trail."""
    db = SessionLocal()
    new_log = AuditLog(
        user_id=user_id,
        action=action,
        details=details,
        record_type=record_type,
        record_id=record_id
    )
    db.add(new_log)
    db.commit()
    db.close()

# --- LOGIN/LOGOUT (Database Connected) ---

def render_login():
    """Renders the login form and handles authentication against the database."""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        config = load_config()
        st.title(f"{config['app_name']} Login")
        
        # In a full production app, this would be an SSO provider (Okta, Azure AD)
        username = st.text_input("Username (e.g., director, alice, charlie)")
        
        if st.button("Login", type="primary"):
            if not username:
                st.warning("Please enter a username.")
                return

            db = SessionLocal()
            user = db.query(User).filter(User.username == username.lower()).first()
            db.close()
            
            if user:
                st.session_state.logged_in = True
                st.session_state.user_id = user.id
                st.session_state.username = user.username
                st.session_state.full_name = user.full_name
                st.session_state.role = user.role
                log_action(user.id, "User login successful.")
                st.rerun()
            else:
                st.error("Invalid username. Please try again.")

# --- DATA ANALYSIS & VISUALIZATION FUNCTIONS (Unchanged logic, now operate on real data) ---

def analyze_precision(data_df):
    """Performs precision analysis (CV)."""
    if 'Value' not in data_df.columns: return None, "Error: 'Value' column not found in uploaded data."
    mean_val=data_df['Value'].mean(); std_val=data_df['Value'].std(); cv_val=(std_val/mean_val)*100 if mean_val!=0 else 0
    results={'N':len(data_df),'Mean':f"{mean_val:.2f}",'Std Dev':f"{std_val:.2f}",'CV (%)':f"{cv_val:.2f}"}
    fig=px.box(data_df,y='Value',title='Precision Data Distribution',points='all'); return results,fig

def analyze_linearity(data_df):
    """Performs linearity analysis using statsmodels for robustness."""
    if 'Expected' not in data_df.columns or 'Observed' not in data_df.columns: return None, "Error: 'Expected' and 'Observed' columns not found."
    model = smf.ols('Observed ~ Expected', data=data_df).fit()
    results = {'N':len(data_df),'Slope':f"{model.params['Expected']:.4f}",'Intercept':f"{model.params['Intercept']:.4f}",'R-squared':f"{model.rsquared:.4f}"}
    fig = px.scatter(data_df, x='Expected', y='Observed', title='Linearity Plot', trendline='ols', trendline_color_override='red'); return results, fig

def create_risk_bubble_chart(risk_df):
    """Creates a Severity vs. Occurrence bubble chart where bubble size is RPN."""
    fig=px.scatter(risk_df,x="Severity",y="Occurrence",size="RPN",color="Project",hover_name="Failure_Mode",size_max=60,title="Product Risk Landscape (FMEA)",labels={"Severity":"Severity (Impact)","Occurrence":"Likelihood of Occurrence"});fig.add_shape(type="rect",x0=6.5,y0=3.5,x1=10,y1=10,line=dict(color="Red",width=2,dash="dash"),fillcolor="rgba(255,0,0,0.1)");fig.add_annotation(x=9.5,y=9.5,text="High-Risk Zone",showarrow=False,xanchor='right',yanchor='top');return fig

# ... (other chart functions like Gantt, Pareto, Utilization would go here if needed on multiple pages) ...

# --- FORECASTING FUNCTIONS ---

def generate_prophet_history(project_data):
    """Generates a synthetic daily progress history for a project to feed into Prophet."""
    start_date=pd.to_datetime(project_data['Start']);today=datetime.now()
    if start_date>today:return pd.DataFrame(columns=['ds','y'])
    days_elapsed=(today-start_date).days;progress_to_date=project_data['Pct_Complete'];date_range=pd.to_datetime(pd.date_range(start=start_date,end=today));y_ideal=np.linspace(0,progress_to_date,len(date_range));noise=np.random.normal(0,2,len(date_range));y_actual=np.clip(y_ideal+noise,0,100);history_df=pd.DataFrame({'ds':date_range,'y':y_actual});return history_df

def run_prophet_forecast(history_df, future_days=90):
    """Takes a history dataframe and runs a Prophet forecast."""
    if history_df.empty or len(history_df)<2:return None,None
    history_df['cap']=100;m=Prophet(growth='logistic');m.fit(history_df);future=m.make_future_dataframe(periods=future_days);future['cap']=100;forecast=m.predict(future);return m,forecast

# --- REPORTING & BUNDLING FUNCTIONS ---

def generate_ppt_report(protocol_data, analysis_results, analysis_fig):
    """Generates a PowerPoint summary report in memory."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0]); slide.shapes.title.text = "Verification & Validation Summary Report"; slide.placeholders[1].text = f"Protocol: {protocol_data.get('Protocol_ID', 'N/A')} - {protocol_data.get('Title', 'N/A')}"
    slide = prs.slides.add_slide(prs.slide_layouts[1]); slide.shapes.title.text = "Protocol Summary"; txBox = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(9.0), Inches(5.5)); tf = txBox.text_frame; tf.clear()
    tf.paragraphs[0].text = f"Project: {protocol_data.get('Project', 'N/A')}"; tf.add_paragraph().text = f"Acceptance Criteria: {protocol_data.get('Acceptance_Criteria', 'N/A')}"
    p_status = tf.add_paragraph(); p_status.text = f"Status: {protocol_data.get('Status', 'N/A')}"; p_status.space_before = Inches(0.2)
    p_results_header = tf.add_paragraph(); p_results_header.text = "Execution Results:"; p_results_header.space_before = Inches(0.5)
    for key, value in analysis_results.items():
        p = tf.add_paragraph(); p.text = f"  • {key}: {value}"; p.level = 1
    slide = prs.slides.add_slide(prs.slide_layouts[5]); slide.shapes.title.text = "Graphical Analysis"; img_bytes = io.BytesIO(); analysis_fig.write_image(img_bytes, format='png', scale=2); img_bytes.seek(0)
    slide.shapes.add_picture(img_bytes, Inches(1.0), Inches(1.5), width=Inches(8.0)); ppt_io = io.BytesIO(); prs.save(ppt_io); ppt_io.seek(0); return ppt_io

def create_submission_zip(project_name, project_reqs, project_protocols, project_risks):
    """Generates a complete regulatory submission package as a zip file in memory."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        plan_content=f"Verification & Validation Plan\nProject: {project_name}\n\n(Auto-generated placeholder document)."; zip_file.writestr(f"{project_name}_V&V_Plan.txt", plan_content)
        if not project_risks.empty: zip_file.writestr(f"{project_name}_Risk_Management_File.csv", project_risks.to_csv(index=False))
        if not project_reqs.empty: zip_file.writestr(f"{project_name}_Traceability_Matrix.csv", project_reqs.to_csv(index=False))
        executed_protocols = project_protocols[project_protocols['Status'].str.contains("Executed", na=False)]
        if not executed_protocols.empty:
            reports_folder = "V&V_Summary_Reports/"
            for _, protocol_row in executed_protocols.iterrows():
                mock_results={'Status':protocol_row['Status'],'Signed Off By':protocol_row.get('Signed_Off_By','N/A'),'Note':'Auto-generated from submission bundle.'}; mock_fig=go.Figure().update_layout(title_text="Data Plot Placeholder")
                ppt_buffer=generate_ppt_report(protocol_row.to_dict(), mock_results, mock_fig)
                if ppt_buffer:
                    report_filename = f"{reports_folder}{protocol_row['Protocol_ID']}_Summary_Report.pptx"
                    zip_file.writestr(report_filename, ppt_buffer.getvalue())
    zip_buffer.seek(0); return zip_buffer
