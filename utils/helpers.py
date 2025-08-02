# utils/helpers.py

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

# --- MOCK DATA GENERATION ---
# In a real application, this would connect to a database (e.g., SQL, Snowflake)

def get_mock_team_data():
    """Generates a mock team dataframe."""
    return pd.DataFrame({
        'Member': ['Alice', 'Bob', 'Charlie', 'Diana', 'Edward'],
        'Role': ['V&V Engineer', 'Sr. V&V Engineer', 'V&V Specialist', 'V&V Engineer', 'Sr. V&V Engineer'],
        'Active_Protocols': [3, 2, 4, 3, 2],
        'Training_Status': ['Compliant', 'Compliant', 'Overdue', 'Compliant', 'Compliant']
    })

def get_mock_projects_data():
    """Generates a mock projects dataframe."""
    today = datetime.now()
    return pd.DataFrame([
        dict(Project="ImmunoPro-A", Start=str(today-timedelta(days=60)), Finish=str(today+timedelta(days=30)), Status='On Track', Owner='Alice'),
        dict(Project="MolecularDX-2", Start=str(today-timedelta(days=90)), Finish=str(today+timedelta(days=10)), Status='At Risk', Owner='Bob'),
        dict(Project="CardioMarker-V", Start=str(today-timedelta(days=20)), Finish=str(today+timedelta(days=90)), Status='On Track', Owner='Charlie'),
        dict(Project="Consumable-X", Start=str(today-timedelta(days=45)), Finish=str(today+timedelta(days=45)), Status='Delayed', Owner='Diana'),
        dict(Project="OncologyPanel-1", Start=str(today-timedelta(days=120)), Finish=str(today+timedelta(days=60)), Status='On Track', Owner='Edward'),
    ])

def get_mock_requirements_data():
    """Generates a mock requirements and traceability matrix."""
    return pd.DataFrame({
        'Req_ID': ['USR-001', 'USR-002', 'FNC-001', 'FNC-002', 'FNC-003', 'SYS-001'],
        'Project': ['ImmunoPro-A', 'ImmunoPro-A', 'ImmunoPro-A', 'MolecularDX-2', 'MolecularDX-2', 'ImmunoPro-A'],
        'Requirement_Type': ['User', 'User', 'Functional', 'Functional', 'Functional', 'System'],
        'Requirement_Text': [
            'The assay shall detect Antigen A with >99% sensitivity.',
            'The assay shall have a specificity of >99%.',
            'The system must provide results within 15 minutes.',
            'The assay shall amplify DNA target region B.',
            'The assay must have a limit of detection of <50 copies/mL.',
            'The software shall display results in a clear format.'
        ],
        'Linked_Protocol_ID': ['IP-PREC-01, IP-SENS-01', 'IP-SPEC-01', 'IP-TTR-01', 'MDX-LOD-01', 'MDX-LOD-01', np.nan],
        'Status': ['Covered', 'Covered', 'Covered', 'Covered', 'Covered', 'Gap']
    })

def get_mock_protocols_data():
    """Generates a mock protocols dataframe."""
    return pd.DataFrame({
        'Protocol_ID': ['IP-PREC-01', 'IP-SENS-01', 'IP-SPEC-01', 'MDX-LOD-01', 'IP-TTR-01'],
        'Project': ['ImmunoPro-A', 'ImmunoPro-A', 'ImmunoPro-A', 'MolecularDX-2', 'ImmunoPro-A'],
        'Title': ['Precision Study (Repeatability)', 'Analytical Sensitivity Study', 'Analytical Specificity Study', 'Limit of Detection', 'Time to Result'],
        'Type': ['Precision', 'Sensitivity', 'Specificity', 'LoD', 'Performance'],
        'Status': ['Executed - Passed', 'Approved', 'Draft', 'Executed - Passed', 'Executed - Failed'],
        'Acceptance_Criteria': ['CV <= 5%', 'Detect 95/100 low positive samples', 'No cross-reactivity with Panel Z', 'LoD < 50 copies/mL', 'Result in < 15 minutes'],
        'Signed_Off_By': ['Alice', np.nan, np.nan, 'Bob', 'Alice'],
        'Sign_Off_Date': [datetime.now()-timedelta(days=5), np.nan, np.nan, datetime.now()-timedelta(days=10), datetime.now()-timedelta(days=2)]
    })

def get_mock_risk_data():
    """Generates a mock FMEA risk dataframe."""
    return pd.DataFrame({
        'Risk_ID': ['R-001', 'R-002', 'R-003', 'R-004'],
        'Project': ['ImmunoPro-A', 'ImmunoPro-A', 'MolecularDX-2', 'MolecularDX-2'],
        'Failure_Mode': ['False Positive due to cross-reactivity', 'Incorrect reagent dispense volume', 'Sample contamination during prep', 'Reagent degradation at room temp'],
        'Severity': [8, 9, 7, 8],
        'Occurrence': [3, 2, 4, 3],
        'Detection': [4, 7, 3, 5],
        'RPN': [96, 126, 84, 120],
        'Mitigation_Action': ['Test with specificity panel', 'Verify dispense volume in OQ', 'Implement new cleaning procedure', 'Conduct real-time stability study'],
        'Linked_Protocol_ID': ['IP-SPEC-01', 'Consumable-OQ-01', 'N/A', 'MDX-STAB-01']
    })

def load_data(source):
    """Dispatcher to load the correct mock data."""
    if 'data_loaded' not in st.session_state:
        st.session_state.projects_df = get_mock_projects_data()
        st.session_state.requirements_df = get_mock_requirements_data()
        st.session_state.protocols_df = get_mock_protocols_data()
        st.session_state.risk_df = get_mock_risk_data()
        st.session_state.team_df = get_mock_team_data()
        st.session_state.audit_log = []
        st.session_state.data_loaded = True
        log_action("SYSTEM", "Initialized mock data sets.")


# --- AUTHENTICATION & LOGGING ---

def render_login():
    """Renders a simple password-based login form."""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.title("AssayVantage Command Center Login")
        with st.form("login_form"):
            st.text_input("Username", value="director", disabled=True)
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                # In a real app, use st.secrets and hash passwords
                if password == "quidel":
                    st.session_state.logged_in = True
                    log_action("director", "User login successful.")
                    st.experimental_rerun()
                else:
                    st.error("Incorrect password")

def log_action(user, action, details=""):
    """Logs an action to the audit trail in session_state."""
    if 'audit_log' not in st.session_state:
        st.session_state.audit_log = []
    
    log_entry = {
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'User': user,
        'Action': action,
        'Details': details
    }
    st.session_state.audit_log.insert(0, log_entry) # Add to the top of the list


# --- PLOTTING & VISUALIZATION FUNCTIONS ---

def create_gantt_chart(df):
    """Creates an interactive Plotly Gantt chart."""
    fig = ff.create_gantt(df, index_col='Status', show_colorbar=True, group_tasks=True,
                          title="Project Portfolio Timelines")
    fig.update_layout(xaxis_title="Date", yaxis_title="Project Status")
    return fig

def create_bar_chart(df, x_col, y_col, title):
    """Creates an interactive Plotly bar chart."""
    fig = px.bar(df, x=x_col, y=y_col, title=title, text_auto=True)
    fig.update_traces(textposition='outside')
    return fig

def create_donut_chart(values, labels, title):
    """Creates a Plotly donut chart."""
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
    fig.update_layout(title_text=title, showlegend=False,
                      annotations=[dict(text=f'{values[0]}%', x=0.5, y=0.5, font_size=20, showarrow=False)])
    return fig


# --- STATISTICAL ANALYSIS FUNCTIONS ---

def analyze_precision(data_df):
    """Performs precision analysis (CV)."""
    if 'Value' not in data_df.columns:
        return None, "Error: 'Value' column not found in uploaded data."
    
    mean = data_df['Value'].mean()
    std = data_df['Value'].std()
    cv = (std / mean) * 100 if mean != 0 else 0
    
    results = {
        'N': len(data_df),
        'Mean': f"{mean:.2f}",
        'Std Dev': f"{std:.2f}",
        'CV (%)': f"{cv:.2f}"
    }
    
    fig = px.box(data_df, y='Value', title='Precision Data Distribution', points='all')
    fig.add_hline(y=mean, line_dash="dot", annotation_text="Mean", annotation_position="bottom right")
    
    return results, fig

def analyze_linearity(data_df):
    """Performs linearity analysis (R^2)."""
    if 'Expected' not in data_df.columns or 'Observed' not in data_df.columns:
        return None, "Error: 'Expected' and 'Observed' columns not found."

    slope, intercept, r_value, p_value, std_err = stats.linregress(data_df['Expected'], data_df['Observed'])
    r_squared = r_value**2

    results = {
        'N': len(data_df),
        'Slope': f"{slope:.4f}",
        'Intercept': f"{intercept:.4f}",
        'R-squared': f"{r_squared:.4f}"
    }

    fig = px.scatter(data_df, x='Expected', y='Observed', title='Linearity Plot',
                     trendline='ols', trendline_color_override='red')
    
    return results, fig


# --- REPORT GENERATION ---

def generate_ppt_report(protocol_data, analysis_results, analysis_fig):
    """Generates a PowerPoint summary report."""
    prs = Presentation()
    
    # Title Slide
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    title.text = "Verification & Validation Summary Report"
    subtitle.text = f"Protocol: {protocol_data['Protocol_ID']} - {protocol_data['Title']}"

    # Summary Slide
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    title.text = "Protocol Summary"
    
    # Define content area
    left = Inches(0.5)
    top = Inches(1.5)
    width = Inches(9.0)
    height = Inches(5.5)
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.clear() 

    p = tf.paragraphs[0]
    p.text = f"Project: {protocol_data['Project']}"
    
    p = tf.add_paragraph()
    p.text = f"Acceptance Criteria: {protocol_data['Acceptance_Criteria']}"
    
    p = tf.add_paragraph()
    p.text = f"Status: {protocol_data['Status']}"
    p.space_before = Inches(0.2)
    
    p = tf.add_paragraph()
    p.text = "Execution Results:"
    p.space_before = Inches(0.5)

    for key, value in analysis_results.items():
        p = tf.add_paragraph()
        p.text = f"  • {key}: {value}"
        p.level = 1

    # Chart Slide
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    title = slide.shapes.title
    title.text = "Graphical Analysis"

    # Save plotly fig to a memory buffer
    img_bytes = io.BytesIO()
    analysis_fig.write_image(img_bytes, format='png', scale=2)
    img_bytes.seek(0)
    
    slide.shapes.add_picture(img_bytes, Inches(1.0), Inches(1.5), width=Inches(8.0))

    # Save presentation to a memory buffer
    ppt_io = io.BytesIO()
    prs.save(ppt_io)
    ppt_io.seek(0)
    
    return ppt_io
