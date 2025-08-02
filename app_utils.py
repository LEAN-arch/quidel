# app_utils.py
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
import io
from scipy import stats

# --- Custom Plotly Template ---
quidelortho_template = {"layout": {"font": {"family": "Arial", "size": 12}}}
pio.templates["quidelortho_sme"] = go.layout.Template(quidelortho_template)
pio.templates.default = "quidelortho_sme"

# === CORE DATA GENERATION ===
def generate_vv_project_data():
    data = {
        'Project/Assay': ['Savanna® RVP12 Assay', 'Sofia® 2 SARS Antigen+ FIA v2', 'Vitros® HIV Combo 5 Test', 'Automated Filler & Capper Line 3', 'Ortho-Vision® Analyzer SW Patch V&V'],
        'Type': ['Assay', 'Assay', 'Assay', 'Equipment', 'Software'], 'Platform': ['Savanna', 'Sofia', 'Vitros', 'Manufacturing', 'Ortho-Vision'],
        'V&V Lead': ['M. Rodriguez', 'J. Chen', 'S. Patel', 'V. Kumar', 'S. Patel'], 'V&V Phase': ['Execution', 'Reporting', 'PQ', 'IQ/OQ', 'Data Analysis'],
        'Overall Status': ['On Track', 'On Track', 'At Risk', 'On Track', 'Behind Schedule'], 'Regulatory Pathway': ['510(k)', 'EUA Modification', 'PMA Supplement', 'N/A', 'Letter to File'],
        'Start Date': [date.today() - timedelta(days=60), date.today() - timedelta(days=90), date.today() - timedelta(days=10), date.today() - timedelta(days=45), date.today() - timedelta(days=30)],
        'Due Date': [date.today() + timedelta(days=45), date.today() + timedelta(days=5), date.today() + timedelta(days=60), date.today() + timedelta(days=75), date.today() + timedelta(days=20)],
        'On Critical Path': [False, True, True, True, False], 'Budget (USD)': [250000, 150000, 500000, 1200000, 75000],
        'Spent (USD)': [150000, 135000, 350000, 450000, 40000], 'Schedule Performance Index (SPI)': [1.05, 1.10, 0.92, 1.02, 0.98],
        'First Time Right %': [98, 99, 95, 92, 100], 'Utilization %': [85, 90, 110, 100, 75]
    }
    df = pd.DataFrame(data)
    df['Start Date'] = pd.to_datetime(df['Start Date']); df['Due Date'] = pd.to_datetime(df['Due Date'])
    return df

def generate_risk_management_data():
    data = {
        'Risk ID': ['R-SAV-001', 'R-EQP-001', 'R-VIT-002', 'R-SFT-001'],
        'Project': ['Savanna® RVP12 Assay', 'Automated Filler & Capper Line 3', 'Vitros® HIV Combo 5 Test', 'Ortho-Vision® Analyzer SW Patch V&V'],
        'Risk Description': ['Cross-reactivity risk.', 'Capper torque inconsistency.', 'Biotin interference.', 'Software regression bug.'],
        'Severity': [4, 5, 5, 4], 'Probability': [3, 3, 2, 2], 'Owner': ['R&D/V&V', 'Automation', 'Clinical/V&V', 'SW V&V']
    }
    df = pd.DataFrame(data); df['Risk_Score'] = df['Severity'] * df['Probability']
    return df.sort_values(by='Risk_Score', ascending=False)

def generate_linearity_data_immunoassay():
    expected_conc = np.array([0, 10, 50, 100, 250, 500]); od_values = 2.5 * expected_conc / (50 + expected_conc)
    observed_od = od_values + np.random.normal(0, 0.03, expected_conc.shape)
    return pd.DataFrame({'Analyte Concentration (ng/mL)': expected_conc, 'Optical Density (OD)': observed_od})

def generate_precision_data_clsi_ep05():
    data = []
    for day in [f'Day {i}' for i in range(1, 6)]:
        for _ in range(2 * 3):
            data.append({'Control': 'Low Positive', 'Day': day, 'S/CO Ratio': np.random.normal(1.8, 0.2)})
            data.append({'Control': 'Moderate Positive', 'Day': day, 'S/CO Ratio': np.random.normal(8.5, 0.8)})
    return pd.DataFrame(data)

def generate_method_comparison_data():
    ref = np.random.uniform(10, 500, 100); cand = 1.05 * ref - 5 + np.random.normal(0, 15, 100)
    return pd.DataFrame({'Reference Method (U/L)': ref, 'Candidate Method (U/L)': cand})

def generate_equipment_pq_data():
    data = []
    for i in range(1, 11):
        mean_fill = 5.02 if i < 7 else 5.08
        for val in np.random.normal(loc=mean_fill, scale=0.05, size=30): data.append({'Batch': f'PQ_Batch_{i}', 'Fill Volume (mL)': val})
    return pd.DataFrame(data)

def generate_traceability_matrix_data():
    return pd.DataFrame({'Requirement ID': ['URS-001', 'SRS-010', 'RISK-CTRL-005'], 'Test Case ID': ['TC-SENS-01-01', 'TC-FLAG-01-03', 'TC-INTER-01-01'], 'Test Result': ['Pass', 'Fail', 'Pass']})

def generate_risk_burndown_data():
    return pd.DataFrame({'Week': [1, 2, 3, 4, 5], 'High': [5, 4, 3, 2, 1], 'Medium': [8, 9, 10, 10, 10], 'Low': [10, 11, 12, 13, 14]})

def generate_change_control_data():
    return pd.DataFrame({'ECO': ['ECO-101', 'ECO-102', 'ECO-105'], 'Product Impacted': ['Sofia® RSV', 'QuickVue® Flu', 'Sofia® RSV'], 'Status': ['Awaiting V&V Plan', 'V&V in Progress', 'V&V Complete']})

def generate_analytical_specificity_data_molecular():
    return pd.DataFrame({'Test Type': ['Inclusivity', 'Exclusivity'], 'Organism': ['SARS-CoV-2', 'MERS-CoV'], 'Result': ['Positive', 'Negative']})

def generate_lot_to_lot_data():
    df1 = pd.DataFrame({'Reagent Lot ID': 'Reference', 'Test Line Intensity': np.random.normal(100, 10, 30)})
    df2 = pd.DataFrame({'Reagent Lot ID': 'Candidate', 'Test Line Intensity': np.random.normal(102, 10, 30)})
    return pd.concat([df1, df2])

def calculate_equivalence(df, group_col, value_col, low, high):
    groups = df[group_col].unique(); data1 = df[df[group_col] == groups[0]][value_col]; data2 = df[df[group_col] == groups[1]][value_col]
    mean_diff = data2.mean() - data1.mean(); p_tost = 0.01 if low < mean_diff < high else 0.99
    return p_tost, 0.5, mean_diff

def generate_defect_trend_data():
    dates = pd.date_range(start="2023-01-01", periods=12, freq='W'); found = np.cumsum(np.random.randint(1, 5, 12))
    open_defects = found - np.cumsum(np.random.randint(0, 4, 12))
    return pd.DataFrame({'Date': dates, 'Total Defects Found': found, 'Open Defects': open_defects.clip(lower=0)})

def generate_submission_package_data(project_name, pathway):
    return pd.DataFrame({'Deliverable': ['V&V Master Plan', 'Risk File', 'RTM', 'V&V Summary Report'], 'Status': ['Approved', 'Approved', 'In Review', 'Drafting'], 'Progress': [100, 100, 85, 15], 'Regulatory_Impact': ['High', 'High', 'High', 'High'], 'Statistical_Robustness': [10, 9, 8, 9]})

def generate_capa_data():
    return pd.DataFrame({'ID': ['CAPA-23-011', 'INV-24-005'], 'Source': ['Complaint Trend', 'V&V Anomaly'], 'Due Date': [date.today() - timedelta(days=10), date.today() + timedelta(days=18)]})

def generate_validation_program_data():
    data = {'System_ID': ['EQP-00123', 'SW-0010', 'EQP-00789'], 'System_Name': ['Filler Line 3', 'LIMS v3.2', 'Lyophilizer #2'], 'Validation_Status': ['Validated', 'Validated', 'Revalidation Due'], 'Last_Validation_Date': [date(2022, 5, 20), date(2023, 1, 10), date(2021, 8, 30)], 'Next_Revalidation_Date': [date(2025, 5, 20), date(2026, 1, 10), date(2024, 8, 30)]}
    df = pd.DataFrame(data); df['Next_Revalidation_Date'] = pd.to_datetime(df['Next_Revalidation_Date'])
    df['Days_Until_Due'] = (df['Next_Revalidation_Date'] - pd.to_datetime(date.today())).dt.days
    df.loc[df['Days_Until_Due'] < 0, 'Validation_Status'] = 'Revalidation Overdue'
    return df

def generate_instrument_schedule_data():
    today=date.today(); return pd.DataFrame({'Instrument': ['Savanna-01', 'Sofia-03', 'Vitros-02'], 'Start': [today, today, today], 'Finish': [today + timedelta(days=1), today + timedelta(days=5), today + timedelta(days=2)], 'Status': ['V&V Execution', 'V&V Execution', 'OOS']})

def generate_training_data_for_heatmap():
    return pd.DataFrame({'Savanna': [2, 1, 0], 'Sofia': [1, 2, 1], 'JMP/R': [2, 1, 1]}, index=['M. Rodriguez', 'J. Chen', 'S. Patel'])

def generate_reagent_lot_status_data():
    return pd.DataFrame({'Material': ['RVP12 Master Mix', 'HIV Control Kit'], 'Lot': ['MM-24A01', 'CTRL-22C09'], 'Expiry': [date.today() + timedelta(days=180), date.today() - timedelta(days=10)], 'Status': ['Qualified', 'Expired']})

def calculate_instrument_utilization(schedule_df):
    return schedule_df

def generate_idp_data():
    return pd.DataFrame({'Team Member': ['J. Chen', 'S. Patel'], 'Development Goal': ['Expert in JMP/R', 'Practitioner on Vitros'], 'Mentor': ['M. Rodriguez', 'Manager'], 'Start Date': [date.today(), date.today()], 'Target Date': [date.today() + timedelta(days=150), date.today() + timedelta(days=90)]})

def generate_process_excellence_data():
    months = pd.date_range(start="2023-01-01", periods=18, freq='M')
    return pd.DataFrame({'Month': months, 'Protocol_Approval_Cycle_Time_Days': 45 - np.arange(18) * 0.5 + np.random.normal(0, 2, 18), 'Report_Rework_Rate_Percent': 8 + np.sin(np.linspace(0, 2*np.pi, 18)) * 2, 'Deviations_per_100_Test_Hours': 3 - np.arange(18) * 0.05})

def generate_workload_forecast_data():
    ds = pd.date_range(start='2022-01-01', periods=24, freq='MS'); trend = np.linspace(500, 800, 24)
    seasonality = 100 * np.sin(np.linspace(0, 4 * np.pi, 24)); y = trend + seasonality + np.random.normal(0, 50, 24)
    return pd.DataFrame({'ds': ds, 'y': y.clip(lower=100)})

def generate_monthly_review_ppt(kpi_data, fig_timeline, fig_risk):
    pres = Presentation(); pres.slide_width = Inches(10); pres.slide_height = Inches(7.5)
    # Title Slide
    title_slide = pres.slides.add_slide(pres.slide_layouts[0]); title_slide.shapes.title.text = "Validation Monthly Management Review"; title_slide.placeholders[1].text = f"Status as of: {date.today().strftime('%B %d, %Y')}"
    # KPI slide
    kpi_slide = pres.slides.add_slide(pres.slide_layouts[5]); kpi_slide.shapes.title.text = "Executive Validation Portfolio Health"
    # Chart slide
    chart_slide = pres.slides.add_slide(pres.slide_layouts[6]); chart_slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(9), Inches(0.8)).text_frame.add_paragraph().text = "Portfolio Timeline & Critical Path"
    img_bytes = pio.to_image(fig_timeline, format="png", width=800, height=450, scale=2); chart_slide.shapes.add_picture(io.BytesIO(img_bytes), Inches(1), Inches(1.2), width=Inches(8))
    ppt_stream = io.BytesIO(); pres.save(ppt_stream); ppt_stream.seek(0)
    return ppt_stream
