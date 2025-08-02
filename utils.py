# utils.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import date, timedelta
from scipy import stats

# --- Custom Plotly Template for QuidelOrtho (SME Grade) ---
quidelortho_template = {
    "layout": {
        "font": {"family": "Arial, Helvetica, sans-serif", "size": 12, "color": "#212529"},
        "title": {"font": {"family": "Arial, Helvetica, sans-serif", "size": 18, "color": "#0039A6"}, "x": 0.05, "xanchor": "left"},
        "plot_bgcolor": "#FFFFFF",
        "paper_bgcolor": "#FFFFFF",
        "colorway": ['#0039A6', '#00AEEF', '#FFC72C', '#F47321', '#7F8C8D', '#DC3545', '#28A745'],
        "xaxis": {"gridcolor": "#E9ECEF", "linecolor": "#CED4DA", "zerolinecolor": "#E9ECEF", "title_font": {"size": 14, "family": "Arial, Helvetica, sans-serif"}},
        "yaxis": {"gridcolor": "#E9ECEF", "linecolor": "#CED4DA", "zerolinecolor": "#E9ECEF", "title_font": {"size": 14, "family": "Arial, Helvetica, sans-serif"}},
        "legend": {"bgcolor": "rgba(255,255,255,0.9)", "bordercolor": "#DEE2E6", "borderwidth": 1, "orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1}
    }
}
pio.templates["quidelortho_sme"] = quidelortho_template
pio.templates.default = "quidelortho_sme"

# === CORE DATA GENERATION (SME ENHANCED) ===

def generate_vv_project_data():
    """Generates enhanced V&V and Validation project data with financial and performance metrics."""
    data = {
        'Project/Assay': [
            'Savanna® RVP12 Assay', 'Sofia® 2 SARS Antigen+ FIA v2', 'Vitros® HIV Combo 5 Test',
            'Triage® BNPNext™ Consumable Change', 'Automated Filler & Capper Line 3', 'Ortho-Vision® Analyzer SW Patch V&V', 'New Blister Pack Sealer Unit 7'
        ],
        'Type': ['Assay', 'Assay', 'Assay', 'Assay', 'Equipment', 'Software', 'Equipment'],
        'Platform': ['Savanna', 'Sofia', 'Vitros', 'Triage', 'Manufacturing', 'Ortho-Vision', 'Packaging'],
        'V&V Lead': ['M. Rodriguez', 'J. Chen', 'S. Patel', 'M. Rodriguez', 'V. Kumar', 'S. Patel', 'V. Kumar'],
        'V&V Phase': ['Execution', 'Reporting', 'PQ', 'On Hold', 'IQ/OQ', 'Data Analysis', 'FAT/SAT'],
        'Overall Status': ['On Track', 'On Track', 'At Risk', 'On Hold', 'On Track', 'On Track', 'Behind Schedule'],
        'Regulatory Pathway': ['510(k)', 'EUA Modification', 'PMA Supplement', 'Letter to File', 'N/A', 'Letter to File', 'N/A'],
        'Key Milestone': ['Final Report for Submission', 'EUA Submission', 'Production Readiness', 'N/A', 'OQ Completion', 'V&V Report for ECO', 'SAT Completion'],
        'Start Date': [date.today() - timedelta(days=60), date.today() - timedelta(days=90), date.today() - timedelta(days=10), date.today() - timedelta(days=120), date.today() - timedelta(days=45), date.today() - timedelta(days=30), date.today() - timedelta(days=90)],
        'Due Date': [date.today() + timedelta(days=45), date.today() + timedelta(days=5), date.today() + timedelta(days=60), date.today() + timedelta(days=15), date.today() + timedelta(days=75), date.today() + timedelta(days=20), date.today() + timedelta(days=30)],
        'On Critical Path': [False, False, True, False, True, False, True],
        'Budget (USD)': [250000, 150000, 500000, 50000, 1200000, 75000, 300000],
        'Spent (USD)': [150000, 135000, 350000, 20000, 450000, 40000, 250000],
        'Schedule Performance Index (SPI)': [1.05, 1.10, 0.92, 1.0, 1.02, 0.98, 0.85],
        'First Time Right %': [98, 99, 95, 100, 92, 100, 88],
        'Utilization %': [85, 90, 110, 0, 100, 75, 120]
    }
    df = pd.DataFrame(data)
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    df['Due Date'] = pd.to_datetime(df['Due Date'])
    return df

def generate_risk_management_data():
    """Generates enhanced risk data for both assay and equipment validation."""
    data = {
        'Risk ID': ['R-SAV-001', 'R-EQP-001', 'R-VIT-002', 'R-TRI-001', 'R-GEN-005', 'R-SFT-001'],
        'Project': [
            'Savanna® RVP12 Assay', 'Automated Filler & Capper Line 3', 'Vitros® HIV Combo 5 Test',
            'Triage® BNPNext™ Consumable Change', 'Savanna® RVP12 Assay', 'Ortho-Vision® Analyzer SW Patch V&V'
        ],
        'Risk Description': [
            'Potential cross-reactivity with emerging non-pathogenic coronavirus strains could lead to false positives.',
            'New automated capper applies inconsistent torque, potentially compromising container closure integrity and product sterility.',
            'Undetected interference from biotin could lead to a false negative result, delaying critical diagnosis.',
            'New plastic supplier for consumable has higher lot-to-lot variability, potentially causing inconsistent test results.',
            'V&V study execution timeline conflicts with key personnel availability, posing a risk of delayed regulatory submission.',
            'Regression testing for software patch fails to cover a legacy edge-case, potentially re-introducing a previously fixed bug.'
        ],
        'Severity': [4, 5, 5, 3, 2, 4],
        'Probability': [3, 3, 2, 3, 4, 2],
        'Owner': ['R&D/V&V', 'Automation/Validation', 'Clinical/V&V', 'Supply Chain/V&V', 'V&V Management', 'SW V&V Team'],
        'Mitigation': [
            'Test against a comprehensive panel of endemic coronaviruses per FDA guidance.',
            'Incorporate torque verification and container closure integrity testing (CCIT) into OQ and PQ protocols.',
            'Include biotin and other interferents in testing per CLSI EP07. Add limitation to Instructions for Use (IFU).',
            'Require tighter Certificate of Analysis (CoA) specs from supplier; perform incoming QC on each lot.',
            'Re-allocate V&V specialist from a lower priority project. Document change in project plan.',
            'Expand regression test suite to include historical defect scenarios.'
        ]
    }
    df = pd.DataFrame(data)
    df['Risk_Score'] = df['Severity'] * df['Probability']
    return df.sort_values(by='Risk_Score', ascending=False)

# === V&V/VALIDATION STUDY DATA GENERATION ===

def generate_linearity_data_immunoassay():
    """Generates realistic linearity data for a quantitative immunoassay."""
    expected_conc = np.array([0, 10, 50, 100, 250, 500, 750, 1000])
    od_values = 2.5 * expected_conc / (50 + expected_conc)
    observed_od = od_values + np.random.normal(0, 0.03, expected_conc.shape)
    observed_od[0] = np.random.uniform(0.04, 0.06)
    return pd.DataFrame({'Analyte Concentration (ng/mL)': expected_conc, 'Optical Density (OD)': observed_od})

def generate_precision_data_clsi_ep05():
    """Generates precision data with nuance for a Sofia® FIA assay."""
    days = [f'Day {i}' for i in range(1, 6)]; runs_per_day = 2; reps_per_run = 3; data = []
    for day_idx, day in enumerate(days):
        day_effect = np.random.normal(0, 0.08)
        for run in range(1, runs_per_day + 1):
            run_effect = np.random.normal(0, 0.05)
            mean_s_co = 1.8 + day_effect + run_effect
            stdev_within_run = 0.12
            values = np.random.normal(loc=mean_s_co, scale=stdev_within_run, size=reps_per_run)
            for val in values: data.append({'Control': 'Low Positive (Near Cutoff)', 'Day': day, 'Run': run, 'S/CO Ratio': val})
    for day_idx, day in enumerate(days):
        day_effect = np.random.normal(0, 0.2)
        for run in range(1, runs_per_day + 1):
            run_effect = np.random.normal(0, 0.1)
            mean_s_co = 8.5 + day_effect + run_effect
            stdev_within_run = 0.4
            values = np.random.normal(loc=mean_s_co, scale=stdev_within_run, size=reps_per_run)
            for val in values: data.append({'Control': 'Moderate Positive', 'Day': day, 'Run': run, 'S/CO Ratio': val})
    return pd.DataFrame(data)

def generate_method_comparison_data():
    """Generates data for a Bland-Altman plot."""
    np.random.seed(123)
    n_samples = 100
    reference_values = np.random.uniform(10, 500, n_samples)
    new_method_values = 1.05 * reference_values - 5 + np.random.normal(0, 15, n_samples)
    return pd.DataFrame({'Reference Method (U/L)': reference_values, 'Candidate Method (U/L)': new_method_values})

def generate_equipment_pq_data():
    """Generates PQ data for an automated filler, including a process shift."""
    np.random.seed(42)
    batches = 10
    samples_per_batch = 30
    data = []
    for i in range(batches):
        mean_fill = 5.02
        if i >= 6: mean_fill = 5.08 # Process shift
        std_dev = 0.05
        values = np.random.normal(loc=mean_fill, scale=std_dev, size=samples_per_batch)
        for val in values:
            data.append({'Batch': f'PQ_Batch_{i+1}', 'Fill Volume (mL)': val})
    return pd.DataFrame(data)


# === REGULATORY & QMS DATA GENERATION ===

def generate_submission_package_data(project_name="Savanna® RVP12 Assay", pathway="510(k)"):
    """Generates a detailed checklist of V&V deliverables for a regulatory submission package."""
    docs = {
        "510(k)": [
            ("V&V Master Plan", "DHF-001", "Approved", "High", 10),
            ("Product Risk Management File (ISO 14971)", "DHF-002", "Approved", "High", 10),
            ("Requirements Traceability Matrix (RTM)", "DHF-003", "Approved", "High", 9),
            ("Software V&V Summary Report", "DHF-005", "Approved", "Medium", 8),
            ("Analytical Sensitivity (LoD) Study Report", "V&V-RPT-010", "Approved", "Medium", 7),
            ("Analytical Specificity (Cross-reactivity) Report", "V&V-RPT-011", "In Review", "High", 9),
            ("Precision/Reproducibility Study Report (CLSI EP05)", "V&V-RPT-012", "In Review", "Low", 5),
            ("Interference Study Report (CLSI EP07)", "V&V-RPT-013", "Data Analysis", "Medium", 6),
            ("Reagent & Consumable Stability Report", "V&V-RPT-014", "Execution", "High", 8),
            ("V&V Master Summary Report", "DHF-020", "Drafting", "High", 9)
        ]
    }
    data = docs.get(pathway, docs["510(k)"])
    df = pd.DataFrame(data, columns=['Deliverable', 'Document ID', 'Status', 'Regulatory_Impact', 'Statistical_Robustness'])
    status_progress_map = {'Drafting': 15, 'Execution': 40, 'Data Analysis': 65, 'In Review': 85, 'Approved': 100}
    df['Progress'] = df['Status'].map(status_progress_map)
    return df

def generate_capa_data():
    """Generates data for V&V-related CAPAs and investigations."""
    data = {
        'ID': ['CAPA-23-011', 'INV-24-005', 'CAPA-24-002', 'INV-24-006'],
        'Source': ['Post-Launch Complaint Trend', 'V&V Study Anomaly', 'Internal Audit Finding', 'Contract Lab Deviation'],
        'Product': ['Sofia® RSV', 'Savanna® RVP12 Assay', 'All Assay Platforms', 'Vitros® HIV Combo 5 Test'],
        'Description': ['Confirmed increase in reported false positives from EU region after lot change.', 'Unexpected weak positive signal for hMPV in Flu A channel during specificity testing.', 'Inconsistent documentation of V&V protocol deviations across multiple projects.', 'External CRO reported temperature excursion during stability sample storage.'],
        'Owner': ['V&V Mgmt / R&D', 'M. Rodriguez', 'QA / V&V Mgmt', 'S. Patel'],
        'Phase': ['Effectiveness Check', 'Root Cause Investigation', 'Implementation', 'Impact Assessment'],
        'Due Date': [date.today() - timedelta(days=10), date.today() + timedelta(days=18), date.today() + timedelta(days=40), date.today() + timedelta(days=5)]
    }
    return pd.DataFrame(data)

def generate_validation_program_data():
    """Generates data for the Validation Master Program overview."""
    data = {
        'System_ID': ['EQP-00123', 'EQP-00456', 'SW-0010', 'EQP-00789', 'UTL-0001'],
        'System_Name': ['Automated Filler & Capper Line 3', 'Blister Pack Sealer Unit 7', 'LIMS v3.2', 'Lyophilizer #2', 'Purified Water (WFI) Loop'],
        'Validation_Status': ['Validated', 'Validation in Progress', 'Validated', 'Revalidation Due', 'Validated'],
        'Last_Validation_Date': [date(2022, 5, 20), date(2024, 6, 15), date(2023, 1, 10), date(2021, 8, 30), date(2024, 2, 28)],
        'Next_Revalidation_Date': [date(2025, 5, 20), date(2027, 6, 15), date(2026, 1, 10), date(2024, 8, 30), date(2025, 2, 28)],
        'Validation_Lead': ['V. Kumar', 'V. Kumar', 'S. Patel', 'V. Kumar', 'Facilities Eng']
    }
    df = pd.DataFrame(data)
    df['Next_Revalidation_Date'] = pd.to_datetime(df['Next_Revalidation_Date'])
    df['Days_Until_Due'] = (df['Next_Revalidation_Date'] - pd.to_datetime(date.today())).dt.days
    return df


# === NEW DATA GENERATION FOR PROCESS EXCELLENCE ===

def generate_process_excellence_data():
    """Generates time-series data for monitoring V&V process performance."""
    months = pd.to_datetime(pd.date_range(start="2023-01-01", periods=18, freq='M'))
    cycle_time = 45 - np.arange(18) * 0.5 + np.random.normal(0, 2, 18)
    rework_rate = 8 + np.sin(np.linspace(0, 2 * np.pi, 18)) * 2 + np.random.normal(0, 0.5, 18)
    rework_rate = np.clip(rework_rate, 3, 12)
    deviation_rate = 3 - np.arange(18) * 0.05 + np.random.normal(0, 0.3, 18)
    deviation_rate = np.clip(deviation_rate, 1, 5)
    return pd.DataFrame({
        'Month': months,
        'Protocol_Approval_Cycle_Time_Days': cycle_time,
        'Report_Rework_Rate_Percent': rework_rate,
        'Deviations_per_100_Test_Hours': deviation_rate
    })

def generate_idp_data():
    """Generates data for tracking Individual Development Plans."""
    data = {
        'Team Member': ['J. Chen', 'S. Patel', 'K. Lee', 'K. Lee', 'V. Kumar'],
        'Development Goal': [
            'Achieve "Expert" rating in Statistical Analysis (JMP/R)',
            'Achieve "Practitioner" rating on Vitros Platform',
            'Achieve "Practitioner" rating on Sofia Platform',
            'Complete V&V Protocol & Report Authoring Training',
            'Lead a Major Capital Project Validation (End-to-End)'
        ],
        'Mentor': ['M. Rodriguez', 'Manager', 'J. Chen', 'M. Rodriguez', 'Manager'],
        'Start Date': [date.today() - timedelta(days=30), date.today(), date.today() - timedelta(days=15), date.today(), date.today() - timedelta(days=45)],
        'Target Date': [date.today() + timedelta(days=150), date.today() + timedelta(days=90), date.today() + timedelta(days=45), date.today() + timedelta(days=75), date.today() + timedelta(days=320)],
        'Status': ['In Progress', 'Not Started', 'In Progress', 'Not Started', 'In Progress'],
        'Linked Project': ['Savanna RVP12', 'Vitros HIV Combo 5', 'Sofia SARS v2', 'N/A - Formal Training', 'Automated Filler & Capper Line 3']
    }
    df = pd.DataFrame(data)
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    df['Target Date'] = pd.to_datetime(df['Target Date'])
    return df
