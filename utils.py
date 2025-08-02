# utils.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import date, timedelta
from scipy import stats

# --- Custom Plotly Template for QuidelOrtho ---
quidelortho_template = {
    "layout": {
        "font": {"family": "Arial, sans-serif", "size": 12, "color": "#333333"},
        "title": {"font": {"family": "Arial, sans-serif", "size": 18, "color": "#0039A6"}, "x": 0.05},
        "plot_bgcolor": "#FFFFFF",
        "paper_bgcolor": "#FFFFFF",
        "colorway": ['#0039A6', '#00AEEF', '#FFC72C', '#F47321', '#7F8C8D'],
        "xaxis": {"gridcolor": "#E5E5E5", "linecolor": "#B0B0B0", "zerolinecolor": "#E5E5E5", "title_font": {"size": 14}},
        "yaxis": {"gridcolor": "#E5E5E5", "linecolor": "#B0B0B0", "zerolinecolor": "#E5E5E5", "title_font": {"size": 14}},
        "legend": {"bgcolor": "rgba(255,255,255,0.85)", "bordercolor": "#CCCCCC", "borderwidth": 1}
    }
}
pio.templates["quidelortho"] = quidelortho_template
pio.templates.default = "quidelortho"

# === CORE DATA GENERATION (V&V DIRECTOR'S VIEW) ===

def generate_vv_project_data():
    """Generates V&V project data reflecting QuidelOrtho's portfolio and regulatory pathways."""
    data = {
        'Project/Assay': [
            'Savanna® RVP12 Assay',
            'Sofia® 2 SARS Antigen+ FIA v2',
            'Vitros® HIV Combo 5 Test',
            'Triage® BNPNext™ Consumable Change',
            'QuickVue® At-Home OTC COVID-19 Test Line Extension'
        ],
        'Platform': ['Savanna', 'Sofia', 'Vitros', 'Triage', 'QuickVue'],
        'V&V Lead': ['M. Rodriguez', 'J. Chen', 'S. Patel', 'M. Rodriguez', 'J. Chen'],
        'V&V Phase': ['Execution', 'Reporting', 'Protocol Development', 'On Hold', 'Planning'],
        'Overall Status': ['On Track', 'On Track', 'At Risk', 'On Hold', 'On Track'],
        'Regulatory Pathway': ['510(k)', 'EUA Modification', 'PMA Supplement', 'Letter to File', '510(k)'],
        'Start Date': [date.today() - timedelta(days=60), date.today() - timedelta(days=90), date.today() - timedelta(days=10), date.today() - timedelta(days=120), date.today() + timedelta(days=15)],
        'Due Date': [date.today() + timedelta(days=45), date.today() + timedelta(days=5), date.today() + timedelta(days=60), date.today() + timedelta(days=15), date.today() + timedelta(days=120)],
    }
    df = pd.DataFrame(data)
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    df['Due Date'] = pd.to_datetime(df['Due Date'])
    return df

def generate_risk_management_data():
    """Generates risk data based on ISO 14971 for V&V activities."""
    data = {
        'Risk ID': ['R-SAV-001', 'R-SOF-003', 'R-VIT-002', 'R-TRI-001', 'R-GEN-005'],
        'Project': [
            'Savanna® RVP12 Assay',
            'Sofia® 2 SARS Antigen+ FIA v2',
            'Vitros® HIV Combo 5 Test',
            'Triage® BNPNext™ Consumable Change',
            'Savanna® RVP12 Assay'
        ],
        'Risk Description': [
            'Potential cross-reactivity with emerging non-pathogenic coronavirus strains leading to false positives.',
            'New swab material shows lower-than-expected analyte recovery near LoD, potentially impacting sensitivity.',
            'Undetected interference from a common therapeutic drug could lead to a false negative result.',
            'New plastic supplier for consumable has higher lot-to-lot variability in material properties.',
            'V&V study execution timeline conflicts with key personnel vacation, risking submission delay.'
        ],
        'Severity': [4, 4, 5, 3, 2], # 1-Negligible, 2-Minor, 3-Serious, 4-Critical, 5-Catastrophic
        'Probability': [3, 4, 2, 3, 4], # 1-Improbable, 2-Remote, 3-Occasional, 4-Probable, 5-Frequent
        'Owner': ['R&D', 'V&V Team', 'Clinical Affairs', 'Supply Chain', 'V&V Mgmt'],
        'Mitigation': ['Test against a comprehensive panel of endemic coronaviruses.', 'Increase sample size for LoD confirmation study; qualify a second swab supplier.', 'Include drug in interference panel testing per CLSI EP07.', 'Require tighter CoA specs from supplier; perform incoming QC on each lot.', 'Re-allocate V&V specialist from a lower priority project.']
    }
    df = pd.DataFrame(data)
    df['Risk_Score'] = df['Severity'] * df['Probability']
    return df.sort_values(by='Risk_Score', ascending=False)

# === V&V STUDY DATA GENERATION ===

def generate_linearity_data_immunoassay():
    """Generates linearity data for a quantitative immunoassay (e.g., Vitros®)."""
    expected_conc = np.array([0, 10, 50, 100, 250, 500])
    # Michaelis-Menten-like curve for immunoassays
    od_values = 2.5 * expected_conc / (50 + expected_conc)
    observed_od = od_values + np.random.normal(0, 0.03, expected_conc.shape)
    observed_od[0] = np.random.uniform(0.04, 0.06) # Blank noise
    return pd.DataFrame({'Analyte Concentration (ng/mL)': expected_conc, 'Optical Density (OD)': observed_od})

def generate_precision_data_clsi_ep05():
    """Generates precision data (Signal-to-Cutoff) for a Sofia® FIA assay, following CLSI EP05."""
    days = [f'Day {i}' for i in range(1, 6)]; runs_per_day = 2; reps_per_run = 3
    data = []
    # Control 1 (Low Positive)
    for day in days:
        for run in range(1, runs_per_day + 1):
            mean_s_co = 1.8 + (days.index(day) * 0.05) # Introduce slight day-to-day drift
            stdev = 0.12
            values = np.random.normal(loc=mean_s_co, scale=stdev, size=reps_per_run)
            for val in values: data.append({'Control': 'Low Positive', 'Day': day, 'Run': run, 'S/CO Ratio': val})
    # Control 2 (High Positive)
    for day in days:
        for run in range(1, runs_per_day + 1):
            mean_s_co = 8.5
            stdev = 0.4
            values = np.random.normal(loc=mean_s_co, scale=stdev, size=reps_per_run)
            for val in values: data.append({'Control': 'High Positive', 'Day': day, 'Run': run, 'S/CO Ratio': val})
    return pd.DataFrame(data)

def generate_analytical_specificity_data_molecular():
    """Generates cross-reactivity data for a Savanna® molecular respiratory panel."""
    np.random.seed(42); data = []
    # Target analyte (e.g., Influenza A)
    data.extend([{'Sample Type': 'Influenza A', 'Result': 'Positive', 'Ct Value': v} for v in np.random.normal(24, 0.5, 10)])
    # High-concentration potential cross-reactants
    data.extend([{'Sample Type': 'Influenza B', 'Result': 'Negative', 'Ct Value': np.nan} for _ in range(5)])
    data.extend([{'Sample Type': 'RSV A', 'Result': 'Negative', 'Ct Value': np.nan} for _ in range(5)])
    data.extend([{'Sample Type': 'SARS-CoV-2', 'Result': 'Negative', 'Ct Value': np.nan} for _ in range(5)])
    # One unexpected weak positive result to flag for investigation
    data.extend([{'Sample Type': 'Human Metapneumovirus', 'Result': 'Positive', 'Ct Value': 37.5}])
    data.extend([{'Sample Type': 'Human Metapneumovirus', 'Result': 'Negative', 'Ct Value': np.nan} for _ in range(4)])
    # Negative control
    data.extend([{'Sample Type': 'Negative Control', 'Result': 'Negative', 'Ct Value': np.nan} for _ in range(10)])
    return pd.DataFrame(data)

def generate_lot_to_lot_data():
    """Generates lot-to-lot data for a QuickVue® lateral flow consumable (e.g., line intensity)."""
    np.random.seed(0)
    lots = ['Lot 23A001 (Reference)', 'Lot 23B005', 'Lot 24A001 (Candidate)']
    data = []
    for lot in lots:
        mean_intensity = 150
        if lot == 'Lot 23B005': mean_intensity = 145 # slight shift
        if lot == 'Lot 24A001 (Candidate)': mean_intensity = 165 # another shift
        stdev = 12 if lot == 'Lot 24A001 (Candidate)' else 8
        values = np.random.normal(loc=mean_intensity, scale=stdev, size=30)
        for val in values: data.append({'Reagent Lot ID': lot, 'Test Line Intensity': val})
    return pd.DataFrame(data)

def calculate_anova(df, group_col, value_col):
    """Performs one-way ANOVA and returns f-statistic and p-value."""
    groups = [group[value_col].values for name, group in df.groupby(group_col)]
    if len(groups) < 2: return np.nan, np.nan
    f_stat, p_value = stats.f_oneway(*groups)
    return f_stat, p_value

# === REGULATORY & QMS DATA GENERATION ===

def generate_submission_package_data(project_name="Savanna® RVP12 Assay", pathway="510(k)"):
    """Generates a checklist of V&V deliverables for a regulatory submission package."""
    docs = {
        "510(k)": [
            ("V&V Master Plan", "DHF-001", "Approved"), ("Product Risk Analysis (ISO 14971)", "DHF-002", "Approved"),
            ("Requirements Traceability Matrix", "DHF-003", "Approved"), ("Analytical Sensitivity (LoD) Study", "DHF-010", "Approved"),
            ("Analytical Specificity (Cross-reactivity)", "DHF-011", "Approved"), ("Precision/Reproducibility Study (CLSI EP05)", "DHF-012", "In Review"),
            ("Interference Study (CLSI EP07)", "DHF-013", "In Review"), ("Reagent & Consumable Stability Study", "DHF-014", "Execution"),
            ("V&V Summary Report", "DHF-020", "Drafting")
        ],
        "PMA Supplement": [
            ("V&V Master Plan", "DHF-001", "Approved"), ("Product Risk Analysis (ISO 14971)", "DHF-002", "Approved"),
            ("Requirements Traceability Matrix", "DHF-003", "Approved"), ("Clinical Study Protocol", "CLIN-001", "Approved"),
            ("Analytical Performance Studies", "DHF-010-SERIES", "Approved"), ("Clinical Study Report", "CLIN-005", "Data Analysis"),
            ("V&V Summary Report for Submission", "DHF-020", "Drafting")
        ]
    }
    data = docs.get(pathway, docs["510(k)"])
    df = pd.DataFrame(data, columns=['Deliverable', 'Document ID', 'Status'])
    status_progress_map = {'Drafting': 25, 'Execution': 50, 'In Review': 75, 'Data Analysis': 85, 'Approved': 100}
    df['Progress'] = df['Status'].map(status_progress_map)
    return df

def generate_capa_data():
    """Generates data for V&V-related CAPAs and investigations."""
    data = {
        'ID': ['CAPA-23-011', 'INV-24-005', 'CAPA-24-002'],
        'Source': ['Post-Launch Complaint Trend', 'V&V Study Anomaly', 'Internal Audit Finding'],
        'Product': ['Sofia® RSV', 'Savanna® RVP12 Assay', 'All Platforms'],
        'Description': ['Increase in reported false positives from EU region after lot change.', 'Unexpected weak positive result for hMPV in Flu A channel during specificity testing.', 'Inconsistent documentation of V&V protocol deviations across projects.'],
        'Owner': ['V&V Mgmt', 'M. Rodriguez', 'QA'],
        'Phase': ['Effectiveness Check', 'Root Cause Investigation', 'Implementation'],
        'Due Date': [date.today() - timedelta(days=10), date.today() + timedelta(days=18), date.today() + timedelta(days=40)]
    }
    return pd.DataFrame(data)

# === V&V LAB OPERATIONS DATA ===

def generate_instrument_schedule_data():
    """Generates schedule data for instruments in the V&V lab."""
    today = pd.Timestamp.now().normalize()
    data = [
        {'Instrument': 'Savanna-V&V-01', 'Start': today, 'Finish': today + timedelta(days=3), 'Status': 'V&V Execution', 'Details': 'RVP12 Reproducibility Study'},
        {'Instrument': 'Savanna-V&V-02', 'Start': today + timedelta(days=4), 'Finish': today + timedelta(days=5), 'Status': 'Scheduled Maintenance', 'Details': 'Annual PM & Calibration'},
        {'Instrument': 'Sofia-V&V-01', 'Start': today - timedelta(days=1), 'Finish': today + timedelta(days=1), 'Status': 'V&V Execution', 'Details': 'SARS v2 LoD Confirmation'},
        {'Instrument': 'Sofia-V&V-02', 'Start': today - timedelta(days=1), 'Finish': today, 'Status': 'OOS', 'Details': 'OOS-24-01: Optics alignment failure'},
        {'Instrument': 'Vitros-DEV-01', 'Start': today, 'Finish': today + timedelta(days=7), 'Status': 'Available for Booking', 'Details': 'Open for R&D or V&V use'},
    ]
    df = pd.DataFrame(data)
    df['Start'] = pd.to_datetime(df['Start'])
    df['Finish'] = pd.to_datetime(df['Finish'])
    return df

def generate_training_data_for_heatmap():
    """Generates a competency matrix for the Assay V&V Team."""
    # Competency Levels: 0=Awareness, 1=Practitioner, 2=Expert/Trainer
    data = {
        'CLSI EP05/EP17 (Precision/LoD)': [2, 2, 1, 1, 0],
        'CLSI EP07 (Interference)': [2, 1, 2, 1, 1],
        'Statistical Analysis (JMP/Python)': [2, 2, 1, 0, 0],
        'Savanna® Platform Operation': [2, 1, 1, 2, 1],
        'Sofia® Platform Operation': [1, 2, 2, 1, 1],
        'Vitros® Platform Operation': [1, 0, 1, 0, 2],
        'Protocol & Report Writing': [2, 2, 2, 1, 1],
    }
    df = pd.DataFrame(data, index=['A. Director (You)', 'M. Rodriguez (Mgr)', 'J. Chen (Sr. Specialist)', 'S. Patel (Specialist II)', 'K. Lee (Specialist I)'])
    return df

def generate_reagent_lot_status_data():
    """Generates status data for critical reagents used in V&V studies."""
    today = date.today()
    data = {
        'Lot ID': ['SAV-RVP12-24A01', 'SOF-SARS-23C15', 'VIT-HIV-23B09', 'SOF-CTRL-23D11'],
        'Reagent/Kit': ['Savanna RVP12 Cartridge', 'Sofia SARS FIA Kit', 'Vitros HIV Combo Reagent Pack', 'Universal FIA Control Swab'],
        'Project Use': ['Savanna RVP12 V&V', 'Sofia SARS v2 V&V', 'Vitros HIV V&V', 'Multiple'],
        'Status': ['In Use', 'In Use', 'Low Inventory', 'Expired'],
        'Expiry Date': [today + timedelta(days=90), today + timedelta(days=25), today + timedelta(days=45), today - timedelta(days=5)],
        'Notes': ['Reference lot for V&V studies.', 'Final LoD studies must complete before expiry.', 'Order placed for next lot.', 'Quarantine and discard. Do not use for V&V.']
    }
    return pd.DataFrame(data)
