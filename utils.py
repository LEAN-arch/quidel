# utils.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import date, timedelta
from scipy import stats

# --- Custom Plotly Template for QuidelOrtho (Commercial Grade) ---
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
pio.templates["quidelortho_commercial"] = quidelortho_template
pio.templates.default = "quidelortho_commercial"

# === CORE DATA GENERATION (V&V DIRECTOR'S VIEW - ENHANCED) ===

def generate_vv_project_data():
    """Generates V&V project data reflecting QuidelOrtho's portfolio, including dependencies and milestones."""
    data = {
        'Project/Assay': [
            'Savanna® RVP12 Assay', 'Sofia® 2 SARS Antigen+ FIA v2', 'Vitros® HIV Combo 5 Test',
            'Triage® BNPNext™ Consumable Change', 'QuickVue® At-Home OTC COVID-19 Test Line Extension', 'Ortho-Vision® Analyzer SW Patch V&V'
        ],
        'Platform': ['Savanna', 'Sofia', 'Vitros', 'Triage', 'QuickVue', 'Ortho-Vision'],
        'V&V Lead': ['M. Rodriguez', 'J. Chen', 'S. Patel', 'M. Rodriguez', 'J. Chen', 'S. Patel'],
        'V&V Phase': ['Execution', 'Reporting', 'Protocol Development', 'On Hold', 'Planning', 'Data Analysis'],
        'Overall Status': ['On Track', 'On Track', 'At Risk', 'On Hold', 'On Track', 'On Track'],
        'Regulatory Pathway': ['510(k)', 'EUA Modification', 'PMA Supplement', 'Letter to File', '510(k)', 'Letter to File'],
        'Upstream Dependency': ['R&D Finalized Assay Design', 'N/A', 'Clinical Sample Acquisition', 'Supply Chain Qualification', 'Marketing Finalizes User Needs', 'SW Development Team'],
        'Key Milestone': ['Final Report for Submission', 'EUA Submission', 'PMA Submission Package Ready', 'N/A', 'V&V Plan Approval', 'V&V Report for ECO'],
        'Start Date': [date.today() - timedelta(days=60), date.today() - timedelta(days=90), date.today() - timedelta(days=10), date.today() - timedelta(days=120), date.today() + timedelta(days=15), date.today() - timedelta(days=30)],
        'Due Date': [date.today() + timedelta(days=45), date.today() + timedelta(days=5), date.today() + timedelta(days=60), date.today() + timedelta(days=15), date.today() + timedelta(days=120), date.today() + timedelta(days=20)],
        'Milestone Date': [date.today() + timedelta(days=40), date.today() + timedelta(days=4), date.today() + timedelta(days=55), pd.NaT, date.today() + timedelta(days=60), date.today() + timedelta(days=18)],
    }
    df = pd.DataFrame(data)
    for col in ['Start Date', 'Due Date', 'Milestone Date']:
        df[col] = pd.to_datetime(df[col])
    return df

def generate_risk_management_data():
    """Generates risk data based on ISO 14971, reflecting a mature risk management process."""
    data = {
        'Risk ID': ['R-SAV-001', 'R-SOF-003', 'R-VIT-002', 'R-TRI-001', 'R-GEN-005', 'R-SFT-001'],
        'Project': [
            'Savanna® RVP12 Assay', 'Sofia® 2 SARS Antigen+ FIA v2', 'Vitros® HIV Combo 5 Test',
            'Triage® BNPNext™ Consumable Change', 'Savanna® RVP12 Assay', 'Ortho-Vision® Analyzer SW Patch V&V'
        ],
        'Risk Description': [
            'Potential cross-reactivity with emerging non-pathogenic coronavirus strains could lead to false positives, resulting in unnecessary treatment.',
            'New swab material shows lower-than-expected analyte recovery near LoD, potentially impacting sensitivity and leading to a false negative.',
            'Undetected interference from a common therapeutic drug (e.g., biotin) could lead to a false negative result, delaying critical diagnosis.',
            'New plastic supplier for consumable has higher lot-to-lot variability in material properties, potentially causing inconsistent test results.',
            'V&V study execution timeline conflicts with key personnel availability, posing a risk of delayed regulatory submission.',
            'Regression testing for software patch fails to cover a legacy edge-case, potentially re-introducing a previously fixed bug.'
        ],
        'Severity': [4, 4, 5, 3, 2, 4],
        'Probability': [3, 4, 2, 3, 4, 2],
        'Owner': ['R&D/V&V', 'V&V Team', 'Clinical/V&V', 'Supply Chain/V&V', 'V&V Management', 'SW V&V Team'],
        'Mitigation': [
            'Test against a comprehensive panel of endemic coronaviruses per FDA guidance. Document results in V&V report.',
            'Increase sample size for LoD confirmation study; qualify a second swab supplier as a risk control measure.',
            'Include biotin and other interferents in testing per CLSI EP07. Add limitation to Instructions for Use (IFU).',
            'Require tighter Certificate of Analysis (CoA) specs from supplier; perform incoming QC on each lot.',
            'Re-allocate V&V specialist from a lower priority project. Document change in project plan.',
            'Expand regression test suite to include historical defect scenarios. Perform targeted black-box testing on affected modules.'
        ]
    }
    df = pd.DataFrame(data)
    df['Risk_Score'] = df['Severity'] * df['Probability']
    return df.sort_values(by='Risk_Score', ascending=False)

# === V&V STUDY DATA GENERATION (ENHANCED) ===

def generate_linearity_data_immunoassay():
    """Generates realistic linearity data for a quantitative immunoassay, including saturation effects."""
    expected_conc = np.array([0, 10, 50, 100, 250, 500, 750, 1000])
    od_values = 2.5 * expected_conc / (50 + expected_conc)
    observed_od = od_values + np.random.normal(0, 0.03, expected_conc.shape)
    observed_od[0] = np.random.uniform(0.04, 0.06) # Blank noise
    observed_od[-2:] += np.random.normal(0, 0.02, 2) # Add noise to saturation
    return pd.DataFrame({'Analyte Concentration (ng/mL)': expected_conc, 'Optical Density (OD)': observed_od})

def generate_precision_data_clsi_ep05():
    """Generates precision data with more nuance, including different variance components for a Sofia® FIA assay."""
    days = [f'Day {i}' for i in range(1, 6)]; runs_per_day = 2; reps_per_run = 3; data = []
    # Control 1 (Low Positive)
    for day_idx, day in enumerate(days):
        day_effect = np.random.normal(0, 0.08)
        for run in range(1, runs_per_day + 1):
            run_effect = np.random.normal(0, 0.05)
            mean_s_co = 1.8 + day_effect + run_effect
            stdev_within_run = 0.12
            values = np.random.normal(loc=mean_s_co, scale=stdev_within_run, size=reps_per_run)
            for val in values: data.append({'Control': 'Low Positive (Near Cutoff)', 'Day': day, 'Run': run, 'S/CO Ratio': val})
    # Control 2 (Moderate Positive)
    for day_idx, day in enumerate(days):
        day_effect = np.random.normal(0, 0.2)
        for run in range(1, runs_per_day + 1):
            run_effect = np.random.normal(0, 0.1)
            mean_s_co = 8.5 + day_effect + run_effect
            stdev_within_run = 0.4
            values = np.random.normal(loc=mean_s_co, scale=stdev_within_run, size=reps_per_run)
            for val in values: data.append({'Control': 'Moderate Positive', 'Day': day, 'Run': run, 'S/CO Ratio': val})
    return pd.DataFrame(data)

def generate_analytical_specificity_data_molecular():
    """Generates cross-reactivity data for a Savanna® panel, including a borderline case for adjudication."""
    np.random.seed(42); data = []
    data.extend([{'Sample ID': f'FLU-A-S{i}', 'Organism Tested': 'Influenza A', 'Result': 'Positive', 'Ct Value': v, 'Notes': 'Inclusivity Panel'} for i, v in enumerate(np.random.normal(24, 0.5, 10), 1)])
    data.extend([{'Sample ID': f'FLU-B-S{i}', 'Organism Tested': 'Influenza B', 'Result': 'Negative', 'Ct Value': np.nan, 'Notes': 'Exclusivity Panel'} for i in range(1, 6)])
    data.extend([{'Sample ID': f'RSV-A-S{i}', 'Organism Tested': 'RSV A', 'Result': 'Negative', 'Ct Value': np.nan, 'Notes': 'Exclusivity Panel'} for i in range(1, 6)])
    data.extend([{'Sample ID': f'hMPV-S1', 'Organism Tested': 'Human Metapneumovirus', 'Result': 'Positive', 'Ct Value': 38.1, 'Notes': 'Potential Cross-reactivity, requires investigation.'}])
    data.extend([{'Sample ID': f'hMPV-S{i}', 'Organism Tested': 'Human Metapneumovirus', 'Result': 'Negative', 'Ct Value': np.nan, 'Notes': 'Exclusivity Panel'} for i in range(2, 6)])
    data.extend([{'Sample ID': f'NTC-S{i}', 'Organism Tested': 'Negative Control', 'Result': 'Negative', 'Ct Value': np.nan, 'Notes': 'Process Control'} for i in range(1, 11)])
    return pd.DataFrame(data)

def generate_lot_to_lot_data():
    """Generates lot-to-lot data for a QuickVue® consumable, suitable for equivalence testing."""
    np.random.seed(0); lots = ['Lot 23A001 (Reference)', 'Lot 24A001 (Candidate)']; data = []
    for lot in lots:
        mean_intensity = 155 if lot.startswith('Lot 23A') else 150
        stdev = 10
        values = np.random.normal(loc=mean_intensity, scale=stdev, size=50)
        for val in values: data.append({'Reagent Lot ID': lot, 'Test Line Intensity': val})
    return pd.DataFrame(data)

def calculate_equivalence(df, group_col, value_col, low_eq_bound, high_eq_bound):
    """Performs a Two-One-Sided T-Test (TOST) for equivalence."""
    groups = df[group_col].unique()
    if len(groups) != 2: return np.nan, np.nan, np.nan
    group1 = df[df[group_col] == groups[0]][value_col]
    group2 = df[df[group_col] == groups[1]][value_col]
    t_stat, p_val_diff = stats.ttest_ind(group1, group2, equal_var=False)
    
    mean_diff = group1.mean() - group2.mean()
    std_err_diff = np.sqrt(group1.var()/len(group1) + group2.var()/len(group2))
    
    t_low = (mean_diff - low_eq_bound) / std_err_diff
    t_high = (mean_diff - high_eq_bound) / std_err_diff
    
    p_low = stats.t.sf(t_low, df=len(group1) + len(group2) - 2)
    p_high = stats.t.cdf(t_high, df=len(group1) + len(group2) - 2)
    
    return max(p_low, p_high), p_val_diff, mean_diff

# === NEW DATA GENERATION FOR ADVANCED VISUALIZATIONS ===

def generate_method_comparison_data():
    """Generates data for a Bland-Altman plot, comparing a new method to a reference."""
    np.random.seed(123)
    n_samples = 100
    reference_values = np.random.uniform(10, 500, n_samples)
    # Introduce a slight proportional bias and random error
    new_method_values = 1.05 * reference_values - 5 + np.random.normal(0, 15, n_samples)
    return pd.DataFrame({'Reference Method (U/L)': reference_values, 'Candidate Method (U/L)': new_method_values})

def generate_risk_burndown_data():
    """Generates time-series data for a risk burndown chart."""
    weeks = pd.to_datetime(pd.date_range(start="2024-01-01", periods=12, freq='W'))
    high_risks = np.array([5, 5, 4, 4, 3, 2, 2, 1, 1, 1, 0, 0])
    medium_risks = np.array([10, 11, 11, 10, 9, 9, 8, 6, 5, 4, 3, 2])
    low_risks = np.array([8, 8, 9, 10, 10, 11, 11, 10, 9, 8, 8, 7])
    return pd.DataFrame({'Week': weeks, 'High': high_risks, 'Medium': medium_risks, 'Low': low_risks})

def generate_defect_trend_data():
    """Generates data for a software defect burn-down chart."""
    days = pd.to_datetime(pd.date_range(start="2024-06-01", periods=30, freq='D'))
    total_defects = np.cumsum(np.random.randint(0, 3, size=30)) + 5
    closed_defects = np.cumsum(np.random.randint(0, 2, size=30))
    closed_defects = np.minimum(closed_defects, total_defects)
    open_defects = total_defects - closed_defects
    return pd.DataFrame({'Date': days, 'Open Defects': open_defects, 'Total Defects Found': total_defects})

def calculate_instrument_utilization(schedule_df):
    """Processes schedule data to calculate utilization stats for a treemap."""
    schedule_df['Duration'] = (schedule_df['Finish'] - schedule_df['Start']).dt.total_seconds() / 3600
    platform_map = {
        'Savanna-V&V-01': 'Savanna', 'Savanna-V&V-02': 'Savanna',
        'Sofia-V&V-01': 'Sofia', 'Sofia-V&V-02': 'Sofia',
        'Vitros-DEV-01': 'Vitros'
    }
    schedule_df['Platform'] = schedule_df['Instrument'].map(platform_map)
    util_df = schedule_df.groupby(['Platform', 'Instrument', 'Status'])['Duration'].sum().reset_index()
    return util_df

# (Other existing functions like generate_submission_package_data, generate_capa_data, etc. remain unchanged from the previous corrected version)

def generate_submission_package_data(project_name="Savanna® RVP12 Assay", pathway="510(k)"):
    """Generates a detailed checklist of V&V deliverables for a regulatory submission package."""
    docs = {
        "510(k)": [
            ("V&V Master Plan", "DHF-001", "Approved"), ("Product Risk Management File (ISO 14971)", "DHF-002", "Approved"),
            ("Requirements Traceability Matrix (RTM)", "DHF-003", "Approved"), ("Software V&V Summary Report", "DHF-005", "Approved"),
            ("Analytical Sensitivity (LoD) Study Report", "V&V-RPT-010", "Approved"), ("Analytical Specificity (Cross-reactivity) Report", "V&V-RPT-011", "In Review"),
            ("Precision/Reproducibility Study Report (CLSI EP05)", "V&V-RPT-012", "In Review"), ("Interference Study Report (CLSI EP07)", "V&V-RPT-013", "Data Analysis"),
            ("Reagent & Consumable Stability Report", "V&V-RPT-014", "Execution"), ("V&V Master Summary Report", "DHF-020", "Drafting")
        ],
        "PMA Supplement": [
            ("V&V Master Plan", "DHF-001", "Approved"), ("Product Risk Management File (ISO 14971)", "DHF-002", "Approved"),
            ("Clinical Study Protocol", "CLIN-001", "Approved"), ("Analytical Performance Studies Package", "V&V-PKG-01", "Approved"),
            ("Clinical Study Final Report", "CLIN-005", "Data Analysis"), ("Labeling & IFU V&V Report", "V&V-RPT-015", "Approved"),
            ("V&V Master Summary Report for Submission", "DHF-020", "Drafting")
        ]
    }
    data = docs.get(pathway, docs["510(k)"])
    df = pd.DataFrame(data, columns=['Deliverable', 'Document ID', 'Status'])
    status_progress_map = {'Drafting': 15, 'Execution': 40, 'Data Analysis': 65, 'In Review': 85, 'Approved': 100}
    df['Progress'] = df['Status'].map(status_progress_map)
    return df

def generate_capa_data():
    """Generates data for V&V-related CAPAs and investigations with enhanced detail."""
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

def generate_instrument_schedule_data():
    """Generates a more dynamic schedule for instruments in the V&V lab."""
    today = pd.Timestamp.now().normalize()
    data = [
        {'Instrument': 'Savanna-V&V-01', 'Start': today, 'Finish': today + timedelta(days=3), 'Status': 'V&V Execution', 'Details': 'RVP12 Reproducibility Study (P.N. V&V-PRO-012)'},
        {'Instrument': 'Savanna-V&V-02', 'Start': today + timedelta(days=4), 'Finish': today + timedelta(days=5), 'Status': 'Calibration/PM', 'Details': 'Annual Preventative Maintenance'},
        {'Instrument': 'Sofia-V&V-01', 'Start': today - timedelta(days=1), 'Finish': today + timedelta(days=1), 'Status': 'V&V Execution', 'Details': 'SARS v2 LoD Confirmation (P.N. V&V-PRO-015)'},
        {'Instrument': 'Sofia-V&V-02', 'Start': today - timedelta(days=2), 'Finish': today + timedelta(days=2), 'Status': 'OOS', 'Details': 'OOS-24-01: Optics alignment failure. Awaiting service engineer.'},
        {'Instrument': 'Vitros-DEV-01', 'Start': today, 'Finish': today + timedelta(days=7), 'Status': 'Available', 'Details': 'Open for booking.'}
    ]
    df = pd.DataFrame(data)
    for col in ['Start', 'Finish']:
        df[col] = pd.to_datetime(df[col])
    return df

def generate_training_data_for_heatmap():
    """Generates an enhanced, role-based competency matrix for the Assay V&V Team."""
    data = {
        'CLSI EP05/EP17 (Precision/LoD)': [2, 2, 1, 1, 0], 'CLSI EP07/EP37 (Interference/Immunoassay)': [2, 1, 2, 1, 1],
        'Statistical Analysis (JMP/R)': [2, 2, 1, 0, 0], 'Savanna® Platform & Assays': [2, 1, 1, 2, 1],
        'Sofia® Platform & Assays': [1, 2, 2, 1, 1], 'Vitros® Platform & Assays': [1, 0, 1, 0, 2],
        'V&V Protocol & Report Authoring': [2, 2, 2, 1, 1], 'Risk Management (ISO 14971)': [2, 2, 1, 1, 0],
        'Software V&V (IEC 62304)': [2, 1, 0, 0, 0]
    }
    df = pd.DataFrame(data, index=['A. Director', 'M. Rodriguez (Mgr)', 'J. Chen (Sr. Specialist)', 'S. Patel (Specialist II)', 'K. Lee (New Hire)'])
    return df

def generate_reagent_lot_status_data():
    """Generates status data for critical reagents used in V&V studies with more detail."""
    today = date.today()
    data = {
        'Lot ID': ['SAV-RVP12-24A01', 'SOF-SARS-23C15', 'VIT-HIV-23B09', 'SOF-CTRL-23D11'],
        'Reagent/Kit': ['Savanna RVP12 Cartridge', 'Sofia SARS FIA Kit', 'Vitros HIV Combo Reagent Pack', 'Universal FIA Control Swab'],
        'Assigned Project': ['Savanna RVP12 V&V', 'Sofia SARS v2 V&V', 'Vitros HIV V&V', 'All Sofia Studies'],
        'Status': ['In Use - Qualified', 'Low Inventory', 'Quarantined - Awaiting CoA', 'Expired - Do Not Use'],
        'Expiry Date': [today + timedelta(days=90), today + timedelta(days=25), today + timedelta(days=180), today - timedelta(days=5)],
        'Notes': ['Reference lot for all V&V studies.', 'Final LoD studies must complete before expiry.', 'Incoming material. Not released for V&V use.', 'Remove from inventory. Document disposal.']
    }
    return pd.DataFrame(data)

def generate_traceability_matrix_data():
    """Generates data for a detailed Requirements Traceability Matrix."""
    data = {
        'Requirement ID': ['URS-001', 'URS-002', 'SRS-005', 'SRS-006', 'Risk-Ctrl-010'],
        'Requirement Type': ['User Need', 'User Need', 'System Requirement', 'Software Requirement', 'Risk Control Measure'],
        'Description': [
            'The Savanna system shall identify Influenza A in a sample.',
            'The assay result should be available in under 30 minutes.',
            'The system shall achieve a clinical sensitivity of >95% for Influenza A.',
            'The software shall flag results with a Ct value > 37 as "indeterminate".',
            'The assay must not cross-react with high-titer Influenza B samples.'
        ],
        'V&V Protocol': ['V&V-PRO-010', 'V&V-PRO-021', 'V&V-PRO-010', 'V&V-PRO-SFT-001', 'V&V-PRO-011'],
        'Test Case ID': ['TC-LOD-01', 'TC-TIME-01-05', 'TC-SENS-01-50', 'TC-FLAG-01-03', 'TC-SPEC-01-10'],
        'Test Result': ['Pass', 'Pass', 'Pass', 'Fail', 'Pass'],
        'V&V Report': ['V&V-RPT-010', 'V&V-RPT-021', 'V&V-RPT-010', 'V&V-RPT-SFT-001', 'V&V-RPT-011']
    }
    return pd.DataFrame(data)

def generate_change_control_data():
    """Generates data for managing post-launch changes via ECOs."""
    data = {
        'ECO Number': ['ECO-01234', 'ECO-01255', 'ECO-01260'],
        'Product Impacted': ['Sofia® 2 SARS Antigen+ FIA', 'Triage® BNPNext™', 'Savanna® RVP12 Assay'],
        'Change Description': [
            'Update IFU to include new, validated swab type.',
            'Qualify second-source supplier for plastic consumable housing.',
            'Minor software update to improve sample traceability logging (no algorithm change).'
        ],
        'V&V Impact Assessment': ['Low. Requires limited V&V to confirm swab equivalency.', 'Medium. Requires full V&V of key performance specs (precision, LoD) with new plastic.', 'Low. Requires targeted software regression testing.'],
        'Assigned V&V Lead': ['J. Chen', 'M. Rodriguez', 'S. Patel'],
        'Status': ['V&V Complete', 'V&V in Progress', 'Awaiting V&V Plan']
    }
    return pd.DataFrame(data)
