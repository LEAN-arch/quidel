# pages/3_Analytical_Studies_Dashboard.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import (generate_linearity_data_immunoassay, generate_precision_data_clsi_ep05,
                   generate_analytical_specificity_data_molecular, generate_lot_to_lot_data, 
                   calculate_equivalence, generate_method_comparison_data)
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Analytical Studies Dashboard | QuidelOrtho",
    layout="wide"
)

st.title("📈 Analytical Performance Studies Dashboard")
st.markdown("### Data Review and Oversight of Key Verification Studies for Regulatory Submissions")

with st.expander("🌐 Director's View: The Role of Analytical Verification", expanded=True):
    st.markdown("""
    This dashboard provides the objective evidence required to verify that our assays meet their design input requirements. My role is to scrutinize this data with my team, ensuring its integrity and confirming that results meet our stringent, pre-defined acceptance criteria. Each "PASS" result on this page becomes a cornerstone of our V&V Summary Reports and, ultimately, our regulatory submissions.

    **Key Regulatory & Quality Frameworks:**
    - **21 CFR 820.30(f) - Design Verification:** This entire page is a testament to this regulation, providing documented evidence that we have confirmed design outputs meet design inputs. The methods, results, and analyses shown here are documented in the Design History File (DHF).
    - **CLSI Guidelines (EP05, EP06, EP07, EP09, EP17):** Our study designs and acceptance criteria are harmonized with these globally recognized standards, ensuring our data is robust and defensible during FDA review. For example, precision studies follow **CLSI EP05-A3** and method comparisons follow **CLSI EP09**.
    - **Risk Management (ISO 14971):** The results of these studies directly inform our risk management file. For example, an LoD result confirms our control over sensitivity risks, while a cross-reactivity study confirms control over specificity risks.
    """)

# --- Data Generation ---
linearity_df = generate_linearity_data_immunoassay()
precision_df = generate_precision_data_clsi_ep05()
specificity_df = generate_analytical_specificity_data_molecular()
lot_df = generate_lot_to_lot_data()
method_comp_df = generate_method_comparison_data()

# --- Page Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "**Precision (CLSI EP05)**",
    "**Method Comparison (CLSI EP09)**",
    "**Specificity (Inclusivity/Exclusivity)**",
    "**Linearity / AMI (CLSI EP06)**",
    "**Lot-to-Lot Equivalence**"
])

with tab1:
    st.header("Precision & Reproducibility (CLSI EP05)")
    st.caption("Example: 5-day, 2-run, 3-replicate study for the Sofia® 2 SARS Antigen+ FIA v2 assay.")
    with st.expander("🔬 Study Design & Acceptance Criteria"):
        st.markdown("""
        **Study Design (CLSI EP05-A3):** To evaluate **reproducibility**, two levels of control material are tested over 5 days, with 2 runs per day, and 3 replicates per run. A nested ANOVA model is used to estimate the variance contributed by each factor.
        **Acceptance Criteria:** **Low Positive Control:** Total %CV must be **≤ 20%**. **Moderate Positive Control:** Total %CV must be **≤ 15%**.
        """)
    
    control_level = st.selectbox("Select Control Level to Analyze", precision_df['Control'].unique(), key='precision_select')
    filtered_df = precision_df[precision_df['Control'] == control_level].copy()
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader(f"Precision Data for {control_level}")
        fig = px.box(filtered_df, x="Day", y="S/CO Ratio", points="all")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Analysis of Variance (ANOVA) Results")
        try:
            model = ols('Q("S/CO Ratio") ~ C(Day) + C(Run):C(Day)', data=filtered_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            anova_table['mean_sq'] = anova_table['sum_sq'] / anova_table['df']
            ms_error = model.scale; ms_run = anova_table.loc['C(Run):C(Day)', 'mean_sq']; ms_day = anova_table.loc['C(Day)', 'mean_sq']
            reps_per_run = filtered_df.groupby(['Day', 'Run']).size().mean(); runs_per_day = filtered_df.groupby('Day')['Run'].nunique().mean()
            var_within = ms_error; var_run = (ms_run - ms_error) / reps_per_run; var_day = (ms_day - ms_run) / (reps_per_run * runs_per_day)
            vc = {'Repeatability (Within-Run)': max(0, var_within), 'Between-Run (Within-Day)': max(0, var_run), 'Between-Day': max(0, var_day)}
            vc_df = pd.DataFrame.from_dict(vc, orient='index', columns=['Variance'])
            vc_df['% Contribution'] = (vc_df['Variance'] / vc_df['Variance'].sum() * 100).round(1)
            st.dataframe(vc_df, use_container_width=True)
            total_cv = (np.sqrt(vc_df['Variance'].sum()) / filtered_df['S/CO Ratio'].mean()) * 100
            acceptance_cv = 20 if "Low Positive" in control_level else 15
            st.metric(f"Total Reproducibility CV vs. Acceptance (≤{acceptance_cv}%)", f"{total_cv:.2f}%")

            if total_cv <= acceptance_cv: st.success(f"**PASS:** Total CV meets acceptance criteria.")
            else: st.error(f"**FAIL:** Total CV exceeds acceptance criteria.")
        except Exception as e: st.error(f"ANOVA calculation failed: {e}")

with tab2:
    st.header("Method Comparison & Bias Estimation (CLSI EP09)")
    st.caption("Comparing the Vitros® Candidate Method to a gold-standard Reference Method using patient samples.")
    with st.expander("🔬 Study Design & Acceptance Criteria"):
        st.markdown("""
        **Study Design (CLSI EP09):** A set of patient samples spanning the assay's measuring range are tested on both the candidate and reference methods. The goal is to quantify bias (systematic error).
        **Statistical Methods:**
        - **Deming Regression:** A regression method that accounts for error in both the X and Y measurements, providing a more accurate estimate of slope and intercept.
        - **Bland-Altman Plot:** A graphical method to visualize the agreement between two quantitative measurements. It plots the difference between measurements against their average.
        **Acceptance Criteria:** **Slope** of Deming regression must be between **0.90-1.10**. **95%** of points must fall within the Bland-Altman **Limits of Agreement**.
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Deming Regression Analysis")
        fig_deming = px.scatter(method_comp_df, x='Reference Method (U/L)', y='Candidate Method (U/L)',
                                title='Deming Regression: Candidate vs. Reference', trendline='ols', trendline_scope='overall')
        # In a real scenario, this would be a proper Deming fit, OLS is a proxy for visualization
        st.plotly_chart(fig_deming, use_container_width=True)
        
    with col2:
        st.subheader("Bland-Altman Agreement Plot")
        method_comp_df['Average'] = (method_comp_df['Reference Method (U/L)'] + method_comp_df['Candidate Method (U/L)']) / 2
        method_comp_df['Difference'] = method_comp_df['Candidate Method (U/L)'] - method_comp_df['Reference Method (U/L)']
        mean_diff = method_comp_df['Difference'].mean()
        std_diff = method_comp_df['Difference'].std()
        upper_loa = mean_diff + 1.96 * std_diff
        lower_loa = mean_diff - 1.96 * std_diff

        fig_bland = px.scatter(method_comp_df, x='Average', y='Difference', title='Bland-Altman Plot')
        fig_bland.add_hline(y=mean_diff, line_dash="solid", line_color="blue", annotation_text=f"Mean Bias: {mean_diff:.2f}")
        fig_bland.add_hline(y=upper_loa, line_dash="dash", line_color="red", annotation_text=f"Upper LoA: {upper_loa:.2f}")
        fig_bland.add_hline(y=lower_loa, line_dash="dash", line_color="red", annotation_text=f"Lower LoA: {lower_loa:.2f}")
        st.plotly_chart(fig_bland, use_container_width=True)

with tab3:
    st.header("Analytical Specificity (Cross-Reactivity & Interference)")
    st.caption("Example: Inclusivity & exclusivity testing for the Savanna® RVP12 molecular assay.")
    with st.expander("🔬 Study Design & Acceptance Criteria"):
        st.markdown("""
        **Study Design:** This study confirms the assay's ability to exclusively detect its intended targets by testing inclusivity (on-target strains) and exclusivity (off-target, high-concentration organisms).
        **Acceptance Criteria:** **100%** of inclusivity strains must be detected. **0%** of exclusivity organisms should be detected. Any confirmed positive is a failure.
        """)
    
    st.dataframe(specificity_df, use_container_width=True, hide_index=True)
    cross_react_failures = specificity_df[specificity_df['Notes'].str.contains("Potential Cross-reactivity")]
    
    st.subheader("Specificity Performance Summary")
    if not cross_react_failures.empty:
        st.error(f"**FAIL: Potential Cross-Reactivity Detected**")
        st.write("The following off-target organism produced a positive signal, requiring root cause investigation.")
        st.dataframe(cross_react_failures)
    else:
        st.success("**PASS:** No unexpected cross-reactivity was detected.")

with tab4:
    st.header("Assay Linearity / Analytical Measuring Interval (AMI) (CLSI EP06)")
    st.caption("Example: Defining the AMI for a Vitros® quantitative immunoassay.")
    with st.expander("🔬 Study Design & Acceptance Criteria"):
        st.markdown("""
        **Study Design:** A dilution series of a high-concentration standard is tested to characterize the assay's response across a wide range of concentrations.
        **Acceptance Criteria:** For the defined linear range: **R-squared ≥ 0.990**. Residuals plot must show no obvious trends.
        """)

    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("Assay Response Curve")
        fig_lin = px.scatter(linearity_df, x='Analyte Concentration (ng/mL)', y='Optical Density (OD)', 
                            title="Full Range Response Curve", trendline='lowess', trendline_color_override="#F47321")
        st.plotly_chart(fig_lin, use_container_width=True)
    with col2:
        st.subheader("Linear Range Analysis")
        linear_range_max = st.number_input("Define Upper Limit of Linear Range for Analysis (ng/mL)", value=500)
        model_df = linearity_df[linearity_df['Analyte Concentration (ng/mL)'] <= linear_range_max].copy()
        X = sm.add_constant(model_df['Analyte Concentration (ng/mL)'])
        model = sm.OLS(model_df['Optical Density (OD)'], X).fit()
        r_squared = model.rsquared
        model_df['Residual'] = model.resid
        fig_res = px.scatter(model_df, x='Analyte Concentration (ng/mL)', y='Residual', title=f"Residuals Plot (R² = {r_squared:.4f})")
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_res, use_container_width=True)
    if r_squared >= 0.990: st.success(f"**PASS:** R² of {r_squared:.4f} meets acceptance criteria.")
    else: st.error(f"**FAIL:** R² of {r_squared:.4f} is below acceptance criteria.")

with tab5:
    st.header("New Reagent Lot Qualification by Equivalence")
    st.caption("Example: Qualifying a new consumable lot for QuickVue® using a Two-One-Sided T-Test (TOST).")
    with st.expander("🔬 Study Design & Acceptance Criteria"):
        st.markdown("""
        **Statistical Method: Equivalence Testing (TOST):** We test to prove *similarity*. We define an **equivalence margin** (e.g., ±15 intensity units) and test the null hypothesis that the lots are different.
        **Acceptance Criteria:** The **TOST P-value** must be **< 0.05** to claim the lots are equivalent.
        """)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("Lot-to-Lot Signal Comparison")
        fig_box = px.violin(lot_df, x='Reagent Lot ID', y='Test Line Intensity',
                            title='Candidate Lot vs. Reference Lot Distribution', box=True, points="all")
        st.plotly_chart(fig_box, use_container_width=True)
    with col2:
        st.subheader("Equivalence Test (TOST) Results")
        low_bound = st.number_input("Lower Equivalence Margin (in intensity units)", value=-15.0, format="%.1f")
        high_bound = st.number_input("Higher Equivalence Margin (in intensity units)", value=15.0, format="%.1f")
        
        p_tost, p_diff, mean_diff = calculate_equivalence(lot_df, 'Reagent Lot ID', 'Test Line Intensity', low_bound, high_bound)
        
        st.metric("Mean Difference (Candidate - Reference)", f"{mean_diff:.2f}")
        st.metric("TOST P-value (for Equivalence)", f"{p_tost:.4f}")
        
        if p_tost < 0.05: st.success(f"**PASS: Lots are Statistically Equivalent.**")
        else: st.error(f"**FAIL: Lots are NOT Statistically Equivalent.**")
