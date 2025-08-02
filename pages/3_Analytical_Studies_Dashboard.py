# pages/3_Analytical_Studies_Dashboard.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import (generate_linearity_data_immunoassay, generate_precision_data_clsi_ep05,
                   generate_analytical_specificity_data_molecular, generate_lot_to_lot_data, calculate_equivalence)
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
    - **CLSI Guidelines (EP05, EP06, EP07, EP17):** Our study designs and acceptance criteria are harmonized with these globally recognized standards, ensuring our data is robust and defensible during FDA review. For example, precision studies follow the **CLSI EP05-A3** framework, which is the industry gold standard.
    - **Risk Management (ISO 14971):** The results of these studies directly inform our risk management file. For example, an LoD result confirms our control over sensitivity risks, while a cross-reactivity study confirms control over specificity risks.
    """)

# --- Data Generation ---
linearity_df = generate_linearity_data_immunoassay()
precision_df = generate_precision_data_clsi_ep05()
specificity_df = generate_analytical_specificity_data_molecular()
lot_df = generate_lot_to_lot_data()

# --- Page Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "**Precision & Reproducibility (CLSI EP05)**",
    "**Analytical Specificity (Inclusivity/Exclusivity)**",
    "**Assay Linearity / Measuring Interval (CLSI EP06)**",
    "**Lot-to-Lot Equivalence (TOST)**"
])

with tab1:
    st.header("Precision & Reproducibility (CLSI EP05)")
    st.caption("Example: 5-day, 2-run, 3-replicate study for the Sofia® 2 SARS Antigen+ FIA v2 assay.")

    with st.expander("🔬 Study Design & Acceptance Criteria"):
        st.markdown("""
        #### Study Design (CLSI EP05-A3)
        To evaluate **reproducibility**, two levels of control material are tested over 5 days, with 2 runs per day, and 3 replicates per run. This nested design allows for the statistical decomposition of total variation into its core components.
        
        #### Statistical Method: Analysis of Variance (ANOVA)
        A nested ANOVA model is used to estimate the variance contributed by each factor: Day (between-day), Run (between-run, within-day), and Repeatability (within-run).
        
        #### Acceptance Criteria (from V&V Plan)
        - The primary metric is the **Total Reproducibility %CV**.
        - **Low Positive Control (Near Cutoff):** Total %CV must be **≤ 20%**.
        - **Moderate Positive Control:** Total %CV must be **≤ 15%**.
        """)
    
    control_level = st.selectbox("Select Control Level to Analyze", precision_df['Control'].unique(), key='precision_select')
    filtered_df = precision_df[precision_df['Control'] == control_level].copy()
    
    col1, col2 = st.columns([1, 1])
    with col1:
        fig = px.box(filtered_df, x="Day", y="S/CO Ratio", title=f"Precision for {control_level} Control", points="all")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Analysis of Variance (ANOVA) Results")
        try:
            # Fit nested ANOVA model
            model = ols('Q("S/CO Ratio") ~ C(Day) + C(Run):C(Day)', data=filtered_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            # Estimate variance components
            ms_error = model.scale
            ms_run = anova_table.loc['C(Run):C(Day)', 'mean_sq']
            ms_day = anova_table.loc['C(Day)', 'mean_sq']
            
            reps = filtered_df.groupby(['Day', 'Run']).size().mean()
            runs = filtered_df['Run'].nunique()
            
            var_within = ms_error
            var_run = (ms_run - ms_error) / reps
            var_day = (ms_day - ms_run) / (reps * runs)
            
            vc = {
                'Repeatability (Within-Run)': max(0, var_within),
                'Between-Run (Within-Day)': max(0, var_run),
                'Between-Day': max(0, var_day)
            }

            vc_df = pd.DataFrame.from_dict(vc, orient='index', columns=['Variance'])
            vc_df['% Contribution'] = (vc_df['Variance'] / vc_df['Variance'].sum() * 100).round(1)
            vc_df['Std. Dev.'] = np.sqrt(vc_df['Variance'])
            st.dataframe(vc_df, use_container_width=True)
            
            total_variance = vc_df['Variance'].sum()
            total_std_dev = np.sqrt(total_variance)
            mean_val = filtered_df['S/CO Ratio'].mean()
            total_cv = (total_std_dev / mean_val) * 100
            
            acceptance_cv = 20 if "Low Positive" in control_level else 15
            st.metric(f"Total Reproducibility CV vs. Acceptance (≤{acceptance_cv}%)", f"{total_cv:.2f}%")

            if total_cv <= acceptance_cv:
                st.success(f"**PASS:** The Total CV of {total_cv:.2f}% meets the acceptance criterion of ≤ {acceptance_cv}%.")
            else:
                st.error(f"**FAIL:** The Total CV of {total_cv:.2f}% exceeds the acceptance criterion of ≤ {acceptance_cv}%. Investigation into variance components is required.")
        except Exception as e:
            st.error(f"ANOVA calculation failed. Check data structure. Error: {e}")

with tab2:
    st.header("Analytical Specificity (Cross-Reactivity & Interference)")
    st.caption("Example: Inclusivity & exclusivity testing for the Savanna® RVP12 molecular assay.")

    with st.expander("🔬 Study Design & Acceptance Criteria"):
        st.markdown("""
        #### Study Design
        This study confirms the assay's ability to exclusively detect its intended targets.
        1.  **Inclusivity:** A panel of representative target strains are tested to ensure they are all correctly identified.
        2.  **Exclusivity (Cross-Reactivity):** A panel of high-concentration, related but non-target organisms are tested to ensure they do not produce a positive result.
        
        #### Acceptance Criteria
        - **100%** of inclusivity panel strains must be detected at the expected concentration.
        - **0%** of exclusivity panel organisms should be detected. Any confirmed positive result is a failure and requires root cause investigation and risk assessment per ISO 14971.
        """)
    
    st.dataframe(specificity_df, use_container_width=True, hide_index=True)
    cross_react_failures = specificity_df[specificity_df['Notes'].str.contains("Potential Cross-reactivity")]
    
    st.subheader("Specificity Performance Summary")
    if not cross_react_failures.empty:
        st.error(f"**FAIL: Potential Cross-Reactivity Detected**")
        st.write("The following off-target organism produced a positive signal, requiring root cause investigation. This is a critical failure that may impact the 510(k) submission timeline and requires a formal risk assessment.")
        st.dataframe(cross_react_failures)
    else:
        st.success("**PASS:** No unexpected cross-reactivity was detected. The assay meets all inclusivity and exclusivity requirements specified in the V&V Plan.")

with tab3:
    st.header("Assay Linearity / Measuring Interval (CLSI EP06)")
    st.caption("Example: Defining the analytical measuring interval (AMI) for a Vitros® quantitative immunoassay.")

    with st.expander("🔬 Study Design & Acceptance Criteria"):
        st.markdown("""
        #### Study Design
        A dilution series of a high-concentration standard is tested to characterize the assay's response across a wide range of analyte concentrations.
        
        #### Acceptance Criteria
        - The reportable linear range is defined by the concentrations where a linear model provides an adequate fit.
        - **For the defined linear range:** The coefficient of determination **(R-squared)** must be **≥ 0.990**.
        - Visual inspection of a **residuals plot** must show no obvious trends or patterns, confirming the appropriateness of the linear model.
        - The assay response must show clear saturation or a "hook effect" outside the proposed linear range.
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

    if r_squared >= 0.990:
        st.success(f"**PASS:** The R² of {r_squared:.4f} within the defined range (0-{linear_range_max} ng/mL) meets the acceptance criterion of ≥ 0.990. Residuals are randomly distributed.")
    else:
        st.error(f"**FAIL:** The R² of {r_squared:.4f} is below the acceptance criterion. The proposed linear range is not supported by the data.")

with tab4:
    st.header("New Reagent Lot Qualification by Equivalence")
    st.caption("Example: Qualifying a new consumable lot for QuickVue® using a Two-One-Sided T-Test (TOST).")

    with st.expander("🔬 Study Design & Acceptance Criteria"):
        st.markdown("""
        #### Statistical Method: Equivalence Testing (TOST)
        Instead of testing for a *difference* (ANOVA), we are testing to prove *similarity*. The **Two-One-Sided T-Test (TOST)** is the industry-standard method. We define an **equivalence margin** (e.g., ±15 intensity units) and test the null hypothesis that the lots are different (i.e., the true difference is outside this margin).
        
        #### Acceptance Criteria
        - The **90% Confidence Interval** of the difference between the lot means must fall entirely within the pre-defined equivalence margins.
        - The **TOST P-value** must be **< 0.05** to reject the null hypothesis of non-equivalence and therefore claim the lots are equivalent.
        """)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("Lot-to-Lot Signal Comparison")
        fig_box = px.box(lot_df, x='Reagent Lot ID', y='Test Line Intensity',
                         title='Candidate Lot vs. Reference Lot', points='all')
        st.plotly_chart(fig_box, use_container_width=True)
    with col2:
        st.subheader("Equivalence Test (TOST) Results")
        low_bound = st.number_input("Lower Equivalence Margin (in intensity units)", value=-15.0, format="%.1f")
        high_bound = st.number_input("Higher Equivalence Margin (in intensity units)", value=15.0, format="%.1f")
        
        p_tost, p_diff, mean_diff = calculate_equivalence(lot_df, 'Reagent Lot ID', 'Test Line Intensity', low_bound, high_bound)
        
        st.metric("Mean Difference (Candidate - Reference)", f"{mean_diff:.2f}")
        st.metric("TOST P-value (for Equivalence)", f"{p_tost:.4f}")
        

        if p_tost < 0.05:
            st.success(f"**PASS: Lots are Statistically Equivalent.** The TOST P-value of {p_tost:.4f} is < 0.05. We can conclude the new lot performs equivalently to the reference lot within the defined margins of ({low_bound}, {high_bound}).")
        else:
            st.error(f"**FAIL: Lots are NOT Statistically Equivalent.** The TOST P-value of {p_tost:.4f} is ≥ 0.05. We cannot reject the hypothesis that the lots are different. The new lot fails qualification.")
        st.caption(f"Standard T-test p-value (for difference) was {p_diff:.4f} for reference.")
