# app.py (Final, Monolithic, World-Class Version with ALL Content and Enhancements)

# --- IMPORTS (CLEANED & ORGANIZED) ---

# Standard Library
from typing import List, Dict, Any, Tuple, Optional, Callable

# Third-Party Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pandas.io.formats.style import Styler
import streamlit as st


# --- PAGE CONFIGURATION (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    layout="wide",
    page_title="V&V Executive Command Center | Portfolio",
    page_icon="🎯"
)


# --- UTILITY & HELPER FUNCTIONS ---

def render_director_briefing(title: str, content: str, reg_refs: str, business_impact: str) -> None:
    """
    Renders a formatted container for strategic context and director-level briefings.

    Args:
        title (str): The title of the briefing.
        content (str): The main markdown content of the briefing.
        reg_refs (str): A string listing relevant regulatory references.
        business_impact (str): A string describing the business impact.
    """
    with st.container(border=True):
        st.subheader(f"🎯 {title}")
        st.markdown(content)
        st.info(f"**Business Impact:** {business_impact}")
        st.warning(f"**Regulatory Mapping:** {reg_refs}")


def render_metric_card(
    title: str, 
    description: str, 
    viz_function: Callable[[str], Any], 
    insight: str, 
    reg_context: str, 
    key: str = ""
) -> None:
    """
    Renders a formatted container for a specific metric or visualization.
    This function is now robustly handling Plotly, Matplotlib, and DataFrame objects.

    Args:
        title (str): The title of the metric card.
        description (str): A markdown description of the metric.
        viz_function (Callable[[str], Any]): The function that generates the visualization.
        insight (str): A string containing the actionable insight derived from the viz.
        reg_context (str): A string describing the regulatory context.
        key (str, optional): A unique key for Streamlit widgets. Defaults to "".
    """
    with st.container(border=True):
        st.subheader(title)
        st.markdown(f"*{description}*")
        st.warning(f"**Regulatory Context:** {reg_context}")
        
        # Generate the visualization object from the provided function
        viz_object = viz_function(key)
        
        # --- ROBUST VISUALIZATION RENDERING ---
        if viz_object is not None:
            if isinstance(viz_object, plt.Figure):
                st.pyplot(viz_object)
            elif isinstance(viz_object, go.Figure):
                st.plotly_chart(viz_object, use_container_width=True)
            elif isinstance(viz_object, pd.io.formats.style.Styler):
                st.dataframe(viz_object, use_container_width=True, hide_index=True)
            elif isinstance(viz_object, pd.DataFrame):
                st.dataframe(viz_object, use_container_width=True, hide_index=True)
            else:
                # Fallback for functions that render their own content and return None
                pass 
        
        st.success(f"**Actionable Insight:** {insight}")


# --- VISUALIZATION & DATA GENERATORS ---
# Note: Functions are standardized to return visualization objects where applicable.

def create_opex_dashboard(key: str) -> Tuple[go.Figure, go.Figure]:
    """
    Generates an enhanced OpEx dashboard with a budget gauge and a burndown chart.

    Args:
        key (str): A unique key for Streamlit widgets (unused here but kept for consistency).

    Returns:
        Tuple[go.Figure, go.Figure]: A tuple containing the gauge figure and the burndown figure.
    """
    budget = 5_000_000
    actual = 4_200_000
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=actual,
        title={'text': "Annual V&V OpEx: Budget vs. Actual"},
        gauge={
            'axis': {'range': [None, budget]},
            'bar': {'color': "cornflowerblue"},
            'steps': [
                {'range': [0, budget * 0.9], 'color': 'lightgreen'},
                {'range': [budget * 0.9, budget], 'color': 'lightyellow'}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': budget}
        }
    ))

    months = pd.date_range(start="2023-01-01", periods=12, freq='ME').strftime('%b')
    monthly_budget = np.ones(12) * (budget / 12)
    np.random.seed(42)
    rng = np.random.default_rng(seed=42)
    actual_spend = monthly_budget + rng.normal(0, 30000, 12)
    df = pd.DataFrame({'Month': months, 'Budget': monthly_budget, 'Actual': actual_spend})
    df['Variance'] = df['Budget'] - df['Actual']
    df['Cumulative Variance'] = df['Variance'].cumsum()

    fig_burn = make_subplots(specs=[[{"secondary_y": True}]])
    fig_burn.add_trace(go.Bar(x=df['Month'], y=df['Actual'], name='Actual Spend', marker_color='cornflowerblue'), secondary_y=False)
    fig_burn.add_trace(go.Scatter(x=df['Month'], y=df['Budget'], name='Budget', mode='lines+markers', line=dict(color='black', dash='dash')), secondary_y=False)
    fig_burn.add_trace(go.Scatter(x=df['Month'], y=df['Cumulative Variance'], name='Cumulative Variance', line=dict(color='red')), secondary_y=True)
    
    fig_burn.update_layout(title_text='Monthly OpEx: Actual vs. Budget & Cumulative Variance')
    fig_burn.update_yaxes(title_text="Monthly Spend ($)", secondary_y=False)
    fig_burn.update_yaxes(title_text="Cumulative Variance ($)", secondary_y=True, showgrid=False)
    
    return fig_gauge, fig_burn


def create_copq_modeler(key: str) -> go.Figure:
    """
    Creates an interactive modeler for Cost of Poor Quality and its relation to V&V spend.
    Also renders the interactive widgets directly.

    Args:
        key (str): A unique key prefix for Streamlit widgets.

    Returns:
        go.Figure: A scatter plot showing the modeled impact of V&V spend on COPQ.
    """
    st.subheader("Interactive Cost of Poor Quality (COPQ) Modeler")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Internal Failure Costs (Annualized)**")
        scrap_rate = st.slider("Scrap Rate (%)", 0.5, 5.0, 2.5, 0.1, key=f"copq_scrap_{key}")
        rework_hours = st.number_input("Rework Hours (per month)", value=250, key=f"copq_rework_{key}")
        
        st.markdown("**External Failure Costs (Annualized)**")
        complaint_hours = st.number_input("Complaint Investigation Hours (per month)", value=150, key=f"copq_complaint_{key}")
        warranty_claims = st.number_input("Warranty Claims ($ per year)", value=75000, key=f"copq_warranty_{key}")
    
    # Constants for calculation
    cost_per_rework_hr = 120
    cost_per_complaint_hr = 150
    cost_of_goods = 10_000_000  # Annual COGS
    
    internal_scrap = (scrap_rate / 100) * cost_of_goods
    internal_rework = rework_hours * cost_per_rework_hr * 12
    external_complaint = complaint_hours * cost_per_complaint_hr * 12
    total_copq = internal_scrap + internal_rework + external_complaint + warranty_claims

    with col2:
        st.metric("Total Annualized COPQ", f"${total_copq:,.0f}", help="Scrap + Rework + Complaints + Warranty")
        st.info("This model quantifies the financial impact of quality failures that robust V&V aims to prevent.")

    # AI/ML Model: Correlation between V&V Spend and COPQ
    np.random.seed(42)
    v_v_spend = np.random.uniform(200_000, 1_000_000, 20)
    copq = 2_500_000 - (1.5 * v_v_spend) + np.random.normal(0, 200_000, 20)
    df_corr = pd.DataFrame({'V&V Spend during Development ($)': v_v_spend, 'Post-Launch COPQ ($)': copq})
    
    fig_corr = px.scatter(df_corr, x='V&V Spend during Development ($)', y='Post-Launch COPQ ($)', 
                          trendline='ols', trendline_color_override='red',
                          title='AI-Modeled Impact of V&V Investment on COPQ')
    return fig_corr


def create_audit_dashboard(key: str) -> Styler:
    """
    Generates a styled DataFrame for the audit readiness dashboard.
    The return type hint has been corrected to use the imported 'Styler' class.
    """
    audit_data = {
        "Audit/Inspection": ["FDA QSR Inspection", "ISO 13485 Recertification", "Internal Audit Q2", "MDSAP Audit"],
        "Date": ["2023-11-15", "2023-08-20", "2023-06-10", "2023-03-05"],
        "V&V-related Findings": [0, 1, 1, 2],
        "Outcome": ["NAI (No Action Indicated)", "Passed w/ Minor Obs.", "Passed", "Passed w/ Minor Obs."]
    }
    df = pd.DataFrame(audit_data)

    def style_outcome(val: str) -> str:
        color = 'white'
        if "NAI" in val or val == "Passed":
            color = 'lightgreen'
        elif "Minor" in val:
            color = 'lightyellow'
        return f'background-color: {color}'
    
    styled_df = df.style.map(style_outcome, subset=['Outcome'])
    return styled_df


def create_qms_kanban(key: str) -> None:
    """
    Simulates and directly renders a Kanban board for V&V tasks in the QMS.

    Args:
        key (str): A unique key for Streamlit widgets (unused).
    """
    tasks = {
        "New": [],
        "Investigation": ["CAPA-2024-001: Investigate Lot A2301-B False Positives", "NCMR-1088: OOT result in stability pull"],
        "Root Cause Analysis": ["CAPA-2023-014: Reagent leak correlation"],
        "Effectiveness Check": ["ECO-091: Verify new packaging seal integrity"],
        "Closed": ["CAPA-2023-009: Software patch for UI freeze", "NCMR-1056: Raw material equivalency"]
    }
    st.subheader("V&V Tasks in the Quality System")
    cols = st.columns(len(tasks))
    for i, (status, items) in enumerate(tasks.items()):
        with cols[i]:
            st.markdown(f"**{status}**")
            if items:
                for item in items:
                    st.info(item)
            else:
                st.markdown("_No items_")


def create_method_transfer_dashboard(key: str) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Generates charts for the global method transfer dashboard.

    Args:
        key (str): A unique key for Streamlit widgets (unused).

    Returns:
        Tuple[go.Figure, pd.DataFrame]: A tuple containing the bar chart and the status DataFrame.
    """
    metrics = ['Precision (%CV)', 'Bias (%)', 'TMV Protocol Pass Rate (%)']
    san_diego = [1.8, 0.5, 100]
    athens_oh = [2.1, -0.8, 95]
    df_metrics = pd.DataFrame({'Metric': metrics, 'San Diego (Source)': san_diego, 'Athens, OH (Receiving)': athens_oh})
    
    fig_bar = px.bar(df_metrics, x='Metric', y=['San Diego (Source)', 'Athens, OH (Receiving)'], 
                     barmode='group', title='Method Performance: Inter-site Comparability')
    
    transfer_status = {
        "Protocol": ["AVP-LOD-01 (LoD)", "AVP-PREC-01 (Precision)", "AVP-LIN-01 (Linearity)", "AVP-STAB-01 (Stability)"],
        "Status": ["Complete", "Complete", "Executing", "Pending Start"]
    }
    df_status = pd.DataFrame(transfer_status)
    return fig_bar, df_status


def create_pipeline_advisor(key: str) -> go.Figure:
    """
    Creates the AI-powered R&D pipeline risk advisor, including UI widgets.

    Args:
        key (str): A unique key prefix for Streamlit widgets.

    Returns:
        go.Figure: A scatter plot functioning as a "Magic Quadrant" for the R&D pipeline.
    """
    # AI/ML Model: Train a model on historical data
    np.random.seed(42)
    historical_data = pd.DataFrame({
        'New_Tech_Count': np.random.randint(0, 5, 20),
        'Complexity_Score': np.random.randint(1, 11, 20),
        'Target_LoD_Tightness': np.random.uniform(0.1, 1, 20),
        'V_V_Duration': np.random.uniform(3, 18, 20)
    })
    feature_names = ['New_Tech_Count', 'Complexity_Score', 'Target_LoD_Tightness']
    X = historical_data[feature_names]
    y = historical_data['V_V_Duration']
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    
    # UI for new project input
    st.subheader("Forecast V&V Risk for New R&D Projects")
    col1, col2, col3 = st.columns(3)
    with col1:
        new_tech = st.slider("Number of New Technologies", 0, 5, 2, key=f"pipe_tech_{key}")
    with col2:
        complexity = st.slider("Assay Complexity Score (1-10)", 1, 10, 5, key=f"pipe_comp_{key}")
    with col3:
        lod_tightness = st.slider("Target LoD Tightness (vs. Predicate)", 0.1, 2.0, 1.0, 0.1, key=f"pipe_lod_{key}")
    
    # Prediction
    new_project_data = pd.DataFrame([[new_tech, complexity, lod_tightness]], columns=feature_names)
    predicted_duration = model.predict(new_project_data)[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted V&V Duration (Months)", f"{predicted_duration:.1f}")
    with col2:
        # Simple heuristic for success probability
        success_prob = max(0, 95 - (predicted_duration * 2))
        st.metric("Predicted Technical Success Probability", f"{success_prob:.1f}%")
    
    # Visualization: Magic Quadrant
    pipeline_projects = pd.DataFrame({
        'Project': ['Project Alpha (LFA)', 'Project Beta (Molecular)', 'Project Gamma (Immuno)', 'Project Delta (Digital)'],
        'V&V Complexity': [3, 8, 6, 9],
        'Predicted ROI (%)': [50, 300, 150, 500],
        'V&V Cost ($M)': [0.5, 2.0, 1.2, 3.5]
    })
    fig_quad = px.scatter(pipeline_projects, x='V&V Complexity', y='Predicted ROI (%)', size='V&V Cost ($M)', 
                          color='Project', text='Project', title='Strategic R&D Pipeline Advisor')
    fig_quad.update_traces(textposition='top center')
    return fig_quad


def create_automation_dashboard(key: str) -> Tuple[go.Figure, go.Figure]:
    """
    Generates charts for the Test Automation Dashboard.

    Args:
        key (str): A unique key for Streamlit widgets (unused).

    Returns:
        Tuple[go.Figure, go.Figure]: A tuple containing the pie chart and the dual-axis bar/line chart.
    """
    automation_data = pd.DataFrame({'Category': ['Automated', 'Manual'], 'Count': [2850, 1300]})
    fig_pie = px.pie(automation_data, values='Count', names='Category', title='Test Case Distribution',
                     color_discrete_map={'Automated': 'cornflowerblue', 'Manual': 'lightgrey'})
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    exec_time = [120, 110, 105, 90, 80, 75]
    coverage = [45, 50, 55, 60, 62, 65]
    fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
    fig_dual.add_trace(go.Bar(x=months, y=exec_time, name='Execution Time (Hours)'), secondary_y=False)
    fig_dual.add_trace(go.Scatter(x=months, y=coverage, name='Automation Coverage (%)', mode='lines+markers'), secondary_y=True)
    fig_dual.update_layout(title_text='Automation Impact on V&V Execution')
    fig_dual.update_yaxes(title_text="Total Execution Time (Hours)", secondary_y=False)
    fig_dual.update_yaxes(title_text="Coverage (%)", secondary_y=True)

    return fig_pie, fig_dual


def create_instrument_utilization_dashboard(key: str) -> Tuple[go.Figure, go.Figure]:
    """
    Generates a heatmap and AI forecast for instrument utilization.

    Args:
        key (str): A unique key for Streamlit widgets (unused).

    Returns:
        Tuple[go.Figure, go.Figure]: A tuple containing the heatmap and the forecast chart.
    """
    np.random.seed(50)
    # Heatmap Data
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hours = [f'{h}:00' for h in range(24)]
    util_data = np.random.randint(0, 100, size=(7, 24))
    util_data[1:5, 8:17] = np.random.randint(70, 100, size=(4, 9))  # Simulate high usage during work hours
    fig_heatmap = px.imshow(util_data, x=hours, y=days, aspect="auto",
                            color_continuous_scale='RdYlGn_r',
                            title='Instrument Utilization Heatmap (Platform X)')
    
    # AI/ML Forecast Data & Model
    months = np.arange(1, 13).reshape(-1, 1)
    historical_util = np.array([40, 42, 45, 50, 55, 60, 62, 68, 70, 75, 80, 85])
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(months, historical_util)
    future_months = np.arange(13, 19).reshape(-1, 1)
    predicted_util = model.predict(future_months)
    
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=np.arange(1, 13), y=historical_util, mode='lines+markers', name='Historical Utilization'))
    fig_forecast.add_trace(go.Scatter(x=np.arange(13, 19), y=predicted_util, mode='lines+markers', name='AI Forecast', line=dict(color='red', dash='dash')))
    fig_forecast.add_hline(y=90, line_dash="dot", line_color="orange", annotation_text="Capacity Threshold")
    fig_forecast.update_layout(title='AI-Powered Utilization Forecast & CapEx Planning', xaxis_title='Month', yaxis_title='Utilization (%)')

    return fig_heatmap, fig_forecast


def create_portfolio_health_dashboard(key: str) -> Styler:
    """
    Generates the RAG status styled DataFrame for the project portfolio.
    The return type hint has been corrected to use the imported 'Styler' class.
    """
    data = {
        'Project': ["ImmunoPro-A", "MolecularDX-2", "CardioScreen-X", "NextGen Platform SW"],
        'Phase': ["System V&V", "Clinical Study", "Feasibility", "Architecture"],
        'Schedule Status': ["Green", "Green", "Amber", "Green"],
        'Budget Status': ["Green", "Amber", "Green", "Green"],
        'Technical Risk': ["Green", "Amber", "Red", "Amber"],
        'Resource Strain': ["Amber", "Green", "Red", "Red"]
    }
    df = pd.DataFrame(data)

    def style_rag(val: str) -> str:
        color = 'white'
        if val == 'Green':
            color = 'lightgreen'
        elif val == 'Amber':
            color = 'lightyellow'
        elif val == 'Red':
            color = '#ffcccb'
        return f'background-color: {color}'

    styled_df = df.style.map(style_rag, subset=['Schedule Status', 'Budget Status', 'Technical Risk', 'Resource Strain'])
    return styled_df

def create_resource_allocation_matrix(key: str) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Generates an enhanced, actionable resource allocation dashboard.
    This function was refactored to return the figure instead of rendering it directly.

    Args:
        key (str): A unique key for Streamlit widgets (unused).

    Returns:
        Tuple[go.Figure, pd.DataFrame]: A tuple containing the allocation bar chart and the DataFrame of over-allocated members.
    """
    team_data = {
        'Team Member': ['Alice', 'Bob', 'Charlie', 'Diana', 'Ethan', 'Fiona'],
        'Primary Skill': ['qPCR', 'Statistics (Python)', 'ELISA', 'ISO 14971', 'Statistics (Python)', 'qPCR (Junior)'],
        'ImmunoPro-A': [50, 50, 0, 0, 0, 100],
        'MolecularDX-2': [10, 25, 60, 5, 0, 0],
        'CardioScreen-X': [40, 0, 40, 60, 50, 0],
        'Sustaining': [0, 25, 10, 35, 50, 0]
    }
    df = pd.DataFrame(team_data)
    df['Total Allocation'] = df[['ImmunoPro-A', 'MolecularDX-2', 'CardioScreen-X', 'Sustaining']].sum(axis=1)
    df['Status'] = df['Total Allocation'].apply(lambda x: 'Over-allocated' if x > 100 else ('At Capacity' if x >= 90 else 'Available'))
    
    fig = px.bar(df.sort_values('Total Allocation'), 
                 x='Total Allocation', 
                 y='Team Member',
                 color='Status',
                 text='Primary Skill',
                 orientation='h',
                 title='V&V Team Capacity & Strategic Alignment',
                 color_discrete_map={'Over-allocated': 'red', 'At Capacity': 'orange', 'Available': 'green'})
    
    fig.add_vline(x=100, line_width=2, line_dash="dash", line_color="black", annotation_text="100% Capacity")
    fig.update_layout(xaxis_title="Total Allocation (%)", yaxis_title="Team Member", legend_title="Status")
    fig.update_traces(textposition='inside', textfont=dict(size=12, color='white'))
    
    over_allocated_df = df[df['Total Allocation'] > 100][['Team Member', 'Total Allocation']]
    return fig, over_allocated_df


def create_lessons_learned_search(key: str) -> None:
    """
    Simulates and directly renders an NLP-powered search engine for the knowledge base.

    Args:
        key (str): A unique key prefix for Streamlit widgets.
    """
    # Mock Knowledge Base
    knowledge_base = {
        "DOC-001 (Immuno- Assay Stability)": "Initial stability run failed at 3 months due to improper blocking agent concentration. Root cause was determined to be a supplier change in BSA. Corrective action involved re-validating the new supplier and adjusting the protocol. See CAPA-2022-012.",
        "DOC-002 (Molecular Assay V&V)": "Cross-reactivity testing for the molecular panel showed minor signal with Influenza C, which was not a specified requirement. Risk assessment concluded this was low risk but it was added to the product insert. This highlights the need for broader cross-reactant panels early in development.",
        "DOC-003 (Software CSV)": "The LIMS integration module failed during system testing due to an undocumented API change from the vendor. This caused a 2-week project delay. Lesson learned: Implement automated API contract testing as part of the CI/CD pipeline for all external integrations.",
        "DOC-004 (ECO-088 Reagent Change)": "A change in a critical buffer supplier post-launch required a full analytical bridging study. The study showed a statistically significant shift in the negative control population, requiring a full re-validation of QC ranges. Emphasizes the risk of seemingly 'equivalent' supplier changes."
    }
    docs = list(knowledge_base.values())
    doc_titles = list(knowledge_base.keys())

    query = st.text_input("Search the V&V Knowledge Base (e.g., 'supplier change' or 'API integration')", key=f"search_{key}")

    if query:
        # AI/ML Model: TF-IDF + Cosine Similarity
        corpus = [query] + docs
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(corpus)
        
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        sim_scores = sorted(list(enumerate(cosine_sim[0])), key=lambda x: x[1], reverse=True)
        
        st.write("#### Search Results:")
        results_found = False
        for i, score in sim_scores[:3]:
            if score > 0.05:
                results_found = True
                with st.container(border=True):
                    st.markdown(f"**Document:** `{doc_titles[i]}`")
                    st.markdown(f"**Relevance Score:** {score:.2f}")
                    st.info(docs[i])
        if not results_found:
            st.warning("No relevant documents found.")


def run_requirement_risk_nlp_model(key: str) -> go.Figure:
    """
    Uses NLP to create a risk quadrant for requirements.

    Args:
        key (str): A unique key for Streamlit widgets (unused).

    Returns:
        go.Figure: A scatter plot visualizing requirement risk.
    """
    reqs = [
        "The system shall process samples in under 5 minutes.", 
        "The assay must have a clinical sensitivity of 95% for Target A.", 
        "The user interface should be intuitive and easy to use.",
        "Results must be displayed quickly after the run is complete.",
        "The software must not crash during normal operation.",
        "The system shall be robust against common user errors.", 
        "Analytical sensitivity (LoD) shall be <= 25 copies/mL.",
    ]
    labels = [0, 0, 1, 1, 0, 1, 0]  # Ambiguity Label (1=ambiguous)
    criticality = [7, 10, 5, 6, 9, 8, 10]  # SME-assigned criticality score
    df = pd.DataFrame({'Requirement': reqs, 'Risk_Label': labels, 'Criticality': criticality})

    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(df['Requirement'])
    y = df['Risk_Label']
    model = LogisticRegression(random_state=42).fit(X, y)

    df['Ambiguity Score'] = model.predict_proba(X)[:, 1]
    
    fig = px.scatter(df, x='Ambiguity Score', y='Criticality', text=df.index,
                     title='Requirement Prioritization Matrix',
                     labels={'Ambiguity Score': 'Predicted Ambiguity / V&V Risk', 'Criticality': 'Business & Patient Safety Criticality'})
    
    fig.update_traces(textposition='top center', textfont=dict(size=12))
    fig.add_vline(x=0.5, line_width=1, line_dash="dash", line_color="black")
    fig.add_hline(y=7.5, line_width=1, line_dash="dash", line_color="black")
    
    fig.add_annotation(x=0.75, y=9, text="High Risk - Clarify Now!", showarrow=False, font=dict(color='red'))
    fig.add_annotation(x=0.25, y=9, text="Low Risk - Proceed", showarrow=False, font=dict(color='green'))
    fig.add_annotation(x=0.75, y=6, text="Clarify if Time Permits", showarrow=False, font=dict(color='orange'))
    fig.add_annotation(x=0.25, y=6, text="Monitor", showarrow=False, font=dict(color='grey'))

    return fig


def run_cqa_forecasting_model(key: str) -> go.Figure:
    """
    Uses a time-series forecasting model (ARIMA) to predict future CQA values.

    Args:
        key (str): A unique key for Streamlit widgets (unused).

    Returns:
        go.Figure: A time-series chart with historical data and the ARIMA forecast.
    """
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100)
    data = 10 + np.random.randn(100).cumsum() * 0.05 + np.linspace(0, 1.5, 100)
    ts = pd.Series(data, index=dates)
    
    model = ARIMA(ts, order=(5, 1, 0)).fit()
    forecast = model.get_forecast(steps=30)
    forecast_df = forecast.summary_frame()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts, mode='lines', name='Historical Data'))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean'], mode='lines', name='Forecast', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean_ci_lower'], fill='tonexty', mode='lines', line_color='rgba(255,0,0,0.2)', name='95% Confidence Interval'))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean_ci_upper'], fill='tonexty', mode='lines', line_color='rgba(255,0,0,0.2)', name='95% Confidence Interval', showlegend=False))
    fig.add_hline(y=12.5, line_dash="dash", line_color="orange", annotation_text="Upper Spec Limit")
    fig.update_layout(title='AI-Powered Forecast of Critical Quality Attribute (CQA)', yaxis_title='CQA Value')
    return fig


def run_defect_root_cause_model(key: str) -> go.Figure:
    """
    Uses NLP to automatically classify defect reports into root cause categories.

    Args:
        key (str): A unique key for Streamlit widgets (unused).

    Returns:
        go.Figure: A pie chart showing the distribution of predicted root causes.
    """
    defects = [
        "Pointer exception in data processing module when sample ID is null.",
        "The system requirement for sample throughput is physically impossible to meet.",
        "UI hangs when the user clicks 'Start' before the reagent is loaded.",
        "The plastic housing for the sensor cracked after 100 cycles.",
        "Requirement URS-015 is in direct conflict with URS-021.",
        "Incorrect driver version for the pump controller was deployed.",
        "The third-party API for LIS is returning a 500 error intermittently.",
        "Off-by-one error in the loop that calculates final concentration.",
    ]
    labels = ["Coding Error", "Requirement Issue", "Coding Error", "Hardware Issue", "Requirement Issue", "Integration Issue", "Integration Issue", "Coding Error"]
    df = pd.DataFrame({'Defect_Description': defects, 'Root_Cause': labels})

    tfidf = TfidfVectorizer(stop_words='english')
    X_train, _, y_train, _ = train_test_split(df['Defect_Description'], df['Root_Cause'], test_size=0.3, random_state=42)
    X_train_vec = tfidf.fit_transform(X_train)
    model = RandomForestClassifier(random_state=42).fit(X_train_vec, y_train)
    
    df['Predicted_Cause'] = model.predict(tfidf.transform(df['Defect_Description']))
    cause_counts = df['Predicted_Cause'].value_counts().reset_index()
    
    fig = px.pie(cause_counts, values='count', names='Predicted_Cause', title='AI-Predicted Defect Root Cause Distribution')
    return fig


def run_sentiment_analysis_model(key: str) -> go.Figure:
    """
    Applies sentiment analysis and creates advanced visualizations. Renders one chart directly.

    Args:
        key (str): A unique key for Streamlit widgets (unused).

    Returns:
        go.Figure: A time-series chart showing the trend of negative sentiment.
    """
    complaint_data = {
        'Date': pd.to_datetime(['2023-01-10', '2023-01-15', '2023-02-05', '2023-02-20', '2023-03-12', '2023-03-25']),
        'Type': ['Reagent Leak', 'Software Glitch', 'Instrument Error', 'Reagent Leak', 'Software Glitch', 'Instrument Error'],
        'Text': [
            "The reagent cartridge was leaking from the bottom seal.",
            "This is the third time this month the software has crashed mid-run. This is unacceptable and costing us a fortune!",
            "Error code 503 appeared on screen, the manual is not clear.",
            "Another leaky pack ruined a full plate. Absolutely terrible product quality.",
            "The new user interface is fantastic, much easier to use!",
            "The machine is making a loud grinding noise."
        ]
    }
    df = pd.DataFrame(complaint_data)
    df = pd.concat([df] * 5, ignore_index=True)  # Amplify data for better viz

    analyzer = SentimentIntensityAnalyzer()
    df['compound'] = df['Text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df['Sentiment'] = df['compound'].apply(lambda c: 'Positive' if c >= 0.05 else ('Negative' if c <= -0.05 else 'Neutral'))

    # Treemap Visualization (Rendered directly)
    fig_tree = px.treemap(df, path=[px.Constant("All Complaints"), 'Type', 'Sentiment'],
                          title='Hierarchical View of Complaint Sentiment by Category',
                          color='compound', color_continuous_scale='RdYlGn',
                          color_continuous_midpoint=0)
    fig_tree.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    st.plotly_chart(fig_tree, use_container_width=True)

    # Time-Series Visualization (Returned)
    df['Month'] = df['Date'].dt.to_period('M').dt.to_timestamp()
    monthly_sentiment = df.groupby('Month')['Sentiment'].value_counts(normalize=True).unstack().fillna(0)
    monthly_sentiment['Negative_Pct'] = monthly_sentiment.get('Negative', 0) * 100
    
    fig_ts = px.area(monthly_sentiment, x=monthly_sentiment.index, y='Negative_Pct',
                     title='Trend of Negative Sentiment in Complaints Over Time')
    fig_ts.update_layout(yaxis_title='Percentage of Complaints with Negative Sentiment (%)')
    
    return fig_ts


def plot_multivariate_anomaly_detection(key: str) -> go.Figure:
    """
    Generates a 3D plot for multivariate anomaly detection using Isolation Forest.

    Args:
        key (str): A unique key for Streamlit widgets (unused).

    Returns:
        go.Figure: A 3D scatter plot showing in-control vs. anomalous data points.
    """
    np.random.seed(101)
    in_control_data = np.random.multivariate_normal(mean=[10, 20, 5], cov=[[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]], size=300)
    anomalies = np.array([[11, 19, 7], [9, 21, 3]])
    data = np.vstack([in_control_data, anomalies])
    df = pd.DataFrame(data, columns=['Temp (°C)', 'Pressure (psi)', 'Flow Rate (mL/min)'])
    
    model = IsolationForest(contamination=0.01, random_state=42)
    df['Anomaly'] = model.fit_predict(df)
    df['Status'] = df['Anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'In Control')
    
    fig = px.scatter_3d(
        df, x='Temp (°C)', y='Pressure (psi)', z='Flow Rate (mL/min)',
        color='Status', color_discrete_map={'In Control': 'blue', 'Anomaly': 'red'},
        symbol='Status', size_max=10, title='AI-Powered Multivariate Process Monitoring'
    )
    fig.update_traces(marker=dict(size=4))
    return fig


def run_predictive_maintenance_model(key: str) -> plt.Figure:
    """
    Simulates instrument sensor data and uses SHAP for an explainable AI model.
    This version is updated to use the modern NumPy random number generator (RNG)
    to eliminate the FutureWarning.
    """
    # --- Data Generation & Model Training ---
    
    # 1. Create a local Random Number Generator instance. This is the new best practice.
    rng = np.random.default_rng(seed=42)
    
    data = []
    for i in range(10):
        will_fail = i >= 7
        for day in range(100):
            laser_drift = (day / 100) * 0.5 if will_fail else 0
            pressure_spike = (day / 100)**2 * 3 if will_fail else 0
            
            # 2. Use the 'rng' instance for all random calls instead of 'np.random'.
            laser_intensity = rng.normal(5 - laser_drift, 0.1)
            pump_pressure = rng.normal(50 + pressure_spike, 0.5)
            temp_fluctuation = rng.normal(37, 0.2 + (day/1000 if will_fail else 0))
            
            failure = 1 if will_fail and day > 95 else 0
            data.append([i, day, laser_intensity, pump_pressure, temp_fluctuation, failure])
    
    df = pd.DataFrame(data, columns=['Instrument_ID', 'Day', 'Laser_Intensity', 'Pump_Pressure', 'Temp_Fluctuation', 'Failure'])
    features = ['Laser_Intensity', 'Pump_Pressure', 'Temp_Fluctuation']
    X = df[features]
    y = df['Failure']
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
    
    # --- SHAP Analysis ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # --- Matplotlib Figure Capture ---
    # This part remains the same as the previous fix.
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Summary Plot: Feature Impact on Failure Prediction")
    fig = plt.gcf()
    
    return fig

def run_nlp_topic_modeling(key: str) -> None:
    """
    Applies NLP Topic Modeling (LDA) to simulated complaint data and renders a table.

    Args:
        key (str): A unique key for Streamlit widgets (unused).
    """
    complaint_docs = [
        "The software froze during a run and I had to restart the instrument.", "Error code 503 appeared on screen, the manual is not clear on this.",
        "The reagent cartridge was leaking from the bottom seal upon opening the box.", "Results seem consistently higher than the previous lot, we suspect a calibration issue.",
        "The machine is making a loud grinding noise during the initial spin cycle.", "I cannot get the system to calibrate properly after the last software update.",
        "The touch screen is unresponsive in the top left corner.", "Another case of a leaky reagent pack, this is the third time this month.",
        "The instrument UI is very slow to respond after starting a new batch.", "Calibration failed multiple times before finally passing.",
        "The seal on the reagent pack was broken, causing a spill inside the machine."
    ] * 5

    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(complaint_docs)
    
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(X)
    
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
        topics[f"Topic {topic_idx+1}"] = ", ".join(top_words)

    st.write("#### Automatically Discovered Complaint Themes:")
    st.table(pd.DataFrame.from_dict(topics, orient='index', columns=["Top Keywords"]))


def create_rtm_data_editor(key: str) -> None:
    """
    Renders an editable DataFrame for the Requirements Traceability Matrix and flags gaps.

    Args:
        key (str): A unique key for Streamlit widgets (unused).
    """
    df_data = [
        {"ID": "URS-001", "Requirement": "Assay must detect Target X with >95% clinical sensitivity.", "Risk": "High", "Linked Test Case": "AVP-SENS-01", "Status": "PASS"},
        {"ID": "DI-002", "Requirement": "Analytical sensitivity (LoD) shall be <= 50 copies/mL.", "Risk": "High", "Linked Test Case": "AVP-LOD-01", "Status": "PASS"},
        {"ID": "SRS-012", "Requirement": "Results screen must display patient ID.", "Risk": "Medium", "Linked Test Case": "SVP-UI-04", "Status": "PASS"},
        {"ID": "DI-003", "Requirement": "Assay must be stable for 12 months at 2-8°C.", "Risk": "High", "Linked Test Case": "AVP-STAB-01", "Status": "IN PROGRESS"},
        {"ID": "URS-003", "Requirement": "Assay must have no cross-reactivity with Influenza B.", "Risk": "Medium", "Linked Test Case": "", "Status": "GAP"},
    ]
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    gaps = df[df["Status"] == "GAP"]
    if not gaps.empty:
        st.error(f"**Critical Finding:** {len(gaps)} traceability gap(s) identified. This is a major audit finding and blocks design release.")


def plot_defect_burnup(key: str) -> go.Figure:
    """
    Generates a defect burnup chart.

    Args:
        key (str): A unique key for Streamlit widgets (unused).

    Returns:
        go.Figure: A burnup chart.
    """
    days = np.arange(1, 46)
    scope = np.ones(45) * 50
    scope[25:] = 60  # Scope creep
    np.random.seed(1)
    closed = np.linspace(0, 45, 45) + np.random.rand(45) * 2
    opened = np.linspace(5, 58, 45) + np.random.rand(45) * 3
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=scope, mode='lines', name='Total Scope', line=dict(dash='dash', color='grey')))
    fig.add_trace(go.Scatter(x=days, y=opened, mode='lines', name='Defects Opened (Cumulative)', fill='tozeroy', line=dict(color='rgba(255,0,0,0.5)')))
    fig.add_trace(go.Scatter(x=days, y=closed, mode='lines', name='Defects Closed (Cumulative)', fill='tozeroy', line=dict(color='rgba(0,128,0,0.5)')))
    fig.update_layout(title='Defect Open vs. Close Trend (Burnup Chart)', xaxis_title='Project Day', yaxis_title='Number of Defects')
    return fig


def plot_cpk_analysis(key: str) -> go.Figure:
    """
    Generates an interactive process capability (CpK) analysis plot.

    Args:
        key (str): A unique key prefix for Streamlit widgets.

    Returns:
        go.Figure: A histogram with interactive specification limits.
    """
    np.random.seed(42)
    data = np.random.normal(loc=10.2, scale=0.25, size=200)
    usl = st.slider("Upper Specification Limit (USL)", 9.0, 12.0, 11.0, key=f"usl_{key}")
    lsl = st.slider("Lower Specification Limit (LSL)", 8.0, 11.0, 9.0, key=f"lsl_{key}")
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    cpk = min((usl - mean) / (3 * std_dev), (mean - lsl) / (3 * std_dev))
    fig = px.histogram(data, nbins=30, title=f"Process Capability (CpK) Analysis | Calculated CpK = {cpk:.2f}")
    fig.add_vline(x=lsl, line_dash="dash", line_color="red", annotation_text="LSL")
    fig.add_vline(x=usl, line_dash="dash", line_color="red", annotation_text="USL")
    fig.add_vline(x=mean, line_dash="dot", line_color="blue", annotation_text="Process Mean")
    return fig


def plot_msa_analysis(key: str) -> go.Figure:
    """
    Generates a Measurement System Analysis (Gage R&R) box plot.

    Args:
        key (str): A unique key for Streamlit widgets (unused).

    Returns:
        go.Figure: A box plot for Gage R&R analysis.
    """
    parts = np.repeat(np.arange(1, 11), 6)
    operators = np.tile(np.repeat(['Alice', 'Bob', 'Charlie'], 2), 10)
    true_values = np.repeat(np.linspace(5, 15, 10), 6)
    operator_bias = np.tile(np.repeat([0, 0.2, -0.15], 2), 10)
    np.random.seed(1)
    measurements = true_values + operator_bias + np.random.normal(0, 0.3, 60)
    df = pd.DataFrame({'Part': parts, 'Operator': operators, 'Measurement': measurements})
    fig = px.box(df, x='Part', y='Measurement', color='Operator', title='Measurement System Analysis (MSA) - Gage R&R')
    return fig


def plot_doe_rsm(key: str) -> go.Figure:
    """
    Generates a Design of Experiments (DOE) Response Surface Methodology (RSM) plot.

    Args:
        key (str): A unique key for Streamlit widgets (unused).

    Returns:
        go.Figure: A 3D surface plot.
    """
    temp = np.linspace(20, 40, 20)
    ph = np.linspace(6.5, 8.5, 20)
    temp_grid, ph_grid = np.meshgrid(temp, ph)
    signal = -(temp_grid - 32)**2 - 2 * (ph_grid - 7.5)**2 + 1000 + np.random.rand(20, 20) * 20
    fig = go.Figure(data=[go.Surface(z=signal, x=temp, y=ph, colorscale='viridis')])
    fig.update_layout(title='Design of Experiments (DOE) Response Surface',
                      scene=dict(xaxis_title='Temperature (°C)', yaxis_title='pH', zaxis_title='Assay Signal'))
    return fig


def plot_enhanced_levey_jennings(key: str) -> None:
    """
    Renders a professional Levey-Jennings plot with programmatic Westgard rule
    detection and a violation log table, mimicking a real LIMS.
    """
    rng = np.random.default_rng(1)
    days = np.arange(1, 31)
    mean, sd = 100, 4
    # Simulate data with specific, detectable violations
    data = rng.normal(mean, sd, 30)
    data[10] = mean + 3.5 * sd  # 1_3s violation
    data[20:22] = mean + 2.2 * sd  # 2_2s violation
    
    # --- Programmatic Westgard Rule Detection ---
    violations = []
    # Check for 1_3s violations
    for i, x in enumerate(data):
        if abs(x - mean) > 3 * sd:
            violations.append({'Day': days[i], 'Value': f'{x:.2f}', 'Rule': '1₃ₛ', 'Interpretation': 'Likely random error; outlier.'})
    # Check for 2_2s violations (two consecutive points > 2SD on the same side)
    for i in range(1, len(data)):
        if (data[i] > mean + 2 * sd and data[i-1] > mean + 2 * sd) or \
           (data[i] < mean - 2 * sd and data[i-1] < mean - 2 * sd):
            violations.append({'Day': days[i], 'Value': f'{data[i]:.2f}', 'Rule': '2₂ₛ', 'Interpretation': 'Systematic error; process shift.'})

    # --- Create Enhanced Visualization ---
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
        row_heights=[0.7, 0.3],
        specs=[[{"type": "xy"}], [{"type": "table"}]]
    )
    
    # 1. Levey-Jennings Plot
    fig.add_trace(go.Scatter(x=days, y=data, mode='lines+markers', name='Control Value', line=dict(color='royalblue')), row=1, col=1)
    for i, color, ls in zip([1, 2, 3], ['green', 'orange', 'red'], ['dot', 'dashdot', 'dash']):
        fig.add_hline(y=mean + i*sd, line_dash=ls, line_color=color, annotation_text=f"+{i}SD", row=1, col=1)
        fig.add_hline(y=mean - i*sd, line_dash=ls, line_color=color, annotation_text=f"-{i}SD", row=1, col=1)

    # Add markers for detected violations
    violation_days = [v['Day'] for v in violations]
    violation_values = [float(v['Value']) for v in violations]
    violation_rules = [v['Rule'] for v in violations]
    fig.add_trace(go.Scatter(x=violation_days, y=violation_values, mode='markers', name='Violation', 
                             marker=dict(color='red', size=12, symbol='x')), row=1, col=1)

    # 2. Violation Log Table
    if violations:
        violations_df = pd.DataFrame(violations)
        fig.add_trace(go.Table(
            header=dict(values=list(violations_df.columns), fill_color='#FFC7CE', align='left'),
            cells=dict(values=[violations_df.Day, violations_df.Value, violations_df.Rule, violations_df.Interpretation], fill_color='#FADBD8', align='left')
        ), row=2, col=1)
        
    fig.update_layout(height=600, title_text='Levey-Jennings Chart with Automated Westgard Rule Violations', showlegend=False)
    fig.update_yaxes(title_text="Control Value", row=1, col=1)
    st.plotly_chart(fig, use_container_width=True)


def run_assay_regression(key: str) -> go.Figure:
    """
    Performs a linear regression analysis and displays the scatter plot and statsmodels summary.
    This function was fixed to ensure a sufficient sample size for statsmodels tests.

    Args:
        key (str): A unique key for Streamlit widgets (unused).

    Returns:
        go.Figure: A scatter plot with an OLS trendline.
    """
    np.random.seed(1)
    # n=8 is the minimum sample size for statsmodels omni_normtest
    conc = np.array([0, 10, 25, 50, 100, 200, 300, 400])
    signal = 50 + 2.5 * conc + np.random.normal(0, 20, 8)
    
    df = pd.DataFrame({'Concentration': conc, 'Signal': signal})
    fig = px.scatter(df, x='Concentration', y='Signal', trendline='ols', title="Assay Performance Regression (Linearity)")
    
    X = sm.add_constant(df['Concentration'])
    model = sm.OLS(df['Signal'], X).fit()
    
    st.code(f"Regression Results (statsmodels summary):\n{model.summary()}")
    return fig


def plot_risk_matrix(key: str) -> go.Figure:
    """
    Generates a product risk matrix plot.

    Args:
        key (str): A unique key for Streamlit widgets (unused).

    Returns:
        go.Figure: A bubble chart representing a risk matrix.
    """
    severity = [9, 10, 6, 8, 7, 5]
    probability = [3, 2, 4, 3, 5, 1]
    risk_level = [s * p for s, p in zip(severity, probability)]
    text = ["False Positive", "False Negative", "Software Crash", "Contamination", "Reagent Exp.", "UI Lag"]
    fig = go.Figure(data=go.Scatter(
        x=probability, y=severity, mode='markers+text', text=text,
        textposition="top center",
        marker=dict(size=risk_level, sizemin=10, color=risk_level, colorscale="Reds", showscale=True, colorbar_title="Risk Score")
    ))
    fig.update_layout(
        title='Risk Matrix (Severity vs. Probability)',
        xaxis_title='Probability of Occurrence', yaxis_title='Severity of Harm',
        xaxis=dict(range=[0, 6]), yaxis=dict(range=[0, 11])
    )
    return fig


def plot_enhanced_spc_charts(key: str) -> None:
    """
    Renders an industrial-grade, interactive X-bar and R chart, the standard
    for monitoring process mean and variability with rational subgroups.
    """
    st.info("Monitor a process by collecting data in rational subgroups (e.g., 5 measurements per batch). The **X-bar Chart** tracks the average between subgroups, while the **R-Chart** tracks the variability within subgroups. Both must be in control.")
    
    process_shift = st.checkbox("Simulate a Process Shift (after Subgroup 15)", key=f"spc_shift_{key}")
    
    # --- Generate Data ---
    rng = np.random.default_rng(42)
    subgroups = 20
    subgroup_size = 5
    data = [rng.normal(10, 0.5, subgroup_size) for _ in range(subgroups)]
    if process_shift:
        data[15:] = [rng.normal(10.8, 0.5, subgroup_size) for _ in range(subgroups - 15)]
        
    df = pd.DataFrame(data, columns=[f'm{i}' for i in range(1, subgroup_size + 1)])
    df.index.name = "Subgroup"
    df['Mean'] = df.mean(axis=1)
    df['Range'] = df.max(axis=1) - df.min(axis=1)

    # --- Calculate Control Limits (using SPC constants for n=5) ---
    # Use data from the first 15 "in-control" subgroups to establish limits
    mean_of_means = df['Mean'][:15].mean()
    mean_of_ranges = df['Range'][:15].mean()
    A2 = 0.577  # SPC constant for n=5
    D4 = 2.114  # SPC constant for n=5
    
    x_bar_ucl, x_bar_lcl = mean_of_means + A2 * mean_of_ranges, mean_of_means - A2 * mean_of_ranges
    r_ucl = D4 * mean_of_ranges
    
    # --- Create Visualization ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("X-bar Chart (Process Mean)", "R-Chart (Process Variability)"))
    
    # X-bar Chart
    fig.add_trace(go.Scatter(x=df.index, y=df['Mean'], name='Subgroup Mean', mode='lines+markers'), row=1, col=1)
    fig.add_hline(y=mean_of_means, line_dash="dash", line_color="green", annotation_text="CL", row=1, col=1)
    fig.add_hline(y=x_bar_ucl, line_dash="dot", line_color="red", annotation_text="UCL", row=1, col=1)
    fig.add_hline(y=x_bar_lcl, line_dash="dot", line_color="red", annotation_text="LCL", row=1, col=1)
    
    # R-Chart
    fig.add_trace(go.Scatter(x=df.index, y=df['Range'], name='Subgroup Range', mode='lines+markers', line=dict(color='orange')), row=2, col=1)
    fig.add_hline(y=mean_of_ranges, line_dash="dash", line_color="green", annotation_text="CL", row=2, col=1)
    fig.add_hline(y=r_ucl, line_dash="dot", line_color="red", annotation_text="UCL", row=2, col=1)
    
    fig.update_layout(height=600, showlegend=False, title_text="X-bar and R Control Charts")
    fig.update_xaxes(title_text="Subgroup Number", row=2, col=1)
    fig.update_yaxes(title_text="Subgroup Average", row=1, col=1)
    fig.update_yaxes(title_text="Subgroup Range", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)


def get_software_risk_data(key: str = "") -> pd.DataFrame:
    """
    Returns a DataFrame of software items and their IEC 62304 classification.
    
    Args:
        key (str, optional): Unused. Defaults to "".
        
    Returns:
        pd.DataFrame: A DataFrame of software risk data.
    """
    return pd.DataFrame([
        {"Software Item": "Patient Result Algorithm", "IEC 62304 Class": "Class C"},
        {"Software Item": "Database Middleware", "IEC 62304 Class": "Class B"},
        {"Software Item": "UI Color Theme Module", "IEC 62304 Class": "Class A"}
    ])


def plot_enhanced_rft_kpi(key: str) -> None:
    """
    Renders a true KPI dashboard for Right-First-Time (RFT), including a gauge,
    a metric with delta, and a historical trend chart.
    """
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("##### Current RFT Rate")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=82.4,
            title={'text': "This Quarter"},
            delta={'reference': 79.1, 'relative': False, 'valueformat': '.1f'},
            gauge={'axis': {'range': [50, 100]}, 'bar': {'color': "cornflowerblue"},
                   'steps': [{'range': [50, 80], 'color': 'lightyellow'}, {'range': [80, 90], 'color': 'lightgreen'}],
                   'threshold': {'line': {'color': "green", 'width': 4}, 'thickness': 0.9, 'value': 95}}
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=40, b=40))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.markdown("##### Historical Trend")
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        rft_rates = [75.2, 78.5, 79.1, 81.3, 80.5, 82.4]
        df = pd.DataFrame({'Month': months, 'RFT Rate (%)': rft_rates})
        
        fig_trend = px.line(df, x='Month', y='RFT Rate (%)', markers=True,
                            title='RFT Performance Over Last 6 Months')
        fig_trend.update_layout(height=300, margin=dict(t=40, b=40), yaxis_range=[70, 90])
        st.plotly_chart(fig_trend, use_container_width=True)


def run_anova_ttest_enhanced(key: str) -> None:
    """
    Renders an interactive module for comparing two groups using a t-test.

    Args:
        key (str): A unique key prefix for Streamlit widgets.
    """
    st.info("Used to determine if there is a statistically significant difference between groups (e.g., reagent lots, instruments, or operators). This is fundamental for method transfer and comparability studies.")
    st.warning("**Regulatory Context:** FDA's guidance on Comparability Protocols; ISO 13485:2016, Section 7.5.6")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_samples = st.slider("Samples per Group", 10, 100, 30, key=f"anova_n_{key}")
        mean_shift = st.slider("Simulated Mean Shift in Lot B", 0.0, 5.0, 0.5, 0.1, key=f"anova_shift_{key}")
        std_dev = st.slider("Group Standard Deviation", 0.5, 5.0, 2.0, 0.1, key=f"anova_std_{key}")
    
    np.random.seed(1)
    group_a = np.random.normal(10, std_dev, n_samples)
    group_b = np.random.normal(10 + mean_shift, std_dev, n_samples)
    df = pd.melt(pd.DataFrame({'Lot A': group_a, 'Lot B': group_b}), var_name='Group', value_name='Measurement')
    
    with col2:
        fig = px.box(df, x='Group', y='Measurement', title="Performance Comparison with Box & Violin Plots", points='all')
        fig.add_trace(go.Violin(x=df['Group'], y=df['Measurement'], box_visible=False, line_color='rgba(0,0,0,0)', fillcolor='rgba(0,0,0,0)', points=False, name='Distribution'))
        st.plotly_chart(fig, use_container_width=True)
    
    t_stat, p_value = stats.ttest_ind(group_a, group_b)
    st.subheader("Statistical Interpretation")
    if p_value < 0.05:
        st.error(f"**Actionable Insight:** The difference is statistically significant (p-value = {p_value:.4f}). Action: An investigation is required. Lot B cannot be considered comparable.")
    else:
        st.success(f"**Actionable Insight:** No statistically significant difference was detected (p-value = {p_value:.4f}). The lots are comparable.")


def run_regression_analysis_stat_enhanced(key: str) -> None:
    """
    Renders an interactive module for linear regression analysis.

    Args:
        key (str): A unique key prefix for Streamlit widgets.
    """
    st.info("Linear regression is critical for verifying linearity and assessing correlation. The statsmodels output provides the detailed metrics required for a regulatory submission.")
    st.warning("**Regulatory Context:** CLSI EP06; FDA Guidance on Bioanalytical Method Validation")
    col1, col2 = st.columns([1, 2])
    with col1:
        noise = st.slider("Measurement Noise (Std Dev)", 0, 50, 15, key=f"regr_noise_{key}")
        bias = st.slider("Systematic Bias", -20, 20, 5, key=f"regr_bias_{key}")
    
    np.random.seed(1)
    conc = np.linspace(0, 400, 15)
    signal = 50 + 2.5 * conc + bias + np.random.normal(0, noise, 15)
    df = pd.DataFrame({'Concentration': conc, 'Signal': signal})
    
    with col2:
        fig = px.scatter(df, x='Concentration', y='Signal', trendline='ols', title="Assay Performance Regression (Linearity)")
        st.plotly_chart(fig, use_container_width=True)
    
    X = sm.add_constant(df['Concentration'])
    model = sm.OLS(df['Signal'], X).fit()
    st.subheader("Statistical Interpretation (Statsmodels OLS Summary)")
    st.code(f"{model.summary()}")
    st.success(f"**Actionable Insight:** The R-squared value of {model.rsquared:.3f} confirms excellent linearity. The p-value for the Concentration coefficient is < 0.001, proving a significant positive relationship.")


def run_descriptive_stats_stat_enhanced(key: str) -> None:
    """
    Renders a module for descriptive statistics analysis.

    Args:
        key (str): A unique key for Streamlit widgets (unused).
    """
    st.info("The foundational analysis for any analytical validation study (e.g., LoD, Precision).")
    st.warning("**Regulatory Context:** CLSI EP17 (Detection Capability); CLSI EP05-A3 (Precision)")
    np.random.seed(1)
    data = np.random.normal(50, 2, 150)
    df = pd.DataFrame(data, columns=["value"])
    fig = px.histogram(df, x="value", marginal="box", nbins=20, title="Descriptive Statistics for Limit of Detection (LoD) Study")
    st.plotly_chart(fig, use_container_width=True)
    
    mean, std = np.mean(data), np.std(data, ddof=1)
    cv = (std / mean) * 100
    ci_95 = stats.t.interval(0.95, len(data) - 1, loc=mean, scale=stats.sem(data))
    st.subheader("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"{mean:.2f}")
    col2.metric("Std Dev", f"{std:.2f}")
    col3.metric("%CV", f"{cv:.2f}%")
    col4.metric("95% CI for Mean", f"{ci_95[0]:.2f} - {ci_95[1]:.2f}")
    st.success("**Actionable Insight:** The low %CV and tight confidence interval provide high confidence that the LoD is reliably at 50 copies/mL, supporting the product claim for the 510(k) submission.")


def run_control_charts_stat_enhanced(key: str) -> None:
    """
    Renders an interactive X-bar and R control chart module.

    Args:
        key (str): A unique key prefix for Streamlit widgets.
    """
    st.info("X-bar & R-charts are used to monitor the mean (X-bar) and variability (R-chart) of a process when data is collected in rational subgroups (e.g., 5 measurements per batch).")
    st.warning("**Regulatory Context:** FDA 21 CFR 820.250 (Statistical Techniques); ISO TR 10017")
    np.random.seed(1)
    data = [np.random.normal(10, 0.5, 5) for _ in range(20)]
    process_shift = st.checkbox("Simulate a Process Shift", key=f"spc_shift_{key}")
    if process_shift:
        data[15:] = [np.random.normal(10.8, 0.5, 5) for _ in range(5)]
    
    df = pd.DataFrame(data, columns=[f'm{i}' for i in range(1, 6)])
    df['mean'] = df.mean(axis=1)
    df['range'] = df.max(axis=1) - df.min(axis=1)
    
    # X-bar Chart Constants (n=5)
    x_bar_cl = df['mean'].mean()
    x_bar_a2 = 0.577
    x_bar_ucl = x_bar_cl + x_bar_a2 * df['range'].mean()
    x_bar_lcl = x_bar_cl - x_bar_a2 * df['range'].mean()
    
    # R Chart Constants (n=5)
    r_cl = df['range'].mean()
    r_d4 = 2.114
    r_ucl = r_d4 * r_cl
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("X-bar Chart (Process Mean)", "R-Chart (Process Variability)"))
    fig.add_trace(go.Scatter(x=df.index, y=df['mean'], name='Subgroup Mean', mode='lines+markers'), row=1, col=1)
    fig.add_hline(y=x_bar_cl, line_dash="dash", line_color="green", annotation_text="CL", row=1, col=1)
    fig.add_hline(y=x_bar_ucl, line_dash="dot", line_color="red", annotation_text="UCL", row=1, col=1)
    fig.add_hline(y=x_bar_lcl, line_dash="dot", line_color="red", annotation_text="LCL", row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['range'], name='Subgroup Range', mode='lines+markers', line=dict(color='orange')), row=2, col=1)
    fig.add_hline(y=r_cl, line_dash="dash", line_color="green", annotation_text="CL", row=2, col=1)
    fig.add_hline(y=r_ucl, line_dash="dot", line_color="red", annotation_text="UCL", row=2, col=1)
    
    fig.update_layout(height=600, title_text="X-bar and R Control Charts")
    st.plotly_chart(fig, use_container_width=True)
    
    if process_shift:
        st.warning("**Actionable Insight:** A clear upward shift is detected in the X-bar chart starting at subgroup 15, while the R-chart remains stable. This indicates a special cause has shifted the process mean without affecting its variability. This requires an immediate investigation.")
    else:
        st.success("**Actionable Insight:** The process is in a state of statistical control. Both the mean and variability are stable and predictable, providing a solid baseline for validation.")


def run_kaplan_meier_stat_enhanced(key: str) -> None:
    """
    Renders a Kaplan-Meier survival plot for shelf-life analysis.

    Args:
        key (str): A unique key for Streamlit widgets (unused).
    """
    st.info("Survival analysis is used to estimate the shelf-life of a product by modeling time-to-failure data, especially when some samples have not failed by the end of the study (censored data).")
    st.warning("**Regulatory Context:** ICH Q1E (Evaluation of Stability Data); FDA Guidance: Q1A(R2)")
    np.random.seed(1)
    time_to_failure = np.random.weibull(2, 50) * 24
    observed = np.random.binomial(1, 0.8, 50)
    df = pd.DataFrame({'Months': time_to_failure, 'Status': ['Failed' if o == 1 else 'Censored' for o in observed]})
    
    fig = px.ecdf(df, x="Months", color="Status", ecdfmode="survival", title="Kaplan-Meier Survival Plot for Shelf-Life Validation")
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Median Survival")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Study Conclusion")
    st.success("**Actionable Insight:** The survival curve demonstrates the probability of a unit remaining stable over time. The point where the curve crosses the 50% line provides the estimated median shelf-life. 'Censored' data points are critical for an accurate model and must be included in the analysis.")


def run_monte_carlo_stat_enhanced(key: str) -> None:
    """
    Renders an interactive Monte Carlo simulation for project timeline risk analysis.

    Args:
        key (str): A unique key prefix for Streamlit widgets.
    """
    st.info("Monte Carlo simulation runs thousands of 'what-if' scenarios on a project plan with uncertain task durations to provide a probabilistic forecast.")
    st.warning("**Regulatory Context:** Aligned with risk-based planning principles in ISO 13485 and Project Management Body of Knowledge (PMBOK).")
    n_sims = st.slider("Number of Simulations", 1000, 10000, 5000, key=f"mc_sims_{key}")
    
    np.random.seed(1)
    task1 = np.random.triangular(8, 10, 15, n_sims)
    task2 = np.random.triangular(15, 20, 30, n_sims)
    task3 = np.random.triangular(5, 8, 12, n_sims)
    total_times = task1 + task2 + task3
    
    p50 = np.percentile(total_times, 50)
    p90 = np.percentile(total_times, 90)
    
    fig = px.histogram(total_times, nbins=50, title="Monte Carlo Simulation of V&V Plan Duration")
    fig.add_vline(x=p50, line_dash="dash", line_color="green", annotation_text=f"P50 (Median) = {p50:.1f} days")
    fig.add_vline(x=p90, line_dash="dash", line_color="red", annotation_text=f"P90 (High Confidence) = {p90:.1f} days")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Risk-Adjusted Planning")
    st.error(f"**Actionable Insight:** While the median completion time is {p50:.1f} days, there is a 10% chance the project will take **{p90:.1f} days or longer**. The P90 estimate must be communicated to the PMO as the commitment date to account for risk.")


def create_v_model_figure(key: str = None) -> go.Figure:
    """
    Generates a plot of the V-Model for software/system development.

    Args:
        key (str, optional): Unused. Defaults to None.

    Returns:
        go.Figure: A plot visualizing the V-Model.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1], mode='lines+markers+text', text=["User Needs", "System Req.", "Architecture", "Module Design"], textposition="top right", line=dict(color='royalblue', width=2), marker=dict(size=10)))
    fig.add_trace(go.Scatter(x=[5, 6, 7, 8], y=[1, 2, 3, 4], mode='lines+markers+text', text=["Unit Test", "Integration Test", "System V&V", "UAT"], textposition="top left", line=dict(color='green', width=2), marker=dict(size=10)))
    for i in range(4):
        fig.add_shape(type="line", x0=4 - i, y0=1 + i, x1=5 + i, y1=1 + i, line=dict(color="grey", width=1, dash="dot"))
    fig.update_layout(title_text=None, showlegend=False, xaxis=dict(showticklabels=False, zeroline=False, showgrid=False), yaxis=dict(showticklabels=False, zeroline=False, showgrid=False))
    return fig


@st.cache_data
def get_complaint_data() -> pd.DataFrame:
    """
    Generates and caches a realistic DataFrame of simulated post-market complaint data.

    Returns:
        pd.DataFrame: A DataFrame of simulated complaint data.
    """
    np.random.seed(42)
    dates = pd.to_datetime(pd.date_range(start="2022-01-01", end="2023-12-31", freq="D"))
    complaint_types = ["False Positive", "Reagent Leak", "Instrument Error", "Software Glitch", "High CV", "No Result"]
    regions = ["AL", "CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC"]
    lots = ["A2201-A", "A2201-B", "A2301-A", "A2301-B"]
    
    n_complaints = 300
    df = pd.DataFrame({
        "Complaint_ID": [f"C-{i+1:04d}" for i in range(n_complaints)],
        "Date": np.random.choice(dates, n_complaints),
        "Lot_Number": np.random.choice(lots, n_complaints, p=[0.2, 0.2, 0.3, 0.3]),
        "Region": np.random.choice(regions, n_complaints),
        "Complaint_Type": np.random.choice(complaint_types, n_complaints, p=[0.15, 0.1, 0.25, 0.1, 0.2, 0.2]),
        "Severity": np.random.choice(["Low", "Medium", "High"], n_complaints, p=[0.6, 0.3, 0.1])
    })
    
    # Inject a specific signal for the CAPA trigger
    n_signal = 15
    signal_df = pd.DataFrame({
        "Complaint_ID": [f"C-{i+301:04d}" for i in range(n_signal)],
        "Date": pd.to_datetime(pd.date_range(start="2023-11-01", periods=n_signal)),
        "Lot_Number": "A2301-B",
        "Region": "CA",
        "Complaint_Type": "False Positive",
        "Severity": "High"
    })
    
    final_df = pd.concat([df, signal_df]).sort_values("Date").reset_index(drop=True)
    return final_df


# --- PAGE RENDERING FUNCTIONS ---

def render_main_page() -> None:
    """Renders the main Executive Summary page."""
    st.title("🎯 The V&V Executive Command Center")
    st.markdown("A definitive showcase of data-driven leadership in a regulated GxP environment.")
    st.markdown("---")
    render_director_briefing(
        "Portfolio Objective", 
        "This interactive application translates the core responsibilities of V&V leadership into a suite of high-density dashboards. It is designed to be an overwhelming and undeniable demonstration of the strategic, technical, and quality systems expertise required for a senior leadership role in the medical device industry.", 
        "ISO 13485, ISO 14971, IEC 62304, 21 CFR 820, 21 CFR Part 11, CLSI Guidelines", 
        "A well-led V&V function directly accelerates time-to-market, reduces compliance risk, lowers the cost of poor quality (COPQ), and builds a culture of data-driven excellence."
    )
    st.info("Please use the navigation sidebar on the left to explore each of the core competency areas.")


def render_design_controls_page() -> None:
    """Renders the Design Controls, Planning & Risk Management page."""
    st.title("🏛️ 1. Design Controls, Planning & Risk Management")
    st.markdown("---")
    render_director_briefing(
        "The Design History File (DHF) as a Strategic Asset", 
        "The DHF is the compilation of records that demonstrates the design was developed in accordance with the design plan and regulatory requirements. An effective V&V leader architects the DHF from day one.", 
        "FDA 21 CFR 820.30 (Design Controls), ISO 13485:2016 (Section 7.3)", 
        "Ensures audit readiness and provides a clear, defensible story of product development to regulatory bodies, accelerating submission review times."
    )
    
    with st.container(border=True):
        st.subheader("The V-Model: A Framework for Compliant V&V")
        st.markdown("The V-Model is the cornerstone of a structured V&V strategy, visually linking the design and development phases (left side) to the corresponding testing and validation phases (right side). This ensures that for every design input, there is a corresponding validation output, forming the basis of a complete and auditable Design History File (DHF).")
        st.plotly_chart(create_v_model_figure(), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### **Left Side: Design & Development (Building it Right)**")
            st.markdown("- **User Needs:** High-level goals from Marketing/customers.\n- **System Requirements:** Detailed functional/performance specs.\n- **Architecture:** High-level system design.\n- **Module Design:** Low-level detailed design.")
        with col2:
            st.markdown("#### **Right Side: Verification & Validation (Proving We Built the Right Thing)**")
            st.markdown("- **Unit Test:** Verifies individual code modules.\n- **Integration Test:** Verifies that modules work together.\n- **System V&V:** Verifies the complete system against requirements.\n- **User Acceptance Testing (UAT):** Validates the system against user needs.")
        st.success("**Actionable Insight:** By enforcing this model, a V&V leader prevents late-stage failures, ensures no requirements are left untested, and provides a clear, defensible V&V narrative to auditors. The horizontal lines represent the core of traceability.")
    
    render_metric_card("Requirements Traceability Matrix (RTM)", "The RTM is the backbone of the DHF, providing an auditable link between user needs, design inputs, V&V activities, and risk controls.", create_rtm_data_editor, "The matrix view instantly flags critical gaps, such as the un-tested cross-reactivity requirement (URS-003), allowing for proactive mitigation before a design freeze.", "FDA 21 CFR 820.30(j) - Design History File (DHF)", key="rtm")
    render_metric_card("Product Risk Management (FMEA & Risk Matrix)", "A systematic process for identifying, analyzing, and mitigating potential failure modes. V&V activities are primary risk mitigations.", plot_risk_matrix, "The risk matrix clearly prioritizes 'False Negative' as the highest risk, ensuring that it receives the most V&V resources and attention. This is a key input for the V&V Master Plan.", "ISO 14971: Application of risk management to medical devices", key="fmea")
    render_metric_card("Design of Experiments (DOE/RSM)", "A powerful statistical tool used to efficiently characterize the product's design space and identify robust operating parameters.", plot_doe_rsm, "The Response Surface Methodology (RSM) plot indicates the assay's optimal performance is at ~32°C and a pH of 7.5. This data forms the basis for setting manufacturing specifications.", "FDA Guidance on Process Validation: General Principles and Practices", key="doe")
    render_metric_card(
        "AI-Powered Requirement Risk Analysis", 
        "This NLP model uses Logistic Regression to analyze the text of a requirement and predict its risk of being ambiguous or untestable. It is trained to identify vague words like 'easy', 'fast', or 'robust'.", 
        run_requirement_risk_nlp_model, 
        "The model flags two requirements with a high probability of ambiguity. This allows the V&V lead to proactively engage with the systems engineering team to clarify and quantify these requirements *before* the design is frozen, preventing costly late-stage rework.", 
        "ISO 13485: 7.3.2 (Design Inputs), FDA Guidance on Design Controls", 
        key="req_risk"
    )


def render_method_validation_page() -> None:
    """Renders the Method Validation & Statistical Rigor page."""
    st.title("🔬 2. Method Validation & Statistical Rigor")
    st.markdown("---")
    render_director_briefing("Ensuring Data Trustworthiness", "Before a product can be validated, the methods used to measure it must be proven reliable. TMV, MSA, and CpK are the statistical pillars that provide objective evidence of this reliability.", "FDA Guidance on Analytical Procedures and Methods Validation; CLSI Guidelines (EP05, EP17, etc.); AIAG MSA Manual", "Prevents costly product failures and batch rejections caused by unreliable or incapable measurement and manufacturing processes. It is the foundation of data integrity.")
    render_metric_card("Process Capability (CpK)", "Measures how well a process can produce output that meets specifications. A CpK > 1.33 is typically considered capable for medical devices.", plot_cpk_analysis, "The interactive slider shows how tightening specification limits directly impacts the CpK value, demonstrating the trade-offs between design margin and manufacturing capability.", "ISO TR 10017 - Guidance on statistical techniques", key="cpk")
    render_metric_card("Measurement System Analysis (MSA/Gage R&R)", "Quantifies the variation in a measurement system attributable to operators and equipment. A key part of Test Method Validation (TMV).", plot_msa_analysis, "The box plot shows that Operator Charlie's measurements are consistently lower than Alice and Bob's, indicating a potential training issue or procedural deviation that requires investigation.", "AIAG MSA Reference Manual", key="msa")
    render_metric_card("Assay Performance Regression Analysis", "Linear regression is used to characterize key assay performance attributes. The full statistical output is critical for regulatory submissions.", run_assay_regression, "The statsmodels summary provides a comprehensive model of the assay's response with high statistical significance (p < 0.001) and an R-squared of 0.99+, confirming excellent linearity.", "CLSI EP06 - Evaluation of the Linearity of Quantitative Measurement Procedures", key="regression")


def render_execution_monitoring_page() -> None:
    """Renders the professionally upgraded Execution Monitoring & Quality Control page."""
    st.title("📈 3. Execution Monitoring & Quality Control")
    st.markdown("---")
    render_director_briefing(
        "Statistical Process Control (SPC) for V&V", 
        "SPC distinguishes between normal process variation ('common cause') and unexpected problems ('special cause') that require immediate investigation. An effective SPC program moves a V&V organization from reactive problem-solving to proactive process control, ensuring data integrity and product consistency.",
        "FDA 21 CFR 820.250 (Statistical Techniques), ISO TR 10017, CLSI C24", 
        "Provides an early warning system for process drifts, reducing the risk of large-scale, costly investigations. It is the foundation for demonstrating a process is in a state of statistical control, a key requirement for process validation (PQ)."
    )
    
    with st.container(border=True):
        st.subheader("Daily Quality Control Monitoring (Levey-Jennings)")
        plot_enhanced_levey_jennings("lj_enhanced")
        st.success("**Actionable Insight:** The system has automatically detected a '2₂ₛ' violation, a classic sign of a systematic error such as a shift in calibration or a degrading reagent. This is a higher priority than the random '1₃ₛ' error. Action: Halt testing, recalibrate the instrument, and re-run controls before processing patient samples.")

    with st.container(border=True):
        st.subheader("Manufacturing Process Monitoring (X-bar & R Charts)")
        plot_enhanced_spc_charts("spc_enhanced")
        st.success("**Actionable Insight:** If a process shift is simulated, the X-bar chart clearly shows the mean shifting out of control, while the R-chart remains stable. This is a powerful diagnostic signal, indicating that the process *variability* is unchanged, but the *average* has moved. This allows the team to rule out causes related to inconsistency (e.g., operator variation) and focus on causes related to a systemic shift (e.g., incorrect buffer concentration, wrong machine setting).")

    with st.container(border=True):
        st.subheader("V&V Protocol Execution Efficiency (Right-First-Time KPI)")
        plot_enhanced_rft_kpi("rft_enhanced")
        st.success("**Actionable Insight:** The RFT rate is currently 82.4%, a 3.3 point improvement over last quarter, showing our process improvement initiatives are working. However, the trend shows we are still far from our goal of 95%. The fact that nearly 1 in 5 protocols requires rework justifies allocating a dedicated resource to root cause analysis of failed/repeated protocols.")

    render_metric_card(
        "AI-Powered Multivariate Anomaly Detection",
        "This model uses an Isolation Forest algorithm to perform multivariate anomaly detection. It identifies outlier data points based on combinations of variables, finding subtle issues that traditional single-variable control charts (like the ones above) would miss.",
        plot_multivariate_anomaly_detection,
        "The model identified two batches as significant outliers by considering Temperature, Pressure, and Flow Rate *simultaneously*. This is powerful because univariate charts would not have flagged these points, as each individual parameter was likely within its own specification limits. The failure is in the *combination* of parameters, indicating a complex process interaction that requires investigation.",
        "AIAG SPC Manual, FDA Guidance on Process Validation",
        key="anomaly_detection"
    )

    render_metric_card(
        "AI-Powered CQA Forecasting", 
        "This ARIMA time-series model forecasts the future trend of a Critical Quality Attribute (CQA) based on its historical performance. This shifts the team from reactive monitoring to proactive intervention.", 
        run_cqa_forecasting_model, 
        "The model predicts that the CQA value will breach the upper specification limit within the next 15 days. This provides an early warning to the manufacturing sciences team to investigate the process *before* a non-conforming lot is produced, saving significant cost and time.", 
        "ICH Q8 (Quality by Design), FDA Guidance on Process Validation", 
        key="cqa_forecast"
    )


def render_quality_management_page() -> None:
    """Renders the Project & Quality Systems Management page."""
    st.title("✅ 4. Project & Quality Systems Management")
    st.markdown("---")
    render_director_briefing("Managing the V&V Ecosystem", "A V&V leader must manage project health, track quality issues, and ensure software compliance. These KPIs provide the necessary oversight to manage timelines, scope, and compliance risks.", "IEC 62304, 21 CFR Part 11, GAMP 5", "Improves project predictability, ensures software compliance (a major source of FDA 483s), and provides transparent reporting to stakeholders.")
    render_metric_card("Defect Open vs. Close Trend (Burnup)", "A burnup chart tracks scope changes and visualizes the rate of work completion against the rate of issue discovery.", plot_defect_burnup, "The widening gap between opened and closed defects indicates that our resolution rate is not keeping up. Action: Allocate additional resources to defect resolution.", "Agile Project Management Principles", key="burnup")
    render_metric_card(
        "AI-Powered Defect Triage", 
        "This NLP model uses a Random Forest Classifier to automatically predict the root cause category of a new defect based on its free-text description. This accelerates the triage process and improves resource allocation.", 
        run_defect_root_cause_model, 
        "The model identifies that 'Requirement Issues' are a significant source of defects, nearly equal to 'Coding Errors'. This is a critical insight, suggesting that improving the requirements engineering process could yield a high ROI in reducing downstream bugs.", 
        "ISO 13485: 8.4 (Analysis of Data)", 
        key="defect_root_cause"
    )
    st.subheader("Software V&V (IEC 62304 & 21 CFR Part 11)")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**IEC 62304 Software Safety Classification**")
            risk_df = get_software_risk_data()
            def classify_color(cls: str) -> str:
                if cls == "Class C": return "background-color: #FF7F7F"
                if cls == "Class B": return "background-color: #FFD700"
                return "background-color: #90EE90"
            st.dataframe(risk_df.style.map(classify_color, subset=['IEC 62304 Class']), use_container_width=True, hide_index=True)
            st.info("V&V rigor is directly tied to this risk classification.")
    with col2:
        with st.container(border=True):
            st.markdown("**21 CFR Part 11 Compliance Checklist**")
            st.checkbox("Validation (11.10a)", value=True, disabled=True)
            st.checkbox("Audit Trails (11.10e)", value=True, disabled=True)
            st.checkbox("Access Controls (11.10d)", value=True, disabled=True)
            st.checkbox("E-Signatures (11.200a)", value=False, disabled=True)
            st.error("Gap identified in E-Signature implementation.")
    
    st.markdown("---")
    st.subheader("Advanced Software V&V & CSV Dashboard")
    st.info("This section provides a detailed view of Computer System Validation (CSV) for GxP systems (like LIMS) and V&V for Software in a Medical Device (SiMD), covering key metrics and compliance with GAMP 5, IEC 62304, and modern cybersecurity standards.")

    tab1, tab2, tab3 = st.tabs(["📊 Key Metrics & KPIs", "📋 GAMP 5 Compliance", "🛡️ Cybersecurity Posture"])
    with tab1:
        st.markdown("##### SiMD V&V Release-Readiness Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Requirements Coverage", "98.7%")
            st.progress(0.987)
        with col2:
            st.metric("Test Case Pass Rate (System V&V)", "99.2%")
            st.progress(0.992)
        with col1:
            st.metric("Defect Density (per KLOC)", "0.85", delta="-0.12", delta_color="inverse", help="Defects per 1,000 Lines of Code. Lower is better.")
        with col2:
            st.metric("Static Analysis Critical Warnings", "3", delta="2", delta_color="inverse", help="Number of critical issues found in automated code scans. Delta from previous build.")
        st.success("**Actionable Insight:** The high requirements coverage and test pass rates indicate strong readiness for a design freeze. The low and decreasing defect density suggests improving code quality. The 3 remaining critical warnings must be adjudicated before final release.")
    with tab2:
        st.info("**GAMP 5** provides a risk-based framework for validating GxP computerized systems (e.g., lab equipment software, LIMS, eDMS). The category determines the validation rigor required.")
        gamp_data = {
            "System": ["Instrument Control SW", "LIMS", "Statistical Analysis SW", "eDMS"],
            "GAMP 5 Category": ["Cat 4: Configured", "Cat 5: Custom", "Cat 3: Standard", "Cat 4: Configured"],
            "Validation Approach": ["Full Validation of Configured Elements", "Full Prospective Validation", "Supplier Assessment & IQ/OQ", "Risk-Based Validation"],
            "Status": ["In Progress", "Planning", "Complete", "Complete"]
        }
        df_gamp = pd.DataFrame(gamp_data)
        def gamp_color(val: str) -> str:
            color = "background-color: "
            if val == "Cat 5: Custom": return color + "#FF7F7F" # Red
            if val == "Cat 4: Configured": return color + "#FFD700" # Yellow
            if val == "Cat 3: Standard": return color + "#90EE90" # Green
            return ""
        st.dataframe(df_gamp.style.map(gamp_color, subset=['GAMP 5 Category']), use_container_width=True, hide_index=True)
        st.success("**Actionable Insight:** The LIMS system, as a GAMP 5 Category 5, requires a full prospective validation effort. This must be prioritized and resourced appropriately. The completed validation for the statistical software provides confidence in its use for regulatory analysis.")
    with tab3:
        st.info("A robust cybersecurity V&V strategy is non-negotiable for connected medical devices. This aligns with **FDA's Premarket Cybersecurity Guidance** and **AAMI TIR57**.")
        st.markdown("##### Cybersecurity V&V Checklist")
        st.checkbox("✅ Threat Modeling (STRIDE) Performed", value=True, disabled=True)
        st.checkbox("✅ Secure Coding Policy in place & training complete", value=True, disabled=True)
        st.checkbox("✅ Software Bill of Materials (SBOM) Generated & Reviewed", value=True, disabled=True)
        st.checkbox("❌ Penetration Testing by Third-Party Vendor", value=False, disabled=True)
        st.checkbox("✅ Vulnerability Scanning Integrated into CI/CD Pipeline", value=True, disabled=True)
        st.error("**Actionable Insight:** A critical gap exists in third-party penetration testing. This is a major finding for any regulatory submission. **Action:** Immediately engage a qualified vendor to perform penetration testing before the final code freeze.")


def render_stats_page() -> None:
    """Renders the Advanced Statistical Methods Workbench page."""
    st.title("📐 5. Advanced Statistical Methods Workbench")
    st.markdown("This interactive workbench demonstrates proficiency in the specific statistical methods required for robust data analysis in a regulated V&V environment.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("Performance Comparison (t-test)")
            run_anova_ttest_enhanced("anova")
        with st.container(border=True):
            st.subheader("Assay Performance (Descriptive Stats)")
            run_descriptive_stats_stat_enhanced("desc")
        with st.container(border=True):
            st.subheader("Shelf-Life & Stability (Kaplan-Meier)")
            run_kaplan_meier_stat_enhanced("km")
    with col2:
        with st.container(border=True):
            st.subheader("Assay Performance (Linearity)")
            run_regression_analysis_stat_enhanced("regr")
        with st.container(border=True):
            st.subheader("Process Monitoring (SPC)")
            run_control_charts_stat_enhanced("spc")
        with st.container(border=True):
            st.subheader("Project Timeline Risk (Monte Carlo)")
            run_monte_carlo_stat_enhanced("mc")


def render_strategic_command_page() -> None:
    """Renders the Strategic Command & Control page."""
    st.title("👑 6. Strategic Command & Control")
    st.markdown("---")
    render_director_briefing("Executive-Level V&V Leadership", "A true V&V leader operates at the intersection of technical execution, financial reality, and cross-functional strategy. This command center demonstrates the tools and mindset required to run V&V not as a cost center, but as a strategic business partner that drives value and mitigates enterprise-level risk.", "ISO 13485 Section 5 (Management Responsibility) & 6 (Resource Management)", "Aligns V&V department with corporate financial goals, improves resource allocation, de-risks regulatory pathways, and enables scalable growth through effective talent management and partner oversight." )

    tab1, tab2, tab3, tab4 = st.tabs(["💰 V&V Cost & ROI Forecaster", "🌐 Regulatory & Partner Dashboard", "🧑‍🔬 Team Competency Matrix", "🔄 ECO Impact Assessment"])

    with tab1:
        st.header("V&V Project Cost & ROI Forecaster")
        st.info("Translate technical plans into financial forecasts to justify resource allocation and demonstrate value to executive leadership.")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Inputs: Project Scope & Resources")
            proj = st.selectbox("Select Project", ["ImmunoPro-A (510k)", "MolecularDX-2 (PMA)", "CardioScreen-X (De Novo)"])
            scenario = st.radio("Select Resourcing Scenario", ["Internal Team", "CRO Outsource"], horizontal=True)
            
            if scenario == "Internal Team":
                st.markdown("**Timeline Estimates**")
                av_weeks = st.slider("Analytical V&V (Weeks)", 1, 26, 8, key="int_av")
                sv_weeks = st.slider("System V&V (Weeks)", 1, 26, 10, key="int_sv")
                sw_weeks = st.slider("Software V&V (Weeks)", 1, 26, 6, key="int_sw")
                cs_weeks = st.slider("Clinical Support (Weeks)", 1, 26, 12, key="int_cs")
                
                st.markdown("**Resource Allocation**")
                fte_sci = st.slider("Number of Scientists (FTEs)", 1, 10, 2, key="int_sci")
                fte_eng = st.slider("Number of Engineers (FTEs)", 1, 10, 1, key="int_eng")

                st.markdown("**Cost Basis**")
                fte_cost = st.number_input("Fully-Burdened Cost per FTE-Week ($)", value=4000, step=100, key="int_fte_cost")
                reagent_cost_per_week = st.number_input("Cost of Reagents per Analytical/System Week ($)", value=7500, step=500, key="int_reagent")
                instrument_cost_per_week = st.number_input("Instrument Time & Maintenance per V&V Week ($)", value=1500, step=100, key="int_instr")
            else: # CRO Outsource Scenario
                st.markdown("**CRO & Internal Oversight Costs**")
                cro_contract_value = st.number_input("CRO Contract Value ($)", value=500000, step=25000)
                mgmt_fte = st.slider("Internal Management Overhead (FTEs)", 0.5, 3.0, 1.0, 0.5)
                mgmt_weeks = st.slider("Project Duration (Weeks)", 10, 52, 36)
                fte_cost = st.number_input("Fully-Burdened Cost per FTE-Week ($)", value=5000, step=100, key="cro_fte_cost")
        with col2:
            st.subheader("Forecasted V&V Budget & ROI")
            
            if scenario == "Internal Team":
                total_personnel_weeks = (av_weeks + sv_weeks + sw_weeks + cs_weeks)
                total_fte = fte_sci + fte_eng
                personnel_cost = total_personnel_weeks * total_fte * fte_cost
                reagent_total_cost = (av_weeks + sv_weeks) * reagent_cost_per_week
                instrument_total_cost = (av_weeks + sv_weeks + sw_weeks) * instrument_cost_per_week
                total_budget = personnel_cost + reagent_total_cost + instrument_total_cost
                cost_data = {'Category': ['Personnel', 'Reagents & Consumables', 'Instrument Time'], 'Cost': [personnel_cost, reagent_total_cost, instrument_total_cost]}
            else: # CRO Outsource Scenario
                personnel_cost = mgmt_fte * mgmt_weeks * fte_cost
                total_budget = cro_contract_value + personnel_cost
                cost_data = {'Category': ['CRO Contract', 'Internal Management'], 'Cost': [cro_contract_value, personnel_cost]}

            st.metric("Total Forecasted V&V Budget", f"${total_budget:,.0f}", help="Calculated based on selected scenario.")
            df_costs = pd.DataFrame(cost_data)
            fig_tree = px.treemap(df_costs, path=['Category'], values='Cost', title='V&V Budget Allocation by Category', color_discrete_map={'(?)':'#2ca02c', 'Personnel':'#1f77b4', 'Reagents & Consumables':'#ff7f0e', 'Instrument Time':'#d62728', 'CRO Contract': '#9467bd', 'Internal Management': '#8c564b'})
            st.plotly_chart(fig_tree, use_container_width=True)

            if scenario == "Internal Team":
                st.subheader("Monthly Personnel Cost Burn")
                total_weeks = av_weeks + sv_weeks + sw_weeks + cs_weeks
                monthly_cost = total_fte * fte_cost * 4.33 # Avg weeks in a month
                burn_df = pd.DataFrame({
                    "Month": pd.date_range(start="2024-01-01", periods=int(total_weeks/4.33)+1, freq="ME"),
                    "Cost": monthly_cost
                })
                fig_burn = px.bar(burn_df, x="Month", y="Cost", title="Projected Monthly Personnel Spend")
                fig_burn.update_layout(yaxis_title="Cost ($)")
                st.plotly_chart(fig_burn, use_container_width=True)

            st.subheader("Return on Investment (ROI) Estimate")
            tpp_revenue = st.number_input("TPP Forecasted 3-Year Revenue ($)", value=15_000_000, step=1_000_000, format="%d")
            if total_budget > 0:
                roi = ((tpp_revenue - total_budget) / total_budget) * 100
                st.metric("High-Level V&V ROI", f"{roi:.1f}%", help="(Forecasted Revenue - V&V Cost) / V&V Cost")
    
    with tab2:
        st.header("Regulatory Strategy & External Partner Dashboard")
        st.info("Dynamically align V&V evidence with submission requirements and manage external vendor performance.")
        sub_type = st.selectbox("Select Submission Type", ["FDA 510(k)", "FDA PMA", "EU IVDR Class C", "EU IVDR Class D"])
        with st.container(border=True):
            st.subheader(f"Dynamic Evidence Checklist for: {sub_type}")
            st.checkbox("✅ Analytical Performance Studies (LoD, Precision, Linearity, etc.)", value=True, disabled=True)
            st.checkbox("✅ Software V&V Documentation (per IEC 62304)", value=True, disabled=True)
            st.checkbox("✅ Risk Management File (per ISO 14971)", value=True, disabled=True)
            st.checkbox("✅ Stability & Shelf-Life Data", value=True, disabled=True)
            if "510(k)" in sub_type: 
                st.checkbox("✅ Substantial Equivalence Testing Data", value=True, disabled=True)
            if "PMA" in sub_type: 
                st.checkbox("🔥 Clinical Validation Data (Pivotal Study Support)", value=True, disabled=True)
                st.checkbox("🔥 PMA Module-Specific Data Packages", value=True, disabled=True)
            if "IVDR" in sub_type:
                st.checkbox("🔥 Scientific Validity Report", value=True, disabled=True)
                st.checkbox("🔥 Clinical Performance Study Report", value=True, disabled=True)
                if "Class D" in sub_type:
                    st.checkbox("🔥 Common Specifications (CS) Conformance Data", value=True, disabled=True)
                    st.checkbox("🔥 Notified Body & EURL Review Support Package", value=True, disabled=True)

        st.subheader("CRO Partner Performance Oversight")
        df_perf = pd.DataFrame({'Metric': ['On-Time Delivery (%)', 'Protocol Deviation Rate (%)', 'Data Quality Score (1-100)'], 'Internal Team': [95, 2.1, 98.5], 'CRO Partner A': [88, 4.5, 96.2]})
        fig = px.bar(df_perf, x='Metric', y=['Internal Team', 'CRO Partner A'], barmode='group', title="Quarterly Performance: Internal Team vs. CRO Partner A")
        fig.update_layout(yaxis_title="Performance Score")
        st.plotly_chart(fig, use_container_width=True)
        st.error("**Actionable Insight:** CRO Partner A is underperforming on On-Time Delivery and has more than double our internal deviation rate. This poses a significant project timeline and data integrity risk. **Action:** Schedule a Quarterly Business Review (QBR) to present this data and establish a formal Performance Improvement Plan (PIP).")

    with tab3:
        st.header("Team Competency & Development Matrix")
        st.info("Proactively manage talent, identify skill gaps for upcoming projects, and drive strategic team development.")
        skills = ['qPCR Method Validation', 'ELISA Development', 'GAMP 5 CSV', 'Statistical Analysis (Python)', 'ISO 14971 Risk Management', 'JMP/Minitab', 'Clinical Study Design']
        team = ['Alice', 'Bob', 'Charlie', 'Diana', 'Ethan']
        np.random.seed(1)
        data = np.random.randint(1, 4, size=(len(team), len(skills)))
        df_skills = pd.DataFrame(data, index=team, columns=skills)
        df_skills.index.name = "Team Member"

        st.subheader("1. Filter for Project Needs")
        required_skills = st.multiselect("Select Required Project Skills", options=skills, default=['qPCR Method Validation', 'ISO 14971 Risk Management', 'Statistical Analysis (Python)'])
        
        st.subheader("2. Analyze Team Readiness")
        def highlight_skills(df: pd.DataFrame) -> pd.DataFrame:
            style = pd.DataFrame('', index=df.index, columns=df.columns)
            for skill in required_skills:
                if skill in df.columns:
                    style.loc[:, skill] = 'background-color: yellow'
            return style
        
        st.dataframe(df_skills.style.apply(highlight_skills, axis=None).background_gradient(cmap='RdYlGn', vmin=1, vmax=3, axis=None).set_caption("Proficiency: 1 (Novice) to 3 (Expert)"), use_container_width=True)

        st.subheader("3. Formulate Action Plan")
        if required_skills:
            team_readiness = df_skills[required_skills].sum(axis=1)
            best_fit = team_readiness.idxmax()
            st.success(f"**Staffing Insight:** **{best_fit}** is the strongest individual lead for this project based on the required skills. However, for ISO 14971, no one is rated as an expert (Level 3).")
            st.warning("**Development Action:** Prioritize ISO 14971 Risk Management training for at least two team members this quarter to mitigate this single-point-of-failure risk.")
        
        csv = df_skills.to_csv().encode('utf-8')
        st.download_button(
            label="Export Full Competency Matrix (CSV)",
            data=csv,
            file_name='team_competency_matrix.csv',
            mime='text/csv',
        )

    with tab4:
        st.header("Interactive ECO Impact Assessment Tool")
        st.info("A logic-driven tool to ensure a consistent, risk-based approach to V&V for post-market changes, ensuring compliance with 21 CFR 820.")
        change_type = st.selectbox("Select Type of Engineering Change Order (ECO)", ["Reagent Formulation Change", "Software (Minor UI change)", "Software (Algorithm update)", "Supplier Change (Critical Component)", "Manufacturing Process Change"])
        
        with st.container(border=True):
            st.subheader("Minimum Required V&V Activities (per SOP-00123)")
            rationale_text = ""
            impact_text = ""
            if change_type == "Reagent Formulation Change":
                st.error("🔴 **Full V&V Suite Required**")
                st.markdown("- Analytical Performance (Precision, LoD, Linearity)\n- Stability Studies (Accelerated & Real-time)\n- Clinical Bridging Study\n- Shipping Validation")
                rationale_text = "Change directly impacts assay performance and patient results. This is a high-risk change requiring comprehensive re-validation and potentially a new regulatory filing."
                impact_text = "**URS-001** (Clinical Sensitivity), **DI-002** (Analytical Sensitivity), **DI-003** (Stability)."
            elif change_type == "Software (Minor UI change)":
                st.success("🟢 **Limited V&V Required**")
                st.markdown("- Software Regression Testing (Targeted)\n- Usability Assessment (Summative if applicable)\n- Documentation Update")
                rationale_text = "Change does not impact the analytical algorithm or patient data integrity. This is a low-risk change focused on user experience."
                impact_text = "**SRS-012** (UI Display)."
            elif change_type == "Software (Algorithm update)":
                st.error("🔴 **Full Software & Analytical V&V Required**")
                st.markdown("- Full Software Validation Suite (per IEC 62304 Class)\n- Analytical Performance regression testing using old vs. new software\n- Full Risk Management File Update")
                rationale_text = "Change to the core algorithm directly impacts patient result calculation. This is the highest software risk category and requires rigorous verification."
                impact_text = "All performance requirements (**URS-001, DI-002**) and software requirements linked to the algorithm."
            elif change_type == "Supplier Change (Critical Component)":
                st.warning("🟡 **Targeted V&V Required**")
                st.markdown("- New Component Qualification (IQC)\n- System-level performance regression testing\n- Limited stability run (bracketing)\n- Comparability Analysis")
                rationale_text = "Change introduces a new variable into the system. This is a medium-risk change requiring confirmation that system performance, reliability, and safety are unaffected."
                impact_text = "All system-level requirements and potentially stability claims (**DI-003**)."
            elif change_type == "Manufacturing Process Change":
                st.warning("🟡 **Process Re-Validation Required**")
                st.markdown("- Process Validation (IQ, OQ, PQ) for the changed step\n- Product Performance Testing on 3 new lots\n- Stability testing on 1 new lot")
                rationale_text = "Change to the manufacturing process could impact product consistency and performance. A risk-based re-validation is required to ensure continued product quality."
                impact_text = "Product specification requirements, stability claims (**DI-003**)."

            st.markdown(f"**Rationale:** {rationale_text}")
            with st.container(border=True):
                st.info(f"**Traceability Impact Analysis:** This change affects the following critical requirements in the RTM: {impact_text}")


def render_post_market_page():
    """
    Renders the complete Post-Market Intelligence & CAPA Feeder page.
    This includes proactive post-market surveillance, AI-driven analysis, 
    and data-driven triggers for the CAPA system.
    """
    st.title("📡 7. Post-Market Intelligence & CAPA Feeder")
    render_director_briefing(
        "Closing the Quality Loop",
        "A mature V&V function extends its influence beyond product launch. This dashboard demonstrates proactive post-market surveillance, using field data to monitor real-world performance, identify emerging trends, and provide data-driven triggers for the CAPA system. This is a critical component of a robust Quality Management System.",
        "21 CFR 820.198 (Complaint files), 21 CFR 820.100 (CAPA), ISO 13485:2016 Section 8.2.2 & 8.5.2",
        "Drives continuous product improvement, reduces the risk of field actions or recalls, and demonstrates a culture of quality and patient safety to regulatory bodies."
    )
    
    # This metric card renders its own content, including a treemap, and returns a time-series figure
    render_metric_card(
        "AI-Powered Complaint Sentiment Analysis", 
        "This VADER sentiment analysis model processes the free-text of complaint records to classify the customer's sentiment as Positive, Neutral, or Negative. This provides a crucial layer of business context on top of the technical data.", 
        run_sentiment_analysis_model, 
        "The analysis reveals that while 'Instrument Error' is the most frequent complaint type, 'Reagent Leak' complaints are overwhelmingly associated with negative sentiment. This indicates that the reagent issue is causing significantly more customer frustration and potential brand damage, and should be prioritized for corrective action.", 
        "ISO 13485: 8.2.1 (Feedback)", 
        key="sentiment_analysis"
    )
    
    # Load complaint data and check for CAPA triggers
    df = get_complaint_data()
    capa_filter = df[(df['Lot_Number'] == 'A2301-B') & (df['Complaint_Type'] == 'False Positive')]
    if len(capa_filter) > 10:
        st.error(
            f"**🔴 CAPA Alert Triggered:** {len(capa_filter)} 'False Positive' complaints for Lot #A2301-B have been received in the last quarter, exceeding the defined threshold of 10. "
            "**Action:** Recommend initiating CAPA-2024-001. V&V to provide resources for investigation and re-validation of retained samples."
        )

    st.subheader("Post-Market Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**Complaint Analysis (Pareto)**")
            complaint_counts = df['Complaint_Type'].value_counts().reset_index()
            complaint_counts.columns = ['Complaint_Type', 'Count']
            complaint_counts['Cumulative_Percentage'] = 100 * complaint_counts['Count'].cumsum() / complaint_counts['Count'].sum()
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=complaint_counts['Complaint_Type'], y=complaint_counts['Count'], name='Count'), secondary_y=False)
            fig.add_trace(go.Scatter(x=complaint_counts['Complaint_Type'], y=complaint_counts['Cumulative_Percentage'], name='Cumulative %', mode='lines+markers'), secondary_y=True)
            fig.update_layout(title_text='Pareto Chart of Complaint Types')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.markdown("**Complaint Trend (Monthly)**")
            monthly_counts = df.resample('ME', on='Date').size().reset_index(name='Count')
            fig_ts = px.line(monthly_counts, x='Date', y='Count', title='Total Complaints per Month')
            st.plotly_chart(fig_ts, use_container_width=True)

    with st.container(border=True):
        st.markdown("**Geographic Complaint Hotspots**")
        region_counts = df['Region'].value_counts().reset_index()
        region_counts.columns = ['Region', 'Count']
        fig_map = px.choropleth(region_counts, locations='Region', locationmode="USA-states", color='Count', scope="usa", title="Complaints by US State", color_continuous_scale="Reds")
        st.plotly_chart(fig_map, use_container_width=True)
    
    # --- COMPLETE AI-DRIVEN ANALYSIS SECTION ---
    st.markdown("---")
    st.subheader("AI-Driven Predictive & Root Cause Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Predictive Maintenance Model")
            st.markdown("This model is trained on historical sensor data to predict the likelihood of an instrument failing in the near future. This enables proactive maintenance, reducing unplanned downtime and costly failed runs.")
            
            # Use st.pyplot for matplotlib figures, as corrected
            fig_pred = run_predictive_maintenance_model("pred_maint")
            st.pyplot(fig_pred)

            st.success("**Actionable Insight:** The model shows that 'Pump_Pressure' is by far the most significant predictor of failure. The maintenance team should prioritize monitoring this parameter and consider it a leading indicator for service scheduling.")

    with col2:
        # This is the completed "missing" part
        with st.container(border=True):
            st.markdown("#### NLP for Complaint Theme Discovery")
            st.markdown("This NLP model uses Latent Dirichlet Allocation (LDA) to analyze free-text from customer complaints, automatically grouping them into distinct topics. This turns unstructured data into actionable, quantifiable themes without manual review.")
            
            # This function renders its own table output
            run_nlp_topic_modeling("nlp_topic")
            
            st.success("**Actionable Insight:** The AI has automatically identified a recurring theme related to 'leaky reagent packs'. This quantitative signal elevates the issue's priority and provides a strong justification for launching a formal investigation with the packaging engineering team.")

# In render_post_market_page()

# ... after the CAPA alert is triggered ...
    if len(capa_filter) > 10:
        st.error(...) # The existing CAPA alert
    
        # --- ENHANCEMENT START ---
        with st.expander("📝 Open V&V Investigation Triage Form"):
            st.subheader("Initial V&V Plan for CAPA-2024-001")
            
            assigned_team = st.multiselect(
                "Assign V&V Team Members for Initial Investigation:",
                ['Alice (Statistics)', 'Bob (Reagent Expert)', 'Charlie (Instrumentation)'],
                default=['Bob (Reagent Expert)']
            )
            
            priority = st.selectbox(
                "Set Investigation Priority:",
                ("High", "Medium", "Low"),
                index=0
            )
            
            hypothesis = st.text_area(
                "Enter Initial Investigation Hypothesis:",
                "Given the complaints are localized to Lot #A2301-B, the initial hypothesis is a raw material deviance or a process error during the manufacturing of this specific lot. We will start by testing retained samples against a control lot."
            )
            
            if st.button("Log Initial Plan & Notify Quality Team"):
                st.success(f"Plan Logged! Assigned to: {', '.join(assigned_team)}. Priority: {priority}. Quality Assurance has been notified to formalize the CAPA record.")
    # --- ENHANCEMENT END ---

def render_dhf_hub_page() -> None:
    """Renders the Digital DHF & Workflow Hub page."""
    st.title("🗂️ 8. The Digital DHF & Workflow Hub")
    render_director_briefing(
        "Orchestrating the Design History File",
        "The DHF is not a static folder; it's a dynamic, living entity that requires active management and cross-functional alignment. This hub demonstrates the ability to manage formal QMS workflows and provides concrete examples of the key documents that V&V is responsible for authoring and maintaining, proving both procedural compliance and documentation excellence.",
        "21 CFR 820.30(j) (DHF), 21 CFR 820.40 (Document Controls), GAMP 5",
        "Ensures audit-proof documentation, accelerates review cycles by providing clear templates and expectations, and fosters seamless collaboration between V&V, R&D, Quality, and Regulatory."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("V&V Master Plan Approval Workflow")
            st.markdown("Status for `VV-MP-001_ImmunoPro-A`:")
            st.markdown("---")
            st.markdown("✔️ **V&V Lead (Self):** Approved `2024-01-15`")
            st.markdown("✔️ **R&D Project Lead:** Approved `2024-01-16`")
            st.markdown("✔️ **Quality Assurance Lead:** Approved `2024-01-17`")
            st.markdown("🟠 **Regulatory Affairs Lead:** Pending Review")
            st.markdown("⬜ **Head of Development:** Not Started")
            st.info("**Insight:** This workflow visualization provides instant status clarity for key deliverables, enabling proactive follow-up to prevent bottlenecks.")

    with col2:
        with st.container(border=True):
            st.subheader("Interactive Document Viewer")
            st.markdown("Click to expand and view mock V&V document templates.")
            
            with st.expander("📄 View Mock V&V Protocol Template"):
                st.markdown("""
                ### V&V Protocol: AVP-LOD-01 - Analytical Sensitivity (LoD)
                **Version:** 1.0
                ---
                **1.0 Purpose:** To determine the Limit of Detection (LoD) of the ImmunoPro-A Assay, defined as the lowest concentration of analyte that can be detected with 95% probability.

                **2.0 Scope:** This protocol applies to the ImmunoPro-A Assay on the QuidelOrtho-100 platform.

                **3.0 Traceability to Requirements:**
                - **DI-002:** Analytical sensitivity (LoD) shall be <= 50 copies/mL.

                **4.0 Method/Procedure:**
                - Prepare a dilution series of the analyte standard from 100 copies/mL down to 5 copies/mL.
                - Test each dilution level with 20 replicates across 3 different reagent lots and 2 instruments.
                - Run a negative control (0 copies/mL) with 60 replicates.
                
                **5.0 Acceptance Criteria:**
                - The hit rate at the claimed LoD (50 copies/mL) must be ≥ 95%.
                - The hit rate for the negative control must be ≤ 5%.

                **6.0 Data Analysis Plan:**
                - Data will be analyzed using Probit regression to calculate the 95% detection probability concentration.
                - Results will be summarized in a table showing hit rates for each level.
                """)

            with st.expander("📋 View Mock V&V Report Template"):
                st.markdown("### V&V Report: AVR-LOD-01 - Analytical Sensitivity (LoD)")
                st.caption("Version: 1.0")
                st.markdown("---")
                
                st.subheader("1.0 Summary")
                st.write("The LoD study was executed per protocol AVP-LOD-01. The results confirm that the ImmunoPro-A Assay meets the required analytical sensitivity.")
                
                st.subheader("2.0 Deviations")
                st.info("**DEV-001:** One replicate at the 25 copies/mL level on Instrument #2 was invalidated due to an operator error. The replicate was repeated successfully. **Impact Assessment:** None.")
                # --- ENHANCEMENT START ---
            with st.container(border=True):
                st.markdown("##### 🛡️ Simulate Audit Defense")
                if st.button("Query this deviation", key="audit_dev_001"):
                    st.warning("**Auditor Query:** 'Your deviation report states 'Impact: None.' Please provide the objective evidence and rationale for this conclusion.'")
                    st.success(
                        """
                        **My Response:** "Certainly. The deviation was a documented operator error on a single replicate out of 60 for that level, and over 300 total data points in the study. The protocol's acceptance criteria are based on the aggregate performance across all valid runs. 
                        
                        We immediately followed our SOP for handling such events, which required invalidating the data point, documenting the root cause, and repeating the replicate. The successful re-test and the statistical power of the overall study design give us high confidence that this isolated, corrected event had no material impact on the final study conclusion or the validity of the LoD claim."
                        """
                    )
# --- ENHANCEMENT END ---

                st.subheader("3.0 Results vs. Acceptance Criteria")
# --- ENHANCEMENT END ---

                st.subheader("3.0 Results vs. Acceptance Criteria")
                st.subheader("3.0 Results vs. Acceptance Criteria")
                
                # Create a DataFrame for the results
                results_data = {
                    'Concentration (copies/mL)': [50, 25, 10, 0],
                    'Replicates': [60, 60, 60, 60],
                    'Hits': [59, 52, 31, 2],
                    'Hit Rate (%)': [98.3, 86.7, 51.7, 3.3],
                    'Acceptance Criteria': ['>= 95%', 'N/A', 'N/A', '<= 5%'],
                    'Pass/Fail': ['PASS', 'N/A', 'N/A', 'PASS']
                }
                results_df = pd.DataFrame(results_data)

                # Define a styling function for Pass/Fail column
                def style_pass_fail(val):
                    if val == 'PASS':
                        color = 'lightgreen'
                    elif val == 'FAIL':
                        color = '#ffcccb'  # light red
                    else:
                        color = 'white'
                    return f'background-color: {color}'

                # Apply the styling to the DataFrame
                styled_df = results_df.style.applymap(style_pass_fail, subset=['Pass/Fail'])
                
                # Display the styled DataFrame
                st.dataframe(styled_df, use_container_width=True, hide_index=True)

                st.subheader("4.0 Conclusion")
                
                # Use columns for a cleaner layout of key metrics
                key_cols = st.columns(2)
                key_cols[0].metric(
                    label="Claimed LoD Performance",
                    value="98.3% Hit Rate",
                    help="Hit Rate @ 50 copies/mL. Meets acceptance criteria of >= 95%."
                )
                key_cols[1].metric(
                    label="Calculated LoD (Probit)",
                    value="38.5 copies/mL",
                    help="The calculated concentration for 95% detection probability."
                )

                st.success("The study successfully demonstrated a hit rate of 98.3% at 50 copies/mL, satisfying the acceptance criteria. The LoD is confirmed to be ≤ 50 copies/mL. Probit analysis provides a significant performance margin.")

                st.subheader("5.0 Traceability")
                st.info("This report provides objective evidence fulfilling requirement **DI-002**.")


def render_operations_page() -> None:
    """Renders the V&V Operations & Automation page."""
    st.title("⚙️ 9. V&V Operations & Automation")
    render_director_briefing(
        "Building a High-Performance V&V Engine",
        "A world-class V&V department is not just a project function; it's a highly efficient operational unit. This dashboard demonstrates leadership in optimizing departmental performance through strategic automation, data-driven capital planning, and a culture of continuous improvement. The goal is to increase V&V throughput, improve data quality, and maximize the value of every resource.",
        "ISO 13485: 6.3 (Infrastructure), Lean Six Sigma Principles, 21 CFR Part 11 (for automated systems)",
        "Reduces project timelines, lowers operational costs, justifies capital expenditures with data, and allows skilled personnel to focus on complex problem-solving rather than repetitive tasks, thereby improving employee engagement and retention."
    )

    st.subheader("🧪 Test Automation & Efficiency")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        fig_pie, fig_dual = create_automation_dashboard("automation")
        with col1:
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.plotly_chart(fig_dual, use_container_width=True)
        
        kpi_cols = st.columns(3)
        kpi_cols[0].metric("Manual Test Rework Rate", "8.2%", delta="1.1%", delta_color="inverse")
        kpi_cols[1].metric("Automated Test Rework Rate", "1.5%", delta="-0.3%")
        kpi_cols[2].metric("Defect Escape Rate (Post-Launch)", "0.1%", delta="-0.05%")
        st.success("**Actionable Insight:** The upward trend in automation coverage directly correlates with a reduction in execution time and rework rates. This provides a clear ROI for continued investment in our test automation framework.")

    st.subheader("🔬 Lab Instrument & Capital Planning")
    with st.container(border=True):
        fig_heatmap, fig_forecast = create_instrument_utilization_dashboard("utilization")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.plotly_chart(fig_forecast, use_container_width=True)
        st.error("**Actionable Insight:** The heatmap reveals our primary analytical platform is severely bottlenecked during standard work hours. The AI forecast confirms that we will exceed 90% capacity within 4 months, jeopardizing the timelines for two upcoming projects. This data forms the basis of a CapEx request for a new instrument in the next budget cycle.")


def render_qms_cockpit_page() -> None:
    """Renders the Integrated QMS Cockpit page."""
    st.title("✅ 10. The Integrated QMS Cockpit")
    render_director_briefing(
        "Ensuring an 'Always Audit Ready' State",
        "A V&V leader is a key steward of the Quality Management System (QMS). This cockpit provides a real-time view of the department's compliance posture, tracking audit performance and the status of V&V's involvement in critical quality records like CAPAs and Non-Conformance Reports (NCMRs).",
        "21 CFR 820 (QSR), ISO 13485: 8.2.4 (Internal Audit), 8.5.2 (CAPA)",
        "Fosters a culture of proactive quality, ensures the department is prepared for unannounced inspections, and provides visibility into V&V's workload and cycle times for supporting the broader QMS, enabling better resource planning."
    )
    
    st.subheader("Audit & Inspection Readiness")
    with st.container(border=True):
        st.dataframe(create_audit_dashboard("audit"), use_container_width=True, hide_index=True)
        kpi_cols = st.columns(3)
        kpi_cols[0].metric("V&V-related Findings (Last 4 Audits)", 4, delta="-2 vs. prior period")
        kpi_cols[1].metric("Avg. Finding Closure Time (Days)", 25, delta="-5 days", delta_color="inverse")
        kpi_cols[2].metric("V&V SOPs Updated (Last 12 Mo.)", "95%")
        st.success("**Actionable Insight:** The downward trend in audit findings and faster closure times are direct results of targeted SOP updates and training initiatives from the previous year, demonstrating a successful continuous improvement loop.")

    with st.container(border=True):
        create_qms_kanban("qms")
        st.info("**Actionable Insight:** The Kanban board shows a potential bottleneck forming in the 'Investigation' phase. This indicates a need to either allocate more resources to initial investigations or to analyze the incoming quality records for recurring themes that could be addressed systemically.")


def render_business_metrics_page() -> None:
    """Renders the V&V Business & Quality Metrics Hub page."""
    st.title("💰 11. V&V Business & Quality Metrics Hub")
    render_director_briefing(
        "Driving V&V as a Business Unit",
        "This dashboard transcends project-level tracking to demonstrate leadership of the V&V department as a financially responsible business unit. It provides visibility into departmental OpEx and, crucially, quantifies V&V's role in reducing the company-wide Cost of Poor Quality (COPQ). This is how a V&V leader communicates value in the language of the C-suite.",
        "ISO 13485: 5.6 (Management Review), 21 CFR 820.20 (Management Responsibility)",
        "Provides clear financial stewardship of departmental budgets and demonstrates the direct, positive ROI of investing in robust V&V by showing its impact on preventing costly internal and external failures."
    )
    
    st.subheader("Departmental Operational Expense (OpEx) Management")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        fig_gauge, fig_bar = create_opex_dashboard("opex")
        with col1:
            st.plotly_chart(fig_gauge, use_container_width=True)
        with col2:
            st.plotly_chart(fig_bar, use_container_width=True)
        st.success("**Actionable Insight:** The department is tracking at 84% of its annual OpEx budget, indicating strong financial control. The consistent monthly burn rate allows for predictable financial forecasting and resource planning for the remainder of the fiscal year.")
    
    with st.container(border=True):
        fig_corr = create_copq_modeler("copq")
        st.plotly_chart(fig_corr, use_container_width=True)
        st.error("**Actionable Insight:** The AI-powered model, trained on historical project data, clearly demonstrates that for every dollar invested in V&V during development, the company saves approximately $1.50 in post-launch COPQ. This provides a powerful, data-driven argument for fully funding V&V activities to maximize long-term profitability.")


def render_portfolio_page() -> None:
    """Renders the V&V Portfolio Command Center page."""
    st.title("📂 12. V&V Portfolio Command Center")
    render_director_briefing(
        "Managing the V&V Portfolio",
        "An effective director manages a portfolio of projects, not just a single timeline. This requires balancing competing priorities, allocating finite resources, and providing clear, high-level status updates to executive leadership. This command center demonstrates the ability to manage these complexities and make data-driven trade-off decisions.",
        "Project Management Body of Knowledge (PMBOK), ISO 13485: 5.6 (Management Review)",
        "Provides executive-level visibility into V&V's contribution to corporate goals, enables proactive risk management across projects, and ensures strategic alignment of the department's most valuable asset: its people."
    )
    
    st.subheader("Project Portfolio Health (RAG Status)")
    with st.container(border=True):
        st.dataframe(create_portfolio_health_dashboard("portfolio"), use_container_width=True, hide_index=True)
        st.error("**Actionable Insight:** The CardioScreen-X project is flagged 'Red' for both Technical Risk and Resource Strain. The V&V team is currently unable to support the aggressive feasibility timeline. **Action:** Escalate to the project core team to either de-scope the initial phase or re-allocate bioinformatics resources from another project.")

    st.subheader("Integrated Resource Allocation Matrix")
    with st.container(border=True):
        # This function was refactored to return the fig and the over_allocated DataFrame
        fig_alloc, over_allocated_df = create_resource_allocation_matrix("allocation")
        st.plotly_chart(fig_alloc, use_container_width=True)
        if not over_allocated_df.empty:
            for _, row in over_allocated_df.iterrows():
                st.warning(f"**⚠️ Over-allocation Alert:** {row['Team Member']} is allocated at {row['Total Allocation']}%. This is unsustainable and poses a risk of burnout and project delays.")

    # In render_portfolio_page()
    
    st.subheader("Project Portfolio Health (RAG Status)")
        with st.container(border=True):
            # This call now only happens when render_portfolio_page() is executed.
            st.dataframe(create_portfolio_health_dashboard("portfolio"), use_container_width=True, hide_index=True)
            
            st.error("**Actionable Insight:** The CardioScreen-X project is flagged 'Red' for both Technical Risk and Resource Strain. The V&V team is currently unable to support the aggressive feasibility timeline.")
            
            # This is the actionability enhancement, correctly placed here.
            if st.button("Simulate Escalation Memo to Core Team", key="escalate_cardio"):
                st.subheader("Generated Escalation Memo")
                st.info("This is the type of clear, data-driven communication required to resolve cross-functional roadblocks.")
                st.text_area(
                    "Memo Draft:",
                    """
    TO: Project Core Team Lead, CardioScreen-X
    FROM: Associate Director, Assay V&V
    SUBJECT: URGENT: V&V Resource Deficit and Risk Assessment for CardioScreen-X
    
    Team,
    
    This memo is to formally escalate a critical resource and technical risk for the CardioScreen-X project, as identified in our V&V Portfolio Command Center.
    
    1.  **The Issue:** The project is currently 'Red' due to a combination of high technical risk and insufficient resource allocation within the V&V team. Our current V&V staffing cannot support the aggressive feasibility timeline without jeopardizing quality or impacting other critical projects.
    
    2.  **The Data:** Our resource matrix confirms that key personnel with the required bioinformatics skills are over-allocated.
    
    3.  **The Recommendation:** I request an immediate core team meeting to discuss one of the following mitigation strategies:
        a) De-scoping the initial feasibility phase to reduce the V&V burden.
        b) Re-allocating dedicated bioinformatics resources from a lower-priority project for the duration of this phase.
    
    Please advise on scheduling this discussion at your earliest convenience. We must address this to maintain the project's viability.
    
    Regards,
    Jose Bautista MSc, LSSBB, PMP
    Associate Director, Assay V&V
                    """,
                    height=350
                )
        # --- END OF THE FIX ---
    
        # The rest of the page content remains the same.
        st.subheader("Integrated Resource Allocation Matrix")
        with st.container(border=True):
            fig_alloc, over_allocated_df = create_resource_allocation_matrix("allocation")
            st.plotly_chart(fig_alloc, use_container_width=True)
            if not over_allocated_df.empty:
                for _, row in over_allocated_df.iterrows():
                    st.warning(f"**⚠️ Over-allocation Alert:** {row['Team Member']} is allocated at {row['Total Allocation']}%. This is unsustainable and poses a risk of burnout and project delays.")
        # --- ENHANCEMENT END ---

def render_learning_hub_page() -> None:
    """Renders the Organizational Learning & Knowledge Hub page."""
    st.title("🧠 13. Organizational Learning & Knowledge Hub")
    render_director_briefing(
        "Building a Learning Organization",
        "A V&V department's greatest asset is its cumulative experience. A key leadership function is to capture, organize, and democratize this knowledge to prevent repeating past mistakes and accelerate future projects. This hub demonstrates a system for turning historical data into an active, intelligent resource.",
        "ISO 13485: 8.4 (Analysis of Data), 8.5.1 (Improvement)",
        "Dramatically reduces the learning curve for new team members, improves the quality of V&V plans by leveraging past data, and fosters a culture of continuous improvement and knowledge sharing. This directly lowers the Cost of Poor Quality (COPQ)."
    )

    st.subheader("AI-Powered Lessons Learned Search Engine")
    with st.container(border=True):
        st.info("This tool uses Natural Language Processing (NLP) to search the entire history of V&V reports, CAPAs, and ECOs to find relevant insights for new projects.")
        create_lessons_learned_search("lessons_learned")


def render_global_strategy_page() -> None:
    """Renders the Global V&V Strategy & R&D Pipeline Advisor page."""
    st.title("🌎 14. Global V&V Strategy & R&D Pipeline Advisor")
    render_director_briefing(
        "Using V&V Data to Drive Corporate Strategy",
        "A visionary V&V leader uses data from past and present projects to de-risk and influence the future. This dashboard demonstrates the capability to manage global, multi-site V&V operations and provides a forward-looking AI tool that advises the R&D and Business Development teams on the risks and costs of the future product pipeline.",
        "ICH Q10 (Pharmaceutical Quality System), GHTF/SG3/N17 (Risk Management)",
        "Ensures consistent product quality and performance across global manufacturing sites, and transforms V&V from a downstream testing function into an upstream strategic partner that helps the company make smarter, data-driven investment decisions."
    )
    
    st.subheader("Global Method Transfer & Harmonization")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        fig_bar, df_status = create_method_transfer_dashboard("transfer")
        with col1:
            st.plotly_chart(fig_bar, use_container_width=True)
        with col2:
            st.markdown("##### Transfer Protocol Status")
            st.dataframe(df_status, use_container_width=True, hide_index=True)
        st.success("**Actionable Insight:** The method transfer is proceeding well, with comparable performance on key metrics. The slight increase in %CV and bias at the receiving site are within acceptable limits defined in the transfer plan. This data provides high confidence for proceeding with the validation of the Athens site.")
        
    st.subheader("AI-Powered R&D Pipeline Risk Advisor")
    with st.container(border=True):
        fig_quad = create_pipeline_advisor("pipeline")
        st.plotly_chart(fig_quad, use_container_width=True)
        st.error("**Actionable Insight:** Our AI model, based on historical data, predicts that Project Delta, while having the highest ROI, also carries the highest V&V cost and complexity. It recommends that executive leadership fully fund the projected V&V budget and timeline for this project, or risk significant delays. Conversely, Project Alpha is a low-risk, quick win that can be executed with minimal resource strain.")


# --- SIDEBAR NAVIGATION AND PAGE ROUTING ---
# Using a dictionary for clean page routing
PAGES = {
    "Executive Summary": render_main_page,
    "1. Design Controls & Planning": render_design_controls_page,
    "2. Method & Process Validation": render_method_validation_page,
    "3. Execution Monitoring & SPC": render_execution_monitoring_page,
    "4. Project & Quality Management": render_quality_management_page,
    "5. Advanced Statistical Methods": render_stats_page,
    "6. Strategic Command & Control": render_strategic_command_page,
    "7. Post-Market Surveillance": render_post_market_page,
    "8. Digital DHF & Workflow Hub": render_dhf_hub_page,
    "9. V&V Operations & Automation": render_operations_page,
    "10. Integrated QMS Cockpit": render_qms_cockpit_page,
    "11. Business & Quality Metrics": render_business_metrics_page,
    "12. Portfolio Command Center": render_portfolio_page,
    "13. Knowledge & Learning Hub": render_learning_hub_page,
    "14. Global Strategy & Pipeline": render_global_strategy_page,
}

st.sidebar.title("V&V Command Center")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# Get the function to render the selected page
page_to_render_func = PAGES[selection]
# Call the function to render the page
page_to_render_func()
