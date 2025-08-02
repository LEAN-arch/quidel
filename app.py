# app.py (CORRECTED)

import streamlit as st
from modules import dashboard, planning_execution, compliance_risk
from utils import helpers

st.set_page_config(
    layout="wide",
    page_title="AssayVantage Command Center",
    page_icon="🔬"
)

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    helpers.render_login()
else:
    if 'data_loaded' not in st.session_state:
        helpers.load_data(None)

    with st.sidebar:
        st.image("https://www.quidelortho.com/etc.clientlibs/quidelortho/clientlibs/clientlib-site/resources/images/logo_white.svg", width=200)
        st.title("AssayVantage")
        st.markdown("### Command Center")
        st.markdown("---")

        PAGES = {
            "Executive Dashboard": dashboard,
            "V&V Lifecycle Management": planning_execution,
            "Compliance & Risk Hub": compliance_risk,
        }
        
        selection = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
        
        st.markdown("---")
        st.info(f"Logged in as: **Director**")
        if st.button("Logout"):
            st.session_state.logged_in = False
            helpers.log_action("director", "User logged out.")
            st.rerun()  # <-- CORRECTED FUNCTION CALL

    page_module = PAGES[selection]
    page_module.render_page()
