# app.py (Enterprise Version)

import streamlit as st
from modules import dashboard, planning_execution, compliance_risk
from utils import helpers
from database import init_db

# --- Initialize Database on first run ---
init_db()

# --- Load Config ---
config = helpers.load_config()

st.set_page_config(
    layout="wide",
    page_title=config['app_name'],
    page_icon="🔬"
)

# --- Authentication Check ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    helpers.render_login()
else:
    # --- Main Application Layout ---
    user = helpers.get_current_user()

    with st.sidebar:
        st.image(config['logo_url'], width=200)
        st.title("AssayVantage")
        st.markdown("### Command Center")
        st.markdown("---")

        PAGES = {
            "Executive Dashboard": dashboard,
            "V&V Lifecycle Management": planning_execution,
        }
        # Only show Compliance Hub to Directors
        if user.role == 'director':
            PAGES["Compliance & Risk Hub"] = compliance_risk
        
        selection = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
        
        st.markdown("---")
        st.info(f"User: **{st.session_state.full_name}**\n\nRole: **{st.session_state.role.capitalize()}**")
        if st.button("Logout"):
            helpers.log_action(user.id, "User logged out.")
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    page_module = PAGES[selection]
    page_module.render_page()
