import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Traffic Emulator Live Dashboard", layout="wide")

st.title("🚀 Traffic Emulator: Live Convergence Dashboard")
st.markdown("This dashboard reads `simulation_metrics.csv` in real-time to visualize system stability.")

# Configuration
METRICS_FILE = "simulation_metrics.csv"
REFRESH_RATE = 2 # seconds

def load_data():
    if not os.path.exists(METRICS_FILE):
        return None
    try:
        return pd.read_csv(METRICS_FILE)
    except Exception as e:
        st.error(f"Error reading metrics file: {e}")
        return None

# Main layout: the fragment re-runs on its own every REFRESH_RATE seconds,
# so each chart key is registered exactly once per run.
@st.fragment(run_every=REFRESH_RATE)
def render_dashboard():
    df = load_data()

    if df is not None and not df.empty:
        # Top Row: KPIs
        col1, col2, col3 = st.columns(3)

        current_obj = df['objective'].iloc[-1]
        prev_obj = df['objective'].iloc[-2] if len(df) > 1 else current_obj
        rel_change = df['rel_change'].iloc[-1]
        max_util = df['max_util'].iloc[-1]

        col1.metric("Current Objective", f"{current_obj:.4f}", f"{current_obj - prev_obj:.4f}")
        col2.metric("Max Utilization", f"{max_util:.2%}")
        col3.metric("Rel. Change", "—" if pd.isna(rel_change) else f"{rel_change:.2e}")

        # Middle Row: Charts
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Objective Convergence")
            fig_obj = px.line(df, x='iteration', y='objective',
                             labels={'iteration': 'Iteration', 'objective': 'Total Delay'},
                             title="System Objective vs Iteration")
            st.plotly_chart(fig_obj, width="stretch", key="chart_objective")

        with chart_col2:
            st.subheader("Convergence Rate")
            fig_conv = px.line(df, x='iteration', y='rel_change',
                             labels={'iteration': 'Iteration', 'rel_change': 'Rel. Change'},
                             title="Relative Change (Convergence Speed)",
                             log_y=True)
            st.plotly_chart(fig_conv, width="stretch", key="chart_convergence")

        # Bottom Row: Utilization
        st.subheader("Max Broker Utilization")
        # Measured (EWMA of arrivals) vs planned (routing / capacity); older
        # metrics files only have max_util
        util_cols = {'max_util': 'Measured', 'max_util_planned': 'Planned'}
        df_util = df.rename(columns=util_cols)
        fig_util = px.line(df_util, x='iteration', y=[c for k, c in util_cols.items() if k in df],
                         labels={'iteration': 'Iteration', 'value': 'Utilization', 'variable': ''},
                         title="Max Utilization over Time")
        st.plotly_chart(fig_util, width="stretch", key="chart_utilization")

    else:
        st.warning("Waiting for simulation data... Please run `python src/cli.py run --config <config>.yaml` in another terminal.")
        if df is None:
            st.info("Simulation metrics file not found. Once the simulation starts, it will appear here.")

render_dashboard()
