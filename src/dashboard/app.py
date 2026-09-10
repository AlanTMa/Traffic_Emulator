import atexit
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.model.config import load_config
from src.model.generate import generate_topology_config

st.set_page_config(page_title="Traffic Emulator Live Dashboard", layout="wide")

st.title("🚀 Traffic Emulator: Live Convergence Dashboard")
st.markdown("Set up a topology in the sidebar and launch a simulation; charts update live from `runs/latest/metrics.jsonl`.")

# Configuration
RUN_DIR = PROJECT_ROOT / "runs"
OUTPUT_DIR = RUN_DIR / "latest"
METRICS_FILE = OUTPUT_DIR / "metrics.jsonl"  # written by the simulation (schema: src/telemetry/schema.py)
RUN_CONFIG = RUN_DIR / "dashboard_run.yaml"
RUN_LOG = RUN_DIR / "dashboard_run.log"
REFRESH_RATE = 2 # seconds

# --- Simulation process management ---

@st.cache_resource
def sim_process() -> dict:
    """The dashboard's simulation subprocess, shared across reruns and browser tabs."""
    holder = {"proc": None, "log": None, "label": ""}
    atexit.register(stop_simulation, holder)
    return holder

def is_running(holder: dict) -> bool:
    return holder["proc"] is not None and holder["proc"].poll() is None

def stop_simulation(holder: dict):
    if is_running(holder):
        holder["proc"].terminate()
        holder["proc"].wait(timeout=10)
    if holder["log"] is not None:
        holder["log"].close()
        holder["log"] = None

def launch_simulation(holder: dict, config: dict, label: str):
    stop_simulation(holder)
    RUN_DIR.mkdir(exist_ok=True)
    RUN_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False))
    METRICS_FILE.unlink(missing_ok=True)  # start the charts from scratch
    holder["log"] = open(RUN_LOG, "w")
    holder["proc"] = subprocess.Popen(
        [sys.executable, "-u", "-m", "src.cli", "run", "--config", str(RUN_CONFIG), "--output-dir", str(OUTPUT_DIR)],
        cwd=PROJECT_ROOT, stdout=holder["log"], stderr=subprocess.STDOUT,
    )
    holder["label"] = label

holder = sim_process()

# --- Sidebar: setup and launch ---

with st.sidebar:
    st.header("Simulation setup")
    topology_source = st.radio("Topology", ["Generate", "Config file"], horizontal=True)

    if topology_source == "Generate":
        n_sources = st.slider("Sources (producers)", 1, 20, 5)
        n_brokers = st.slider("Brokers (service nodes)", 1, 10, 3)
        load = st.slider("Target load", 0.05, 0.90, 0.30, 0.05,
                         help="Total offered rate as a fraction of total broker capacity")
        seed = st.number_input("Seed", min_value=0, value=42, step=1)
        with st.expander("Controller parameters"):
            window = st.number_input("Window (s)", 0.5, 60.0, 5.0, 0.5,
                                     help="Seconds per controller iteration")
            warmup = st.number_input("Warm-up windows", 0, 100, 4)
            eta = st.slider("eta (split inertia)", 0.01, 1.0, 0.35)
            gamma = st.slider("gamma (price smoothing)", 0.01, 1.0, 0.50)
            beta = st.slider("beta (arrival-rate EWMA)", 0.01, 1.0, 0.30)
    else:
        config_files = sorted((PROJECT_ROOT / "config").glob("*.yaml"))
        config_file = st.selectbox("Config", config_files, format_func=lambda p: p.name)

    launch_col, stop_col = st.columns(2)
    if launch_col.button("Launch", type="primary", width="stretch"):
        try:
            if topology_source == "Generate":
                config = {
                    "simulation": {"window": window, "warmup": int(warmup), "controller_mode": "windowed_stochastic"},
                    "algorithm": {"eta": eta, "gamma": gamma, "beta": beta},
                    "topology": generate_topology_config(n_sources, n_brokers, load, int(seed)),
                }
                label = f"Generated {n_sources}×{n_brokers}, load {load:.0%}, seed {seed}"
            else:
                config = load_config(config_file)
                label = config_file.name
            launch_simulation(holder, config, label)
            st.rerun()
        except ValueError as e:
            st.error(str(e))
    if stop_col.button("Stop", width="stretch", disabled=not is_running(holder)):
        stop_simulation(holder)
        st.rerun()

if RUN_CONFIG.exists() and holder["label"]:
    with st.expander(f"Topology: {holder['label']}"):
        topo_cfg = yaml.safe_load(RUN_CONFIG.read_text())["topology"]
        src_col, brk_col = st.columns(2)
        src_col.dataframe(pd.DataFrame(topo_cfg["sources"]).rename(columns={"rate": "rate (pkt/s)"}), hide_index=True)
        brk_col.dataframe(pd.DataFrame(topo_cfg["brokers"]).rename(columns={"capacity": "capacity (pkt/s)"}), hide_index=True)

# --- Live charts ---

def load_data():
    if not METRICS_FILE.exists():
        return None
    try:
        df = pd.read_json(METRICS_FILE, lines=True)
    except ValueError:
        return None  # empty file, or a line still being written
    if df.empty:
        return df
    # Scalar views of the per-broker records for the summary charts
    df["timestamp"] = df["sim_time"]
    df["rel_change"] = df[["route_rel", "price_rel"]].max(axis=1)
    df["max_util_planned"] = df["util_j"].apply(max)
    measured = df["util_measured_j"] if "util_measured_j" in df else df["util_j"]
    df["max_util"] = measured.apply(max)
    return df

def render_status(df):
    proc = holder["proc"]
    progress = f" · iteration {int(df['iteration'].iloc[-1])}, t = {df['timestamp'].iloc[-1]:.0f}s" if df is not None and not df.empty else ""
    if is_running(holder):
        st.success(f"Running: {holder['label']}{progress}")
    elif proc is not None:
        code = proc.returncode
        if code == 0:
            st.info(f"Finished: {holder['label']}{progress}")
        else:
            st.warning(f"Stopped: {holder['label']}{progress}")
            log_tail = RUN_LOG.read_text(errors="replace").strip().splitlines()[-15:] if RUN_LOG.exists() else []
            if any("Traceback" in line or "Error" in line for line in log_tail):
                st.code("\n".join(log_tail))

# Main layout: the fragment re-runs on its own every REFRESH_RATE seconds,
# so each chart key is registered exactly once per run.
@st.fragment(run_every=REFRESH_RATE)
def render_dashboard():
    df = load_data()
    render_status(df)

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

    elif is_running(holder):
        st.info("Simulation started; the first point appears after the first window.")
    else:
        st.info("No simulation data yet. Choose a topology in the sidebar and press **Launch**.")

render_dashboard()
