import atexit
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.controller.diagnostics import DEFAULT_TOLERANCES
from src.model.config import load_config
from src.model.generate import generate_topology_config
from src.telemetry.metrics import JsonlTail

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
MAX_ROWS = 3000  # most recent iterations kept for plotting

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
                    "simulation": {"window": window, "warmup": int(warmup), "controller_mode": "windowed_stochastic",
                                   "seed": int(seed)},
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
        src_col.dataframe(pd.DataFrame(topo_cfg["sources"]).rename(columns={"rate": "rate (work units/s)"}), hide_index=True)
        brk_col.dataframe(pd.DataFrame(topo_cfg["brokers"]).rename(columns={"capacity": "capacity (work units/s)"}), hide_index=True)

# --- Live charts ---

VIEWS = ["Overview", "Brokers", "Routing", "Optimality", "Queues & latency"]
view = st.segmented_control("View", VIEWS, default="Overview", key="view") or "Overview"

def load_data():
    # Incremental: each refresh parses only the rows appended since the last one
    if "metrics_tail" not in st.session_state:
        st.session_state.metrics_tail = JsonlTail(METRICS_FILE, max_rows=MAX_ROWS)
    tail = st.session_state.metrics_tail
    rows = tail.read()
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.attrs["total_rows"] = tail.total_rows
    # Scalar views of the per-broker records for the summary charts
    df["timestamp"] = df["sim_time"]
    df["rel_change"] = df[["route_rel", "price_rel"]].max(axis=1)
    df["max_util_planned"] = df["util_j"].apply(max)
    measured = df["util_measured_j"] if "util_measured_j" in df else df["util_j"]
    df["max_util"] = measured.apply(max)
    return df

def node_names(df):
    """Source and broker ids from the run's run.json when it matches, else P0.. / SN1.."""
    n_sources, n_brokers = np.array(df["lambda_ij"].iloc[-1]).shape
    try:
        meta = json.loads((OUTPUT_DIR / "run.json").read_text())
        if (meta["n_sources"], meta["n_brokers"]) == (n_sources, n_brokers):
            return meta["sources"], meta["brokers"]
    except (OSError, KeyError, ValueError):
        pass
    return [f"P{i}" for i in range(n_sources)], [f"SN{j + 1}" for j in range(n_brokers)]

def per_broker(df, column, brokers):
    """Long-form frame (iteration, broker, value) from a per-broker list column."""
    values = np.array(df[column].tolist(), dtype=float)
    out = pd.DataFrame(values, columns=brokers)
    out["iteration"] = df["iteration"].to_numpy()
    return out.melt(id_vars="iteration", var_name="broker", value_name=column)

def per_source(df, column, sources):
    """Long-form frame (iteration, source, value) from a per-source list column."""
    return per_broker(df, column, sources).rename(columns={"broker": "source"})

def per_route(df, column, sources, brokers):
    """Long-form frame (iteration, source, broker, value) from a [source][broker] matrix column."""
    values = np.array(df[column].tolist(), dtype=float)          # (T, N, M)
    T, N, M = values.shape
    return pd.DataFrame({
        "iteration": np.repeat(df["iteration"].to_numpy(), N * M),
        "source": np.tile(np.repeat(sources, M), T),
        "broker": np.tile(brokers, T * N),
        column: values.reshape(-1),
    })

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

def render_overview(df):
    col1, col2, col3 = st.columns(3)
    current_obj = df['objective'].iloc[-1]
    prev_obj = df['objective'].iloc[-2] if len(df) > 1 else current_obj
    rel_change = df['rel_change'].iloc[-1]
    col1.metric("Current Objective", f"{current_obj:.4f}", f"{current_obj - prev_obj:.4f}")
    col2.metric("Max Utilization", f"{df['max_util'].iloc[-1]:.2%}")
    col3.metric("Rel. Change", "—" if pd.isna(rel_change) else f"{rel_change:.2e}")

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
                           title="Relative Change (Convergence Speed)", log_y=True)
        st.plotly_chart(fig_conv, width="stretch", key="chart_convergence")

def render_brokers(df):
    _, brokers = node_names(df)
    measured = "util_measured_j" in df
    st.subheader("Broker utilization")
    util = per_broker(df, "util_j", brokers)
    if measured:
        # windowed_stochastic: planned (routing / capacity) vs measured (EWMA of arrivals)
        util["kind"] = "planned"
        util_m = per_broker(df, "util_measured_j", brokers).rename(columns={"util_measured_j": "util_j"})
        util_m["kind"] = "measured"
        util = pd.concat([util, util_m])
        fig = px.line(util, x="iteration", y="util_j", color="broker", line_dash="kind",
                      labels={"iteration": "Iteration", "util_j": "Utilization Λ_j / μ_j"})
    else:
        fig = px.line(util, x="iteration", y="util_j", color="broker",
                      labels={"iteration": "Iteration", "util_j": "Utilization Λ_j / μ_j"})
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, width="stretch", key="chart_broker_util")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Broker loads Λ_j")
        fig = px.line(per_broker(df, "load_j", brokers), x="iteration", y="load_j", color="broker",
                      labels={"iteration": "Iteration", "load_j": "Load Λ_j (work units/s)"})
        st.plotly_chart(fig, width="stretch", key="chart_broker_load")
    with col2:
        st.subheader("Congestion prices p_j")
        fig = px.line(per_broker(df, "price_j", brokers), x="iteration", y="price_j", color="broker",
                      labels={"iteration": "Iteration", "price_j": "Price p_j"})
        st.plotly_chart(fig, width="stretch", key="chart_broker_price")

    last = df.iloc[-1]
    table = pd.DataFrame({"load Λ_j": last["load_j"], "utilization": last["util_j"], "price p_j": last["price_j"]},
                         index=brokers)
    if measured:
        table["measured utilization"] = last["util_measured_j"]
    st.dataframe(table.style.format({c: "{:.2%}" if "utilization" in c else "{:.6g}" for c in table.columns}))

def render_routing(df):
    sources, brokers = node_names(df)
    st.subheader("Routing fractions λ_ij / λ_i")
    frac = per_route(df, "fraction_ij", sources, brokers)
    fig = px.line(frac, x="iteration", y="fraction_ij", color="broker", facet_col="source",
                  facet_col_wrap=min(len(sources), 5),
                  labels={"iteration": "Iteration", "fraction_ij": "Fraction"})
    fig.update_yaxes(range=[-0.02, 1.02])
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(fig, width="stretch", key="chart_routing_fractions")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current split")
        current = pd.DataFrame(df["fraction_ij"].iloc[-1], index=sources, columns=brokers)
        fig = px.imshow(current, text_auto=".1%", zmin=0, zmax=1, color_continuous_scale="Blues",
                        labels={"x": "Broker", "y": "Source", "color": "Fraction"}, aspect="auto")
        st.plotly_chart(fig, width="stretch", key="chart_routing_heatmap")
    with col2:
        st.subheader("Per-source mean end-to-end delay (model)")
        fig = px.line(per_source(df, "e2e_i", sources), x="iteration", y="e2e_i", color="source",
                      labels={"iteration": "Iteration",
                              "e2e_i": "Σ_j λ_ij (D_ij + D_j) / λ_i  (s)"})
        st.plotly_chart(fig, width="stretch", key="chart_source_delay")

RESIDUALS = {  # telemetry key -> (label, certificate tolerance key)
    "r_conservation": ("conservation", "r_conservation"),
    "r_capacity": ("capacity", "r_service_capacity"),
    "r_price": ("price consistency", "r_price"),
    "r_fixed_point": ("fixed point", "r_fixed_point"),
    "r_kkt_stationarity": ("KKT stationarity", "r_active_stationarity"),
    "r_kkt_complementarity": ("KKT complementarity", "r_kkt_complementarity"),
    "r_inactive_complementarity": ("unused-route KKT", "r_inactive_complementarity"),
}

def render_optimality(df):
    sources, brokers = node_names(df)
    last = df.iloc[-1]

    # Proposition 1 certificate for the latest state (never a convergence claim)
    if last["certified"]:
        st.success(f"PASS · {last['certificate_status']} (iteration {int(last['iteration'])})")
    else:
        st.error(f"FAIL · {last['certificate_status']} (iteration {int(last['iteration'])})")
    if last["controller_mode"] == "windowed_stochastic":
        st.caption("windowed_stochastic prices follow noisy measured rates, so price consistency and the "
                   "fixed point are not expected to hold exactly.")

    # Total marginal cost per route; unused routes (in the latest state) dotted
    st.subheader("Total marginal cost C_ij + C_j")
    marg = per_route(df, "M_ij", sources, brokers)
    active_now = pd.DataFrame(last["active_ij"], index=sources, columns=brokers)
    marg["route"] = np.where([active_now.loc[s, b] for s, b in zip(marg["source"], marg["broker"])], "used", "unused")
    fig = px.line(marg, x="iteration", y="M_ij", color="broker", line_dash="route", facet_col="source",
                  facet_col_wrap=min(len(sources), 5), log_y=True,
                  line_dash_map={"used": "solid", "unused": "dot"},
                  labels={"iteration": "Iteration", "M_ij": "C_ij + C_j"})
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(fig, width="stretch", key="chart_marginal_cost")

    col1, col2 = st.columns(2)
    with col1:
        # KKT / Wardrop: used routes equalize at alpha_i, unused routes lie at or above it
        st.subheader("KKT / Wardrop equalization (latest)")
        current = pd.DataFrame({
            "source": np.repeat(sources, len(brokers)),
            "broker": np.tile(brokers, len(sources)),
            "M_ij": np.array(last["M_ij"]).reshape(-1),
            "route": np.where(np.array(last["active_ij"]).reshape(-1), "used", "unused"),
        })
        fig = px.scatter(current, x="source", y="M_ij", color="broker", symbol="route", log_y=True,
                         symbol_map={"used": "circle", "unused": "circle-open"},
                         labels={"M_ij": "C_ij + C_j"})
        fig.add_scatter(x=sources, y=last["alpha_i"], mode="markers", name="α_i",
                        marker=dict(symbol="line-ew-open", size=28, color="black"))
        fig.update_traces(marker_size=11, selector=dict(mode="markers", type="scatter"))
        st.plotly_chart(fig, width="stretch", key="chart_kkt")
    with col2:
        st.subheader("Common safe step s_t")
        if df["s_t"].notna().any():
            steps = per_broker(df[df["s_j"].notna()], "s_j", brokers)
            fig = px.line(steps, x="iteration", y="s_j", color="broker",
                          labels={"iteration": "Iteration", "s_j": "Step bound"})
            fig.add_scatter(x=df["iteration"], y=df["s_t"], mode="lines", name="s_t = min_j s_j",
                            line=dict(color="black", width=3, dash="dash"))
            fig.update_yaxes(range=[0, 1.05])
            st.plotly_chart(fig, width="stretch", key="chart_safe_step")
        else:
            st.info("This controller mode has no safe step (windowed_stochastic); see static_algorithm1.")

    st.subheader("Residual diagnostics")
    floor = 1e-18  # exact zeros cannot be drawn on a log axis
    resid = pd.DataFrame({label: df[key].astype(float).clip(lower=floor)
                          for key, (label, _) in RESIDUALS.items() if key in df})
    resid["iteration"] = df["iteration"].to_numpy()
    fig = px.line(resid.melt(id_vars="iteration", var_name="residual", value_name="value"),
                  x="iteration", y="value", color="residual", log_y=True,
                  labels={"iteration": "Iteration", "value": f"Residual (zeros drawn at {floor:g})"})
    st.plotly_chart(fig, width="stretch", key="chart_residuals")
    table = pd.DataFrame([{"residual": label, "latest": last[key], "tolerance": DEFAULT_TOLERANCES[tol_key],
                           "pass": bool(abs(last[key]) <= DEFAULT_TOLERANCES[tol_key])}
                          for key, (label, tol_key) in RESIDUALS.items()])
    st.dataframe(table.style.format({"latest": "{:.3e}", "tolerance": "{:.0e}"}), hide_index=True)

def render_queues(df):
    if "latency_mean" not in df:
        st.info("Queue and latency metrics come from the event-driven simulation (windowed_stochastic); "
                "static_algorithm1 has no events or queues.")
        return
    sources, brokers = node_names(df)
    last = df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    st.caption("Each event is one normalized unit of work (5x3 instance: 1 unit = 1 MB); latency is the "
               "end-to-end sojourn time of a work unit, not a message latency.")
    col1.metric("Work units completed", f"{int(last['completed_total']):,}")
    col2.metric("Mean latency (run)", f"{last['latency_mean_total']:.4f} s")
    col3.metric("p95 (last window)", f"{last['latency_p95']:.4f} s")
    col4.metric("p99 (last window)", f"{last['latency_p99']:.4f} s")

    st.subheader("End-to-end work-unit sojourn time per window")
    lat = df[["iteration", "latency_mean", "latency_p50", "latency_p95", "latency_p99"]].rename(
        columns={"latency_mean": "mean", "latency_p50": "p50", "latency_p95": "p95", "latency_p99": "p99"})
    fig = px.line(lat.melt(id_vars="iteration", var_name="statistic", value_name="latency"),
                  x="iteration", y="latency", color="statistic",
                  labels={"iteration": "Window", "latency": "Sojourn time (s)"})
    st.plotly_chart(fig, width="stretch", key="chart_latency")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Per-source mean latency: measured vs model")
        measured = per_source(df, "latency_mean_i", sources).rename(columns={"latency_mean_i": "latency"})
        measured["kind"] = "measured"
        model = per_source(df, "e2e_i", sources).rename(columns={"e2e_i": "latency"})
        model["kind"] = "model"
        fig = px.line(pd.concat([measured, model]), x="iteration", y="latency", color="source", line_dash="kind",
                      labels={"iteration": "Window", "latency": "Sojourn time (s)"})
        st.plotly_chart(fig, width="stretch", key="chart_source_latency")
    with col2:
        st.subheader("Broker queue lengths")
        fig = px.line(per_broker(df, "queue_broker_j", brokers), x="iteration", y="queue_broker_j", color="broker",
                      labels={"iteration": "Window", "queue_broker_j": "In system (waiting + in service)"})
        st.plotly_chart(fig, width="stretch", key="chart_broker_queue")

    st.subheader("Access queue lengths (latest window)")
    fig = px.imshow(pd.DataFrame(last["queue_access_ij"], index=sources, columns=brokers), text_auto=True,
                    color_continuous_scale="Oranges", labels={"x": "Broker", "y": "Source", "color": "In system"},
                    aspect="auto")
    st.plotly_chart(fig, width="stretch", key="chart_access_queue")

RENDERERS = {"Overview": render_overview, "Brokers": render_brokers, "Routing": render_routing,
             "Optimality": render_optimality, "Queues & latency": render_queues}

# The fragment re-runs on its own every REFRESH_RATE seconds, so each chart
# key is registered exactly once per run.
@st.fragment(run_every=REFRESH_RATE)
def render_dashboard():
    df = load_data()
    render_status(df)

    if df is not None and not df.empty:
        if df.attrs["total_rows"] > len(df):
            st.caption(f"Showing the latest {len(df):,} of {df.attrs['total_rows']:,} iterations; "
                       f"the full log is {METRICS_FILE.relative_to(PROJECT_ROOT)}.")
        RENDERERS[view](df)
    elif is_running(holder):
        st.info("Simulation started; the first point appears after the first window.")
    else:
        st.info("No simulation data yet. Choose a topology in the sidebar and press **Launch**.")

render_dashboard()
