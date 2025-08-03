# tranche_model_phase44.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Streamlit Setup ---
st.set_page_config(layout="wide")
st.title("💵 Credit Tranche Pricing App – Phase 4.4 (PnL + Live Calibration)")

# --- Market Data Fetch (Live Calibration) ---
@st.cache_data(ttl=3600)
def fetch_cdx_5y_spread():
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2"
    try:
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        return df.iloc[-1, 0]  # latest spread in bps
    except:
        return None

spread5y = fetch_cdx_5y_spread()
if spread5y is not None:
    st.sidebar.metric("Live CDX 5Y Spread (FRED)", f"{spread5y:.1f} bps")
    r0_default = (spread5y / 10000) / 0.6  # assume 40% recovery
else:
    r0_default = 0.05

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("📌 Simulation Inputs")
    r0 = st.slider("Initial Intensity r₀", 0.005, 0.15, float(r0_default), step=0.005)
    sigma = st.slider("Volatility σ", 0.01, 0.5, 0.15, step=0.01)
    kappa = st.slider("Mean Reversion κ", 0.01, 1.0, 0.3, step=0.01)
    theta = st.slider("Long-term Intensity θ", 0.01, 0.2, 0.07, step=0.005)
    exposure_factor = st.slider("Exposure Factor", 0.05, 0.5, 0.2, step=0.01)

    st.header("📈 Tranche Structure")
    attachment = st.slider("Attachment (%)", 0.0, 1.0, 0.03, step=0.01)
    detachment = st.slider("Detachment (%)", attachment + 0.01, 1.0, 0.07, step=0.01)
    running_spread = st.slider("Running Spread (bps)", 0, 2000, 500, step=25) / 10000
    forward_start_year = st.slider("Forward-Start Year", 0.0, 4.9, 0.0, step=0.25)
    callable_years = st.multiselect("Callable At (Years)", [1, 2, 3, 4], default=[])

    st.header("🧨 Shock & Hedge")
    shock_r0 = st.slider("Shock to r₀ (bps)", -100, 100, 0, step=10) / 10000
    shock_sigma = st.slider("Shock to σ", -0.2, 0.2, 0.0, step=0.01)
    hedge_ratio = st.slider("Index Hedge Ratio", 0.0, 2.0, 1.0, step=0.1)

# --- Constants ---
T = 5
n_steps = int(T * 12)
dt = 1 / 12
time_grid = np.linspace(0, T, n_steps + 1)
discount_curve = np.exp(-0.01 * time_grid)
n_paths = 500
total_index_size = 125_000_000

# --- CIR Simulation ---
def simulate_cir_paths(r0, kappa, theta, sigma, n_paths, n_steps, dt):
    r = np.zeros((n_paths, n_steps + 1))
    r[:, 0] = r0
    for t in range(1, n_steps + 1):
        sqrt_r = np.sqrt(np.maximum(r[:, t-1], 0))
        dW = np.random.normal(0, np.sqrt(dt), size=n_paths)
        dr = kappa * (theta - r[:, t-1]) * dt + sigma * sqrt_r * dW
        r[:, t] = np.maximum(r[:, t-1] + dr, 0)
    return r

# --- Tranche Valuation ---
def compute_tranche_valuation(r_paths):
    forward_idx = int(forward_start_year / dt)
    r_paths = r_paths[:, forward_idx:]
    local_time = time_grid[:len(r_paths[0])]

    cum_intensity = np.cumsum(r_paths * dt * exposure_factor, axis=1)
    cum_intensity = np.minimum(cum_intensity, 1.0)

    tranche_loss = np.clip(cum_intensity - attachment, 0, detachment - attachment) / (detachment - attachment)
    notional = 1 - tranche_loss
    discount = np.exp(-0.01 * local_time)

    call_idx = min([int(y / dt) for y in callable_years], default=len(local_time))

    running_leg = np.sum(notional[:, :call_idx] * running_spread * dt * discount[:call_idx], axis=1)
    protection_leg = tranche_loss[:, -1]

    tranche_notional = (detachment - attachment) * total_index_size
    running_pv = np.mean(running_leg) * tranche_notional
    protection_pv = np.mean(protection_leg) * tranche_notional
    net_cost = running_pv - protection_pv
    avg_notional = np.mean(np.sum(notional * dt * discount, axis=1)) * tranche_notional
    eq_spread = running_pv / avg_notional if avg_notional > 0 else 0

    return {
        "time": local_time,
        "r_paths": r_paths,
        "notional": notional,
        "loss": tranche_loss,
        "running_leg": running_pv,
        "protection_leg": protection_pv,
        "cost": net_cost,
        "eq_spread": eq_spread
    }

# --- Simulations ---
base_paths = simulate_cir_paths(r0, kappa, theta, sigma, n_paths, n_steps, dt)
shock_paths = simulate_cir_paths(r0 + shock_r0, kappa, theta, sigma + shock_sigma, n_paths, n_steps, dt)

res_base = compute_tranche_valuation(base_paths)
res_shock = compute_tranche_valuation(shock_paths)

# --- Display Metrics ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("🔵 Base Scenario (in USD)")
    st.metric("Equivalent Spread", f"{res_base['eq_spread']*10000:.2f} bps")
    st.metric("Running Leg PV", f"${res_base['running_leg'] / 1e6:,.3f} M")
    st.metric("Protection Leg PV", f"${res_base['protection_leg'] / 1e6:,.3f} M")
    st.metric("Net PV to Buyer", f"${res_base['cost'] / 1e6:,.3f} M")

with col2:
    st.subheader("🟠 Shocked Scenario (in USD)")
    st.metric("Eq Spread (Shocked)", f"{res_shock['eq_spread']*10000:.2f} bps")
    st.metric("Net PV to Buyer (Shocked)", f"${res_shock['cost'] / 1e6:,.3f} M")
    pnl = (res_base['eq_spread'] - res_shock['eq_spread']) * total_index_size * hedge_ratio
    st.metric("Estimated Hedge PnL", f"${pnl / 1e6:,.3f} M")

# --- PnL Attribution ---
st.subheader("📉 PnL Attribution (Buyer Perspective)")
dp_cost = res_shock['cost'] - res_base['cost']
dp_running = res_shock['running_leg'] - res_base['running_leg']
dp_protection = res_shock['protection_leg'] - res_base['protection_leg']

st.write(f"- Δ Running Leg PV: ${dp_running/1e6:.3f} M")
st.write(f"- Δ Protection Leg PV: ${dp_protection/1e6:.3f} M")
st.write(f"- → Net Δ Cost to Buyer: ${dp_cost/1e6:.3f} M")

# --- Charts ---
st.subheader("📉 Average Notional Remaining")
fig1, ax1 = plt.subplots()
ax1.plot(res_base["time"], np.mean(res_base["notional"], axis=0), label="Base", lw=2)
ax1.plot(res_shock["time"], np.mean(res_shock["notional"], axis=0), label="Shocked", lw=2, linestyle="--")
ax1.set_ylabel("Remaining Tranche Notional")
ax1.set_xlabel("Time (years)")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)

st.subheader("📊 Inferred Correlation Proxy (Std Dev of Loss)")
corr_proxy = np.std(res_base["loss"], axis=0)
fig2, ax2 = plt.subplots()
ax2.plot(res_base["time"], corr_proxy, color="purple")
ax2.set_ylabel("Std Dev of Loss")
ax2.set_xlabel("Time (years)")
ax2.grid(True)
st.pyplot(fig2)

st.subheader("🧮 Cumulative Defaults (Expected)")
avg_cum = np.mean(1 - res_base["notional"], axis=0) * 125
fig3, ax3 = plt.subplots()
ax3.step(res_base["time"], avg_cum, where="post", color="darkgreen")
ax3.set_ylabel("Expected Defaulted Names")
ax3.set_xlabel("Time (years)")
ax3.grid(True)
st.pyplot(fig3)
