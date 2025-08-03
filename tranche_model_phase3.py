import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Streamlit App Config ---
st.set_page_config(layout="wide")
st.title("📘 Credit Tranche Pricing App – Phase 4.2")

# --- GUI Inputs ---
with st.sidebar:
    st.header("📌 Simulation Inputs")
    r0 = st.slider("Initial Intensity r₀", 0.005, 0.15, 0.05, step=0.005)
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
dt = 1 / 12
n_steps = int(T / dt)
time_grid = np.linspace(0, T, n_steps + 1)
discount_curve = np.exp(-0.01 * time_grid)
n_paths = 500

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

    call_idx = min([int(y/dt) for y in callable_years], default=len(local_time))

    running_leg = np.sum(notional[:, :call_idx] * running_spread * dt * discount[:call_idx], axis=1)
    protection_leg = tranche_loss[:, -1]

    running_pv = np.mean(running_leg)
    protection_pv = np.mean(protection_leg)
    net_cost = running_pv - protection_pv

    avg_notional = np.mean(np.sum(notional * dt * discount, axis=1))
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

# --- Simulate ---
base_paths = simulate_cir_paths(r0, kappa, theta, sigma, n_paths, n_steps, dt)
shock_paths = simulate_cir_paths(r0 + shock_r0, kappa, theta, sigma + shock_sigma, n_paths, n_steps, dt)

res_base = compute_tranche_valuation(base_paths)
res_shock = compute_tranche_valuation(shock_paths)

# --- Output Metrics ---
colA, colB = st.columns(2)
with colA:
    st.subheader("🔵 Base Scenario")
    st.metric("Equivalent Spread", f"{res_base['eq_spread']*10000:.2f} bps")
    st.metric("Total Cost to Buyer", f"{res_base['cost']:.5f} units")
    st.metric("PV of Running Leg", f"{res_base['running_leg']:.5f}")
    st.metric("PV of Protection Leg", f"{res_base['protection_leg']:.5f}")

with colB:
    st.subheader("🟠 Shocked Scenario")
    st.metric("Eq Spread (Shocked)", f"{res_shock['eq_spread']*10000:.2f} bps")
    st.metric("Cost to Buyer (Shocked)", f"{res_shock['cost']:.5f} units")
    pnl = (res_base['eq_spread'] - res_shock['eq_spread']) * 10000 * hedge_ratio
    st.metric("Estimated Hedge PnL", f"{pnl:.2f} bps")

# --- Charts ---
st.subheader("📉 Average Notional Remaining")
fig1, ax1 = plt.subplots()
ax1.plot(res_base["time"], np.mean(res_base["notional"], axis=0), label="Base", lw=2)
ax1.plot(res_shock["time"], np.mean(res_shock["notional"], axis=0), label="Shocked", lw=2, linestyle="--")
ax1.set_ylabel("Remaining Notional")
ax1.set_xlabel("Time (years)")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)

st.subheader("📊 Inferred Correlation Proxy (Std of Loss)")
corr_proxy = np.std(res_base["loss"], axis=0)
fig2, ax2 = plt.subplots()
ax2.plot(res_base["time"], corr_proxy, color="purple", label="Std Dev of Loss")
ax2.set_ylabel("Standard Deviation")
ax2.set_xlabel("Time (years)")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

st.subheader("🧮 Expected Cumulative Defaults")
avg_cum = np.mean(1 - res_base["notional"], axis=0)
fig3, ax3 = plt.subplots()
ax3.step(res_base["time"], avg_cum * 125, where="post", label="Expected Defaults", color="darkgreen")
ax3.set_ylabel("# of Defaults (out of 125)")
ax3.set_xlabel("Time (years)")
ax3.grid(True)
ax3.legend()
st.pyplot(fig3)
