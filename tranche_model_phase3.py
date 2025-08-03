import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 Tranche Pricing Model – Phase 4.1")

# === User Controls ===
col1, col2 = st.columns(2)
with col1:
    r0 = st.slider("Initial Intensity (r₀)", 0.005, 0.15, 0.05)
    sigma = st.slider("Volatility (σ)", 0.01, 0.5, 0.15)
    kappa = st.slider("Mean Reversion Speed (κ)", 0.01, 1.0, 0.3)
    theta = st.slider("Long-Term Intensity (θ)", 0.01, 0.2, 0.07)
    exposure_factor = st.slider("Exposure Factor", 0.05, 0.5, 0.2)

with col2:
    attachment = st.slider("Attachment Point", 0.0, 1.0, 0.03)
    detachment = st.slider("Detachment Point", attachment + 0.01, 1.0, 0.07)
    running_spread = st.slider("Running Spread (bps)", 0, 2000, 500) / 10000
    callable_years = st.multiselect("Callable At Year", [1, 2, 3, 4], default=[])
    forward_start_year = st.slider("Forward-Start Year", 0.0, 4.9, 0.0)

# === Shock Inputs ===
st.sidebar.header("🧨 Market Shocks")
shock_r0 = st.sidebar.slider("Shock to r₀ (bps)", -100, 100, 0) / 10000
shock_sigma = st.sidebar.slider("Shock to σ", -0.2, 0.2, 0.0)
hedge_ratio = st.sidebar.slider("Index Hedge Ratio", 0.0, 2.0, 1.0)

# === Simulation Settings ===
T, dt = 5, 1/12
n_steps = int(T / dt)
time_grid = np.linspace(0, T, n_steps + 1)
n_paths = 500
discount_curve = np.exp(-0.01 * time_grid)

def simulate_cir_paths(r0, kappa, theta, sigma, n_steps, dt, n_paths):
    r = np.zeros((n_paths, n_steps + 1))
    r[:, 0] = r0
    for t in range(1, n_steps + 1):
        sqrt_r = np.sqrt(np.maximum(r[:, t-1], 0))
        dW = np.random.normal(0, np.sqrt(dt), size=n_paths)
        dr = kappa * (theta - r[:, t-1]) * dt + sigma * sqrt_r * dW
        r[:, t] = np.maximum(r[:, t-1] + dr, 0)
    return r

def compute_tranche_values(r_paths):
    shifted_time = int(forward_start_year / dt)
    r_paths = r_paths[:, shifted_time:]
    time_adj = time_grid[:len(r_paths[0])]

    cum_intensity = np.cumsum(r_paths * dt * exposure_factor, axis=1)
    cum_intensity = np.minimum(cum_intensity, 1.0)

    loss = np.clip(cum_intensity - attachment, 0, detachment - attachment) / (detachment - attachment)
    notional = 1 - loss

    discount = np.exp(-0.01 * time_adj)
    callable_index = min([int(y/dt) for y in callable_years], default=len(time_adj))

    running_leg = np.sum(notional[:, :callable_index] * running_spread * dt * discount[:callable_index], axis=1)
    protection_leg = loss[:, -1]

    expected_running = np.mean(running_leg)
    expected_protection = np.mean(protection_leg)
    cost_buyer = expected_running - expected_protection
    avg_notional = np.mean(np.sum(notional * dt * discount, axis=1))
    eq_spread = expected_running / avg_notional if avg_notional > 0 else 0

    return {
        "loss": loss,
        "notional": notional,
        "running_leg": expected_running,
        "protection_leg": expected_protection,
        "cost": cost_buyer,
        "eq_spread": eq_spread,
        "time": time_adj
    }

base_r = simulate_cir_paths(r0, kappa, theta, sigma, n_steps, dt, n_paths)
shock_r = simulate_cir_paths(r0 + shock_r0, kappa, theta, sigma + shock_sigma, n_steps, dt, n_paths)

res_base = compute_tranche_values(base_r)
res_shock = compute_tranche_values(shock_r)
hedge_pnl = hedge_ratio * shock_r0 * 10000  # bps estimate

# === Display ===
col1, col2 = st.columns(2)
with col1:
    st.metric("Equivalent Spread (Base)", f"{res_base['eq_spread']*10000:.1f} bps")
    st.metric("Cost to Buyer (Base)", f"{res_base['cost']:.4f}")
    st.metric("Running PV", f"{res_base['running_leg']:.4f}")
    st.metric("Protection PV", f"{res_base['protection_leg']:.4f}")
with col2:
    st.metric("Eq Spread (Shocked)", f"{res_shock['eq_spread']*10000:.1f} bps")
    st.metric("Shocked Cost", f"{res_shock['cost']:.4f}")
    st.metric("Estimated Hedge PnL", f"{hedge_pnl:.1f} bps")


st.subheader("📉 Average Notional Remaining Over Time")
fig1, ax1 = plt.subplots()
ax1.plot(res_base["time"], np.mean(res_base["notional"], axis=0), label="Base", lw=2)
ax1.plot(res_shock["time"], np.mean(res_shock["notional"], axis=0), label="Shocked", lw=2, linestyle="--")
ax1.set_ylabel("Notional"), ax1.set_xlabel("Time")
ax1.grid(True), ax1.legend()
st.pyplot(fig1)

st.subheader("📈 Inferred Correlation Proxy (StdDev of Loss)")
corr_proxy = np.std(res_base["loss"], axis=0)
fig2, ax2 = plt.subplots()
ax2.plot(res_base["time"], corr_proxy, label="Loss Volatility", color="purple")
ax2.set_ylabel("Std Dev"), ax2.set_xlabel("Time")
ax2.grid(True)
st.pyplot(fig2)

st.subheader("🧮 Cumulative Defaults")
avg_cum = np.mean(1 - res_base["notional"], axis=0)
fig3, ax3 = plt.subplots()
ax3.step(res_base["time"], avg_cum * 125, where="post", label="Expected Defaults")
ax3.set_ylabel("# Defaults"), ax3.set_xlabel("Time")
ax3.grid(True)
st.pyplot(fig3)
