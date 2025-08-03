import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Config
st.set_page_config(layout="wide")
st.title("Tranche Pricing – Phase 4: Callable, Shocks, and Hedging")

# === Inputs ===
r0 = st.slider("Initial Intensity (r₀)", 0.01, 0.15, 0.05)
sigma = st.slider("Volatility (σ)", 0.01, 0.5, 0.15)
callable_years = st.multiselect("Callable at year", [1, 2, 3, 4], default=[2])

# Shocks
shock_r0 = st.slider("Shock to r₀ (bps)", -100, 100, 0) / 10000
shock_sigma = st.slider("Shock to σ", -0.1, 0.1, 0.0)
hedge_ratio = st.slider("Index Notional for Hedge", 0.0, 2.0, 1.0)

# === Constants ===
kappa, theta = 0.3, 0.07
T, dt = 5, 1/12
n_steps = int(T / dt)
time_grid = np.linspace(0, T, n_steps + 1)
n_paths = 500
attachment, detachment = 0.03, 0.07
exposure_factor = 0.2
running_spread = 0.05  # 500 bps

# === CIR Simulation ===
def simulate_cir_paths(r0, kappa, theta, sigma, T, n_paths):
    rates = np.zeros((n_paths, n_steps + 1))
    rates[:, 0] = r0
    for t in range(1, n_steps + 1):
        sqrt_r = np.sqrt(np.maximum(rates[:, t-1], 0))
        dW = np.random.normal(0, np.sqrt(dt), size=n_paths)
        dr = kappa * (theta - rates[:, t-1]) * dt + sigma * sqrt_r * dW
        rates[:, t] = np.maximum(rates[:, t-1] + dr, 0)
    return rates

# === Tranche PV ===
def compute_tranche_pv(rates, callable_years):
    cum_losses = np.cumsum(rates * dt * exposure_factor, axis=1)
    cum_losses = np.minimum(cum_losses, 1.0)
    tranche_losses = np.clip(cum_losses - attachment, 0, detachment - attachment) / (detachment - attachment)
    notional_remaining = 1 - tranche_losses
    discount_factors = np.exp(-0.01 * time_grid)

    # Handle callable
    if callable_years:
        call_idx = int(min(callable_years) / dt)
        running_leg = np.sum(notional_remaining[:, :call_idx] * running_spread * dt * discount_factors[:call_idx], axis=1)
    else:
        running_leg = np.sum(notional_remaining * running_spread * dt * discount_factors, axis=1)

    protection_leg = tranche_losses[:, -1]
    expected_running_leg = np.mean(running_leg)
    expected_protection = np.mean(protection_leg)
    total_cost = expected_running_leg - expected_protection
    avg_notional = np.mean(np.sum(notional_remaining * dt * discount_factors, axis=1))
    eq_spread = expected_running_leg / avg_notional if avg_notional > 0 else 0

    return {
        "tranche_losses": tranche_losses,
        "notional_path": notional_remaining,
        "running_leg": expected_running_leg,
        "protection_leg": expected_protection,
        "total_cost": total_cost,
        "equivalent_spread": eq_spread
    }

# === Simulate ===
base_rates = simulate_cir_paths(r0, kappa, theta, sigma, T, n_paths)
shock_rates = simulate_cir_paths(r0 + shock_r0, kappa, theta, sigma + shock_sigma, T, n_paths)

res_base = compute_tranche_pv(base_rates, callable_years)
res_shock = compute_tranche_pv(shock_rates, callable_years)

# === Hedging Estimate ===
hedge_pnl = hedge_ratio * shock_r0 * 10000  # Rough approximation

# === Output ===
col1, col2 = st.columns(2)
with col1:
    st.metric("Base Equivalent Spread", f"{res_base['equivalent_spread']*10000:.1f} bps")
    st.metric("Base Total Cost (Buyer)", f"{res_base['total_cost']:.4f}")
    st.metric("Base Running PV", f"{res_base['running_leg']:.4f}")
    st.metric("Base Protection PV", f"{res_base['protection_leg']:.4f}")

with col2:
    st.metric("Shocked Equivalent Spread", f"{res_shock['equivalent_spread']*10000:.1f} bps")
    st.metric("Shocked Total Cost (Buyer)", f"{res_shock['total_cost']:.4f}")
    st.metric("Estimated Hedge PnL", f"{hedge_pnl:.1f} bps")

# === Chart ===
st.subheader("Average Notional Remaining")
fig, ax = plt.subplots()
ax.plot(time_grid, np.mean(res_base["notional_path"], axis=0), label="Base", lw=2)
ax.plot(time_grid, np.mean(res_shock["notional_path"], axis=0), label="Shocked", lw=2, linestyle="--")
ax.set_xlabel("Time")
ax.set_ylabel("Notional")
ax.grid(True)
ax.legend()
st.pyplot(fig)
