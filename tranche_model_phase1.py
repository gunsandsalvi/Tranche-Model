
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# CIR simulation parameters
st.title("Tranche Pricing and Greeks (Phase 1 Upgrade)")
st.sidebar.header("CIR Intensity Model")

r0 = st.sidebar.slider("Initial Intensity (r₀)", 0.01, 0.15, 0.05)
kappa = st.sidebar.slider("Mean Reversion Speed (κ)", 0.01, 1.0, 0.3)
theta = st.sidebar.slider("Long Term Mean (θ)", 0.01, 0.20, 0.07)
sigma = st.sidebar.slider("Volatility (σ)", 0.01, 0.5, 0.15)
T = st.sidebar.slider("Maturity (Years)", 1, 10, 5)
n_paths = st.sidebar.selectbox("Number of Monte Carlo Paths", [100, 500, 1000], index=1)
dt = 1/12
n_steps = int(T / dt)

def simulate_cir_paths(r0, kappa, theta, sigma, T, n_paths):
    dt = 1/12
    n_steps = int(T / dt)
    rates = np.zeros((n_paths, n_steps + 1))
    rates[:, 0] = r0
    for t in range(1, n_steps + 1):
        sqrt_r = np.sqrt(np.maximum(rates[:, t-1], 0))
        dW = np.random.normal(0, np.sqrt(dt), size=n_paths)
        dr = kappa * (theta - rates[:, t-1]) * dt + sigma * sqrt_r * dW
        rates[:, t] = np.maximum(rates[:, t-1] + dr, 0)
    return rates

# Simulate intensity paths
rates = simulate_cir_paths(r0, kappa, theta, sigma, T, n_paths)
expected_rate_path = np.mean(rates, axis=0)

# Tranche parameters
st.sidebar.header("Tranche Structure")
attachment = st.sidebar.slider("Attachment Point", 0.00, 0.3, 0.03, step=0.01)
detachment = st.sidebar.slider("Detachment Point", 0.05, 1.0, 0.07, step=0.01)
exposure_factor = st.sidebar.slider("Portfolio Sensitivity Factor", 0.05, 0.5, 0.2, step=0.01)

# Calculate portfolio loss (cumulative)
cum_losses = np.cumsum(rates * dt * exposure_factor, axis=1)
cum_losses = np.minimum(cum_losses, 1.0)

# Calculate tranche loss for each path
tranche_losses = np.clip(cum_losses - attachment, 0, detachment - attachment) / (detachment - attachment)
notional_remaining = 1 - tranche_losses

# Averaged tranche loss (path-dependent)
avg_loss = np.mean(tranche_losses, axis=0)
avg_notional = 1 - avg_loss

# Charts: Sample paths
st.subheader("Sample CIR Intensity Paths")
fig1, ax1 = plt.subplots()
for i in range(min(10, n_paths)):
    ax1.plot(np.linspace(0, T, n_steps + 1), rates[i, :], alpha=0.6)
ax1.plot(np.linspace(0, T, n_steps + 1), expected_rate_path, color='black', linewidth=2, label="Mean")
ax1.set_xlabel("Time (Years)")
ax1.set_ylabel("Intensity")
ax1.legend()
st.pyplot(fig1)

st.subheader("Sample Tranche Loss Paths")
fig2, ax2 = plt.subplots()
for i in range(min(10, n_paths)):
    ax2.plot(np.linspace(0, T, n_steps + 1), tranche_losses[i, :], alpha=0.6)
ax2.plot(np.linspace(0, T, n_steps + 1), avg_loss, color='black', linewidth=2, label="Mean")
ax2.set_xlabel("Time (Years)")
ax2.set_ylabel("Tranche Loss")
ax2.legend()
st.pyplot(fig2)

# Output: Average final results
st.subheader("Final Metrics")
st.write(f"**Average Final Tranche Loss**: {avg_loss[-1]:.4f}")
st.write(f"**Average Remaining Notional**: {avg_notional[-1]:.4f}")
