
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# App setup
st.set_page_config(layout="wide")
st.title("Credit Tranche Pricing with Optionality (Phase 2)")

# Sidebar controls
st.sidebar.header("Simulation Controls")
run_sim = st.sidebar.button("🔁 Run Simulation")

# CIR Parameters
st.sidebar.header("CIR Intensity Model")
r0 = st.sidebar.slider("Initial Intensity (r₀)", 0.01, 0.15, 0.05)
kappa = st.sidebar.slider("Mean Reversion Speed (κ)", 0.01, 1.0, 0.3)
theta = st.sidebar.slider("Long Term Mean (θ)", 0.01, 0.20, 0.07)
sigma = st.sidebar.slider("Volatility (σ)", 0.01, 0.5, 0.15)
T = st.sidebar.slider("Maturity (Years)", 1, 10, 5)
dt = 1/12
n_steps = int(T / dt)
time_grid = np.linspace(0, T, n_steps + 1)

# Tranche structure
st.sidebar.header("Tranche Structure")
attachment = st.sidebar.slider("Attachment Point", 0.00, 0.3, 0.03, step=0.01)
detachment = st.sidebar.slider("Detachment Point", 0.05, 1.0, 0.07, step=0.01)
exposure_factor = st.sidebar.slider("Portfolio Sensitivity", 0.05, 0.5, 0.2, step=0.01)
n_paths = st.sidebar.selectbox("Number of Monte Carlo Paths", [100, 500, 1000], index=1)

# Option type
st.sidebar.header("Option Type")
option_type = st.sidebar.selectbox("Option Type", ["None", "Forward-Start Payer", "Callable Tranche"])
option_start = st.sidebar.slider("Forward/Call Start (Years)", 0.5, T-0.5, 1.0, step=0.5)
option_strike = st.sidebar.slider("Option Strike (as % loss)", 0.0, 1.0, 0.05, step=0.05)

# State for simulation trigger
if "rates" not in st.session_state or run_sim:
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

    st.session_state.rates = simulate_cir_paths(r0, kappa, theta, sigma, T, n_paths)

rates = st.session_state.rates
cum_losses = np.cumsum(rates * dt * exposure_factor, axis=1)
cum_losses = np.minimum(cum_losses, 1.0)

# Tranche loss and average
tranche_losses = np.clip(cum_losses - attachment, 0, detachment - attachment) / (detachment - attachment)
avg_loss = np.mean(tranche_losses, axis=0)
avg_notional = 1 - avg_loss

# Optionality logic
option_index = int(option_start / dt)
if option_type == "Forward-Start Payer":
    payoff_paths = np.maximum(tranche_losses[:, -1] - option_strike, 0)
    option_value = np.mean(payoff_paths)
elif option_type == "Callable Tranche":
    # Callable if loss exceeds strike at option_start
    called = tranche_losses[:, option_index] >= option_strike
    payoff_paths = np.where(called, tranche_losses[:, option_index], tranche_losses[:, -1])
    option_value = np.mean(payoff_paths)
else:
    option_value = None

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("CIR Intensity Paths")
    fig1, ax1 = plt.subplots()
    for i in range(min(10, n_paths)):
        ax1.plot(time_grid, rates[i, :], alpha=0.5)
    ax1.plot(time_grid, np.mean(rates, axis=0), label="Mean", color='black')
    ax1.set_xlabel("Time (Years)")
    ax1.set_ylabel("Intensity")
    ax1.legend()
    st.pyplot(fig1)

with col2:
    st.subheader("Tranche Loss Paths")
    fig2, ax2 = plt.subplots()
    for i in range(min(10, n_paths)):
        ax2.plot(time_grid, tranche_losses[i, :], alpha=0.5)
    ax2.plot(time_grid, avg_loss, label="Mean", color='black')
    ax2.set_xlabel("Time (Years)")
    ax2.set_ylabel("Tranche Loss")
    ax2.legend()
    st.pyplot(fig2)

# Results
st.subheader("Final Metrics")
st.write(f"**Average Final Tranche Loss**: {avg_loss[-1]:.4f}")
st.write(f"**Average Remaining Notional**: {avg_notional[-1]:.4f}")
if option_value is not None:
    st.write(f"**{option_type} Option Value**: {option_value:.4f}")
