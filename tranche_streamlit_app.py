
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# CIR simulation parameters
st.title("Tranche Pricing and Greeks (Mobile Friendly)")
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

# Display chart
st.subheader("Average CIR Intensity Path")
fig, ax = plt.subplots()
ax.plot(np.linspace(0, T, n_steps + 1), expected_rate_path, label="Mean Intensity")
ax.set_xlabel("Time (Years)")
ax.set_ylabel("Intensity")
ax.legend()
st.pyplot(fig)

# Placeholder tranche modeling logic
st.subheader("Tranche Loss (Simplified)")
attachment = 0.03
detachment = 0.07
expected_portfolio_loss = np.minimum(expected_rate_path.cumsum() * dt * 0.2, 1.0)
tranche_loss = np.clip(expected_portfolio_loss - attachment, 0, detachment - attachment) / (detachment - attachment)
notional_remaining = 1 - tranche_loss

fig2, ax2 = plt.subplots()
ax2.plot(np.linspace(0, T, n_steps + 1), tranche_loss, label="Tranche Loss")
ax2.plot(np.linspace(0, T, n_steps + 1), notional_remaining, label="Tranche Notional Remaining")
ax2.set_xlabel("Time (Years)")
ax2.set_ylabel("Loss / Notional")
ax2.legend()
st.pyplot(fig2)

# Greeks approximation (finite difference on initial intensity)
epsilon = 0.005
rates_up = simulate_cir_paths(r0 + epsilon, kappa, theta, sigma, T, n_paths)
rates_down = simulate_cir_paths(r0 - epsilon, kappa, theta, sigma, T, n_paths)
loss_up = np.clip(np.minimum(np.mean(rates_up, axis=0).cumsum() * dt * 0.2, 1.0) - attachment, 0, detachment - attachment) / (detachment - attachment)
loss_down = np.clip(np.minimum(np.mean(rates_down, axis=0).cumsum() * dt * 0.2, 1.0) - attachment, 0, detachment - attachment) / (detachment - attachment)

delta = (loss_up[-1] - loss_down[-1]) / (2 * epsilon)
gamma = (loss_up[-1] - 2 * tranche_loss[-1] + loss_down[-1]) / (epsilon**2)

st.subheader("Greeks (Final Time Step)")
st.write(f"**Delta** (∂Loss/∂r₀): {delta:.4f}")
st.write(f"**Gamma** (∂²Loss/∂r₀²): {gamma:.4f}")
