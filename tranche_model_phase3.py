
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(layout="wide")
st.title("Tranche Pricing with Greeks and Optionality (Phase 3)")
n_names = 125  # typical for CDX index

# === SIDEBAR ===
st.sidebar.header("Simulation Controls")
run_sim = st.sidebar.button("🔁 Run Simulation")

# CIR Parameters
st.sidebar.header("CIR Intensity Model")
r0 = st.sidebar.slider("Initial Intensity (r₀, %)", 1.0, 15.0, 5.0, step=0.1) / 100
kappa = st.sidebar.slider("Mean Reversion Speed (κ)", 0.01, 1.0, 0.3)
theta = st.sidebar.slider("Long-Term Mean (θ, %)", 1.0, 20.0, 7.0, step=0.1) / 100
sigma = st.sidebar.slider("Volatility (σ, %)", 1.0, 50.0, 15.0, step=0.5) / 100
T = st.sidebar.slider("Maturity (Years)", 1, 10, 5)
dt = 1/12
n_steps = int(T / dt)
time_grid = np.linspace(0, T, n_steps + 1)

# Tranche Structure
st.sidebar.header("Tranche Parameters")
attachment = st.sidebar.slider("Attachment Point", 0.00, 0.3, 0.03, step=0.01)
detachment = st.sidebar.slider("Detachment Point", 0.05, 1.0, 0.07, step=0.01)
exposure_factor = st.sidebar.slider("Portfolio Sensitivity (factor)", 0.05, 0.5, 0.2, step=0.01)
notional = 1e7  # $10 million

# Simulation Size
n_paths = st.sidebar.selectbox("Monte Carlo Paths", [100, 500, 1000], index=1)

# Optionality
st.sidebar.header("Embedded Option")
option_type = st.sidebar.selectbox("Option Type", ["None", "Forward-Start Payer", "Callable Tranche"])
option_start = st.sidebar.slider("Option Start (Years)", 0.5, T-0.5, 1.0, step=0.5)
option_strike = st.sidebar.slider("Option Strike (as % tranche loss)", 0.0, 1.0, 0.05, step=0.01)

# === MODEL ===
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

# Tranche loss
tranche_losses = np.clip(cum_losses - attachment, 0, detachment - attachment) / (detachment - attachment)
avg_loss = np.mean(tranche_losses, axis=0)
avg_notional = 1 - avg_loss
tranche_cashflows = avg_notional[:-1] * dt  # approximation

# Stepwise cumulative number of defaulted names
defaulted_names = np.round(cum_losses * n_names).astype(int)

# Inferred correlation proxy (loss variance normalized)
expected_loss = np.mean(tranche_losses[:, -1])
loss_variance = np.var(tranche_losses[:, -1])
correlation_proxy = (
    loss_variance / (expected_loss * (1 - expected_loss))
    if expected_loss * (1 - expected_loss) > 0 else 0
)

# PUF and Spread Estimation
discount_factor = np.exp(-r0 * T)
puf = (1 - avg_loss[-1]) * discount_factor
spread = (np.sum(tranche_cashflows) / np.sum(avg_notional[:-1])) * 10000  # in bps

# Greeks (finite differences on r0 and sigma)
def compute_greek_shifted(base_param, shift, param_name):
    kwargs = {"r0": r0, "sigma": sigma}
    kwargs[param_name] = base_param + shift
    shifted_rates = simulate_cir_paths(**kwargs, kappa=kappa, theta=theta, T=T, n_paths=n_paths)
    shifted_losses = np.cumsum(shifted_rates * dt * exposure_factor, axis=1)
    shifted_losses = np.minimum(shifted_losses, 1.0)
    shifted_tranche_losses = np.clip(shifted_losses - attachment, 0, detachment - attachment) / (detachment - attachment)
    return np.mean(shifted_tranche_losses, axis=0)

epsilon = 0.005
delta_path = (compute_greek_shifted(r0, epsilon, "r0") - compute_greek_shifted(r0, -epsilon, "r0")) / (2 * epsilon)
vega_path = (compute_greek_shifted(sigma, epsilon, "sigma") - compute_greek_shifted(sigma, -epsilon, "sigma")) / (2 * epsilon)

# Option Payoff (if any)
option_index = int(option_start / dt)
if option_type == "Forward-Start Payer":
    payoff_paths = np.maximum(tranche_losses[:, -1] - option_strike, 0)
    option_value = np.mean(payoff_paths)
elif option_type == "Callable Tranche":
    called = tranche_losses[:, option_index] >= option_strike
    payoff_paths = np.where(called, tranche_losses[:, option_index], tranche_losses[:, -1])
    option_value = np.mean(payoff_paths)
else:
    option_value = None

# === OUTPUT ===
col1, col2 = st.columns(2)

with col1:
    st.subheader("CIR Intensity Paths")
    fig1, ax1 = plt.subplots()
    for i in range(min(10, n_paths)):
        ax1.plot(time_grid, rates[i, :], alpha=0.4)
    ax1.plot(time_grid, np.mean(rates, axis=0), label="Mean", color='black')
    ax1.set_ylabel("Intensity (annualized)")
    ax1.set_xlabel("Time (Years)")
    ax1.legend()
    st.pyplot(fig1)

with col2:
    st.subheader("Tranche Loss Paths")
    fig2, ax2 = plt.subplots()
    for i in range(min(10, n_paths)):
        ax2.plot(time_grid, tranche_losses[i, :], alpha=0.4)
    ax2.plot(time_grid, avg_loss, label="Mean", color='black')
    ax2.set_ylabel("Tranche Loss")
    ax2.set_xlabel("Time (Years)")
    ax2.legend()
    st.pyplot(fig2)

# Greeks
st.subheader("Greeks Over Time")
fig3, ax3 = plt.subplots()
ax3.plot(time_grid, delta_path, label="Delta (∂Loss/∂r₀)")
ax3.plot(time_grid, vega_path, label="Vega (∂Loss/∂σ)")
ax3.set_xlabel("Time (Years)")
ax3.set_ylabel("Greek Value")
ax3.legend()
st.pyplot(fig3)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Cumulative Defaulted Names")
    fig_def, ax_def = plt.subplots()
    for i in range(min(10, n_paths)):
        ax_def.step(time_grid, defaulted_names[i, :], where='post', alpha=0.4)
    ax_def.plot(time_grid, np.mean(defaulted_names, axis=0), label="Mean", color='black')
    ax_def.set_xlabel("Time (Years)")
    ax_def.set_ylabel("Number of Defaults")
    ax_def.set_ylim(0, n_names)
    ax_def.legend()
    st.pyplot(fig_def)

with col4:
    st.subheader("Inferred Correlation Proxy")
    st.write(f"**Proxy ρ ≈ {correlation_proxy:.3f}** (based on tranche loss variance)")
    fig_corr, ax_corr = plt.subplots()
    ax_corr.hist(tranche_losses[:, -1], bins=30, alpha=0.6, color='skyblue', edgecolor='black')
    ax_corr.axvline(expected_loss, color='red', linestyle='--', label='Mean Loss')
    ax_corr.set_title("Distribution of Final Tranche Losses")
    ax_corr.set_xlabel("Final Tranche Loss")
    ax_corr.set_ylabel("Frequency")
    ax_corr.legend()
    st.pyplot(fig_corr)

# Final Metrics
st.subheader("Final Metrics")
st.write(f"**Final Tranche Loss**: {avg_loss[-1]*100:.2f} %")
st.write(f"**Price Up Front (PUF)**: {puf*100:.2f} %")
st.write(f"**Equivalent Spread**: {spread:.2f} bps")
if option_value is not None:
    st.write(f"**{option_type} Value**: {option_value:.4f} (per unit notional)")
