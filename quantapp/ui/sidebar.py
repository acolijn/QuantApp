"""
Sidebar controls
================
Renders all sidebar widgets and returns a config dict consumed by the tabs.
"""

import streamlit as st
from quantapp.potentials import POTENTIALS


def render_sidebar():
    """Draw all sidebar controls and return a dict of parameter values.

    Returns
    -------
    dict with keys:
        potential_name, x_min, x_max, N, init_mode,
        x0, sigma, k0, eigen_n, n1, n2, ratio,
        dt, steps_per_frame, n_frames, speed, time_per_frame,
        show_components, show_potential, show_eigenstates, n_eigenstates,
        y_max_prob
    """
    # ── Potential ──────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Setup")
    potential_name = st.sidebar.selectbox(
        "Potential V(x)", list(POTENTIALS.keys()), index=3,
    )
    pot_info = POTENTIALS[potential_name]
    st.sidebar.caption(pot_info["description"])

    # Reset wavepacket defaults when the potential changes
    if st.session_state.get("last_potential") != potential_name:
        st.session_state.last_potential = potential_name
        st.session_state.x0 = pot_info["default_x0"]
        st.session_state.sigma = pot_info["default_sigma"]
        st.session_state.k0 = pot_info["default_k0"]

    x_min, x_max = pot_info["x_range"]
    N = st.sidebar.select_slider("Grid points", options=[256, 512, 1024, 2048], value=1024)

    # ── Initial wavepacket ────────────────────────────────────────────
    st.sidebar.header("🌊 Initial wavepacket")
    init_mode = st.sidebar.radio(
        "Initialization",
        ["Gaussian wavepacket", "Energy eigenstate", "Superposition"],
    )

    # defaults for inactive modes
    x0 = sigma = k0 = 0.0
    eigen_n = 0
    n1, n2 = 0, 1
    ratio = 0.5

    if init_mode == "Gaussian wavepacket":
        c1, c2 = st.sidebar.columns(2)
        x0 = c1.slider("x₀ (center)", float(x_min), float(x_max), step=0.1, key="x0")
        sigma = c2.slider("σ (width)", 0.1, 5.0, step=0.1, key="sigma")
        k0 = st.sidebar.slider("k₀ (momentum)", -10.0, 10.0, step=0.1, key="k0")
    elif init_mode == "Energy eigenstate":
        eigen_n = st.sidebar.slider("Quantum number n", 0, 15, 0)
    elif init_mode == "Superposition":
        st.sidebar.markdown("Superposition of two eigenstates:")
        c1, c2 = st.sidebar.columns(2)
        n1 = c1.number_input("n₁", 0, 15, 0)
        n2 = c2.number_input("n₂", 0, 15, 1)
        ratio = st.sidebar.slider("Mixing ratio (n₁ weight)", 0.0, 1.0, 0.5, 0.05)

    # ── Time evolution ────────────────────────────────────────────────
    st.sidebar.header("⏱️ Time evolution")
    dt = st.sidebar.select_slider("Time step dt",
                                   options=[0.001, 0.005, 0.01, 0.02, 0.05], value=0.01)
    steps_per_frame = st.sidebar.slider("Steps per frame", 1, 20, 5,
                                         help="More steps = faster evolution, fewer frames")
    max_time = st.sidebar.slider("Max time", 1.0, 100.0, 30.0, 1.0)
    speed = st.sidebar.slider("Animation speed (ms/frame)", 10, 200, 50, 10)

    # Derive frame count (cap at 500 for browser performance)
    time_per_frame = steps_per_frame * dt
    n_frames = max(1, int(max_time / time_per_frame))
    if n_frames > 500:
        steps_per_frame = max(1, int(max_time / (500 * dt)))
        time_per_frame = steps_per_frame * dt
        n_frames = max(1, int(max_time / time_per_frame))
    st.sidebar.caption(
        f"{n_frames} frames · Δt/frame = {time_per_frame:.3f} · "
        f"total = {n_frames * time_per_frame:.1f}"
    )

    # ── Display options ───────────────────────────────────────────────
    st.sidebar.header("📊 Display")
    show_components = st.sidebar.checkbox("Show Re(ψ) and Im(ψ)", value=False)
    show_potential = st.sidebar.checkbox("Show potential V(x)", value=True)
    show_eigenstates = st.sidebar.checkbox("Show energy eigenstates", value=False)
    n_eigenstates = 4
    if show_eigenstates:
        n_eigenstates = st.sidebar.slider("Number of eigenstates", 1, 10, 4)
    y_max_prob = st.sidebar.slider("|ψ|² y-axis max", 0.1, 2.0, 0.8, 0.05)

    # ── Footer ────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**QuantApp** — Quantum Mechanics Simulator  \n"
        "Built with Streamlit + NumPy + Plotly  \n"
        "Split-Operator FFT method"
    )

    return dict(
        potential_name=potential_name,
        x_min=x_min, x_max=x_max, N=N,
        init_mode=init_mode,
        x0=x0, sigma=sigma, k0=k0,
        eigen_n=eigen_n, n1=n1, n2=n2, ratio=ratio,
        dt=dt, steps_per_frame=steps_per_frame,
        n_frames=n_frames, speed=speed,
        time_per_frame=time_per_frame,
        show_components=show_components,
        show_potential=show_potential,
        show_eigenstates=show_eigenstates,
        n_eigenstates=n_eigenstates,
        y_max_prob=y_max_prob,
    )
