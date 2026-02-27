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
    with st.sidebar.expander("⚙️ Setup", expanded=True):
        potential_name = st.selectbox(
            "Potential V(x)", list(POTENTIALS.keys()), index=3,
        )
        pot_info = POTENTIALS[potential_name]
        st.caption(pot_info["description"])

        # ── Per-potential parameter controls ───────────────────────────
        # pot_kwargs is a hashable tuple of (name, value) pairs used as a
        # cache key and passed to _resolve_potential_func.
        params = pot_info.get("params", [])
        param_values = {}
        if params:
            st.markdown("**Potential parameters**")
            for p in params:
                # Conditional visibility (e.g. seed only when jitter > 0)
                show_if = p.get("show_if")
                if show_if:
                    ref_val = param_values.get(show_if["param"])
                    if ref_val is None or not (ref_val > show_if.get("gt", float("-inf"))):
                        continue

                pkey = f"pot_{potential_name}_{p['name']}"
                help_text = p.get("help")
                if isinstance(p["default"], int) and isinstance(p["min"], int):
                    val = st.slider(p["label"], int(p["min"]), int(p["max"]),
                                    int(p["default"]), int(p["step"]),
                                    help=help_text, key=pkey)
                else:
                    val = st.slider(p["label"], float(p["min"]), float(p["max"]),
                                    float(p["default"]), float(p["step"]),
                                    help=help_text, key=pkey)
                param_values[p["name"]] = val
        pot_kwargs = tuple(sorted(param_values.items()))

        # Reset wavepacket defaults when the potential changes
        if st.session_state.get("last_potential") != potential_name:
            st.session_state.last_potential = potential_name
            st.session_state.x0 = pot_info["default_x0"]
            st.session_state.sigma = pot_info["default_sigma"]
            st.session_state.k0 = pot_info["default_k0"]

        # ── Spatial domain ─────────────────────────────────────────────
        default_xmin, default_xmax = pot_info["x_range"]
        st.markdown("**Spatial domain**")
        c1, c2 = st.columns(2)
        x_min = c1.number_input("x_min", min_value=-500.0, max_value=500.0,
                                value=float(default_xmin), step=1.0,
                                key=f"xrange_{potential_name}_min")
        x_max = c2.number_input("x_max", min_value=-500.0, max_value=500.0,
                                value=float(default_xmax), step=1.0,
                                key=f"xrange_{potential_name}_max")
        if x_min >= x_max:
            st.error("x_min must be less than x_max")
            x_min, x_max = float(default_xmin), float(default_xmax)

        N = st.select_slider("Grid points", options=[256, 512, 1024, 2048], value=1024)

        # For multi-well: round N up to a multiple of n_wells so the grid
        # samples each well identically (avoids tiny asymmetry in eigenstates).
        if potential_name == "Multi-well (band structure)":
            mw_n_wells = param_values.get("n_wells", 1)
            if mw_n_wells > 1 and N % mw_n_wells != 0:
                N = N + (mw_n_wells - N % mw_n_wells)

    # ── Initial wavepacket ────────────────────────────────────────────
    with st.sidebar.expander("🌊 Initial wavepacket", expanded=False):
        init_mode = st.radio(
            "Initialization",
            ["Gaussian wavepacket", "Energy eigenstate", "Superposition"],
        )

        # defaults for inactive modes
        x0 = sigma = k0 = 0.0
        eigen_n = 0
        n1, n2 = 0, 1
        ratio = 0.5

        if init_mode == "Gaussian wavepacket":
            c1, c2 = st.columns(2)
            x0 = c1.slider("x₀ (center)", float(x_min), float(x_max), step=0.1, key="x0")
            sigma = c2.slider("σ (width)", 0.1, 5.0, step=0.1, key="sigma")
            k0 = st.slider("k₀ (momentum)", -10.0, 10.0, step=0.1, key="k0")
        elif init_mode == "Energy eigenstate":
            eigen_n = st.slider("Quantum number n", 0, 15, 0)
        elif init_mode == "Superposition":
            st.markdown("Superposition of two eigenstates:")
            c1, c2 = st.columns(2)
            n1 = c1.number_input("n₁", 0, 15, 0)
            n2 = c2.number_input("n₂", 0, 15, 1)
            ratio = st.slider("Mixing ratio (n₁ weight)", 0.0, 1.0, 0.5, 0.05)

    # ── Time evolution ────────────────────────────────────────────────
    with st.sidebar.expander("⏱️ Time evolution", expanded=False):
        dt = st.select_slider("Time step dt",
                               options=[0.001, 0.005, 0.01, 0.02, 0.05], value=0.01)
        steps_per_frame = st.slider("Steps per frame", 1, 20, 5,
                                     help="More steps = faster evolution, fewer frames")
        max_time = st.slider("Max time", 1.0, 100.0, 30.0, 1.0)
        speed = st.slider("Animation speed (ms/frame)", 10, 200, 50, 10)

        # Derive frame count (cap at 500 for browser performance)
        time_per_frame = steps_per_frame * dt
        n_frames = max(1, int(max_time / time_per_frame))
        if n_frames > 500:
            steps_per_frame = max(1, int(max_time / (500 * dt)))
            time_per_frame = steps_per_frame * dt
            n_frames = max(1, int(max_time / time_per_frame))
        st.caption(
            f"{n_frames} frames · Δt/frame = {time_per_frame:.3f} · "
            f"total = {n_frames * time_per_frame:.1f}"
        )

    # ── Display options ───────────────────────────────────────────────
    with st.sidebar.expander("📊 Display", expanded=False):
        show_components = st.checkbox("Show Re(ψ) and Im(ψ)", value=False)
        show_potential = st.checkbox("Show potential V(x)", value=True)
        show_eigenstates = st.checkbox("Show energy eigenstates", value=False)
        n_eigenstates = 4
        if show_eigenstates:
            n_eigenstates = st.slider("Number of eigenstates", 1, 10, 4)
        y_max_prob = st.slider("|ψ|² y-axis max", 0.1, 2.0, 0.8, 0.05)

    # ── Footer ────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**QuantApp** — Quantum Mechanics Simulator  \n"
        "Built with Streamlit + NumPy + Plotly  \n"
        "Split-Operator FFT method"
    )

    return dict(
        potential_name=potential_name,
        pot_kwargs=pot_kwargs,
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
