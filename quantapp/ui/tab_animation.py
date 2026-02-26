"""
Tab 1 — Time Evolution Animation
=================================
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from quantapp.simulation import (
    compute_eigenstates_cached,
    create_quantum_system,
    precompute_animation,
)
from quantapp.plots import (
    build_wavefunction_traces,
    build_frame_data,
    y_axis_config,
    frame_title,
    scale_potential_for_display,
)


def render(cfg, x_anim, V_scaled, eigenstates, stride):
    """Render the Time Evolution tab.

    Parameters
    ----------
    cfg : dict   — sidebar config from render_sidebar()
    x_anim       — downsampled x grid
    V_scaled     — scaled potential for overlay
    eigenstates  — cached eigenstates array
    stride       — downsample stride
    """
    col_ctrl, col_plot = st.columns([1, 3])

    # Stable cache keys — use defaults for inactive modes
    _x0 = cfg["x0"] if cfg["init_mode"] == "Gaussian wavepacket" else 0.0
    _sigma = cfg["sigma"] if cfg["init_mode"] == "Gaussian wavepacket" else 1.0
    _k0 = cfg["k0"] if cfg["init_mode"] == "Gaussian wavepacket" else 0.0
    _eigen_n = cfg["eigen_n"] if cfg["init_mode"] == "Energy eigenstate" else 0
    _n1 = cfg["n1"] if cfg["init_mode"] == "Superposition" else 0
    _n2 = cfg["n2"] if cfg["init_mode"] == "Superposition" else 1
    _ratio = cfg["ratio"] if cfg["init_mode"] == "Superposition" else 0.5

    with col_ctrl:
        compute_btn = st.button("▶️ Compute & Play",
                                use_container_width=True, type="primary")
        st.markdown("---")
        info_box = st.empty()

    with col_plot:
        plot_placeholder = st.empty()

        if compute_btn:
            _render_animation(cfg, plot_placeholder, info_box,
                              x_anim, V_scaled, stride,
                              _x0, _sigma, _k0, _eigen_n, _n1, _n2, _ratio)
        else:
            _render_static(cfg, plot_placeholder, info_box,
                           x_anim, V_scaled, eigenstates, stride,
                           _x0, _sigma, _k0, _eigen_n, _n1, _n2, _ratio)


# ── Private helpers ────────────────────────────────────────────────────

def _render_animation(cfg, plot_placeholder, info_box,
                      x_anim, V_scaled, stride,
                      x0, sigma, k0, eigen_n, n1, n2, ratio):
    """Pre-compute and display the animated figure."""
    precompute_animation.clear()
    total_t = cfg["n_frames"] * cfg["steps_per_frame"] * cfg["dt"]

    with st.spinner(f"Pre-computing {cfg['n_frames']} frames (t = 0 → {total_t:.1f}) …"):
        (probs, re_parts, im_parts, times, energies,
         exp_xs, delta_xs, norms_arr) = precompute_animation(
            cfg["potential_name"], cfg["init_mode"],
            x0, sigma, k0, eigen_n, n1, n2, ratio,
            cfg["dt"], cfg["steps_per_frame"], cfg["n_frames"],
            cfg["N"], cfg["x_min"], cfg["x_max"], stride,
            pot_kwargs=cfg["pot_kwargs"],
        )

    # Build initial traces
    traces = build_wavefunction_traces(
        x_anim, probs[0], re_parts[0], im_parts[0],
        V_scaled, cfg["show_components"], cfg["show_potential"],
    )

    # Build animation frames
    plotly_frames, slider_steps = [], []
    for i in range(len(times)):
        fd = build_frame_data(probs[i], re_parts[i], im_parts[i],
                              V_scaled, cfg["show_components"], cfg["show_potential"])
        plotly_frames.append(go.Frame(
            data=fd, name=str(i),
            layout=go.Layout(title=frame_title(
                times[i], energies[i], exp_xs[i], delta_xs[i])),
        ))
        slider_steps.append(dict(
            args=[[str(i)], dict(frame=dict(duration=0, redraw=False),
                                 mode="immediate",
                                 transition=dict(duration=0))],
            label=f"{times[i]:.2f}",
            method="animate",
        ))

    # Assemble figure
    y_range, y_title = y_axis_config(cfg["y_max_prob"], cfg["show_components"])
    fig = go.Figure(data=traces, frames=plotly_frames)
    fig.update_layout(
        template="plotly_white", height=500,
        xaxis=dict(title="x", range=[cfg["x_min"], cfg["x_max"]]),
        yaxis=dict(range=y_range, title=y_title),
        title=frame_title(times[0], energies[0], exp_xs[0], delta_xs[0]),
        margin=dict(l=50, r=20, t=80, b=120),
        legend=dict(orientation="h", y=1.12),
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=-0.15, x=0.0, xanchor="left",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(
                         frame=dict(duration=cfg["speed"], redraw=False),
                         fromcurrent=True,
                         transition=dict(duration=0),
                         mode="immediate")]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(
                         frame=dict(duration=0, redraw=False),
                         mode="immediate",
                         transition=dict(duration=0))]),
                dict(label="⏮ Reset", method="animate",
                     args=[["0"], dict(
                         frame=dict(duration=0, redraw=True),
                         mode="immediate",
                         transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(
            active=0,
            yanchor="top", y=-0.22, xanchor="left", x=0.0, len=1.0,
            currentvalue=dict(prefix="t = ", visible=True, xanchor="center"),
            transition=dict(duration=0),
            steps=slider_steps,
        )],
    )
    plot_placeholder.plotly_chart(fig, use_container_width=True, key="animation")

    info_box.markdown(f"""
    **Frames**: {len(times)}  
    **Time range**: 0 → {times[-1]:.2f}  
    **⟨E⟩ drift**: {energies[0]:.4f} → {energies[-1]:.4f}  
    **Norm drift**: {norms_arr[0]:.6f} → {norms_arr[-1]:.6f}  
    
    Use **▶ Play** / **⏸ Pause** below  
    the plot, or drag the **slider**.
    """)


def _render_static(cfg, plot_placeholder, info_box,
                   x_anim, V_scaled, eigenstates, stride,
                   x0, sigma, k0, eigen_n, n1, n2, ratio):
    """Show the initial (static) wavefunction."""
    qs0 = create_quantum_system(
        cfg["potential_name"], cfg["x_min"], cfg["x_max"], cfg["N"],
        cfg["init_mode"], eigenstates,
        pot_kwargs=cfg["pot_kwargs"],
        x0=x0, sigma=sigma, k0=k0,
        eigen_n=eigen_n, n1=n1, n2=n2, ratio=ratio,
    )
    E0 = qs0.expectation_energy()
    prob0 = np.abs(qs0.psi[::stride]) ** 2

    traces0 = build_wavefunction_traces(
        x_anim, prob0,
        np.real(qs0.psi[::stride]), np.imag(qs0.psi[::stride]),
        V_scaled, cfg["show_components"], cfg["show_potential"],
    )

    y_range, y_title = y_axis_config(cfg["y_max_prob"], cfg["show_components"])
    fig0 = go.Figure(data=traces0)
    fig0.update_layout(
        template="plotly_white", height=500,
        xaxis=dict(title="x", range=[cfg["x_min"], cfg["x_max"]]),
        yaxis=dict(range=y_range, title=y_title),
        title=f"|ψ(x)|²   —   t = 0.000    ⟨E⟩ = {E0:.3f}",
        margin=dict(l=50, r=20, t=50, b=50),
        legend=dict(orientation="h", y=1.12),
    )
    plot_placeholder.plotly_chart(fig0, use_container_width=True, key="initial")

    info_box.markdown(f"""
    **t** = 0.000  
    **⟨E⟩** = {E0:.4f}  
    **⟨x⟩** = {qs0.expectation_x():.4f}  
    **Δx** = {qs0.uncertainty_x():.4f}  
    **Norm** = {qs0.norm():.6f}  
    
    Press **▶️ Compute & Play** to  
    start the animation.
    """)
