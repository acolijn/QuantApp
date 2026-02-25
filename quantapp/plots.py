"""
Plotting helpers
================
All Plotly figure-building logic, decoupled from the Streamlit UI.
"""

import numpy as np
import plotly.graph_objects as go


def scale_potential_for_display(V, y_max, stride):
    """Scale the potential V(x) for overlay on the wavefunction plot.

    Returns the scaled array on the downsampled grid.
    Infinite walls (V > 1e4) are rendered as tall vertical lines.
    """
    V_ds = V[::stride].copy()
    is_wall = V_ds > 1e4

    V_finite = V_ds.copy()
    V_finite[is_wall] = 0
    V_finite = np.clip(V_finite, np.min(V_finite), 500)

    V_range = np.max(V_finite) - np.min(V_finite)
    if V_range > 0:
        V_scaled = (V_finite - np.min(V_finite)) / V_range * y_max * 0.5
    elif np.max(np.abs(V_finite)) > 0:
        V_scaled = V_finite / np.max(np.abs(V_finite)) * y_max * 0.5
    else:
        V_scaled = V_finite.copy()

    V_scaled[is_wall] = y_max * 0.95
    return V_scaled


def build_wavefunction_traces(x, prob, re_psi, im_psi, V_scaled,
                              show_components, show_potential):
    """Build the initial Plotly traces for the wavefunction plot."""
    traces = []

    if show_components:
        env = np.sqrt(prob)
        traces += [
            go.Scatter(x=x, y=env, mode="lines", name="|ψ|",
                       line=dict(color="#2196F3", width=2, dash="dot")),
            go.Scatter(x=x, y=-env, mode="lines", name="-|ψ|",
                       line=dict(color="#2196F3", width=2, dash="dot"),
                       showlegend=False),
            go.Scatter(x=x, y=re_psi, mode="lines", name="Re(ψ)",
                       line=dict(color="#4CAF50", width=1.5)),
            go.Scatter(x=x, y=im_psi, mode="lines", name="Im(ψ)",
                       line=dict(color="#FF9800", width=1.5)),
        ]
    else:
        traces.append(
            go.Scatter(x=x, y=prob, mode="lines", name="|ψ(x)|²",
                       line=dict(color="#2196F3", width=2.5),
                       fill="tozeroy", fillcolor="rgba(33,150,243,0.15)")
        )

    if show_potential:
        traces.append(
            go.Scatter(x=x, y=V_scaled, mode="lines", name="V(x) (scaled)",
                       line=dict(color="rgba(255,87,34,0.5)", width=1.5, dash="dash"))
        )

    return traces


def build_frame_data(prob, re_psi, im_psi, V_scaled,
                     show_components, show_potential):
    """Build the Scatter updates for a single animation frame."""
    data = []

    if show_components:
        env = np.sqrt(prob)
        data += [
            go.Scatter(y=env),
            go.Scatter(y=-env),
            go.Scatter(y=re_psi),
            go.Scatter(y=im_psi),
        ]
    else:
        data.append(go.Scatter(y=prob))

    if show_potential:
        data.append(go.Scatter(y=V_scaled))

    return data


def y_axis_config(y_max_prob, show_components):
    """Return (range, title) for the y-axis."""
    if show_components:
        amp = np.sqrt(y_max_prob)
        return [-amp, max(amp, y_max_prob)], "ψ(x)"
    return [0, y_max_prob], "|ψ(x)|²"


def frame_title(t, E, x_mean, dx):
    """One-line title string for the plot."""
    return f"|ψ(x)|²   —   t = {t:.3f}    ⟨E⟩ = {E:.3f}    ⟨x⟩ = {x_mean:.3f}    Δx = {dx:.3f}"


# ── Eigenstate colours ─────────────────────────────────────────────────

EIGENSTATE_COLORS = [
    "#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0",
    "#00BCD4", "#FF5722", "#607D8B", "#795548", "#CDDC39",
    "#3F51B5", "#009688",
]
