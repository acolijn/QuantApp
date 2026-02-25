"""
Tab 2 — Energy Eigenstates
===========================
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quantapp.plots import EIGENSTATE_COLORS


def render(cfg, x_grid, eigenenergies, eigenstates):
    """Render the Energy Eigenstates tab.

    Parameters
    ----------
    cfg           — sidebar config dict
    x_grid        — full x grid
    eigenenergies — array of eigenvalues
    eigenstates   — matrix of eigenvectors
    """
    st.subheader(f"Energy eigenstates — {cfg['potential_name']}")
    n_show = st.slider("Number of eigenstates to display", 1, 12, 5, key="n_eigen_show")

    if eigenstates.size == 0 or eigenstates.shape[1] < n_show:
        st.warning("Could not compute eigenstates for this potential. "
                   "Try adjusting the grid range.")
        return

    # ── Probability densities ──────────────────────────────────────────
    fig_eigen = make_subplots(
        rows=n_show, cols=1, shared_xaxes=True, vertical_spacing=0.02,
    )
    for i in range(n_show):
        col = EIGENSTATE_COLORS[i % len(EIGENSTATE_COLORS)]
        r, g, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
        fig_eigen.add_trace(go.Scatter(
            x=x_grid, y=eigenstates[:, i] ** 2, mode="lines",
            name=f"n={i}  E={eigenenergies[i]:.3f}",
            line=dict(color=col, width=2),
            fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.1)",
        ), row=i + 1, col=1)
        fig_eigen.update_yaxes(title_text=f"n={i}", row=i + 1, col=1,
                               showticklabels=False)

    fig_eigen.update_layout(
        height=150 * n_show, template="plotly_white",
        title="Probability densities |ψₙ(x)|²",
        showlegend=True, legend=dict(orientation="h", y=1.05),
        xaxis=dict(title="x"),
    )
    st.plotly_chart(fig_eigen, use_container_width=True)

    # ── Energy level diagram ───────────────────────────────────────────
    st.subheader("Energy levels")
    fig_levels = go.Figure()
    for i in range(min(n_show, len(eigenenergies))):
        col = EIGENSTATE_COLORS[i % len(EIGENSTATE_COLORS)]
        fig_levels.add_trace(go.Scatter(
            x=[-1, 1], y=[eigenenergies[i], eigenenergies[i]],
            mode="lines+text",
            line=dict(color=col, width=3),
            text=[f"n={i}", f"E={eigenenergies[i]:.3f}"],
            textposition=["middle left", "middle right"],
            name=f"n={i}", showlegend=False,
        ))
    fig_levels.update_layout(
        height=400, template="plotly_white",
        xaxis=dict(showticklabels=False, range=[-2, 3], title=""),
        yaxis=dict(title="Energy"),
        title="Energy spectrum",
    )
    st.plotly_chart(fig_levels, use_container_width=True)

    # ── Energy differences ─────────────────────────────────────────────
    if len(eigenenergies) > 1:
        st.subheader("Energy differences")
        diffs = np.diff(eigenenergies[:n_show])
        st.json({f"E_{i+1} - E_{i}": f"{d:.4f}" for i, d in enumerate(diffs)})
