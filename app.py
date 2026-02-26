"""
QuantApp — Interactive Quantum Mechanics Simulator
===================================================
A Streamlit app for demonstrating solutions to the Schrödinger equation.

Run with:  streamlit run app.py
"""

import streamlit as st

from quantapp.simulation import compute_eigenstates_cached
from quantapp.plots import scale_potential_for_display
from quantapp.ui.sidebar import render_sidebar
from quantapp.ui import tab_animation, tab_eigenstates, tab_theory


# ═══════════════════════════════════════════════════════════════════════
#  Page config & header
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Quantum Mechanics Simulator", page_icon="⚛️", layout="wide")
st.title("⚛️ Quantum Mechanics Simulator")
st.markdown("""
Explore solutions to the **Schrödinger equation** interactively.  
Watch wavepackets evolve in real time under different potentials using the
**Split-Operator FFT** method — a unitary, stable numerical scheme.
""")


# ═══════════════════════════════════════════════════════════════════════
#  Sidebar & shared data
# ═══════════════════════════════════════════════════════════════════════

cfg = render_sidebar()

x_grid, V_grid, eigenenergies, eigenstates = compute_eigenstates_cached(
    cfg["potential_name"], cfg["x_min"], cfg["x_max"], cfg["N"], n_eigen=25,
    pot_kwargs=cfg["pot_kwargs"],
)

stride = max(1, cfg["N"] // 300)
x_anim = x_grid[::stride]
V_scaled = scale_potential_for_display(V_grid, cfg["y_max_prob"], stride)


# ═══════════════════════════════════════════════════════════════════════
#  Tabs
# ═══════════════════════════════════════════════════════════════════════

tab_anim, tab_eigen, tab_info = st.tabs([
    "🎬 Time Evolution", "📐 Energy Eigenstates", "📖 Theory",
])

with tab_anim:
    tab_animation.render(cfg, x_anim, V_scaled, eigenstates, stride)

with tab_eigen:
    tab_eigenstates.render(cfg, x_grid, eigenenergies, eigenstates)

with tab_info:
    tab_theory.render()
