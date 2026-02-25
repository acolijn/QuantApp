"""QuantApp — Interactive Quantum Mechanics Simulator.

Solves the 1-D time-dependent Schrödinger equation using the
Split-Operator FFT method with Trotter-Suzuki decomposition.

Subpackages
-----------
ui : Streamlit interface components (sidebar, tabs).

Modules
-------
solver      : QuantumSystem class — core physics engine.
potentials  : Predefined potential functions and registry.
simulation  : Caching, wavefunction initialisation, animation precomputation.
plots       : Plotly figure builders.
"""
