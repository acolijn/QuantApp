"""
Tab 3 — Theory
===============
Static informational content about the Split-Operator FFT method.
"""

import streamlit as st


def render():
    """Render the Theory tab."""
    st.subheader("The Schrödinger Equation")
    st.latex(
        r"i\hbar \frac{\partial}{\partial t}\Psi(x,t) "
        r"= \hat{H}\Psi(x,t) "
        r"= \left[-\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2} + V(x)\right]\Psi(x,t)"
    )

    st.markdown("### Split-Operator FFT Method\n\nThe time evolution operator is:")
    st.latex(r"\hat{U}(\Delta t) = e^{-i\hat{H}\Delta t/\hbar}")

    st.markdown("Using the **Trotter-Suzuki decomposition** (2nd order):")
    st.latex(
        r"\hat{U}(\Delta t) \approx "
        r"e^{-i\hat{V}\Delta t/2\hbar} \cdot "
        r"e^{-i\hat{T}\Delta t/\hbar} \cdot "
        r"e^{-i\hat{V}\Delta t/2\hbar} "
        r"+ \mathcal{O}(\Delta t^3)"
    )

    st.markdown(r"""
**The key insight:**
- The potential operator $e^{-iV\Delta t/2\hbar}$ is **diagonal in position space** → simple multiplication
- The kinetic operator $e^{-iT\Delta t/\hbar}$ is **diagonal in momentum space** → multiply in Fourier space
- We use FFT to switch between representations efficiently

**Algorithm per time step:**
1. Multiply $\psi(x)$ by $e^{-iV(x)\Delta t / 2\hbar}$ (half-step potential)
2. FFT to momentum space → $\tilde{\psi}(k)$
3. Multiply by $e^{-i\hbar k^2 \Delta t / 2m}$ (full-step kinetic)
4. Inverse FFT back to position space
5. Multiply by $e^{-iV(x)\Delta t / 2\hbar}$ (half-step potential)

### Properties
- **Unitary**: Preserves normalization exactly (unlike finite-difference schemes)
- **Stable**: No numerical instabilities
- **Efficient**: $\mathcal{O}(N \log N)$ per time step via FFT
- **Accurate**: Global error $\mathcal{O}(\Delta t^2)$

---

### Physics demonstrations

| Potential | Key concept |
|-----------|-------------|
| **Free particle** | Wavepacket spreading, group vs phase velocity |
| **Infinite well** | Quantized energy levels, standing waves |
| **Finite well** | Tunneling into classically forbidden regions |
| **Harmonic oscillator** | Equally spaced levels, coherent states |
| **Double well** | Quantum tunneling between wells |
| **Barrier** | Transmission and reflection, tunneling probability |
| **Morse potential** | Anharmonic molecular vibrations |
| **Step potential** | Partial reflection at energy boundaries |

---
*We use natural units where $\hbar = 1$ and $m = 1$.*
""")
