"""
Tab 3 — Theory
===============
Static informational content about the numerical methods and physics.
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

### Hard-wall potentials: DST propagator

The standard FFT assumes **periodic boundary conditions**, which causes amplitude
to leak through infinite potential walls. For potentials with hard walls (e.g. the
infinite square well), we use a **Discrete Sine Transform (DST-I)** for the kinetic
step instead. The DST basis functions $\sin(n\pi x/L)$ are exactly zero at the
walls, naturally enforcing Dirichlet boundary conditions $\psi = 0$.

This eliminates leakage entirely and conserves both **norm** and **energy** to
machine precision (~$10^{-13}$) — no artificial renormalization needed.

The DST-based energy expectation value uses the Parseval relation in the
sine basis:

$$\langle T \rangle = \frac{\Delta x}{2(N_\mathrm{int}+1)} \sum_k T_k |\tilde\psi_k|^2$$

where $T_k = \hbar^2 (k\pi/L)^2 / 2m$ are the kinetic eigenvalues and
$\tilde\psi_k$ are the DST coefficients.

### Boundary handling for scattering potentials

For scattering potentials (barrier, step, free particle), the wavepacket can
travel toward the edges of the simulation grid. With the FFT's periodic
boundaries, the wavepacket would "wrap around" to the other side, creating
artificial high-$k$ modes whose energy grows as $k_\text{max}^2 \propto 1/\Delta x^2$.

We handle this by using **wide simulation domains** (e.g. $x \in [-50, 50]$ for
the barrier and step potentials) so that the wavepacket does not reach the
grid boundaries during typical simulation times. This avoids the need for
artificial boundary treatments that can interfere with the physical
reflection and transmission amplitudes.

As a safety measure, the expectation energy is always computed as a
**per-particle value** $\langle E \rangle / \langle\psi|\psi\rangle$, which
stays accurate even if small amounts of probability leak through the
periodic boundaries.

> **Note:** The solver also includes an optional complex absorbing potential
> (CAP) infrastructure that can damp outgoing waves near the grid edges.
> This is disabled by default to preserve the physical norm, but can be
> enabled for specialised use cases where very long simulation times are
> needed.

### Smooth potential edges

Discontinuous potentials (e.g. square barrier, step) cause the Trotter splitting
error to diverge because the commutator $[T, V]$ involves $\nabla V$, which is
infinite at a sharp edge. We replace sharp discontinuities with smooth $\tanh$
profiles:

$$V_\text{step}(x) = \tfrac{1}{2}\bigl(1 + \tanh[s\,(x - x_\text{edge})]\bigr)$$

with steepness $s = 5$, giving a transition width of $\sim$0.4 length units.
This keeps the physics essentially unchanged while reducing energy drift to
$< 10^{-4}$ relative.

The `smooth_rect` helper combines two such steps to form the finite well and
barrier shapes.

---

### Physics demonstrations

| Potential | Key concept |
|-----------|-------------|
| **Free particle** | Wavepacket spreading, group vs phase velocity |
| **Infinite well** | Quantized energy levels, standing waves (DST propagator) |
| **Finite well** | Tunneling into classically forbidden regions |
| **Harmonic oscillator** | Equally spaced levels, coherent states |
| **Double well** | Quantum tunneling between wells |
| **Barrier** | Transmission and reflection, tunneling probability |
| **Morse potential** | Anharmonic molecular vibrations |
| **Step potential** | Partial reflection at energy boundaries |

---
*We use natural units where $\hbar = 1$ and $m = 1$.*
""")
