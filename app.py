"""
QuantApp — Interactive Quantum Mechanics Simulator
===================================================
A Streamlit app for demonstrating solutions to the Schrödinger equation.

Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quantum_solver import QuantumSystem, POTENTIALS

# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quantum Mechanics Simulator",
    page_icon="⚛️",
    layout="wide",
)

st.title("⚛️ Quantum Mechanics Simulator")
st.markdown("""
Explore solutions to the **Schrödinger equation** interactively.  
Watch wavepackets evolve in real time under different potentials using the
**Split-Operator FFT** method — a unitary, stable numerical scheme.
""")

# ── Sidebar controls ───────────────────────────────────────────────────
st.sidebar.header("⚙️ Setup")

# Potential selection
potential_name = st.sidebar.selectbox(
    "Potential V(x)",
    list(POTENTIALS.keys()),
    index=3,  # Default: harmonic oscillator
)
pot_info = POTENTIALS[potential_name]
st.sidebar.caption(pot_info["description"])

# Reset wavepacket defaults when the selected potential changes
if "last_potential" not in st.session_state or st.session_state.last_potential != potential_name:
    st.session_state.last_potential = potential_name
    st.session_state.x0 = pot_info["default_x0"]
    st.session_state.sigma = pot_info["default_sigma"]
    st.session_state.k0 = pot_info["default_k0"]

# Grid parameters
x_min, x_max = pot_info["x_range"]
N = st.sidebar.select_slider("Grid points", options=[256, 512, 1024, 2048], value=1024)

st.sidebar.header("🌊 Initial wavepacket")

init_mode = st.sidebar.radio(
    "Initialization",
    ["Gaussian wavepacket", "Energy eigenstate", "Superposition"],
    index=0,
)

if init_mode == "Gaussian wavepacket":
    col1, col2 = st.sidebar.columns(2)
    x0 = col1.slider("x₀ (center)", float(x_min), float(x_max),
                      step=0.1, key="x0")
    sigma = col2.slider("σ (width)", 0.1, 5.0,
                        step=0.1, key="sigma")
    k0 = st.sidebar.slider("k₀ (momentum)", -10.0, 10.0,
                            step=0.1, key="k0")
elif init_mode == "Energy eigenstate":
    eigen_n = st.sidebar.slider("Quantum number n", 0, 15, 0)
elif init_mode == "Superposition":
    st.sidebar.markdown("Superposition of two eigenstates:")
    col1, col2 = st.sidebar.columns(2)
    n1 = col1.number_input("n₁", 0, 15, 0)
    n2 = col2.number_input("n₂", 0, 15, 1)
    ratio = st.sidebar.slider("Mixing ratio (n₁ weight)", 0.0, 1.0, 0.5, 0.05)

st.sidebar.header("⏱️ Time evolution")
dt = st.sidebar.select_slider("Time step dt", options=[0.001, 0.005, 0.01, 0.02, 0.05], value=0.01)
steps_per_frame = st.sidebar.slider("Steps per frame", 1, 20, 5,
                                     help="More steps = faster evolution, fewer frames")
max_time = st.sidebar.slider("Max time", 1.0, 100.0, 30.0, 1.0)
speed = st.sidebar.slider("Animation speed (ms/frame)", 10, 200, 50, 10)

st.sidebar.header("📊 Display")
show_components = st.sidebar.checkbox("Show Re(ψ) and Im(ψ)", value=False)
show_potential = st.sidebar.checkbox("Show potential V(x)", value=True)
show_eigenstates = st.sidebar.checkbox("Show energy eigenstates", value=False)
if show_eigenstates:
    n_eigenstates = st.sidebar.slider("Number of eigenstates", 1, 10, 4)
y_max_prob = st.sidebar.slider("|ψ|² y-axis max", 0.1, 2.0, 0.8, 0.05)

# ── Initialize quantum system ──────────────────────────────────────────

@st.cache_data
def create_system_and_eigenstates(potential_name, x_min, x_max, N, n_eigen=10):
    """Cache the system setup and eigenstate computation."""
    qs = QuantumSystem(x_min=x_min, x_max=x_max, N=N)
    qs.set_potential(POTENTIALS[potential_name]["func"])
    try:
        energies, states = qs.compute_eigenstates(n_states=n_eigen)
    except Exception:
        energies, states = np.array([]), np.array([])
    return qs.x.copy(), qs.V.copy(), energies, states


x_grid, V_grid, eigenenergies, eigenstates = create_system_and_eigenstates(
    potential_name, x_min, x_max, N, n_eigen=16
)

def init_system():
    """Create a fresh quantum system with the chosen initial state."""
    qs = QuantumSystem(x_min=x_min, x_max=x_max, N=N)
    qs.set_potential(POTENTIALS[potential_name]["func"])

    if init_mode == "Gaussian wavepacket":
        qs.set_gaussian_wavepacket(x0=x0, sigma=sigma, k0=k0)
    elif init_mode == "Energy eigenstate":
        if eigenstates.size > 0 and eigen_n < eigenstates.shape[1]:
            qs.psi = eigenstates[:, eigen_n].astype(complex).copy()
        else:
            qs.set_gaussian_wavepacket(x0=0, sigma=1.0, k0=0)
    elif init_mode == "Superposition":
        if eigenstates.size > 0 and max(n1, n2) < eigenstates.shape[1]:
            c1 = np.sqrt(ratio)
            c2 = np.sqrt(1 - ratio)
            qs.psi = (c1 * eigenstates[:, n1] + c2 * eigenstates[:, n2]).astype(complex)
            qs.psi /= np.sqrt(np.sum(np.abs(qs.psi) ** 2) * qs.dx)
        else:
            qs.set_gaussian_wavepacket(x0=0, sigma=1.0, k0=0)
    return qs


# ── Tabs ────────────────────────────────────────────────────────────────
tab_animate, tab_eigenstates, tab_info = st.tabs([
    "🎬 Time Evolution", "📐 Energy Eigenstates", "📖 Theory"
])

# ── TAB 1: Time Evolution Animation ────────────────────────────────────
with tab_animate:
    col_ctrl, col_plot = st.columns([1, 3])

    with col_ctrl:
        compute_btn = st.button("▶️ Compute & Play", use_container_width=True, type="primary")
        n_frames = st.slider("Number of frames", 50, 600, 200, 10,
                              help="More frames = longer computation but smoother result")
        st.markdown("---")
        info_box = st.empty()

    # Downsample the x-grid for the animation to keep data payload small
    stride = max(1, N // 300)
    x_anim = x_grid[::stride]

    # Scale potential once for display
    V_anim = V_grid[::stride].copy()
    # For display: detect "infinite" walls (V > 1e4) and render them as tall lines
    is_wall = V_anim > 1e4
    V_finite = V_anim.copy()
    V_finite[is_wall] = 0  # Remove walls from the finite part
    V_finite = np.clip(V_finite, np.min(V_finite), 500)
    V_range = np.max(V_finite) - np.min(V_finite)
    if V_range > 0:
        # Map the potential into [0, y_max_prob * 0.5] so it's always visible
        V_scaled = (V_finite - np.min(V_finite)) / V_range * y_max_prob * 0.5
    elif np.max(np.abs(V_finite)) > 0:
        V_scaled = V_finite / np.max(np.abs(V_finite)) * y_max_prob * 0.5
    else:
        V_scaled = V_finite.copy()
    # Put walls back as tall vertical lines at the y-axis max
    V_scaled[is_wall] = y_max_prob * 0.95

    @st.cache_data
    def precompute_animation(potential_name, init_mode_key, x0_v, sigma_v, k0_v,
                              eigen_n_v, n1_v, n2_v, ratio_v,
                              _dt, _steps_per_frame, _n_frames, _N,
                              _x_min, _x_max, _stride):
        """Pre-compute all animation frames (cached)."""
        qs = QuantumSystem(x_min=_x_min, x_max=_x_max, N=_N)
        qs.set_potential(POTENTIALS[potential_name]["func"])

        # Initialize state
        if init_mode_key == "Gaussian wavepacket":
            qs.set_gaussian_wavepacket(x0=x0_v, sigma=sigma_v, k0=k0_v)
        elif init_mode_key == "Energy eigenstate":
            _, _, _, _eigs = create_system_and_eigenstates(potential_name, _x_min, _x_max, _N, n_eigen=16)
            if _eigs.size > 0 and eigen_n_v < _eigs.shape[1]:
                qs.psi = _eigs[:, eigen_n_v].astype(complex).copy()
            else:
                qs.set_gaussian_wavepacket(x0=0, sigma=1.0, k0=0)
        elif init_mode_key == "Superposition":
            _, _, _, _eigs = create_system_and_eigenstates(potential_name, _x_min, _x_max, _N, n_eigen=16)
            if _eigs.size > 0 and max(n1_v, n2_v) < _eigs.shape[1]:
                c1, c2 = np.sqrt(ratio_v), np.sqrt(1 - ratio_v)
                qs.psi = (c1 * _eigs[:, n1_v] + c2 * _eigs[:, n2_v]).astype(complex)
                qs.psi /= np.sqrt(np.sum(np.abs(qs.psi) ** 2) * qs.dx)
            else:
                qs.set_gaussian_wavepacket(x0=0, sigma=1.0, k0=0)

        # Collect frames
        probs = []            # |ψ|² downsampled
        re_parts = []         # Re(ψ) downsampled
        im_parts = []         # Im(ψ) downsampled
        times = []
        energies = []
        exp_x = []
        delta_x = []
        norms = []

        # Frame 0
        probs.append(np.abs(qs.psi[::_stride]) ** 2)
        re_parts.append(np.real(qs.psi[::_stride]))
        im_parts.append(np.imag(qs.psi[::_stride]))
        times.append(qs.time)
        energies.append(qs.expectation_energy())
        exp_x.append(qs.expectation_x())
        delta_x.append(qs.uncertainty_x())
        norms.append(qs.norm())

        for _ in range(_n_frames):
            for __ in range(_steps_per_frame):
                qs.step(_dt)
            probs.append(np.abs(qs.psi[::_stride]) ** 2)
            re_parts.append(np.real(qs.psi[::_stride]))
            im_parts.append(np.imag(qs.psi[::_stride]))
            times.append(qs.time)
            energies.append(qs.expectation_energy())
            exp_x.append(qs.expectation_x())
            delta_x.append(qs.uncertainty_x())
            norms.append(qs.norm())

        return (np.array(probs), np.array(re_parts), np.array(im_parts),
                np.array(times), np.array(energies),
                np.array(exp_x), np.array(delta_x), np.array(norms))

    # Gather init params (use defaults for unused modes so cache key is stable)
    _x0 = x0 if init_mode == "Gaussian wavepacket" else 0.0
    _sigma = sigma if init_mode == "Gaussian wavepacket" else 1.0
    _k0 = k0 if init_mode == "Gaussian wavepacket" else 0.0
    _eigen_n = eigen_n if init_mode == "Energy eigenstate" else 0
    _n1 = n1 if init_mode == "Superposition" else 0
    _n2 = n2 if init_mode == "Superposition" else 1
    _ratio = ratio if init_mode == "Superposition" else 0.5

    with col_plot:
        plot_placeholder = st.empty()

        if compute_btn:
            with st.spinner("Pre-computing time evolution..."):
                (probs, re_parts, im_parts, times, energies,
                 exp_xs, delta_xs, norms_arr) = precompute_animation(
                    potential_name, init_mode, _x0, _sigma, _k0,
                    _eigen_n, _n1, _n2, _ratio,
                    dt, steps_per_frame, n_frames, N,
                    x_min, x_max, stride,
                )

            # ── Build Plotly figure with native animation frames ──
            # Initial data traces
            traces = [
                go.Scatter(
                    x=x_anim, y=probs[0], mode='lines',
                    name='|ψ(x)|²',
                    line=dict(color='#2196F3', width=2.5),
                    fill='tozeroy', fillcolor='rgba(33,150,243,0.15)',
                ),
            ]
            trace_count = 1

            if show_potential:
                traces.append(go.Scatter(
                    x=x_anim, y=V_scaled, mode='lines',
                    name='V(x) (scaled)',
                    line=dict(color='rgba(255,87,34,0.5)', width=1.5, dash='dash'),
                ))
                trace_count += 1

            if show_components:
                traces.append(go.Scatter(
                    x=x_anim, y=re_parts[0], mode='lines',
                    name='Re(ψ)', line=dict(color='#4CAF50', width=1.5),
                ))
                traces.append(go.Scatter(
                    x=x_anim, y=im_parts[0], mode='lines',
                    name='Im(ψ)', line=dict(color='#FF9800', width=1.5),
                ))
                trace_count += 2

            # Build animation frames
            plotly_frames = []
            slider_steps = []
            for i in range(len(times)):
                frame_data = [go.Scatter(y=probs[i])]  # Update |ψ|²

                if show_potential:
                    frame_data.append(go.Scatter(y=V_scaled))  # Potential doesn't change

                if show_components:
                    frame_data.append(go.Scatter(y=re_parts[i]))
                    frame_data.append(go.Scatter(y=im_parts[i]))

                plotly_frames.append(go.Frame(
                    data=frame_data,
                    name=str(i),
                    layout=go.Layout(
                        title=f"|ψ(x)|²   —   t = {times[i]:.3f}    ⟨E⟩ = {energies[i]:.3f}    ⟨x⟩ = {exp_xs[i]:.3f}    Δx = {delta_xs[i]:.3f}"
                    ),
                ))

                # Slider steps (show a subset of labels to avoid clutter)
                slider_steps.append(dict(
                    args=[[str(i)], dict(frame=dict(duration=0, redraw=False),
                                         mode="immediate",
                                         transition=dict(duration=0))],
                    label=f"{times[i]:.2f}",
                    method="animate",
                ))

            # Assemble the animated figure
            fig = go.Figure(data=traces, frames=plotly_frames)

            y_range_main = [0, y_max_prob]
            y_range_comp = [-np.sqrt(y_max_prob), np.sqrt(y_max_prob)]

            fig.update_layout(
                template="plotly_white",
                height=500 if not show_components else 500,
                xaxis=dict(title="x", range=[x_min, x_max]),
                yaxis=dict(range=y_range_main, title="|ψ(x)|²"),
                title=f"|ψ(x)|²   —   t = {times[0]:.3f}    ⟨E⟩ = {energies[0]:.3f}    ⟨x⟩ = {exp_xs[0]:.3f}    Δx = {delta_xs[0]:.3f}",
                margin=dict(l=50, r=20, t=80, b=120),
                legend=dict(orientation="h", y=1.12),
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        y=-0.15, x=0.0, xanchor="left",
                        buttons=[
                            dict(label="▶ Play",
                                 method="animate",
                                 args=[None, dict(
                                     frame=dict(duration=speed, redraw=False),
                                     fromcurrent=True,
                                     transition=dict(duration=0),
                                     mode="immediate",
                                 )]),
                            dict(label="⏸ Pause",
                                 method="animate",
                                 args=[[None], dict(
                                     frame=dict(duration=0, redraw=False),
                                     mode="immediate",
                                     transition=dict(duration=0),
                                 )]),
                            dict(label="⏮ Reset",
                                 method="animate",
                                 args=[["0"], dict(
                                     frame=dict(duration=0, redraw=True),
                                     mode="immediate",
                                     transition=dict(duration=0),
                                 )]),
                        ],
                    ),
                ],
                sliders=[dict(
                    active=0,
                    yanchor="top", y=-0.22, xanchor="left", x=0.0,
                    len=1.0,
                    currentvalue=dict(prefix="t = ", visible=True, xanchor="center"),
                    transition=dict(duration=0),
                    steps=slider_steps,
                )],
            )

            plot_placeholder.plotly_chart(fig, use_container_width=True, key="animation")

            # Summary stats
            info_box.markdown(f"""
            **Frames**: {len(times)}  
            **Time range**: 0 → {times[-1]:.2f}  
            **⟨E⟩ drift**: {energies[0]:.4f} → {energies[-1]:.4f}  
            **Norm drift**: {norms_arr[0]:.6f} → {norms_arr[-1]:.6f}  
            
            Use **▶ Play** / **⏸ Pause** below  
            the plot, or drag the **slider**.
            """)
        else:
            # Show static initial frame
            qs0 = init_system()
            E0 = qs0.expectation_energy()
            prob0 = np.abs(qs0.psi[::stride]) ** 2

            fig0_traces = [
                go.Scatter(
                    x=x_anim, y=prob0, mode='lines',
                    name='|ψ(x)|²',
                    line=dict(color='#2196F3', width=2.5),
                    fill='tozeroy', fillcolor='rgba(33,150,243,0.15)',
                ),
            ]
            if show_potential:
                fig0_traces.append(go.Scatter(
                    x=x_anim, y=V_scaled, mode='lines',
                    name='V(x) (scaled)',
                    line=dict(color='rgba(255,87,34,0.5)', width=1.5, dash='dash'),
                ))
            if show_components:
                fig0_traces.append(go.Scatter(
                    x=x_anim, y=np.real(qs0.psi[::stride]), mode='lines',
                    name='Re(ψ)', line=dict(color='#4CAF50', width=1.5),
                ))
                fig0_traces.append(go.Scatter(
                    x=x_anim, y=np.imag(qs0.psi[::stride]), mode='lines',
                    name='Im(ψ)', line=dict(color='#FF9800', width=1.5),
                ))

            fig0 = go.Figure(data=fig0_traces)
            fig0.update_layout(
                template="plotly_white", height=500,
                xaxis=dict(title="x", range=[x_min, x_max]),
                yaxis=dict(range=[0, y_max_prob], title="|ψ(x)|²"),
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


# ── TAB 2: Energy Eigenstates ──────────────────────────────────────────
with tab_eigenstates:
    st.subheader(f"Energy eigenstates — {potential_name}")

    n_show = st.slider("Number of eigenstates to display", 1, 12, 5, key="n_eigen_show")

    if eigenstates.size > 0 and eigenstates.shape[1] >= n_show:
        fig_eigen = make_subplots(
            rows=n_show, cols=1, shared_xaxes=True,
            vertical_spacing=0.02,
        )

        colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0',
                  '#00BCD4', '#FF5722', '#607D8B', '#795548', '#CDDC39',
                  '#3F51B5', '#009688']

        for i in range(n_show):
            psi_n = eigenstates[:, i]
            prob_n = psi_n ** 2

            fig_eigen.add_trace(go.Scatter(
                x=x_grid, y=prob_n, mode='lines',
                name=f'n={i}  E={eigenenergies[i]:.3f}',
                line=dict(color=colors[i % len(colors)], width=2),
                fill='tozeroy',
                fillcolor=f'rgba({int(colors[i % len(colors)][1:3], 16)},'
                          f'{int(colors[i % len(colors)][3:5], 16)},'
                          f'{int(colors[i % len(colors)][5:7], 16)},0.1)',
            ), row=i + 1, col=1)

            fig_eigen.update_yaxes(
                title_text=f"n={i}", row=i + 1, col=1,
                showticklabels=False,
            )

        fig_eigen.update_layout(
            height=150 * n_show,
            template="plotly_white",
            title="Probability densities |ψₙ(x)|²",
            showlegend=True,
            legend=dict(orientation="h", y=1.05),
            xaxis=dict(title="x"),
        )
        st.plotly_chart(fig_eigen, use_container_width=True)

        # Energy level diagram
        st.subheader("Energy levels")
        fig_levels = go.Figure()
        for i in range(min(n_show, len(eigenenergies))):
            fig_levels.add_trace(go.Scatter(
                x=[-1, 1], y=[eigenenergies[i], eigenenergies[i]],
                mode='lines+text',
                line=dict(color=colors[i % len(colors)], width=3),
                text=[f'n={i}', f'E={eigenenergies[i]:.3f}'],
                textposition=['middle left', 'middle right'],
                name=f'n={i}',
                showlegend=False,
            ))
        fig_levels.update_layout(
            height=400, template="plotly_white",
            xaxis=dict(showticklabels=False, range=[-2, 3], title=""),
            yaxis=dict(title="Energy"),
            title="Energy spectrum",
        )
        st.plotly_chart(fig_levels, use_container_width=True)

        # Energy differences
        if len(eigenenergies) > 1:
            st.subheader("Energy differences")
            diffs = np.diff(eigenenergies[:n_show])
            diff_data = {f"E_{i+1} - E_{i}": f"{d:.4f}" for i, d in enumerate(diffs)}
            st.json(diff_data)
    else:
        st.warning("Could not compute eigenstates for this potential. Try adjusting the grid range.")


# ── TAB 3: Theory ──────────────────────────────────────────────────────
with tab_info:
    st.subheader("The Schrödinger Equation")
    st.latex(r"i\hbar \frac{\partial}{\partial t}\Psi(x,t) = \hat{H}\Psi(x,t) = \left[-\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2} + V(x)\right]\Psi(x,t)")

    st.markdown("""
    ### Split-Operator FFT Method

    The time evolution operator is:
    """)
    st.latex(r"\hat{U}(\Delta t) = e^{-i\hat{H}\Delta t/\hbar}")

    st.markdown("Using the **Trotter-Suzuki decomposition** (2nd order):")
    st.latex(r"\hat{U}(\Delta t) \approx e^{-i\hat{V}\Delta t/2\hbar} \cdot e^{-i\hat{T}\Delta t/\hbar} \cdot e^{-i\hat{V}\Delta t/2\hbar} + \mathcal{O}(\Delta t^3)")

    st.markdown("""
    **The key insight:**
    - The potential operator $e^{-iV\\Delta t/2\\hbar}$ is **diagonal in position space** → simple multiplication
    - The kinetic operator $e^{-iT\\Delta t/\\hbar}$ is **diagonal in momentum space** → multiply in Fourier space
    - We use FFT to switch between representations efficiently

    **Algorithm per time step:**
    1. Multiply $\\psi(x)$ by $e^{-iV(x)\\Delta t / 2\\hbar}$ (half-step potential)
    2. FFT to momentum space → $\\tilde{\\psi}(k)$
    3. Multiply by $e^{-i\\hbar k^2 \\Delta t / 2m}$ (full-step kinetic)
    4. Inverse FFT back to position space
    5. Multiply by $e^{-iV(x)\\Delta t / 2\\hbar}$ (half-step potential)

    ### Properties
    - **Unitary**: Preserves normalization exactly (unlike finite-difference schemes)
    - **Stable**: No numerical instabilities
    - **Efficient**: $\\mathcal{O}(N \\log N)$ per time step via FFT
    - **Accurate**: Global error $\\mathcal{O}(\\Delta t^2)$

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
    *We use natural units where $\\hbar = 1$ and $m = 1$.*
    """)


# ── Footer ──────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**QuantApp** — Quantum Mechanics Simulator  \n"
    "Built with Streamlit + NumPy + Plotly  \n"
    "Split-Operator FFT method"
)
