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


# ═══════════════════════════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data
def compute_eigenstates_cached(potential_name, x_min, x_max, N, n_eigen=10):
    """Compute and cache eigenstates for the given potential."""
    qs = QuantumSystem(x_min=x_min, x_max=x_max, N=N)
    qs.set_potential(POTENTIALS[potential_name]["func"])
    try:
        energies, states = qs.compute_eigenstates(n_states=n_eigen)
    except Exception:
        energies, states = np.array([]), np.array([])
    return qs.x.copy(), qs.V.copy(), energies, states


def create_quantum_system(potential_name, x_min, x_max, N, init_mode,
                          x0, sigma, k0, eigen_n, n1, n2, ratio, eigenstates):
    """Create a QuantumSystem with the chosen initial state."""
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
            c1, c2 = np.sqrt(ratio), np.sqrt(1 - ratio)
            qs.psi = (c1 * eigenstates[:, n1] + c2 * eigenstates[:, n2]).astype(complex)
            qs.psi /= np.sqrt(np.sum(np.abs(qs.psi) ** 2) * qs.dx)
        else:
            qs.set_gaussian_wavepacket(x0=0, sigma=1.0, k0=0)
    return qs


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


@st.cache_data
def precompute_animation(potential_name, init_mode_key, x0_v, sigma_v, k0_v,
                         eigen_n_v, n1_v, n2_v, ratio_v,
                         _dt, _steps_per_frame, _n_frames,
                         _N, _x_min, _x_max, _stride):
    """Pre-compute all animation frames (cached).

    Returns arrays: probs, re_parts, im_parts, times, energies,
                    exp_x, delta_x, norms
    """
    # Build system
    _, _, _, eigs = compute_eigenstates_cached(potential_name, _x_min, _x_max, _N, n_eigen=16)
    qs = QuantumSystem(x_min=_x_min, x_max=_x_max, N=_N)
    qs.set_potential(POTENTIALS[potential_name]["func"])

    # Initialise wavefunction
    if init_mode_key == "Gaussian wavepacket":
        qs.set_gaussian_wavepacket(x0=x0_v, sigma=sigma_v, k0=k0_v)
    elif init_mode_key == "Energy eigenstate":
        if eigs.size > 0 and eigen_n_v < eigs.shape[1]:
            qs.psi = eigs[:, eigen_n_v].astype(complex).copy()
        else:
            qs.set_gaussian_wavepacket(x0=0, sigma=1.0, k0=0)
    elif init_mode_key == "Superposition":
        if eigs.size > 0 and max(n1_v, n2_v) < eigs.shape[1]:
            c1, c2 = np.sqrt(ratio_v), np.sqrt(1 - ratio_v)
            qs.psi = (c1 * eigs[:, n1_v] + c2 * eigs[:, n2_v]).astype(complex)
            qs.psi /= np.sqrt(np.sum(np.abs(qs.psi) ** 2) * qs.dx)
        else:
            qs.set_gaussian_wavepacket(x0=0, sigma=1.0, k0=0)

    # Collect frames
    def snapshot():
        return dict(
            prob=np.abs(qs.psi[::_stride]) ** 2,
            re=np.real(qs.psi[::_stride]),
            im=np.imag(qs.psi[::_stride]),
            time=qs.time,
            energy=qs.expectation_energy(),
            exp_x=qs.expectation_x(),
            delta_x=qs.uncertainty_x(),
            norm=qs.norm(),
        )

    frames = [snapshot()]
    for _ in range(_n_frames):
        for __ in range(_steps_per_frame):
            qs.step(_dt)
        frames.append(snapshot())

    # Unpack into arrays
    keys = ("prob", "re", "im", "time", "energy", "exp_x", "delta_x", "norm")
    arrays = {k: np.array([f[k] for f in frames]) for k in keys}
    return (arrays["prob"], arrays["re"], arrays["im"],
            arrays["time"], arrays["energy"],
            arrays["exp_x"], arrays["delta_x"], arrays["norm"])


# ── Plot builders ──────────────────────────────────────────────────────

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
#  Sidebar controls
# ═══════════════════════════════════════════════════════════════════════

# ── Potential ──────────────────────────────────────────────────────────
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

# ── Initial wavepacket ────────────────────────────────────────────────
st.sidebar.header("🌊 Initial wavepacket")
init_mode = st.sidebar.radio(
    "Initialization",
    ["Gaussian wavepacket", "Energy eigenstate", "Superposition"],
)

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

# ── Time evolution ────────────────────────────────────────────────────
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
    f"{n_frames} frames · Δt/frame = {time_per_frame:.3f} · total = {n_frames * time_per_frame:.1f}"
)

# ── Display options ───────────────────────────────────────────────────
st.sidebar.header("📊 Display")
show_components = st.sidebar.checkbox("Show Re(ψ) and Im(ψ)", value=False)
show_potential = st.sidebar.checkbox("Show potential V(x)", value=True)
show_eigenstates = st.sidebar.checkbox("Show energy eigenstates", value=False)
if show_eigenstates:
    n_eigenstates = st.sidebar.slider("Number of eigenstates", 1, 10, 4)
y_max_prob = st.sidebar.slider("|ψ|² y-axis max", 0.1, 2.0, 0.8, 0.05)


# ═══════════════════════════════════════════════════════════════════════
#  Compute eigenstates & derived data
# ═══════════════════════════════════════════════════════════════════════

x_grid, V_grid, eigenenergies, eigenstates = compute_eigenstates_cached(
    potential_name, x_min, x_max, N, n_eigen=16,
)

stride = max(1, N // 300)
x_anim = x_grid[::stride]
V_scaled = scale_potential_for_display(V_grid, y_max_prob, stride)

# Stable cache keys — use defaults for inactive modes
_x0 = x0 if init_mode == "Gaussian wavepacket" else 0.0
_sigma = sigma if init_mode == "Gaussian wavepacket" else 1.0
_k0 = k0 if init_mode == "Gaussian wavepacket" else 0.0
_eigen_n = eigen_n if init_mode == "Energy eigenstate" else 0
_n1 = n1 if init_mode == "Superposition" else 0
_n2 = n2 if init_mode == "Superposition" else 1
_ratio = ratio if init_mode == "Superposition" else 0.5


# ═══════════════════════════════════════════════════════════════════════
#  Tabs
# ═══════════════════════════════════════════════════════════════════════

tab_animate, tab_eigenstates, tab_info = st.tabs([
    "🎬 Time Evolution", "📐 Energy Eigenstates", "📖 Theory",
])


# ═══════════════════════════════════════════════════════════════════════
#  TAB 1 — Time Evolution Animation
# ═══════════════════════════════════════════════════════════════════════

with tab_animate:
    col_ctrl, col_plot = st.columns([1, 3])

    with col_ctrl:
        compute_btn = st.button("▶️ Compute & Play",
                                use_container_width=True, type="primary")
        st.markdown("---")
        info_box = st.empty()

    with col_plot:
        plot_placeholder = st.empty()

        if compute_btn:
            # ── Run simulation ─────────────────────────────────────
            precompute_animation.clear()
            total_t = n_frames * steps_per_frame * dt
            with st.spinner(f"Pre-computing {n_frames} frames (t = 0 → {total_t:.1f}) …"):
                (probs, re_parts, im_parts, times, energies,
                 exp_xs, delta_xs, norms_arr) = precompute_animation(
                    potential_name, init_mode, _x0, _sigma, _k0,
                    _eigen_n, _n1, _n2, _ratio,
                    dt, steps_per_frame, n_frames, N, x_min, x_max, stride,
                )

            # ── Initial traces ─────────────────────────────────────
            traces = build_wavefunction_traces(
                x_anim, probs[0], re_parts[0], im_parts[0],
                V_scaled, show_components, show_potential,
            )

            # ── Animation frames ──────────────────────────────────
            plotly_frames, slider_steps = [], []
            for i in range(len(times)):
                fd = build_frame_data(probs[i], re_parts[i], im_parts[i],
                                      V_scaled, show_components, show_potential)
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

            # ── Assemble figure ────────────────────────────────────
            y_range, y_title = y_axis_config(y_max_prob, show_components)
            fig = go.Figure(data=traces, frames=plotly_frames)
            fig.update_layout(
                template="plotly_white", height=500,
                xaxis=dict(title="x", range=[x_min, x_max]),
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
                                 frame=dict(duration=speed, redraw=False),
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

            # ── Summary stats ──────────────────────────────────────
            info_box.markdown(f"""
            **Frames**: {len(times)}  
            **Time range**: 0 → {times[-1]:.2f}  
            **⟨E⟩ drift**: {energies[0]:.4f} → {energies[-1]:.4f}  
            **Norm drift**: {norms_arr[0]:.6f} → {norms_arr[-1]:.6f}  
            
            Use **▶ Play** / **⏸ Pause** below  
            the plot, or drag the **slider**.
            """)

        else:
            # ── Static initial frame ───────────────────────────────
            qs0 = create_quantum_system(
                potential_name, x_min, x_max, N, init_mode,
                _x0, _sigma, _k0, _eigen_n, _n1, _n2, _ratio, eigenstates,
            )
            E0 = qs0.expectation_energy()
            prob0 = np.abs(qs0.psi[::stride]) ** 2

            traces0 = build_wavefunction_traces(
                x_anim, prob0,
                np.real(qs0.psi[::stride]), np.imag(qs0.psi[::stride]),
                V_scaled, show_components, show_potential,
            )

            y_range, y_title = y_axis_config(y_max_prob, show_components)
            fig0 = go.Figure(data=traces0)
            fig0.update_layout(
                template="plotly_white", height=500,
                xaxis=dict(title="x", range=[x_min, x_max]),
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


# ═══════════════════════════════════════════════════════════════════════
#  TAB 2 — Energy Eigenstates
# ═══════════════════════════════════════════════════════════════════════

EIGENSTATE_COLORS = [
    "#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0",
    "#00BCD4", "#FF5722", "#607D8B", "#795548", "#CDDC39",
    "#3F51B5", "#009688",
]

with tab_eigenstates:
    st.subheader(f"Energy eigenstates — {potential_name}")
    n_show = st.slider("Number of eigenstates to display", 1, 12, 5, key="n_eigen_show")

    if eigenstates.size > 0 and eigenstates.shape[1] >= n_show:
        # ── Probability densities ──────────────────────────────────
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

        # ── Energy level diagram ───────────────────────────────────
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

        # ── Energy differences ─────────────────────────────────────
        if len(eigenenergies) > 1:
            st.subheader("Energy differences")
            diffs = np.diff(eigenenergies[:n_show])
            st.json({f"E_{i+1} - E_{i}": f"{d:.4f}" for i, d in enumerate(diffs)})
    else:
        st.warning("Could not compute eigenstates for this potential. "
                   "Try adjusting the grid range.")


# ═══════════════════════════════════════════════════════════════════════
#  TAB 3 — Theory
# ═══════════════════════════════════════════════════════════════════════

with tab_info:
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


# ═══════════════════════════════════════════════════════════════════════
#  Footer
# ═══════════════════════════════════════════════════════════════════════

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**QuantApp** — Quantum Mechanics Simulator  \n"
    "Built with Streamlit + NumPy + Plotly  \n"
    "Split-Operator FFT method"
)
