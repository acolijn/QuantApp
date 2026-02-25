# QuantApp — Quantum Mechanics Simulator

An interactive 1D quantum mechanics simulator built with
[Streamlit](https://streamlit.io/), [NumPy](https://numpy.org/), and
[Plotly](https://plotly.com/). It solves the time-dependent Schrödinger
equation in real time using the **Split-Operator FFT** method.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

---

## Features

- **8 built-in potentials** — free particle, infinite/finite square well,
  harmonic oscillator, double well, tunneling barrier, Morse potential,
  step potential.
- **Three initialisation modes** — Gaussian wavepacket, energy eigenstates,
  superposition of two eigenstates.
- **Animated time evolution** — pre-computed frames with Play / Pause /
  Slider controls; real-time diagnostics (⟨E⟩, ⟨x⟩, Δx, norm).
- **Eigenstate viewer** — probability densities, energy level diagram, and
  energy differences.
- **Theory tab** — in-app documentation of the numerical methods and physics.

## Quick Start

### Local (venv)

```bash
# Clone and enter the project
cd QuantApp

# Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**.

### Docker

```bash
docker compose up --build
```

---

## Project Structure

```
QuantApp/
├── app.py                      # Streamlit entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── quantapp/                   # Core package
│   ├── __init__.py
│   ├── solver.py               # QuantumSystem — physics engine
│   ├── potentials.py           # Potential functions & registry
│   ├── simulation.py           # Caching, wavefunction init, animation precomputation
│   ├── plots.py                # Plotly figure builders
│   └── ui/                     # Streamlit UI components
│       ├── __init__.py
│       ├── sidebar.py          # Sidebar controls → config dict
│       ├── tab_animation.py    # Time Evolution tab
│       ├── tab_eigenstates.py  # Energy Eigenstates tab
│       └── tab_theory.py       # Theory / documentation tab
│
└── tests/
    ├── test_solver.py          # 18 tests — solver, DST, energy conservation
    └── test_potentials.py      # 11 tests — potential functions, smooth edges, registry
```

---

## Architecture

### Solver (`quantapp/solver.py`)

`QuantumSystem` is the core physics engine. It maintains a 1D spatial grid,
a complex wavefunction `psi`, and the potential `V(x)`.

**Time evolution** uses the Trotter-Suzuki split-operator method:

```
ψ(t+dt) = exp(-iV dt/2) · FFT⁻¹[ exp(-iT dt) · FFT[ exp(-iV dt/2) · ψ(t) ] ]
```

This is O(N log N) per step, unconditionally stable, and unitary (norm-preserving).

Two propagator backends are available:

| Mode | When used | Boundary conditions |
|------|-----------|---------------------|
| **FFT** | Default for all potentials | Periodic (open systems) |
| **DST-I** | Automatically selected when V has hard walls (> 10⁴) | Dirichlet (ψ = 0 at walls) |

The DST propagator achieves machine-precision energy conservation (~10⁻¹³
relative drift) for the infinite square well.

**Observables:** norm, ⟨x⟩, ⟨x²⟩, ⟨p⟩, ⟨E⟩, Δx. Energy is computed as a
per-particle value ⟨E⟩/⟨ψ|ψ⟩ in the FFT path, and via the DST Parseval
relation in the DST path.

**Eigenstates** are computed from the tridiagonal finite-difference
Hamiltonian using `scipy.linalg.eigh_tridiagonal`.

### Potentials (`quantapp/potentials.py`)

Each potential is a pure function `V(x) → np.ndarray`. A `POTENTIALS` registry
dict maps display names to `{func, description, x_range, default_x0, ...}`.

**Smooth edges:** Discontinuous potentials (finite well, barrier, step) use
`tanh`-based profiles (`smooth_step`, `smooth_rect`) with `DEFAULT_STEEPNESS = 5`
(transition width ≈ 0.4 length units). This prevents the Trotter splitting error
from diverging at sharp edges.

### Simulation (`quantapp/simulation.py`)

Streamlit-specific caching and wavefunction DRY initialisation:

- `compute_eigenstates_cached()` — `@st.cache_data` wrapper for eigenstate
  computation.
- `initialize_wavefunction()` — single source of truth for setting ψ₀ from
  any initialisation mode.
- `precompute_animation()` — cached computation of all animation frames
  (probabilities, Re/Im parts, times, energies, ⟨x⟩, Δx, norms).

### Plots (`quantapp/plots.py`)

Framework-agnostic Plotly figure builders:

- `scale_potential_for_display()` — scales V(x) for overlay on ψ plots.
- `build_wavefunction_traces()` / `build_frame_data()` — trace construction.
- `y_axis_config()`, `frame_title()` — layout helpers.

### UI (`quantapp/ui/`)

Each Streamlit tab is a separate module with a `render()` function:

| Module | Tab | Description |
|--------|-----|-------------|
| `sidebar.py` | — | All sidebar widgets; returns a config dict |
| `tab_animation.py` | Time Evolution | Animated + static wavefunction display |
| `tab_eigenstates.py` | Eigenstates | Probability densities, energy levels |
| `tab_theory.py` | Theory | In-app documentation |

---

## Numerical Methods in Detail

### Split-Operator FFT

The Hamiltonian is split into kinetic (T) and potential (V) parts. The
second-order Trotter-Suzuki decomposition gives O(Δt³) local error and
O(Δt²) global error:

```
U(Δt) ≈ exp(-iV Δt/2) · exp(-iT Δt) · exp(-iV Δt/2)
```

The potential half-steps are diagonal multiplications in position space.
The kinetic step is a diagonal multiplication in momentum space, accessed
via FFT/IFFT.

### DST Propagator for Hard Walls

When the potential contains hard walls (V > 10⁴), the solver detects a
contiguous interior region and switches to a **Discrete Sine Transform
(type I)** for the kinetic step. The DST basis functions sin(nπx/L) are
exactly zero at the boundaries, enforcing Dirichlet conditions without
leakage.

Energy is measured via the DST Parseval relation:

```
⟨T⟩ = [Δx / 2(N_int + 1)] Σ_k T_k |ψ̃_k|²
```

### Smooth Potential Edges

Sharp discontinuities make the commutator [T, V] ∝ ∇V diverge, causing
O(Δt³ · |∇V|) Trotter error per step. We replace discontinuities with
smooth tanh profiles (steepness = 5, transition ≈ 0.4 length units).

### Boundary Strategy for Scattering Potentials

Scattering potentials (barrier, step, free particle) use wide spatial
domains (e.g. x ∈ [-50, 50]) so the wavepacket doesn't reach the FFT's
periodic boundaries during typical simulation times. The solver also
includes optional Complex Absorbing Potential (CAP) infrastructure that
can be enabled for specialised long-time simulations.

---

## Adding a New Potential

1. **Define the function** in `quantapp/potentials.py`:

   ```python
   def my_potential(x, param=1.0):
       """Description of the potential."""
       return param * x**2  # example
   ```

2. **Add a registry entry** to the `POTENTIALS` dict:

   ```python
   "My potential": {
       "func": lambda x: my_potential(x, param=2.0),
       "description": "A custom potential — short description",
       "x_range": (-10, 10),
       "default_x0": 0.0,
       "default_k0": 0.0,
       "default_sigma": 1.0,
   },
   ```

3. **Add tests** in `tests/test_potentials.py` (shape, symmetry, limits).

The new potential will automatically appear in the sidebar dropdown.

---

## Running Tests

```bash
pytest -v
```

Currently **29 tests** covering:

- Grid setup and normalisation
- Time evolution unitarity (norm and energy conservation)
- Eigenstate computation (harmonic oscillator eigenvalues, orthonormality)
- DST propagator (activation, norm, energy, long-run stability)
- Smooth-edge energy conservation (finite well, barrier, step)
- N-independence regression (N=1024 vs N=2048 energy agreement)
- Potential functions (shapes, limits, symmetry, registry completeness)

---

## Configuration

Key parameters adjustable from the sidebar:

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| Grid points (N) | 256–2048 | 1024 | Spatial resolution |
| Time step (dt) | 0.001–0.05 | 0.01 | Accuracy vs speed |
| Steps per frame | 1–20 | 5 | Animation smoothness |
| Max time | 1–100 | 30 | Total simulation time |

---

## Known Limitations

- **N = 2048 + step potential:** At very high grid resolution, the maximum
  momentum k_max doubles, making FFT periodic wrap-around 4× worse. For
  most potentials this is negligible due to wide grids, but the step
  potential's reflected wave can accumulate enough aliased energy to
  measurably drift. Use N ≤ 1024 for the step potential, or enable the
  CAP absorber for long-time runs.
- **3D / multi-particle:** Only 1D single-particle systems are supported.
- **Relativistic effects:** Non-relativistic Schrödinger equation only.

---

## License

This project is provided as-is for educational purposes.
