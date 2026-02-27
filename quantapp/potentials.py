"""
Predefined Potential Functions
==============================
Each potential is a callable V(x) → np.ndarray, plus metadata for the UI.
To add a new potential, define the function and add an entry to POTENTIALS.
"""

import numpy as np


# ── Smooth step helper ─────────────────────────────────────────────────

def smooth_step(x, x_edge, steepness):
    """Smooth step from 0 to 1 at x = x_edge.

    Uses tanh with the given steepness (1/transition_width).
    The transition spans roughly ±2/steepness around x_edge.
    """
    return 0.5 * (1.0 + np.tanh(steepness * (x - x_edge)))


def smooth_rect(x, center, half_width, steepness):
    """Smooth rectangular window: ~1 inside [center-hw, center+hw], ~0 outside."""
    return smooth_step(x, center - half_width, steepness) * (1.0 - smooth_step(x, center + half_width, steepness))


# Default edge steepness — sharp enough to be physical, smooth enough
# for the split-operator method.  Transition width ~ 2/steepness ≈ 0.4.
DEFAULT_STEEPNESS = 5.0


# ── Potential functions ────────────────────────────────────────────────

def free_particle(x):
    """V(x) = 0"""
    return np.zeros_like(x)


def infinite_square_well(x, width=10.0):
    """Infinite square well centered at origin."""
    V = np.zeros_like(x)
    V[np.abs(x) > width / 2] = 1e6
    return V


def finite_square_well(x, width=6.0, depth=50.0, steepness=DEFAULT_STEEPNESS):
    """Finite square well: V ≈ -depth inside, 0 outside.

    Uses smooth tanh edges for numerical stability.
    """
    return -depth * smooth_rect(x, 0.0, width / 2, steepness)


def harmonic_oscillator(x, omega=1.0, mass=1.0):
    """V(x) = ½mω²x²"""
    return 0.5 * mass * omega ** 2 * x ** 2


def double_well(x, separation=3.0, depth=8.0):
    """Double well: two Gaussian wells."""
    return -depth * (np.exp(-((x - separation / 2) ** 2))
                     + np.exp(-((x + separation / 2) ** 2)))


def potential_barrier(x, width=1.0, height=20.0, steepness=DEFAULT_STEEPNESS):
    """Rectangular barrier for tunneling demonstrations.

    Uses smooth tanh edges for numerical stability.
    """
    return height * smooth_rect(x, 0.0, width / 2, steepness)


def asymmetric_double_well(x, offset=2.0):
    """Asymmetric double well: x⁴ - 5x² + offset·x."""
    return 0.05 * (x ** 4 - 5 * x ** 2 + offset * x) + 5


def morse_potential(x, D=10.0, a=0.5, x0=0.0):
    """Morse potential: V = D(1 - e^{-a(x-x0)})² — models molecular bonds."""
    return D * (1 - np.exp(-a * (x - x0))) ** 2


def step_potential(x, height=15.0, steepness=DEFAULT_STEEPNESS):
    """Step potential: V ≈ height for x > 0.

    Uses smooth tanh edge for numerical stability.
    """
    return height * smooth_step(x, 0.0, steepness)


def periodic_potential(x, depth=5.0, period=3.0):
    """Cosine lattice: V(x) = depth · [1 − cos(2πx/a)] / 2.

    Produces a periodic array of wells with spacing *period* (= lattice
    constant *a*).  The potential ranges from 0 (at the well minima) to
    *depth* (at the maxima).  Ideal for demonstrating energy band
    structure and Bloch's theorem — the FFT's periodic boundary
    conditions are the *correct* BCs for this potential.

    Parameters
    ----------
    x : np.ndarray
    depth : float  (default 5.0)
        Peak-to-trough amplitude.
    period : float  (default 3.0)
        Lattice constant *a*.
    """
    return depth * (1.0 - np.cos(2 * np.pi * x / period)) / 2.0


def multi_well(x, n_wells=5, well_width=1.0, depth=30.0,
               jitter=0.0, seed=42, steepness=DEFAULT_STEEPNESS):
    """N rectangular wells on a periodic lattice, with optional position jitter.

    Creates *n_wells* negative rectangular wells on a periodic grid with
    spacing L/n_wells, where L is the total interval length.  Because the
    grid uses periodic boundary conditions the spacing wraps around: the
    gap from the last well through the boundary back to the first well
    equals the gap between any other adjacent pair.

    Each well centre can be randomly displaced by up to +/- *jitter* from
    its lattice site (useful for modelling disorder in liquid noble gases).
    An overlap guard clamps the effective jitter so adjacent well edges
    never touch.

    The potential is constructed to be *exactly* periodic even with smooth
    tanh edges, by folding each point's distance to the well centre into
    the range [-spacing/2, spacing/2) before evaluating the smooth window.

    Parameters
    ----------
    x : np.ndarray
        Spatial grid (assumed periodic: x[0] and x[-1]+dx wrap around).
    n_wells : int  (default 5)
        Number of wells.
    well_width : float  (default 1.0)
        Full width of each rectangular well.
    depth : float  (default 30.0)
        Depth of each well (potential is -depth inside).
    jitter : float  (default 0.0)
        Maximum random displacement of each well centre from its
        lattice site.  Set to 0 for a perfect periodic lattice.
    seed : int  (default 42)
        Random seed for reproducibility (only matters when jitter > 0).
    steepness : float
        Edge steepness for the smooth rectangular windows.

    Returns
    -------
    np.ndarray
        Potential array V(x) with values in [-depth, 0].
    """
    x_lo = float(x[0])
    dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
    L = len(x) * dx  # full periodic domain length (N·dx = b − a)

    # Periodic lattice spacing
    spacing = L / n_wells

    # Centres placed so that the wrap-around gap equals the inter-well gap
    centers = x_lo + (np.arange(n_wells) + 0.5) * spacing

    # Apply jitter with overlap guard (respects periodicity)
    if jitter > 0 and n_wells > 1:
        max_jitter = (spacing - well_width) / 2  # keeps edges from touching
        safe_jitter = max(0.0, min(jitter, max_jitter))
        rng = np.random.default_rng(int(seed))
        centers = centers + rng.uniform(-safe_jitter, safe_jitter, size=n_wells)

    V = np.zeros_like(x)
    hw = well_width / 2
    for c in centers:
        # Fold distance to centre into [-L/2, L/2) so every well sees the
        # same local coordinate — guarantees exact periodicity.
        dx = (x - c + L / 2) % L - L / 2
        V -= depth * smooth_rect(dx, 0.0, hw, steepness)
    return V


# ── Registry ───────────────────────────────────────────────────────────
# Each entry carries: func, description, x_range, and wavepacket defaults.

POTENTIALS = {
    "Free particle": {
        "func": free_particle,
        "params": [],
        "description": "V(x) = 0 — Free propagation, wavepacket spreading",
        "x_range": (-40, 40),
        "default_x0": -5.0,
        "default_k0": 3.0,
        "default_sigma": 1.5,
        "periodic": True,
    },
    "Infinite square well": {
        "func": infinite_square_well,
        "params": [
            {"name": "width", "label": "Well width", "min": 2.0, "max": 30.0, "default": 10.0, "step": 0.5},
        ],
        "description": "Particle in a box — quantized energy levels",
        "x_range": (-10, 10),
        "default_x0": 0.0,
        "default_k0": 0.0,
        "default_sigma": 1.0,
    },
    "Finite square well": {
        "func": finite_square_well,
        "params": [
            {"name": "width", "label": "Well width", "min": 1.0, "max": 20.0, "default": 6.0, "step": 0.5},
            {"name": "depth", "label": "Well depth", "min": 5.0, "max": 200.0, "default": 50.0, "step": 5.0},
        ],
        "description": "Finite depth well — bound & scattering states (smooth edges)",
        "x_range": (-15, 15),
        "default_x0": 0.0,
        "default_k0": 0.0,
        "default_sigma": 1.0,
        "periodic": True,
    },
    "Harmonic oscillator": {
        "func": harmonic_oscillator,
        "params": [
            {"name": "omega", "label": "Frequency ω", "min": 0.1, "max": 5.0, "default": 1.0, "step": 0.1},
        ],
        "description": "V = ½mω²x² — Equally spaced energy levels",
        "x_range": (-10, 10),
        "default_x0": -3.0,
        "default_k0": 0.0,
        "default_sigma": 0.5,
    },
    "Double well": {
        "func": double_well,
        "params": [
            {"name": "separation", "label": "Well separation", "min": 1.0, "max": 10.0, "default": 4.0, "step": 0.5},
            {"name": "depth", "label": "Well depth", "min": 1.0, "max": 30.0, "default": 10.0, "step": 1.0},
        ],
        "description": "Two symmetric wells — quantum tunneling between wells",
        "x_range": (-10, 10),
        "default_x0": -2.0,
        "default_k0": 0.0,
        "default_sigma": 0.7,
        "periodic": True,
    },
    "Potential barrier (tunneling)": {
        "func": potential_barrier,
        "params": [
            {"name": "width", "label": "Barrier width", "min": 0.1, "max": 5.0, "default": 1.0, "step": 0.1},
            {"name": "height", "label": "Barrier height", "min": 1.0, "max": 100.0, "default": 20.0, "step": 1.0},
        ],
        "description": "Rectangular barrier — quantum tunneling (smooth edges)",
        "x_range": (-50, 50),
        "default_x0": -5.0,
        "default_k0": 4.0,
        "default_sigma": 1.0,
        "periodic": True,
    },
    "Morse potential": {
        "func": morse_potential,
        "params": [
            {"name": "D", "label": "Dissociation energy D", "min": 1.0, "max": 30.0, "default": 12.0, "step": 1.0},
            {"name": "a", "label": "Width parameter a", "min": 0.1, "max": 2.0, "default": 0.4, "step": 0.05},
            {"name": "x0", "label": "Equilibrium position x₀", "min": -10.0, "max": 10.0, "default": -3.0, "step": 0.5},
        ],
        "description": "Anharmonic oscillator — models molecular vibrations",
        "x_range": (-15, 15),
        "default_x0": -3.0,
        "default_k0": 0.0,
        "default_sigma": 0.8,
        "periodic": True,
    },
    "Step potential": {
        "func": step_potential,
        "params": [
            {"name": "height", "label": "Step height", "min": 1.0, "max": 50.0, "default": 15.0, "step": 1.0},
        ],
        "description": "V ≈ V₀ for x > 0 — partial reflection & transmission (smooth edge)",
        "x_range": (-50, 50),
        "default_x0": -8.0,
        "default_k0": 4.0,
        "default_sigma": 1.5,
        "periodic": True,
    },
    "Periodic lattice": {
        "func": periodic_potential,
        "params": [
            {"name": "depth", "label": "Lattice depth", "min": 1.0, "max": 20.0, "default": 5.0, "step": 0.5},
            {"name": "period", "label": "Lattice period", "min": 1.0, "max": 10.0, "default": 3.0, "step": 0.5},
        ],
        "description": "Cosine lattice V = V₀[1−cos(2πx/a)]/2 — energy bands & Bloch waves",
        "x_range": (-15, 15),
        "default_x0": 0.0,
        "default_k0": 0.0,
        "default_sigma": 2.0,
        "periodic": True,
    },
    "Multi-well (band structure)": {
        "func": multi_well,
        "params": [
            {"name": "n_wells", "label": "Number of wells", "min": 1, "max": 20, "default": 5, "step": 1},
            {"name": "well_width", "label": "Well width", "min": 0.2, "max": 5.0, "default": 2.0, "step": 0.1},
            {"name": "depth", "label": "Well depth", "min": 1.0, "max": 100.0, "default": 10.0, "step": 1.0},
            {"name": "jitter", "label": "Position jitter", "min": 0.0, "max": 5.0, "default": 0.0, "step": 0.1,
             "help": "Max random shift of each well from its lattice site. "
                     "Automatically clamped to prevent overlap."},
            {"name": "seed", "label": "Random seed", "min": 0, "max": 9999, "default": 42, "step": 1,
             "show_if": {"param": "jitter", "gt": 0}},
        ],
        "description": "N rectangular wells with tunable jitter \u2014 band structure & disorder",
        "x_range": (-15, 15),
        "default_x0": 0.0,
        "default_k0": 0.0,
        "default_sigma": 1.5,
        "periodic": True,
    },
}
