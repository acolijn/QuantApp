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


# ── Registry ───────────────────────────────────────────────────────────
# Each entry carries: func, description, x_range, and wavepacket defaults.

POTENTIALS = {
    "Free particle": {
        "func": free_particle,
        "description": "V(x) = 0 — Free propagation, wavepacket spreading",
        "x_range": (-40, 40),
        "default_x0": -5.0,
        "default_k0": 3.0,
        "default_sigma": 1.5,
        "periodic": True,
    },
    "Infinite square well": {
        "func": lambda x: infinite_square_well(x, width=10.0),
        "description": "Particle in a box — quantized energy levels",
        "x_range": (-10, 10),
        "default_x0": 0.0,
        "default_k0": 0.0,
        "default_sigma": 1.0,
    },
    "Finite square well": {
        "func": lambda x: finite_square_well(x, width=6.0, depth=50.0),
        "description": "Finite depth well — bound & scattering states (smooth edges)",
        "x_range": (-15, 15),
        "default_x0": 0.0,
        "default_k0": 0.0,
        "default_sigma": 1.0,
        "periodic": True,
    },
    "Harmonic oscillator": {
        "func": lambda x: harmonic_oscillator(x, omega=1.0),
        "description": "V = ½mω²x² — Equally spaced energy levels",
        "x_range": (-10, 10),
        "default_x0": -3.0,
        "default_k0": 0.0,
        "default_sigma": 0.5,
    },
    "Double well": {
        "func": lambda x: double_well(x, separation=4.0, depth=10.0),
        "description": "Two symmetric wells — quantum tunneling between wells",
        "x_range": (-10, 10),
        "default_x0": -2.0,
        "default_k0": 0.0,
        "default_sigma": 0.7,
        "periodic": True,
    },
    "Potential barrier (tunneling)": {
        "func": lambda x: potential_barrier(x, width=1.0, height=20.0),
        "description": "Rectangular barrier — quantum tunneling (smooth edges)",
        "x_range": (-50, 50),
        "default_x0": -5.0,
        "default_k0": 4.0,
        "default_sigma": 1.0,
        "periodic": True,
    },
    "Morse potential": {
        "func": lambda x: morse_potential(x, D=12.0, a=0.4, x0=-3.0),
        "description": "Anharmonic oscillator — models molecular vibrations",
        "x_range": (-15, 15),
        "default_x0": -3.0,
        "default_k0": 0.0,
        "default_sigma": 0.8,
        "periodic": True,
    },
    "Step potential": {
        "func": lambda x: step_potential(x, height=15.0),
        "description": "V ≈ V₀ for x > 0 — partial reflection & transmission (smooth edge)",
        "x_range": (-50, 50),
        "default_x0": -8.0,
        "default_k0": 4.0,
        "default_sigma": 1.5,
        "periodic": True,
    },
    "Periodic lattice": {
        "func": lambda x: periodic_potential(x, depth=5.0, period=3.0),
        "description": "Cosine lattice V = V₀[1−cos(2πx/a)]/2 — energy bands & Bloch waves",
        "x_range": (-15, 15),
        "default_x0": 0.0,
        "default_k0": 0.0,
        "default_sigma": 2.0,
        "periodic": True,
    },
}
