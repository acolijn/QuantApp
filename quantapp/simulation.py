"""
Simulation helpers
==================
Caching, wavefunction initialisation, and pre-computation of animation frames.
All Streamlit cache decorators live here so the solver stays framework-agnostic.
"""

import numpy as np
import streamlit as st

from quantapp.solver import QuantumSystem
from quantapp.potentials import POTENTIALS


# ── Eigenstate caching ─────────────────────────────────────────────────

# Bump _CACHE_VERSION when potential code changes to invalidate stale
# Streamlit caches (st.cache_data does not track transitive dependencies).
_CACHE_VERSION = 4


def _resolve_potential_func(potential_name, pot_kwargs=()):
    """Return the potential callable, applying pot_kwargs for the given potential.

    pot_kwargs is a tuple of (name, value) pairs produced by the sidebar.
    When non-empty it is converted to keyword arguments for the raw
    potential function.
    """
    pot_info = POTENTIALS[potential_name]
    kwargs = dict(pot_kwargs)
    if kwargs:
        func = pot_info["func"]
        return lambda x: func(x, **kwargs)
    return pot_info["func"]


@st.cache_data
def compute_eigenstates_cached(potential_name, x_min, x_max, N,
                               n_eigen=10, pot_kwargs=(),
                               _v=_CACHE_VERSION):
    """Compute and cache eigenstates for the given potential."""
    pot_info = POTENTIALS[potential_name]
    func = _resolve_potential_func(potential_name, pot_kwargs)
    qs = QuantumSystem(x_min=x_min, x_max=x_max, N=N)
    qs.set_potential(func, periodic=pot_info.get("periodic", False))
    try:
        energies, states = qs.compute_eigenstates(n_states=n_eigen)
    except Exception:
        energies, states = np.array([]), np.array([])
    return qs.x.copy(), qs.V.copy(), energies, states


# ── Shared wavefunction initialisation ─────────────────────────────────

def initialize_wavefunction(qs, init_mode, eigenstates, *,
                            x0=0.0, sigma=1.0, k0=0.0,
                            eigen_n=0, n1=0, n2=1, ratio=0.5):
    """Set the wavefunction on *qs* according to *init_mode*.

    This is the single source of truth — used by both the static preview
    and the animation pre-computation.
    """
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


def create_quantum_system(potential_name, x_min, x_max, N, init_mode,
                          eigenstates, pot_kwargs=(), **init_kwargs):
    """Build a QuantumSystem and initialise its wavefunction."""
    pot_info = POTENTIALS[potential_name]
    func = _resolve_potential_func(potential_name, pot_kwargs)
    qs = QuantumSystem(x_min=x_min, x_max=x_max, N=N)
    qs.set_potential(func, periodic=pot_info.get("periodic", False))
    initialize_wavefunction(qs, init_mode, eigenstates, **init_kwargs)
    return qs


# ── Animation pre-computation ──────────────────────────────────────────

@st.cache_data
def precompute_animation(potential_name, init_mode_key,
                         x0_v, sigma_v, k0_v,
                         eigen_n_v, n1_v, n2_v, ratio_v,
                         _dt, _steps_per_frame, _n_frames,
                         _N, _x_min, _x_max, _stride,
                         pot_kwargs=(),
                         _v=_CACHE_VERSION):
    """Pre-compute all animation frames (cached).

    Returns arrays: probs, re_parts, im_parts, times, energies,
                    exp_x, delta_x, norms
    """
    _, _, _, eigs = compute_eigenstates_cached(
        potential_name, _x_min, _x_max, _N, n_eigen=16,
        pot_kwargs=pot_kwargs,
    )
    qs = create_quantum_system(
        potential_name, _x_min, _x_max, _N, init_mode_key, eigs,
        pot_kwargs=pot_kwargs,
        x0=x0_v, sigma=sigma_v, k0=k0_v,
        eigen_n=eigen_n_v, n1=n1_v, n2=n2_v, ratio=ratio_v,
    )

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

    keys = ("prob", "re", "im", "time", "energy", "exp_x", "delta_x", "norm")
    arrays = {k: np.array([f[k] for f in frames]) for k in keys}
    return (arrays["prob"], arrays["re"], arrays["im"],
            arrays["time"], arrays["energy"],
            arrays["exp_x"], arrays["delta_x"], arrays["norm"])
