"""
Quantum Mechanics Solver
========================
Numerical solutions to the time-dependent Schrödinger equation using the
Split-Operator FFT method.

The idea: The time evolution operator is  U(dt) = exp(-i H dt / ħ)
where H = T + V (kinetic + potential).

Using the Trotter-Suzuki decomposition:
    U(dt) ≈ exp(-i V dt/2ħ) · exp(-i T dt/ħ) · exp(-i V dt/2ħ)

- exp(-i V dt/2ħ) is diagonal in position space
- exp(-i T dt/ħ) is diagonal in momentum space (apply via FFT)

This gives O(dt³) accuracy per step and is unconditionally stable + unitary.

We work in natural units where ħ = 1, m = 1 (adjustable).
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal


class QuantumSystem:
    """A 1D quantum system with configurable potential."""

    def __init__(self, x_min=-15.0, x_max=15.0, N=1024, mass=1.0, hbar=1.0):
        self.hbar = hbar
        self.mass = mass
        self.N = N
        self.x = np.linspace(x_min, x_max, N, endpoint=False)
        self.dx = self.x[1] - self.x[0]
        self.L = x_max - x_min

        # Momentum space grid
        self.dk = 2 * np.pi / self.L
        self.k = np.fft.fftfreq(N, d=self.dx) * 2 * np.pi

        # Kinetic energy in momentum space: T = ħ²k²/(2m)
        self.kinetic_energy_k = (self.hbar ** 2 * self.k ** 2) / (2 * self.mass)

        # State
        self.psi = np.zeros(N, dtype=complex)
        self.V = np.zeros(N)
        self.time = 0.0

    def set_potential(self, potential_func):
        """Set the potential V(x)."""
        self.V = potential_func(self.x)
        # Automatically set up hard-wall enforcement for infinite potentials
        if np.any(self.V > 1e4):
            self.set_hard_wall_mask(wall_threshold=1e4)

    def set_gaussian_wavepacket(self, x0=0.0, sigma=1.0, k0=0.0):
        """Initialize ψ(x,0) as a Gaussian wavepacket.
        
        ψ(x) = (2πσ²)^{-1/4} exp(-(x-x0)²/(4σ²)) exp(ik0·x)
        """
        self.psi = ((2 * np.pi * sigma ** 2) ** (-0.25)
                     * np.exp(-(self.x - x0) ** 2 / (4 * sigma ** 2))
                     * np.exp(1j * k0 * self.x))
        # Enforce hard walls on initial state
        if hasattr(self, 'hard_wall_mask'):
            self.psi *= self.hard_wall_mask
        self.psi /= np.sqrt(np.sum(np.abs(self.psi) ** 2) * self.dx)
        self.time = 0.0

    def set_eigenstate(self, n, potential_type="harmonic"):
        """Initialize as the n-th energy eigenstate (0-indexed)."""
        if potential_type == "harmonic":
            self._set_harmonic_eigenstate(n)
        elif potential_type == "infinite_well":
            self._set_infinite_well_eigenstate(n)
        else:
            # Numerical eigenstate via finite differences
            self._set_numerical_eigenstate(n)

    def _set_harmonic_eigenstate(self, n):
        """Analytic harmonic oscillator eigenstates."""
        from scipy.special import hermite
        from math import factorial
        # For ω=1, m=1, ħ=1: ψ_n(x) = (mω/πħ)^{1/4} * 1/√(2^n n!) * H_n(ξ) * exp(-ξ²/2)
        omega = 1.0  # Assume V = 0.5 * m * omega^2 * x^2 with omega=1
        xi = np.sqrt(self.mass * omega / self.hbar) * self.x
        Hn = hermite(n)
        norm = ((self.mass * omega / (np.pi * self.hbar)) ** 0.25
                / np.sqrt(2 ** n * factorial(n)))
        self.psi = (norm * Hn(xi) * np.exp(-xi ** 2 / 2)).astype(complex)
        self.psi /= np.sqrt(np.sum(np.abs(self.psi) ** 2) * self.dx)
        self.time = 0.0

    def _set_infinite_well_eigenstate(self, n):
        """Analytic infinite square well eigenstates."""
        # Well from x_min to x_max where V=0
        L = self.x[-1] - self.x[0]
        x_shifted = self.x - self.x[0]
        self.psi = (np.sqrt(2 / L) * np.sin((n + 1) * np.pi * x_shifted / L)).astype(complex)
        self.psi /= np.sqrt(np.sum(np.abs(self.psi) ** 2) * self.dx)
        self.time = 0.0

    def _set_numerical_eigenstate(self, n):
        """Compute eigenstate numerically using finite difference Hamiltonian."""
        energies, states = self.compute_eigenstates(n_states=n + 1)
        self.psi = states[:, n].astype(complex)
        self.psi /= np.sqrt(np.sum(np.abs(self.psi) ** 2) * self.dx)
        self.time = 0.0

    def compute_eigenstates(self, n_states=10):
        """Compute the lowest n energy eigenstates using finite differences.
        
        Returns (energies, eigenstates) where eigenstates[:,i] is the i-th state.
        """
        # Tridiagonal Hamiltonian: H = -ħ²/(2m dx²) * tridiag(1,-2,1) + diag(V)
        coeff = self.hbar ** 2 / (2 * self.mass * self.dx ** 2)
        diagonal = 2 * coeff + self.V
        off_diagonal = -coeff * np.ones(self.N - 1)

        n_states = min(n_states, self.N - 2)
        energies, states = eigh_tridiagonal(diagonal, off_diagonal,
                                             select='i', select_range=(0, n_states - 1))
        # Normalize
        for i in range(states.shape[1]):
            states[:, i] /= np.sqrt(np.sum(states[:, i] ** 2) * self.dx)

        return energies, states

    def set_hard_wall_mask(self, wall_threshold=1e4):
        """Create a mask that forces ψ=0 where V > threshold.
        
        This enforces Dirichlet boundary conditions for infinite wells,
        which is more physical than relying on a large finite potential.
        """
        self.hard_wall_mask = self.V < wall_threshold

    def step(self, dt):
        """Advance the wavefunction by one time step dt using split-operator FFT.
        
        ψ(t+dt) = e^{-iVdt/2ħ} · FFT⁻¹[ e^{-iTdt/ħ} · FFT[ e^{-iVdt/2ħ} ψ(t) ] ]
        """
        # Clamp the potential phase to avoid overflow with very large V (e.g. 1e6)
        V_eff = np.clip(self.V, -1e4, 1e4)

        # Half-step in position space (potential)
        self.psi *= np.exp(-1j * V_eff * dt / (2 * self.hbar))

        # Full step in momentum space (kinetic)
        psi_k = np.fft.fft(self.psi)
        psi_k *= np.exp(-1j * self.kinetic_energy_k * dt / self.hbar)
        self.psi = np.fft.ifft(psi_k)

        # Half-step in position space (potential)
        self.psi *= np.exp(-1j * V_eff * dt / (2 * self.hbar))

        # Enforce hard walls: ψ = 0 where V is "infinite"
        # The FFT kinetic step leaks amplitude through the walls (periodic BCs).
        # Zeroing the leaked part and renormalizing approximates perfect reflection,
        # preserving unitarity so the wavefunction doesn't slowly vanish.
        if hasattr(self, 'hard_wall_mask'):
            norm_before = np.sum(np.abs(self.psi) ** 2) * self.dx
            self.psi *= self.hard_wall_mask
            norm_after = np.sum(np.abs(self.psi) ** 2) * self.dx
            if norm_after > 1e-15:
                self.psi *= np.sqrt(norm_before / norm_after)

        self.time += dt

    def evolve(self, total_time, dt=0.01):
        """Evolve the system for a given total time."""
        n_steps = int(total_time / dt)
        for _ in range(n_steps):
            self.step(dt)

    def probability_density(self):
        """Return |ψ(x)|²."""
        return np.abs(self.psi) ** 2

    def norm(self):
        """Return ∫|ψ|² dx (should stay ≈ 1)."""
        return np.sum(self.probability_density()) * self.dx

    def expectation_x(self):
        """⟨x⟩ = ∫ ψ* x ψ dx."""
        return np.real(np.sum(np.conj(self.psi) * self.x * self.psi) * self.dx)

    def expectation_x2(self):
        """⟨x²⟩."""
        return np.real(np.sum(np.conj(self.psi) * self.x ** 2 * self.psi) * self.dx)

    def expectation_p(self):
        """⟨p⟩ computed in momentum space."""
        psi_k = np.fft.fft(self.psi) * self.dx  # Normalize FFT
        return np.real(np.sum(np.conj(psi_k) * self.hbar * self.k * psi_k) * self.dk / (2 * np.pi))

    def expectation_energy(self):
        """⟨E⟩ = ⟨T⟩ + ⟨V⟩."""
        # Kinetic in k-space
        psi_k = np.fft.fft(self.psi) * self.dx
        T = np.real(np.sum(np.conj(psi_k) * self.kinetic_energy_k * psi_k) * self.dk / (2 * np.pi))
        # Potential in x-space
        V = np.real(np.sum(np.conj(self.psi) * self.V * self.psi) * self.dx)
        return T + V

    def uncertainty_x(self):
        """Δx = √(⟨x²⟩ - ⟨x⟩²)."""
        return np.sqrt(max(0, self.expectation_x2() - self.expectation_x() ** 2))


# ── Predefined Potentials ──────────────────────────────────────────────

def free_particle(x):
    """V(x) = 0"""
    return np.zeros_like(x)


def infinite_square_well(x, width=10.0):
    """Infinite square well centered at origin."""
    V = np.zeros_like(x)
    V[np.abs(x) > width / 2] = 1e6  # Large but finite for numerics
    return V


def finite_square_well(x, width=6.0, depth=50.0):
    """Finite square well: V = -depth inside, 0 outside."""
    V = np.zeros_like(x)
    V[np.abs(x) <= width / 2] = -depth
    return V


def harmonic_oscillator(x, omega=1.0, mass=1.0):
    """V(x) = ½mω²x²"""
    return 0.5 * mass * omega ** 2 * x ** 2


def double_well(x, separation=3.0, depth=8.0):
    """Double well: two Gaussian wells."""
    return -depth * (np.exp(-((x - separation / 2) ** 2))
                     + np.exp(-((x + separation / 2) ** 2)))


def potential_barrier(x, width=1.0, height=20.0):
    """Rectangular barrier for tunneling demonstrations."""
    V = np.zeros_like(x)
    V[np.abs(x) < width / 2] = height
    return V


def asymmetric_double_well(x, offset=2.0):
    """Asymmetric double well: x⁴ - 5x² + offset·x."""
    return 0.05 * (x ** 4 - 5 * x ** 2 + offset * x) + 5


def morse_potential(x, D=10.0, a=0.5, x0=0.0):
    """Morse potential: V = D(1 - e^{-a(x-x0)})² — models molecular bonds."""
    return D * (1 - np.exp(-a * (x - x0))) ** 2


def step_potential(x, height=15.0):
    """Step potential: V = height for x > 0."""
    V = np.zeros_like(x)
    V[x > 0] = height
    return V


# Dictionary of available potentials with display names and default parameters
POTENTIALS = {
    "Free particle": {
        "func": free_particle,
        "description": "V(x) = 0 — Free propagation, wavepacket spreading",
        "x_range": (-20, 20),
        "default_x0": -5.0,
        "default_k0": 3.0,
        "default_sigma": 1.5,
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
        "description": "Finite depth well — bound & scattering states",
        "x_range": (-15, 15),
        "default_x0": 0.0,
        "default_k0": 0.0,
        "default_sigma": 1.0,
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
    },
    "Potential barrier (tunneling)": {
        "func": lambda x: potential_barrier(x, width=1.0, height=20.0),
        "description": "Rectangular barrier — quantum tunneling demonstration",
        "x_range": (-15, 15),
        "default_x0": -5.0,
        "default_k0": 4.0,
        "default_sigma": 1.0,
    },
    "Morse potential": {
        "func": lambda x: morse_potential(x, D=12.0, a=0.4, x0=-3.0),
        "description": "Anharmonic oscillator — models molecular vibrations",
        "x_range": (-15, 15),
        "default_x0": -3.0,
        "default_k0": 0.0,
        "default_sigma": 0.8,
    },
    "Step potential": {
        "func": lambda x: step_potential(x, height=15.0),
        "description": "V = V₀ for x > 0 — partial reflection & transmission",
        "x_range": (-20, 20),
        "default_x0": -8.0,
        "default_k0": 4.0,
        "default_sigma": 1.5,
    },
}
