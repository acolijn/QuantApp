"""
Quantum Mechanics Solver
========================
Numerical solutions to the time-dependent Schrödinger equation using the
Split-Operator FFT method.

The time evolution operator is  U(dt) = exp(-i H dt / ħ)
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

    # ── Potential setup ────────────────────────────────────────────────

    def set_potential(self, potential_func):
        """Set the potential V(x)."""
        self.V = potential_func(self.x)
        if np.any(self.V > 1e4):
            self.set_hard_wall_mask(wall_threshold=1e4)

    def set_hard_wall_mask(self, wall_threshold=1e4):
        """Create a mask that forces ψ=0 where V > threshold.

        Enforces Dirichlet boundary conditions for infinite wells.
        """
        self.hard_wall_mask = self.V < wall_threshold

    # ── State initialisation ───────────────────────────────────────────

    def set_gaussian_wavepacket(self, x0=0.0, sigma=1.0, k0=0.0):
        """Initialize ψ(x,0) as a Gaussian wavepacket.

        ψ(x) = (2πσ²)^{-1/4} exp(-(x-x0)²/(4σ²)) exp(ik0·x)
        """
        self.psi = ((2 * np.pi * sigma ** 2) ** (-0.25)
                     * np.exp(-(self.x - x0) ** 2 / (4 * sigma ** 2))
                     * np.exp(1j * k0 * self.x))
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
            self._set_numerical_eigenstate(n)

    def _set_harmonic_eigenstate(self, n):
        """Analytic harmonic oscillator eigenstates."""
        from scipy.special import hermite
        from math import factorial
        omega = 1.0
        xi = np.sqrt(self.mass * omega / self.hbar) * self.x
        Hn = hermite(n)
        norm = ((self.mass * omega / (np.pi * self.hbar)) ** 0.25
                / np.sqrt(2 ** n * factorial(n)))
        self.psi = (norm * Hn(xi) * np.exp(-xi ** 2 / 2)).astype(complex)
        self.psi /= np.sqrt(np.sum(np.abs(self.psi) ** 2) * self.dx)
        self.time = 0.0

    def _set_infinite_well_eigenstate(self, n):
        """Analytic infinite square well eigenstates."""
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

    # ── Eigenstate computation ─────────────────────────────────────────

    def compute_eigenstates(self, n_states=10):
        """Compute the lowest n energy eigenstates using finite differences.

        Returns (energies, eigenstates) where eigenstates[:,i] is the i-th state.
        """
        coeff = self.hbar ** 2 / (2 * self.mass * self.dx ** 2)
        diagonal = 2 * coeff + self.V
        off_diagonal = -coeff * np.ones(self.N - 1)

        n_states = min(n_states, self.N - 2)
        energies, states = eigh_tridiagonal(diagonal, off_diagonal,
                                             select='i', select_range=(0, n_states - 1))
        for i in range(states.shape[1]):
            states[:, i] /= np.sqrt(np.sum(states[:, i] ** 2) * self.dx)

        return energies, states

    # ── Time evolution ─────────────────────────────────────────────────

    def step(self, dt):
        """Advance the wavefunction by one time step dt using split-operator FFT.

        ψ(t+dt) = e^{-iVdt/2ħ} · FFT⁻¹[ e^{-iTdt/ħ} · FFT[ e^{-iVdt/2ħ} ψ(t) ] ]
        """
        V_eff = np.clip(self.V, -1e4, 1e4)

        # Half-step potential
        self.psi *= np.exp(-1j * V_eff * dt / (2 * self.hbar))

        # Full step kinetic (momentum space)
        psi_k = np.fft.fft(self.psi)
        psi_k *= np.exp(-1j * self.kinetic_energy_k * dt / self.hbar)
        self.psi = np.fft.ifft(psi_k)

        # Half-step potential
        self.psi *= np.exp(-1j * V_eff * dt / (2 * self.hbar))

        # Enforce hard walls with norm preservation
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

    # ── Observables ────────────────────────────────────────────────────

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
        psi_k = np.fft.fft(self.psi) * self.dx
        return np.real(np.sum(np.conj(psi_k) * self.hbar * self.k * psi_k) * self.dk / (2 * np.pi))

    def expectation_energy(self):
        """⟨E⟩ = ⟨T⟩ + ⟨V⟩."""
        psi_k = np.fft.fft(self.psi) * self.dx
        T = np.real(np.sum(np.conj(psi_k) * self.kinetic_energy_k * psi_k) * self.dk / (2 * np.pi))
        V = np.real(np.sum(np.conj(self.psi) * self.V * self.psi) * self.dx)
        return T + V

    def uncertainty_x(self):
        """Δx = √(⟨x²⟩ - ⟨x⟩²)."""
        return np.sqrt(max(0, self.expectation_x2() - self.expectation_x() ** 2))
