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
from scipy.fft import dst, idst
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
        If the interior region is contiguous, sets up a DST-based
        kinetic propagator that naturally enforces ψ=0 at the walls.
        """
        self.hard_wall_mask = self.V < wall_threshold
        self._setup_dst_propagator()

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

    # ── DST propagator setup ────────────────────────────────────────────

    def _setup_dst_propagator(self):
        """Set up a DST-based kinetic propagator for hard-wall potentials.

        The Discrete Sine Transform naturally enforces ψ=0 at the walls
        (Dirichlet BCs), eliminating the amplitude leakage that occurs
        with the FFT's periodic BCs.  This preserves both norm and energy
        to machine precision.

        Requires the interior region (V < threshold) to be contiguous.
        Falls back to FFT with simple masking otherwise.
        """
        interior_idx = np.where(self.hard_wall_mask)[0]
        if len(interior_idx) < 2:
            self._use_dst = False
            return

        start, end = interior_idx[0], interior_idx[-1] + 1  # exclusive end

        # Interior must be contiguous for DST
        if end - start != len(interior_idx):
            self._use_dst = False
            return

        self._use_dst = True
        self._interior_slice = slice(start, end)
        N_int = end - start

        # The DST-I assumes ψ=0 at positions one grid spacing outside
        # the data, so the effective well width is (N_int + 1) * dx.
        L_well = (N_int + 1) * self.dx
        n_modes = np.arange(1, N_int + 1)
        self._dst_kinetic = (
            (self.hbar ** 2 * (n_modes * np.pi / L_well) ** 2)
            / (2 * self.mass)
        )

    # ── Time evolution ─────────────────────────────────────────────────

    def step(self, dt):
        """Advance ψ by one time step dt.

        Automatically selects the DST-based kinetic propagator for
        hard-wall potentials, or the standard FFT propagator otherwise.
        """
        if getattr(self, '_use_dst', False):
            self._step_dst(dt)
        else:
            self._step_fft(dt)
        self.time += dt

    def _step_fft(self, dt):
        """Split-operator FFT step (for open / periodic systems)."""
        V_eff = np.clip(self.V, -1e4, 1e4)

        self.psi *= np.exp(-1j * V_eff * dt / (2 * self.hbar))

        psi_k = np.fft.fft(self.psi)
        psi_k *= np.exp(-1j * self.kinetic_energy_k * dt / self.hbar)
        self.psi = np.fft.ifft(psi_k)

        self.psi *= np.exp(-1j * V_eff * dt / (2 * self.hbar))

        # For hard walls without a contiguous interior (no DST),
        # apply the mask but do NOT renormalise — renormalisation
        # pumps energy into the system, causing the drift.
        if hasattr(self, 'hard_wall_mask'):
            self.psi *= self.hard_wall_mask

    def _step_dst(self, dt):
        """Split-operator DST step (for hard-wall bounded systems).

        Uses Discrete Sine Transform (type I) for the kinetic step.
        The DST basis functions are sin(nπx/L) which are exactly zero
        at the walls, so there is no amplitude leakage and both norm
        and energy are conserved to machine precision.
        """
        sl = self._interior_slice
        V_int = self.V[sl]

        # Half-step potential (interior only)
        self.psi[sl] *= np.exp(-1j * V_int * dt / (2 * self.hbar))

        # Full step kinetic via DST-I
        psi_int = self.psi[sl]
        psi_k = dst(psi_int.real, type=1) + 1j * dst(psi_int.imag, type=1)
        psi_k *= np.exp(-1j * self._dst_kinetic * dt / self.hbar)
        self.psi[sl] = idst(psi_k.real, type=1) + 1j * idst(psi_k.imag, type=1)

        # Half-step potential
        self.psi[sl] *= np.exp(-1j * V_int * dt / (2 * self.hbar))

        # Ensure zero outside walls (should already be, but be safe)
        self.psi[:sl.start] = 0
        self.psi[sl.stop:] = 0

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
        if getattr(self, '_use_dst', False):
            return self._expectation_p_dst()
        psi_k = np.fft.fft(self.psi) * self.dx
        return np.real(np.sum(np.conj(psi_k) * self.hbar * self.k * psi_k) * self.dk / (2 * np.pi))

    def _expectation_p_dst(self):
        """⟨p⟩ via DST — vanishes by symmetry of sin basis, but compute anyway."""
        # In the DST basis, p is off-diagonal; compute via finite differences
        psi_int = self.psi[self._interior_slice]
        dpsi = np.gradient(psi_int, self.dx)
        return np.real(-1j * self.hbar * np.sum(np.conj(psi_int) * dpsi) * self.dx)

    def expectation_energy(self):
        """⟨E⟩ = ⟨T⟩ + ⟨V⟩.

        Uses the DST basis for ⟨T⟩ when hard walls are present,
        avoiding Gibbs artifacts from the FFT at wall boundaries.
        """
        if getattr(self, '_use_dst', False):
            return self._expectation_energy_dst()
        psi_k = np.fft.fft(self.psi) * self.dx
        T = np.real(np.sum(np.conj(psi_k) * self.kinetic_energy_k * psi_k) * self.dk / (2 * np.pi))
        V = np.real(np.sum(np.conj(self.psi) * self.V * self.psi) * self.dx)
        return T + V

    def _expectation_energy_dst(self):
        """⟨E⟩ via DST for hard-wall systems.

        DST Parseval relation: Σ_n |ψ_n|² = Σ_k |ψ̃_k|² / (2(N+1))
        so ⟨T⟩ = [dx / (2(N+1))] Σ_k T_k |ψ̃_k|²
        """
        sl = self._interior_slice
        psi_int = self.psi[sl]
        N_int = len(psi_int)

        # ⟨T⟩ in DST basis
        psi_k = dst(psi_int.real, type=1) + 1j * dst(psi_int.imag, type=1)
        T = np.real(np.sum(np.abs(psi_k) ** 2 * self._dst_kinetic)) * self.dx / (2 * (N_int + 1))

        # ⟨V⟩ in position space (only interior contributes; V·ψ = 0 at walls)
        V = np.real(np.sum(np.conj(psi_int) * self.V[sl] * psi_int)) * self.dx

        return T + V

    def uncertainty_x(self):
        """Δx = √(⟨x²⟩ - ⟨x⟩²)."""
        return np.sqrt(max(0, self.expectation_x2() - self.expectation_x() ** 2))
