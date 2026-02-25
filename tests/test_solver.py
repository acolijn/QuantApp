"""Tests for quantapp.solver — QuantumSystem."""

import numpy as np
import pytest

from quantapp.solver import QuantumSystem
from quantapp.potentials import infinite_square_well


class TestQuantumSystemBasics:
    """Sanity checks on grid setup and initialisation."""

    def test_grid_shape(self):
        qs = QuantumSystem(x_min=-5, x_max=5, N=256)
        assert qs.x.shape == (256,)
        assert qs.k.shape == (256,)

    def test_gaussian_normalised(self):
        qs = QuantumSystem(N=1024)
        qs.set_potential(lambda x: np.zeros_like(x))
        qs.set_gaussian_wavepacket(x0=0, sigma=1.0, k0=0)
        assert abs(qs.norm() - 1.0) < 1e-6

    def test_expectation_x_centered(self):
        qs = QuantumSystem(N=1024)
        qs.set_potential(lambda x: np.zeros_like(x))
        qs.set_gaussian_wavepacket(x0=2.0, sigma=1.0, k0=0)
        assert abs(qs.expectation_x() - 2.0) < 0.05


class TestTimeEvolution:
    """Verify that the split-operator propagator is unitary."""

    def test_norm_preserved_free(self):
        qs = QuantumSystem(N=512)
        qs.set_potential(lambda x: np.zeros_like(x))
        qs.set_gaussian_wavepacket(x0=0, sigma=1.0, k0=3.0)
        qs.evolve(total_time=1.0, dt=0.01)
        assert abs(qs.norm() - 1.0) < 1e-4

    def test_norm_preserved_harmonic(self):
        qs = QuantumSystem(x_min=-10, x_max=10, N=512)
        qs.set_potential(lambda x: 0.5 * x ** 2)
        qs.set_gaussian_wavepacket(x0=-3, sigma=0.5, k0=0)
        qs.evolve(total_time=2.0, dt=0.01)
        assert abs(qs.norm() - 1.0) < 1e-4

    def test_energy_conserved_harmonic(self):
        qs = QuantumSystem(x_min=-10, x_max=10, N=512)
        qs.set_potential(lambda x: 0.5 * x ** 2)
        qs.set_gaussian_wavepacket(x0=-3, sigma=0.5, k0=0)
        E0 = qs.expectation_energy()
        qs.evolve(total_time=2.0, dt=0.01)
        assert abs(qs.expectation_energy() - E0) / abs(E0) < 1e-3


class TestEigenstates:
    """Test eigenstate computation."""

    def test_harmonic_eigenvalues(self):
        """Harmonic oscillator: E_n = (n + 1/2) ħω."""
        qs = QuantumSystem(x_min=-10, x_max=10, N=512)
        qs.set_potential(lambda x: 0.5 * x ** 2)
        energies, _ = qs.compute_eigenstates(n_states=5)
        expected = np.arange(5) + 0.5
        np.testing.assert_allclose(energies, expected, atol=0.02)

    def test_eigenstates_orthonormal(self):
        qs = QuantumSystem(x_min=-10, x_max=10, N=512)
        qs.set_potential(lambda x: 0.5 * x ** 2)
        _, states = qs.compute_eigenstates(n_states=4)
        # Check orthonormality
        overlap = states.T @ states * qs.dx
        np.testing.assert_allclose(overlap, np.eye(4), atol=0.01)


class TestDSTPropagator:
    """Tests for the DST-based propagator used with hard-wall potentials."""

    def _make_infinite_well_system(self, N=512, width=10.0):
        qs = QuantumSystem(x_min=-10, x_max=10, N=N)
        qs.set_potential(lambda x: infinite_square_well(x, width=width))
        return qs

    def test_dst_activated(self):
        """DST propagator should be activated for the infinite well."""
        qs = self._make_infinite_well_system()
        assert hasattr(qs, '_use_dst') and qs._use_dst

    def test_norm_preserved_infinite_well(self):
        qs = self._make_infinite_well_system()
        qs.set_gaussian_wavepacket(x0=0, sigma=1.0, k0=3.0)
        qs.evolve(total_time=5.0, dt=0.01)
        assert abs(qs.norm() - 1.0) < 1e-8

    def test_energy_conserved_infinite_well(self):
        """⟨E⟩ should be conserved to high precision with the DST propagator."""
        qs = self._make_infinite_well_system()
        qs.set_gaussian_wavepacket(x0=0, sigma=1.0, k0=3.0)
        E0 = qs.expectation_energy()
        qs.evolve(total_time=5.0, dt=0.01)
        E_final = qs.expectation_energy()
        # With DST this should be conserved to ~1e-6 or better
        assert abs(E_final - E0) / abs(E0) < 1e-4, (
            f"Energy drifted: {E0:.6f} → {E_final:.6f} "
            f"(relative: {abs(E_final - E0) / abs(E0):.2e})"
        )

    def test_energy_conserved_long_run(self):
        """Even over a long simulation, energy should stay stable."""
        qs = self._make_infinite_well_system()
        qs.set_gaussian_wavepacket(x0=-2.0, sigma=0.8, k0=0.0)
        E0 = qs.expectation_energy()
        qs.evolve(total_time=20.0, dt=0.01)
        E_final = qs.expectation_energy()
        assert abs(E_final - E0) / abs(E0) < 1e-3, (
            f"Energy drifted over long run: {E0:.6f} → {E_final:.6f}"
        )
