"""Tests for quantapp.solver — QuantumSystem."""

import numpy as np
import pytest

from quantapp.solver import QuantumSystem
from quantapp.potentials import (
    infinite_square_well, finite_square_well,
    potential_barrier, step_potential,
)


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


class TestSmoothedPotentialEnergy:
    """Verify energy conservation for potentials with smooth edges."""

    @pytest.mark.parametrize("name,pot_func,xr,x0,sigma,k0", [
        ("finite_well", lambda x: finite_square_well(x, width=6.0, depth=50.0),
         (-15, 15), 0.0, 1.0, 0.0),
        ("barrier", lambda x: potential_barrier(x, width=1.0, height=20.0),
         (-50, 50), -5.0, 1.0, 4.0),
        ("step", lambda x: step_potential(x, height=15.0),
         (-50, 50), -8.0, 1.5, 4.0),
    ])
    def test_energy_conserved(self, name, pot_func, xr, x0, sigma, k0):
        """⟨E⟩ should be conserved with smooth potential edges.

        Evolution time is 8s — long enough to demonstrate the physics
        but short enough that scattered waves stay inside the absorbing
        boundary layer.
        """
        qs = QuantumSystem(x_min=xr[0], x_max=xr[1], N=1024)
        qs.set_potential(pot_func)
        qs.set_gaussian_wavepacket(x0=x0, sigma=sigma, k0=k0)
        E0 = qs.expectation_energy()
        qs.evolve(total_time=8.0, dt=0.01)
        E_final = qs.expectation_energy()
        assert abs(E_final - E0) / abs(E0) < 1e-3, (
            f"{name}: Energy drifted {E0:.4f} → {E_final:.4f} "
            f"(relative: {abs(E_final - E0) / abs(E0):.2e})"
        )


class TestAbsorbingBoundaries:
    """Verify absorbing boundaries prevent N-dependent aliasing."""

    def test_absorber_inactive_for_dst(self):
        """DST propagator ignores the absorber (hard walls enforce BCs)."""
        qs = QuantumSystem(x_min=-10, x_max=10, N=512)
        qs.set_potential(lambda x: infinite_square_well(x, width=10.0))
        qs.set_gaussian_wavepacket(x0=0, sigma=1.0, k0=3.0)
        E0 = qs.expectation_energy()
        qs.evolve(total_time=5.0, dt=0.01)
        assert abs(qs.norm() - 1.0) < 1e-8
        assert abs(qs.expectation_energy() - E0) / abs(E0) < 1e-4

    @pytest.mark.parametrize("pot_func,xr,x0,sigma,k0", [
        (lambda x: potential_barrier(x, width=1.0, height=20.0),
         (-50, 50), -5.0, 1.0, 4.0),
        (lambda x: step_potential(x, height=15.0),
         (-50, 50), -8.0, 1.5, 4.0),
    ])
    def test_n_independent(self, pot_func, xr, x0, sigma, k0):
        """Energy at N=1024 and N=2048 must agree within 1%.

        This is the regression test for the N=2048 aliasing bug.
        """
        energies = {}
        for N in [1024, 2048]:
            qs = QuantumSystem(x_min=xr[0], x_max=xr[1], N=N)
            qs.set_potential(pot_func)
            qs.set_gaussian_wavepacket(x0=x0, sigma=sigma, k0=k0)
            qs.evolve(total_time=5.0, dt=0.01)
            energies[N] = qs.expectation_energy()
        rel_diff = abs(energies[1024] - energies[2048]) / abs(energies[1024])
        assert rel_diff < 1e-2, (
            f"N-dependent energy: N=1024→{energies[1024]:.4f}, "
            f"N=2048→{energies[2048]:.4f} (rel={rel_diff:.2e})"
        )
