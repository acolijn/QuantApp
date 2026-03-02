"""Tests for quantapp.potentials."""

import numpy as np
import pytest

from quantapp.potentials import (
    POTENTIALS,
    free_particle,
    harmonic_oscillator,
    infinite_square_well,
    potential_barrier,
    morse_potential,
    finite_square_well,
    step_potential,
    periodic_potential,
    multi_well,
    smooth_step,
    smooth_rect,
)


class TestPotentialFunctions:

    def test_free_particle_zero(self):
        x = np.linspace(-10, 10, 100)
        np.testing.assert_array_equal(free_particle(x), np.zeros(100))

    def test_harmonic_oscillator_symmetric(self):
        x = np.linspace(-5, 5, 101)
        V = harmonic_oscillator(x)
        # V(x) == V(-x)
        np.testing.assert_allclose(V, V[::-1], atol=1e-12)

    def test_infinite_well_walls(self):
        x = np.linspace(-10, 10, 1000)
        V = infinite_square_well(x, width=6.0)
        assert np.all(V[np.abs(x) > 3.0] == 1e6)
        assert np.all(V[np.abs(x) <= 3.0] == 0.0)

    def test_barrier_height(self):
        x = np.linspace(-5, 5, 1000)
        V = potential_barrier(x, width=2.0, height=30.0)
        # Center of barrier should be very close to the full height
        assert abs(V[len(x) // 2] - 30.0) < 0.1
        # Far from edges should be near zero
        assert V[0] < 0.1
        assert V[-1] < 0.1

    def test_morse_minimum(self):
        x = np.linspace(-10, 10, 1000)
        V = morse_potential(x, D=10.0, a=0.5, x0=0.0)
        assert abs(V[np.argmin(np.abs(x))]) < 0.01  # min near x0

    def test_periodic_potential_range(self):
        """Periodic potential oscillates between -depth and 0."""
        x = np.linspace(-15, 15, 10001)
        V = periodic_potential(x, depth=5.0, period=3.0)
        assert np.max(V) <= 1e-10  # never positive
        assert abs(np.min(V) - (-5.0)) < 0.01  # trough = -depth
        assert abs(np.max(V)) < 0.01  # peak = 0

    def test_periodic_potential_periodicity(self):
        """V(x + a) == V(x) for lattice constant a."""
        a = 3.0
        x = np.linspace(0, a, 500, endpoint=False)
        V1 = periodic_potential(x, depth=5.0, period=a)
        V2 = periodic_potential(x + a, depth=5.0, period=a)
        np.testing.assert_allclose(V1, V2, atol=1e-12)

    def test_periodic_potential_symmetric(self):
        """V(x) == V(-x) (even function)."""
        x = np.linspace(-15, 15, 1001)
        V = periodic_potential(x, depth=5.0, period=3.0)
        np.testing.assert_allclose(V, V[::-1], atol=1e-10)


class TestSmoothEdges:
    """Verify smooth potential properties."""

    def test_smooth_step_limits(self):
        x = np.linspace(-10, 10, 1001)  # odd count so x=0 is on the grid
        s = smooth_step(x, 0.0, 15.0)
        assert s[0] < 1e-6  # ~0 far left
        assert s[-1] > 1 - 1e-6  # ~1 far right
        assert abs(s[len(x) // 2] - 0.5) < 0.01  # 0.5 at edge

    def test_finite_well_depth(self):
        x = np.linspace(-15, 15, 1000)
        V = finite_square_well(x, width=6.0, depth=50.0)
        # Center should be close to -depth
        assert abs(V[len(x) // 2] - (-50.0)) < 0.5
        # Far edges should be close to 0
        assert abs(V[0]) < 0.5
        assert abs(V[-1]) < 0.5

    def test_step_potential_levels(self):
        x = np.linspace(-20, 20, 1000)
        V = step_potential(x, height=15.0)
        assert V[0] < 0.1  # ~0 far left
        assert abs(V[-1] - 15.0) < 0.1  # ~15 far right

    def test_smooth_potentials_symmetric(self):
        """finite_square_well and barrier should be symmetric."""
        x = np.linspace(-10, 10, 1001)  # odd so center exists
        for func in [finite_square_well, potential_barrier]:
            V = func(x)
            np.testing.assert_allclose(V, V[::-1], atol=1e-10)


class TestMultiWell:
    """Tests for the multi-well (band structure) potential."""

    def test_equidistant_depth(self):
        x = np.linspace(-20, 20, 4000)
        V = multi_well(x, n_wells=5, well_width=1.0, depth=30.0)
        # Minimum should be close to -depth
        assert abs(np.min(V) - (-30.0)) < 1.0

    def test_single_well_centered(self):
        """One well on a periodic lattice sits at the interval midpoint."""
        x = np.linspace(-20, 20, 4000)
        V = multi_well(x, n_wells=1, well_width=2.0, depth=20.0)
        assert abs(x[np.argmin(V)]) < 1.0

    def test_n_wells_count(self):
        """Count local minima -- should match n_wells for zero jitter."""
        x = np.linspace(-20, 20, 4000)
        for n in [3, 5, 8]:
            V = multi_well(x, n_wells=n, well_width=1.0, depth=30.0, jitter=0.0)
            below = V < -15.0
            entries = np.sum(np.diff(below.astype(int)) == 1)
            assert entries == n, f"Expected {n} wells, found {entries}"

    def test_potential_non_positive(self):
        x = np.linspace(-20, 20, 2000)
        V = multi_well(x, n_wells=5, well_width=1.0, depth=30.0)
        assert np.all(V <= 1e-10)

    def test_jitter_reproducible(self):
        x = np.linspace(-20, 20, 2000)
        V1 = multi_well(x, n_wells=5, jitter=1.0, seed=99)
        V2 = multi_well(x, n_wells=5, jitter=1.0, seed=99)
        np.testing.assert_array_equal(V1, V2)

    def test_jitter_different_seeds(self):
        x = np.linspace(-20, 20, 2000)
        V1 = multi_well(x, n_wells=5, jitter=1.0, seed=1)
        V2 = multi_well(x, n_wells=5, jitter=1.0, seed=2)
        assert not np.allclose(V1, V2)

    def test_jitter_no_overlap(self):
        """Even with huge requested jitter, wells must not overlap."""
        x = np.linspace(-20, 20, 4000)
        V = multi_well(x, n_wells=8, well_width=2.0, depth=30.0,
                        jitter=100.0, seed=7)  # absurd jitter clamped
        # Depth should still be ~-30, not deeper (no stacking)
        assert np.min(V) > -31.0

    def test_zero_jitter_translational_symmetry(self):
        """With zero jitter the potential is exactly periodic over one cell."""
        n_wells = 5
        N = 4000  # divisible by n_wells
        x = np.linspace(-20, 20, N, endpoint=False)
        V = multi_well(x, n_wells=n_wells, well_width=1.0, depth=30.0, jitter=0.0)
        cell = N // n_wells
        # Each cell must be identical (exact translational symmetry)
        for i in range(1, n_wells):
            np.testing.assert_allclose(V[:cell], V[i * cell:(i + 1) * cell],
                                       atol=1e-10)

    def test_periodic_continuity(self):
        """Potential at left edge should match potential at right edge (periodic)."""
        x = np.linspace(-20, 20, 4000, endpoint=False)
        V = multi_well(x, n_wells=5, well_width=1.0, depth=30.0, jitter=0.0)
        # With periodic placement, V[0] should be close to V[-1]
        assert abs(V[0] - V[-1]) < 0.5


class TestPotentialRegistry:

    def test_all_potentials_callable(self):
        for name, info in POTENTIALS.items():
            x = np.linspace(*info["x_range"], 100)
            V = info["func"](x)
            assert V.shape == x.shape, f"{name}: shape mismatch"

    def test_all_have_required_keys(self):
        required = {"func", "description", "x_range", "default_x0", "default_k0", "default_sigma"}
        for name, info in POTENTIALS.items():
            assert required.issubset(info.keys()), f"{name}: missing keys {required - info.keys()}"
