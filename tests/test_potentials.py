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
        assert np.all(V[np.abs(x) < 1.0] == 30.0)

    def test_morse_minimum(self):
        x = np.linspace(-10, 10, 1000)
        V = morse_potential(x, D=10.0, a=0.5, x0=0.0)
        assert abs(V[np.argmin(np.abs(x))]) < 0.01  # min near x0


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
