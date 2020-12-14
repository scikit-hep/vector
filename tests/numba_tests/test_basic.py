# -*- coding: utf-8 -*-

import numba

from pytest import approx

from vector.numba.lorentz.xyzt import LorentzXYZTFree


# Wrapped so the syntax remains the same with main test file
def assert_almost_equal(actual, expected):
    assert actual == expected


@numba.njit
def pass_through(obj):
    return obj


def test_simple_check():
    testit = LorentzXYZTFree(1, 2, 3, 4)
    assert repr(testit) == "Lxyz(1 2 3 4)"


def test_passthrough():
    assert repr(pass_through(LorentzXYZTFree(1, 2, 3, 4))) == repr(
        LorentzXYZTFree(1.0, 2.0, 3.0, 4.0)
    )


def test_constructor():
    @numba.njit
    def test_constructor():
        return LorentzXYZTFree(1.1, 2.2, 3.3, 4.4)

    assert repr(test_constructor()) == repr(LorentzXYZTFree(1.1, 2.2, 3.3, 4.4))


# test_mag
# test_pt
# test_eta
# test_phi
def test_free_elements():
    @numba.njit
    def try_it_out(testit):
        return testit.x, testit["x"], testit.pt, testit.eta, testit.phi, testit.mag

    testit = LorentzXYZTFree(1, 2, 3, 4)
    x1, x2, pt, eta, phi, mass = try_it_out(testit)

    assert testit.x == x1
    assert testit["x"] == x2
    assert testit.pt == pt
    assert testit.eta == eta
    assert testit.phi == phi
    assert testit.mass == mass


def test_add_vector():
    @numba.njit
    def test_addition():
        return LorentzXYZTFree(1.0, 2.0, 3.0, 4.0) + LorentzXYZTFree(4.0, 3.0, 2.0, 1.0)

    assert repr(test_addition()) == repr(LorentzXYZTFree(5, 5, 5, 5))


def test_add_scalar():
    @numba.njit
    def test_addition_scalar():
        return (
            LorentzXYZTFree(1.0, 2.0, 3.0, 4.0) + 1.0,
            1.0 + LorentzXYZTFree(1.0, 2.0, 3.0, 4.0),
        )

    v12, v21 = test_addition_scalar()

    assert_almost_equal(2, v12.x)
    assert_almost_equal(3, v12.y)
    assert_almost_equal(4, v12.z)
    assert_almost_equal(5, v12.t)

    assert_almost_equal(2, v21.x)
    assert_almost_equal(3, v21.y)
    assert_almost_equal(4, v21.z)
    assert_almost_equal(5, v21.t)


# test_multiply_vec
def test_dot():
    @numba.njit
    def test_multiply():
        v1 = LorentzXYZTFree(1.0, 2.0, 3.0, 4.0)
        v2 = LorentzXYZTFree(0.0, 2.0, 2.0, 5.0)
        return (v1.dot(v2), v1 * v2)

    dot, mul = test_multiply()
    assert dot == approx(20 - 6 - 4)
    assert mul == approx(20 - 6 - 4)


def test_multiply_scalar():
    @numba.njit
    def test_multiply_scalar(v):
        return (v * 2.0, 2.0 * v)

    v1 = LorentzXYZTFree(1.0, 2.0, 3.0, 4.0)
    v12, v21 = test_multiply_scalar(v1)

    assert_almost_equal(v1.x * 2, v12.x)
    assert_almost_equal(v1.y * 2, v12.y)
    assert_almost_equal(v1.z * 2, v12.z)
    assert_almost_equal(v1.t * 2, v12.t)

    assert_almost_equal(v1.x * 2, v21.x)
    assert_almost_equal(v1.y * 2, v21.y)
    assert_almost_equal(v1.z * 2, v21.z)
    assert_almost_equal(v1.t * 2, v21.t)
