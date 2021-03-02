# -*- coding: utf-8 -*-
import numba
from pytest import approx

from vector.numba.lorentz.xyzt import LorentzXYZTFree


def test_passthrough():
    @numba.njit
    def pass_through(obj):
        return obj

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

    ans = LorentzXYZTFree(1.0, 2.0, 3.0, 4.0) + 1
    assert ans == v12
    assert ans == v21


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

    assert v1 * 2 == v12
    assert v1 * 2 == v21
