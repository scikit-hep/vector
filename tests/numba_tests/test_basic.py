import pytest

import numba
from vector.numba.lorentz.xyzt import LorentzXYZTFree


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


@numba.njit
def try_it_out(testit):
    return testit.x, testit["x"], testit.pt, testit.eta, testit.phi, testit.mass


def test_free_elements():
    testit = LorentzXYZTFree(1, 2, 3, 4)
    x1, x2, pt, eta, phi, mass = try_it_out(testit)

    assert testit.x == x1
    assert testit["x"] == x2
    assert testit.pt == pt
    assert testit.eta == eta
    assert testit.phi == phi
    assert testit.mass == mass
