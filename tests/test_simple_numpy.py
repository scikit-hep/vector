# -*- coding: utf-8 -*-
from vector.numpy.lorentz.xyzt import LorentzXYZT


def test_basic_vector():
    v = LorentzXYZT(0, 3, 4, 5)
    assert v.mag == 0.0


def test_basic_addition():
    v1 = LorentzXYZT(1, 2, 3, 4)
    v2 = LorentzXYZT(4, 3, 2, 1)

    v12 = v1 + v2

    assert v12.x == 5
    assert v12.y == 5
    assert v12.z == 5
    assert v12.t == 5


def test_basic_scalar():
    v1 = LorentzXYZT(1, 2, 3, 4)
    v12 = v1 * 2
    assert v12.x == v1.x * 2
    assert v12.y == v1.y * 2
    assert v12.z == v1.z * 2
    assert v12.t == v1.t * 2


def test_dot():
    v1 = LorentzXYZT(1, 2, 3, 4)
    v2 = LorentzXYZT(0, 2, 2, 5)

    res = v1.dot(v2)

    assert res == 20 - 6 - 4


def test_mul_vec():
    v1 = LorentzXYZT(1, 2, 3, 4)
    v2 = LorentzXYZT(0, 2, 2, 5)

    res = v1 * v2

    assert res == 20 - 6 - 4


def test_mul_scalar():
    v1 = LorentzXYZT(1, 2, 3, 4)
    v12 = v1.mul(2)
    assert v12.x == v1.x * 2
    assert v12.y == v1.y * 2
    assert v12.z == v1.z * 2
    assert v12.t == v1.t * 2
