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
