# -*- coding: utf-8 -*-
from vector.numpy.lorentz.xyzt import LorentzXYZT


def test_basic_vector():
    v = LorentzXYZT(0, 3, 4, 5)
    assert v.mag == 0.0
