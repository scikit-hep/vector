# -*- coding: utf-8 -*-
import vector
import vector.single.lorentz.xyzt


def test_import():
    assert vector.__version__ is not None


def test_vector_compare():
    v1 = vector.single.lorentz.xyzt.LorentzXYZTFree(1, 2, 3, 4)
    v1p = vector.single.lorentz.xyzt.LorentzXYZTFree(1, 2, 3, 4)
    v2 = vector.single.lorentz.xyzt.LorentzXYZTFree(1, 2, 3, 4.0000000001)
    v3 = vector.single.lorentz.xyzt.LorentzXYZTFree(1, 2, 5, 4)

    assert v1 == v1
    assert v1 == v1p
    assert v1 == v2
    assert not v1 != v2
    assert v1 != v3
    assert not v1 == v3
