# -*- coding: utf-8 -*-

import ROOT
from pytest import approx

from vector.single.lorentz.xyzt import LorentzXYZTFree


def test_simple():
    v1 = LorentzXYZTFree(1, 2, 3, 4)
    v2 = ROOT.TLorentzVector(1, 2, 3, 4)

    assert v1.x == approx(v2.X())
    assert v1.y == approx(v2.Y())
    assert v1.z == approx(v2.Z())
    assert v1.t == approx(v2.T())
    assert v1.pt == approx(v2.Pt())
    assert v1.eta == approx(v2.Eta())
    assert v1.phi == approx(v2.Phi())
    assert v1.mass == approx(v2.M())
