from vector.numpy.lorentz.xyzt import LorentzXYZT


def test_basic_vector():
    v = LorentzXYZT(0, 3, 4, 5)
    assert v.mass == 0.0
