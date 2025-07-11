# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

sympy = pytest.importorskip("sympy")

pytestmark = pytest.mark.sympy

x, y, z = sympy.symbols("x y z", real=True)
values = {x: 0.4, y: 0.5, z: 0.6}


def test_spatial_sympy():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
    )
    out = vec.rotate_euler(0.1, 0.2, 0.3)
    assert isinstance(out, vector.VectorSympy3D)
    assert isinstance(out.azimuthal, vector.backends.sympy.AzimuthalSympyXY)
    assert isinstance(out.longitudinal, vector.backends.sympy.LongitudinalSympyZ)
    # cannot equate the sympy expressions because of floating point errors
    assert out.x.subs(values) == pytest.approx(0.5956646364506655)
    assert out.y.subs(values) == pytest.approx(0.409927258162962)
    assert out.z.subs(values) == pytest.approx(0.4971350761081869)

    for t in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        tvec = getattr(vec, "to_" + t)()
        out = tvec.rotate_euler(0.1, 0.2, 0.3)
        assert isinstance(out, vector.VectorSympy3D)
        assert isinstance(out.azimuthal, vector.backends.sympy.AzimuthalSympyXY)
        assert isinstance(out.longitudinal, vector.backends.sympy.LongitudinalSympyZ)
        # cannot equate the sympy expressions because of floating point errors
        assert out.x.subs(values) == pytest.approx(0.5956646364506655)
        assert out.y.subs(values) == pytest.approx(0.409927258162962)
        assert out.z.subs(values) == pytest.approx(0.4971350761081869)
