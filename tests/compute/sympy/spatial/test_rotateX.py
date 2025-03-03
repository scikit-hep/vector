# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector
import vector._methods

sympy = pytest.importorskip("sympy")

pytestmark = pytest.mark.sympy

x, y, z, time = sympy.symbols("x y z time", real=True)
values = {x: 0.1, y: 0.2, z: 0.3}


def test_spatial_sympy():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
    )
    out = vec.rotateX(0.25)
    assert isinstance(out.azimuthal, vector._methods.AzimuthalXY)
    assert isinstance(out.longitudinal, vector._methods.LongitudinalZ)
    assert out.x == x
    # cannot equate the sympy expressions because of floating point errors
    assert out.y.subs(values) == pytest.approx(0.1195612965657721)
    assert out.z.subs(values) == pytest.approx(0.340154518364098)

    for t in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        out = getattr(vec, "to_" + t)().rotateX(0.25)
        assert isinstance(out.azimuthal, vector._methods.AzimuthalXY)
        assert isinstance(out.longitudinal, vector._methods.LongitudinalZ)
        assert out.x.subs(values) == pytest.approx(0.1)
        assert out.y.subs(values) == pytest.approx(0.1195612965657721)
        assert out.z.subs(values) == pytest.approx(0.340154518364098)


def test_lorentz_sympy():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyT(time),
    )
    out = vec.rotateX(0.25)
    assert isinstance(out.azimuthal, vector._methods.AzimuthalXY)
    assert isinstance(out.longitudinal, vector._methods.LongitudinalZ)
    assert hasattr(out, "temporal")
    assert out.x == x
    # cannot equate the sympy expressions because of floating point errors
    assert out.y.subs(values) == pytest.approx(0.1195612965657721)
    assert out.z.subs(values) == pytest.approx(0.340154518364098)

    for t in (
        "xyzt",
        "xythetat",
        "xyetat",
        "rhophizt",
        "rhophithetat",
        "rhophietat",
        "xyztau",
        "xythetatau",
        "xyetatau",
        "rhophiztau",
        "rhophithetatau",
        "rhophietatau",
    ):
        out = getattr(vec, "to_" + t)().rotateX(0.25)
        assert isinstance(out.azimuthal, vector._methods.AzimuthalXY)
        assert isinstance(out.longitudinal, vector._methods.LongitudinalZ)
        assert hasattr(out, "temporal")
        assert out.x.subs(values) == pytest.approx(0.1)
        assert out.y.subs(values) == pytest.approx(0.1195612965657721)
        assert out.z.subs(values) == pytest.approx(0.340154518364098)
