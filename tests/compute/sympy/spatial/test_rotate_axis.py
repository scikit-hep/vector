# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

sympy = pytest.importorskip("sympy")

pytestmark = pytest.mark.sympy

x, y, z, nx, ny, nz, t = sympy.symbols("x y z nx ny nz t", real=True)
values = {x: 0.1, y: 0.2, z: 0.3, nx: 0.4, ny: 0.5, nz: 0.6}


def test_spatial_sympy():
    axis = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
    )
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(nx, ny),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(nz),
    )
    out = vec.rotate_axis(axis, 0.25)
    assert isinstance(out, vector.VectorSympy3D)
    assert isinstance(out.azimuthal, vector.backends.sympy.AzimuthalSympyXY)
    assert isinstance(out.longitudinal, vector.backends.sympy.LongitudinalSympyZ)
    assert out.x.subs(values) == pytest.approx(0.374834254043358)
    assert out.y.subs(values) == pytest.approx(0.5383405688588193)
    assert out.z.subs(values) == pytest.approx(0.5828282027463345)

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        for t2 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
            taxis, tvec = (
                getattr(axis, "to_" + t1)(),
                getattr(vec, "to_" + t2)(),
            )
            out = tvec.rotate_axis(taxis, 0.25)
            assert isinstance(out, vector.VectorSympy3D)
            assert isinstance(out.azimuthal, vector.backends.sympy.AzimuthalSympyXY)
            assert isinstance(
                out.longitudinal, vector.backends.sympy.LongitudinalSympyZ
            )
            assert out.x.subs(values) == pytest.approx(0.374834254043358)
            assert out.y.subs(values) == pytest.approx(0.5383405688588193)
            assert out.z.subs(values) == pytest.approx(0.5828282027463345)


def test_lorentz_sympy():
    axis = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(nx, ny),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(nz),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    with pytest.raises(TypeError):
        out = vec.rotate_axis(axis, 0.25)
    out = vec.rotate_axis(axis.to_Vector3D(), 0.25)
    assert isinstance(out, vector.VectorSympy4D)
    assert isinstance(out.azimuthal, vector.backends.sympy.AzimuthalSympyXY)
    assert isinstance(out.longitudinal, vector.backends.sympy.LongitudinalSympyZ)
    assert hasattr(out, "temporal")
    assert out.x.subs(values) == pytest.approx(0.374834254043358)
    assert out.y.subs(values) == pytest.approx(0.5383405688588193)
    assert out.z.subs(values) == pytest.approx(0.5828282027463345)

    for t1 in (
        "xyz",
        "xytheta",
        "xyeta",
        "rhophiz",
        "rhophitheta",
        "rhophieta",
    ):
        for t2 in (
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
            taxis, tvec = (
                getattr(axis, "to_" + t1)(),
                getattr(vec, "to_" + t2)(),
            )
            out = tvec.rotate_axis(taxis, 0.25)
            assert isinstance(out, vector.VectorSympy4D)
            assert isinstance(out.azimuthal, vector.backends.sympy.AzimuthalSympyXY)
            assert isinstance(
                out.longitudinal, vector.backends.sympy.LongitudinalSympyZ
            )
            assert hasattr(out, "temporal")
            assert out.x.subs(values) == pytest.approx(0.374834254043358)
            assert out.y.subs(values) == pytest.approx(0.5383405688588193)
            assert out.z.subs(values) == pytest.approx(0.5828282027463345)
