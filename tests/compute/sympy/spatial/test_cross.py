# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

sympy = pytest.importorskip("sympy")

pytestmark = pytest.mark.sympy

x, y, z, nx, ny, nz = sympy.symbols("x y z nx ny nz", real=True)
values = {x: 0.1, y: 0.2, z: 0.3, nx: 0.4, ny: 0.5, nz: 0.6}


def test_spatial_sympy():
    v1 = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
    )
    v2 = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(nx, ny),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(nz),
    )
    with pytest.raises(TypeError):
        out = v1.to_Vector4D().cross(v2)
    out = v1.cross(v2)
    assert isinstance(out, vector.VectorSympy3D)
    assert isinstance(out.azimuthal, vector.backends.sympy.AzimuthalSympyXY)
    assert isinstance(out.longitudinal, vector.backends.sympy.LongitudinalSympyZ)
    assert (out.x, out.y, out.z) == (
        -ny * z + nz * y,
        nx * z - nz * x,
        -nx * y + ny * x,
    )

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        for t2 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
            transformed1, transformed2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            out = transformed1.cross(transformed2)
            assert isinstance(out, vector.VectorSympy3D)
            assert isinstance(out.azimuthal, vector.backends.sympy.AzimuthalSympyXY)
            assert isinstance(
                out.longitudinal, vector.backends.sympy.LongitudinalSympyZ
            )
            assert (
                out.x.subs(values),
                out.y.subs(values),
                out.z.subs(values),
            ) == pytest.approx((-0.03, 0.06, -0.03))
