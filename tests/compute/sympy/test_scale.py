# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

sympy = pytest.importorskip("sympy")

pytestmark = pytest.mark.sympy

x, y, z, t = sympy.symbols("x y z t", real=True)
values = {x: 1, y: 2, z: 3, t: 10}


def test_planar_posfactor():
    vec = vector.VectorSympy2D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
    )
    out = vec.scale(1.75)
    assert type(out.azimuthal) == type(vec.azimuthal)  # noqa: E721
    assert out.x == x * 1.75
    assert out.y == y * 1.75

    for t1 in ("xy", "rhophi"):
        tvec = getattr(vec, "to_" + t1)()
        out = tvec.scale(1.75)
        assert type(out.azimuthal) == type(tvec.azimuthal)  # noqa: E721
        assert out.x.subs(values).evalf() == pytest.approx(1 * 1.75)
        assert out.y.subs(values).evalf() == pytest.approx(2 * 1.75)


def test_planar_negfactor():
    vec = vector.VectorSympy2D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
    )
    out = vec.scale(-1.75)
    assert type(out.azimuthal) == type(vec.azimuthal)  # noqa: E721
    assert out.x == x * -1.75
    assert out.y == y * -1.75

    for t1 in ("xy", "rhophi"):
        tvec = getattr(vec, "to_" + t1)()
        out = tvec.scale(-1.75)
        assert type(out.azimuthal) == type(tvec.azimuthal)  # noqa: E721
        assert out.x.subs(values).evalf() == pytest.approx(1 * -1.75)
        assert out.y.subs(values).evalf() == pytest.approx(2 * -1.75)


def test_spatial_posfactor():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
    )
    out = vec.scale(1.75)
    assert type(out.azimuthal) == type(vec.azimuthal)  # noqa: E721
    assert type(out.longitudinal) == type(vec.longitudinal)  # noqa: E721
    assert out.x == x * 1.75
    assert out.y == y * 1.75
    assert out.z == z * 1.75

    for t1 in (
        "xyz",
        "xytheta",
        "xyeta",
        "rhophiz",
        "rhophitheta",
        "rhophieta",
    ):
        tvec = getattr(vec, "to_" + t1)()
        out = tvec.scale(1.75)
        assert type(out.azimuthal) == type(tvec.azimuthal)  # noqa: E721
        assert type(out.longitudinal) == type(tvec.longitudinal)  # noqa: E721
        assert out.x.subs(values).evalf() == pytest.approx(1 * 1.75)
        assert out.y.subs(values).evalf() == pytest.approx(2 * 1.75)
        assert out.z.subs(values).evalf() == pytest.approx(3 * 1.75)


def test_spatial_negfactor():
    vec = vector.VectorSympy3D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
    )
    out = vec.scale(-1.75)
    assert type(out.azimuthal) == type(vec.azimuthal)  # noqa: E721
    assert type(out.longitudinal) == type(vec.longitudinal)  # noqa: E721
    assert out.x == x * -1.75
    assert out.y == y * -1.75
    assert out.z == z * -1.75

    for t1 in (
        "xyz",
        "xytheta",
        "xyeta",
        "rhophiz",
        "rhophitheta",
        "rhophieta",
    ):
        tvec = getattr(vec, "to_" + t1)()
        out = tvec.scale(-1.75)
        assert type(out.azimuthal) == type(tvec.azimuthal)  # noqa: E721
        assert type(out.longitudinal) == type(tvec.longitudinal)  # noqa: E721
        assert out.x.subs(values).evalf() == pytest.approx(1 * -1.75)
        assert out.y.subs(values).evalf() == pytest.approx(2 * -1.75)
        assert out.z.subs(values).evalf() == pytest.approx(3 * -1.75)


def test_lorentz_postime_posfactor():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    out = vec.scale(1.75)
    assert type(out.azimuthal) == type(vec.azimuthal)  # noqa: E721
    assert type(out.longitudinal) == type(vec.longitudinal)  # noqa: E721
    assert type(out.temporal) == type(vec.temporal)  # noqa: E721
    assert out.x == x * 1.75
    assert out.y == y * 1.75
    assert out.z == z * 1.75
    assert out.t == t * 1.75

    for t1 in (
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
        tvec = getattr(vec, "to_" + t1)()
        out = tvec.scale(1.75)
        assert type(out.azimuthal) == type(tvec.azimuthal)  # noqa: E721
        assert type(out.longitudinal) == type(tvec.longitudinal)  # noqa: E721
        assert type(out.temporal) == type(tvec.temporal)  # noqa: E721
        assert out.x.subs(values).evalf() == pytest.approx(1 * 1.75)
        assert out.y.subs(values).evalf() == pytest.approx(2 * 1.75)
        assert out.z.subs(values).evalf() == pytest.approx(3 * 1.75)
        assert out.t.subs(values).evalf() == pytest.approx(10 * 1.75)


def test_lorentz_postime_negfactor():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyT(t),
    )
    out = vec.scale(-1.75)
    assert type(out.azimuthal) == type(vec.azimuthal)  # noqa: E721
    assert type(out.longitudinal) == type(vec.longitudinal)  # noqa: E721
    assert type(out.temporal) == type(vec.temporal)  # noqa: E721
    assert out.x == x * -1.75
    assert out.y == y * -1.75
    assert out.z == z * -1.75
    assert out.t == t * -1.75

    for t1 in (
        "xyzt",
        "xythetat",
        "xyetat",
        "rhophizt",
        "rhophithetat",
        "rhophietat",
    ):
        tvec = getattr(vec, "to_" + t1)()
        out = tvec.scale(-1.75)
        assert type(out.azimuthal) == type(tvec.azimuthal)  # noqa: E721
        assert type(out.longitudinal) == type(tvec.longitudinal)  # noqa: E721
        assert type(out.temporal) == type(tvec.temporal)  # noqa: E721
        assert out.x.subs(values).evalf() == pytest.approx(1 * -1.75)
        assert out.y.subs(values).evalf() == pytest.approx(2 * -1.75)
        assert out.z.subs(values).evalf() == pytest.approx(3 * -1.75)
        assert out.t.subs(values).evalf() == pytest.approx(10 * -1.75)

    for t1 in (
        "xyztau",
        "xythetatau",
        "xyetatau",
        "rhophiztau",
        "rhophithetatau",
        "rhophietatau",
    ):
        tvec = getattr(vec, "to_" + t1)()
        out = tvec.scale(-1.75)
        assert type(out.azimuthal) == type(tvec.azimuthal)  # noqa: E721
        assert type(out.longitudinal) == type(tvec.longitudinal)  # noqa: E721
        assert type(out.temporal) == type(tvec.temporal)  # noqa: E721
        assert out.x.subs(values).evalf() == pytest.approx(1 * -1.75)
        assert out.y.subs(values).evalf() == pytest.approx(2 * -1.75)
        assert out.z.subs(values).evalf() == pytest.approx(3 * -1.75)
        assert out.t.subs(values).evalf() == pytest.approx(10 * 1.75)


def test_lorentz_negtime_posfactor():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyT(-t),
    )
    out = vec.scale(1.75)
    assert type(out.azimuthal) == type(vec.azimuthal)  # noqa: E721
    assert type(out.longitudinal) == type(vec.longitudinal)  # noqa: E721
    assert type(out.temporal) == type(vec.temporal)  # noqa: E721
    assert out.x == x * 1.75
    assert out.y == y * 1.75
    assert out.z == z * 1.75
    assert out.t == -t * 1.75

    for t1 in (
        "xyzt",
        "xythetat",
        "xyetat",
        "rhophizt",
        "rhophithetat",
        "rhophietat",
    ):
        tvec = getattr(vec, "to_" + t1)()
        out = tvec.scale(1.75)
        assert type(out.azimuthal) == type(tvec.azimuthal)  # noqa: E721
        assert type(out.longitudinal) == type(tvec.longitudinal)  # noqa: E721
        assert type(out.temporal) == type(tvec.temporal)  # noqa: E721
        assert out.x.subs(values).evalf() == pytest.approx(1 * 1.75)
        assert out.y.subs(values).evalf() == pytest.approx(2 * 1.75)
        assert out.z.subs(values).evalf() == pytest.approx(3 * 1.75)
        assert out.t.subs(values).evalf() == pytest.approx(10 * -1.75)

    for t1 in (
        "xyztau",
        "xythetatau",
        "xyetatau",
        "rhophiztau",
        "rhophithetatau",
        "rhophietatau",
    ):
        tvec = getattr(vec, "to_" + t1)()
        out = tvec.scale(1.75)
        assert type(out.azimuthal) == type(tvec.azimuthal)  # noqa: E721
        assert type(out.longitudinal) == type(tvec.longitudinal)  # noqa: E721
        assert type(out.temporal) == type(tvec.temporal)  # noqa: E721
        assert out.x.subs(values).evalf() == pytest.approx(1 * 1.75)
        assert out.y.subs(values).evalf() == pytest.approx(2 * 1.75)
        assert out.z.subs(values).evalf() == pytest.approx(3 * 1.75)
        assert out.t.subs(values).evalf() == pytest.approx(10 * 1.75)


def test_lorentz_negtime_negfactor():
    vec = vector.VectorSympy4D(
        azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y),
        longitudinal=vector.backends.sympy.LongitudinalSympyZ(z),
        temporal=vector.backends.sympy.TemporalSympyT(-t),
    )
    out = vec.scale(-1.75)
    assert type(out.azimuthal) == type(vec.azimuthal)  # noqa: E721
    assert type(out.longitudinal) == type(vec.longitudinal)  # noqa: E721
    assert type(out.temporal) == type(vec.temporal)  # noqa: E721
    assert out.x == x * -1.75
    assert out.y == y * -1.75
    assert out.z == z * -1.75
    assert out.t == -t * -1.75

    for t1 in (
        "xyzt",
        "xythetat",
        "xyetat",
        "rhophizt",
        "rhophithetat",
        "rhophietat",
    ):
        tvec = getattr(vec, "to_" + t1)()
        out = tvec.scale(-1.75)
        assert type(out.azimuthal) == type(tvec.azimuthal)  # noqa: E721
        assert type(out.longitudinal) == type(tvec.longitudinal)  # noqa: E721
        assert type(out.temporal) == type(tvec.temporal)  # noqa: E721
        assert out.x.subs(values).evalf() == pytest.approx(1 * -1.75)
        assert out.y.subs(values).evalf() == pytest.approx(2 * -1.75)
        assert out.z.subs(values).evalf() == pytest.approx(3 * -1.75)
        assert out.t.subs(values).evalf() == pytest.approx(-10 * -1.75)

    for t1 in (
        "xyztau",
        "xythetatau",
        "xyetatau",
        "rhophiztau",
        "rhophithetatau",
        "rhophietatau",
    ):
        tvec = getattr(vec, "to_" + t1)()
        out = tvec.scale(-1.75)
        assert type(out.azimuthal) == type(tvec.azimuthal)  # noqa: E721
        assert type(out.longitudinal) == type(tvec.longitudinal)  # noqa: E721
        assert type(out.temporal) == type(tvec.temporal)  # noqa: E721
        assert out.x.subs(values).evalf() == pytest.approx(1 * -1.75)
        assert out.y.subs(values).evalf() == pytest.approx(2 * -1.75)
        assert out.z.subs(values).evalf() == pytest.approx(3 * -1.75)
        assert out.t.subs(values).evalf() == pytest.approx(10 * 1.75)
