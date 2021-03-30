# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import pytest

import vector._backends.numpy_
import vector._backends.object_


def test_planar_posfactor():
    vec = vector._backends.object_.VectorObject2D(
        vector._backends.object_.AzimuthalObjectXY(1, 2),
    )
    out = vec.scale(1.75)
    assert type(out.azimuthal) == type(vec.azimuthal)  # noqa: E721
    assert out.x == pytest.approx(1 * 1.75)
    assert out.y == pytest.approx(2 * 1.75)

    for t1 in ("xy", "rhophi"):
        tvec = getattr(vec, "to_" + t1)()
        out = tvec.scale(1.75)
        assert type(out.azimuthal) == type(tvec.azimuthal)  # noqa: E721
        assert out.x == pytest.approx(1 * 1.75)
        assert out.y == pytest.approx(2 * 1.75)


def test_planar_negfactor():
    vec = vector._backends.object_.VectorObject2D(
        vector._backends.object_.AzimuthalObjectXY(1, 2),
    )
    out = vec.scale(-1.75)
    assert type(out.azimuthal) == type(vec.azimuthal)  # noqa: E721
    assert out.x == pytest.approx(1 * -1.75)
    assert out.y == pytest.approx(2 * -1.75)

    for t1 in ("xy", "rhophi"):
        tvec = getattr(vec, "to_" + t1)()
        out = tvec.scale(-1.75)
        assert type(out.azimuthal) == type(tvec.azimuthal)  # noqa: E721
        assert out.x == pytest.approx(1 * -1.75)
        assert out.y == pytest.approx(2 * -1.75)


def test_spatial_posfactor():
    vec = vector._backends.object_.VectorObject3D(
        vector._backends.object_.AzimuthalObjectXY(1, 2),
        vector._backends.object_.LongitudinalObjectZ(3),
    )
    out = vec.scale(1.75)
    assert type(out.azimuthal) == type(vec.azimuthal)  # noqa: E721
    assert type(out.longitudinal) == type(vec.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(1 * 1.75)
    assert out.y == pytest.approx(2 * 1.75)
    assert out.z == pytest.approx(3 * 1.75)

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
        assert out.x == pytest.approx(1 * 1.75)
        assert out.y == pytest.approx(2 * 1.75)
        assert out.z == pytest.approx(3 * 1.75)


def test_spatial_negfactor():
    vec = vector._backends.object_.VectorObject3D(
        vector._backends.object_.AzimuthalObjectXY(1, 2),
        vector._backends.object_.LongitudinalObjectZ(3),
    )
    out = vec.scale(-1.75)
    assert type(out.azimuthal) == type(vec.azimuthal)  # noqa: E721
    assert type(out.longitudinal) == type(vec.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(1 * -1.75)
    assert out.y == pytest.approx(2 * -1.75)
    assert out.z == pytest.approx(3 * -1.75)

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
        assert out.x == pytest.approx(1 * -1.75)
        assert out.y == pytest.approx(2 * -1.75)
        assert out.z == pytest.approx(3 * -1.75)


def test_lorentz_postime_posfactor():
    vec = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectXY(1, 2),
        vector._backends.object_.LongitudinalObjectZ(3),
        vector._backends.object_.TemporalObjectT(4),
    )
    out = vec.scale(1.75)
    assert type(out.azimuthal) == type(vec.azimuthal)  # noqa: E721
    assert type(out.longitudinal) == type(vec.longitudinal)  # noqa: E721
    assert type(out.temporal) == type(vec.temporal)  # noqa: E721
    assert out.x == pytest.approx(1 * 1.75)
    assert out.y == pytest.approx(2 * 1.75)
    assert out.z == pytest.approx(3 * 1.75)
    assert out.t == pytest.approx(4 * 1.75)

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
        assert out.x == pytest.approx(1 * 1.75)
        assert out.y == pytest.approx(2 * 1.75)
        assert out.z == pytest.approx(3 * 1.75)
        assert out.t == pytest.approx(4 * 1.75)


def test_lorentz_postime_negfactor():
    vec = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectXY(1, 2),
        vector._backends.object_.LongitudinalObjectZ(3),
        vector._backends.object_.TemporalObjectT(4),
    )
    out = vec.scale(-1.75)
    assert type(out.azimuthal) == type(vec.azimuthal)  # noqa: E721
    assert type(out.longitudinal) == type(vec.longitudinal)  # noqa: E721
    assert type(out.temporal) == type(vec.temporal)  # noqa: E721
    assert out.x == pytest.approx(1 * -1.75)
    assert out.y == pytest.approx(2 * -1.75)
    assert out.z == pytest.approx(3 * -1.75)
    assert out.t == pytest.approx(4 * -1.75)

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
        assert out.x == pytest.approx(1 * -1.75)
        assert out.y == pytest.approx(2 * -1.75)
        assert out.z == pytest.approx(3 * -1.75)
        assert out.t == pytest.approx(4 * -1.75)

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
        assert out.x == pytest.approx(1 * -1.75)
        assert out.y == pytest.approx(2 * -1.75)
        assert out.z == pytest.approx(3 * -1.75)
        assert out.t == pytest.approx(6.06217782649107)


def test_lorentz_negtime_posfactor():
    vec = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectXY(1, 2),
        vector._backends.object_.LongitudinalObjectZ(3),
        vector._backends.object_.TemporalObjectT(-1.5),
    )
    out = vec.scale(1.75)
    assert type(out.azimuthal) == type(vec.azimuthal)  # noqa: E721
    assert type(out.longitudinal) == type(vec.longitudinal)  # noqa: E721
    assert type(out.temporal) == type(vec.temporal)  # noqa: E721
    assert out.x == pytest.approx(1 * 1.75)
    assert out.y == pytest.approx(2 * 1.75)
    assert out.z == pytest.approx(3 * 1.75)
    assert out.t == pytest.approx(-1.5 * 1.75)

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
        assert out.x == pytest.approx(1 * 1.75)
        assert out.y == pytest.approx(2 * 1.75)
        assert out.z == pytest.approx(3 * 1.75)
        assert out.t == pytest.approx(-1.5 * 1.75)

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
        assert out.x == pytest.approx(1 * 1.75)
        assert out.y == pytest.approx(2 * 1.75)
        assert out.z == pytest.approx(3 * 1.75)
        assert out.t == pytest.approx(1.5 * 1.75)


def test_lorentz_negtime_negfactor():
    vec = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectXY(1, 2),
        vector._backends.object_.LongitudinalObjectZ(3),
        vector._backends.object_.TemporalObjectT(-1.5),
    )
    out = vec.scale(-1.75)
    assert type(out.azimuthal) == type(vec.azimuthal)  # noqa: E721
    assert type(out.longitudinal) == type(vec.longitudinal)  # noqa: E721
    assert type(out.temporal) == type(vec.temporal)  # noqa: E721
    assert out.x == pytest.approx(1 * -1.75)
    assert out.y == pytest.approx(2 * -1.75)
    assert out.z == pytest.approx(3 * -1.75)
    assert out.t == pytest.approx(-1.5 * -1.75)

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
        assert out.x == pytest.approx(1 * 1.75)
        assert out.y == pytest.approx(2 * 1.75)
        assert out.z == pytest.approx(3 * 1.75)
        assert out.t == pytest.approx(-1.5 * 1.75)

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
        assert out.x == pytest.approx(1 * -1.75)
        assert out.y == pytest.approx(2 * -1.75)
        assert out.z == pytest.approx(3 * -1.75)
        assert out.t == pytest.approx(8.880280119455692)
