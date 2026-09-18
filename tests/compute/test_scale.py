# Copyright (c) 2019, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import math

import pytest

import vector.backends.numpy
import vector.backends.object


def test_planar_posfactor():
    vec = vector.backends.object.VectorObject2D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(1, 2),
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
    vec = vector.backends.object.VectorObject2D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(1, 2),
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
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(1, 2),
        longitudinal=vector.backends.object.LongitudinalObjectZ(3),
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
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(1, 2),
        longitudinal=vector.backends.object.LongitudinalObjectZ(3),
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
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(1, 2),
        longitudinal=vector.backends.object.LongitudinalObjectZ(3),
        temporal=vector.backends.object.TemporalObjectT(4),
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
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(1, 2),
        longitudinal=vector.backends.object.LongitudinalObjectZ(3),
        temporal=vector.backends.object.TemporalObjectT(4),
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
        # Tau coordinates do not store the sign of t, so |t| is reconstructed.
        # Scaling preserves the timelike character (tau stays positive), so the
        # magnitude matches |4 * -1.75| = 7.0 (previously this gave the buggy
        # spacelike value 6.0621... because tau's sign was flipped).
        assert out.t == pytest.approx(4 * 1.75)


def test_lorentz_negtime_posfactor():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(1, 2),
        longitudinal=vector.backends.object.LongitudinalObjectZ(3),
        temporal=vector.backends.object.TemporalObjectT(-1.5),
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
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(1, 2),
        longitudinal=vector.backends.object.LongitudinalObjectZ(3),
        temporal=vector.backends.object.TemporalObjectT(-1.5),
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
        out = tvec.scale(-1.75)
        assert type(out.azimuthal) == type(tvec.azimuthal)  # noqa: E721
        assert type(out.longitudinal) == type(tvec.longitudinal)  # noqa: E721
        assert type(out.temporal) == type(tvec.temporal)  # noqa: E721
        assert out.x == pytest.approx(1 * -1.75)
        assert out.y == pytest.approx(2 * -1.75)
        assert out.z == pytest.approx(3 * -1.75)
        assert out.t == pytest.approx(-1.5 * -1.75)

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
        # This vector is spacelike (|t| < |p|, tau < 0). Scaling preserves the
        # spacelike character, so the reconstructed |t| now matches the
        # t-coordinate result |-1.5 * -1.75| = 2.625 (previously this gave the
        # buggy value 8.8802... because tau's sign was flipped to positive).
        assert out.t == pytest.approx(-1.5 * -1.75)


def test_lorentz_scale_tau_sign_invariant():
    # Scaling by a factor lambda multiplies t**2 - mag**2 by lambda**2, so the
    # causal character (sign of tau) is invariant: |tau| scales by |lambda| and
    # the sign is preserved. A negative factor must NOT flip a timelike vector
    # into the spacelike (negative-tau) encoding.
    # timelike vector (tau > 0)
    mag = math.sqrt(1.0 + 4.0 + 9.0)
    t_vec = vector.obj(x=1.0, y=2.0, z=3.0, t=math.sqrt(mag**2 + 25.0))  # tau = 5
    assert t_vec.tau == pytest.approx(5.0)

    tau_vec = vector.obj(x=1.0, y=2.0, z=3.0, tau=5.0)
    scaled_tau = tau_vec.scale(-2.0)
    scaled_t = t_vec.scale(-2.0)
    # tau magnitude scales by |factor|, sign stays positive (timelike)
    assert scaled_tau.tau == pytest.approx(10.0)
    assert scaled_t.tau == pytest.approx(10.0)
    assert scaled_tau.tau == pytest.approx(scaled_t.tau)

    # spacelike vector (tau < 0) stays spacelike under negative scaling
    spacelike = vector.obj(x=3.0, y=4.0, z=0.0, tau=-2.0)
    assert spacelike.scale(-3.0).tau == pytest.approx(-6.0)
