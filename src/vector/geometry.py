# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.


class Vector:
    pass


class Vector2D(Vector):
    pass


class Vector3D(Vector):
    pass


class Vector4D(Vector):
    pass


class Coordinates:
    pass


class Azimuthal(Coordinates):
    @property
    def elements(self):
        "azimuthal elements docs"
        raise AssertionError


class Longitudinal(Coordinates):
    @property
    def elements(self):
        "longitudinal elements docs"
        raise AssertionError


class Temporal(Coordinates):
    @property
    def elements(self):
        "temporal elements docs"
        raise AssertionError


class AzimuthalXY(Azimuthal):
    pass


class AzimuthalRhoPhi(Azimuthal):
    pass


class LongitudinalZ(Longitudinal):
    pass


class LongitudinalTheta(Longitudinal):
    pass


class LongitudinalEta(Longitudinal):
    pass


class TemporalT(Temporal):
    pass


class TemporalTau(Temporal):
    pass


def aztype(obj):
    for t in type(obj.azimuthal).__mro__:
        if t in (AzimuthalXY, AzimuthalRhoPhi):
            return t
    else:
        return None


def ltype(obj):
    for t in type(obj.longitudinal).__mro__:
        if t in (LongitudinalZ, LongitudinalTheta, LongitudinalEta):
            return t
    else:
        return None


def ttype(obj):
    for t in type(obj.temporal).__mro__:
        if t in (TemporalT, TemporalTau):
            return t
    else:
        return None


_coordinate_class_to_names = {
    AzimuthalXY: ("x", "y"),
    AzimuthalRhoPhi: ("rho", "phi"),
    LongitudinalZ: ("z",),
    LongitudinalTheta: ("theta",),
    LongitudinalEta: ("eta",),
    TemporalT: ("t",),
    TemporalTau: ("tau",),
}
