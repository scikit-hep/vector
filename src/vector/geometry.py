# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.


class Tuple:
    pass


class Vector(Tuple):
    pass


class PlanarVector(Vector):
    pass


class SpatialVector(Vector):
    pass


class LorentzVector(Vector):
    pass


class Point(Tuple):
    pass


class PlanarPoint(Point):
    pass


class SpatialPoint(Point):
    pass


class LorentzPoint(Point):
    pass


class Azimuthal:
    @property
    def elements(self):
        "azimuthal elements docs"
        raise AssertionError


class Longitudinal:
    @property
    def elements(self):
        "longitudinal elements docs"
        raise AssertionError


class Temporal:
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


class LongitudinalW(Longitudinal):
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
        raise AssertionError(type(obj.azimuthal))


def ltype(obj):
    for t in type(obj.longitudinal).__mro__:
        if t in (LongitudinalZ, LongitudinalTheta, LongitudinalEta, LongitudinalW):
            return t
    else:
        raise AssertionError(type(obj.longitudinal))


def ttype(obj):
    for t in type(obj.temporal).__mro__:
        if t in (TemporalT, TemporalTau):
            return t
    else:
        raise AssertionError(type(obj.temporal))


class Transform:
    def __call__(self, v):
        "call docs"
        return self.apply(v)


class FrameTransform:
    def __call__(self, v):
        "call docs"
        return self.apply(v)
