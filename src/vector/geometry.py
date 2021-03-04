# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.


class Tuple:
    pass


class Vector(Tuple):
    pass


class Point(Tuple):
    pass


class Transform:
    pass


class FrameTransform:
    pass


class Planar:
    @property
    def x(self):
        "x docs"
        raise NotImplementedError

    @property
    def y(self):
        "y docs"
        raise NotImplementedError

    @property
    def rho(self):
        "rho docs"
        raise NotImplementedError

    @property
    def phi(self):
        "phi docs"
        raise NotImplementedError

    @property
    def rho2(self):
        "rho2 docs"
        raise NotImplementedError


class Spatial:
    pass


class Lorentz:
    pass


class Azimuthal:
    pass


class Longitudinal:
    pass


class Temporal:
    pass


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
