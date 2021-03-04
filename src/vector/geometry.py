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
