# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import vector.compute
import vector.geometry


class Planar:
    @property
    def x(self):
        "x docs"
        return vector.compute.planar.x.dispatch(self)

    @property
    def y(self):
        "y docs"
        return vector.compute.planar.y.dispatch(self)

    @property
    def rho(self):
        "rho docs"
        return vector.compute.planar.rho.dispatch(self)

    @property
    def phi(self):
        "phi docs"
        return vector.compute.planar.phi.dispatch(self)

    @property
    def rho2(self):
        "rho2 docs"
        return vector.compute.planar.rho2.dispatch(self)


class Spatial(Planar):
    @property
    def z(self):
        "z docs"
        return vector.compute.planar.z.dispatch(self)

    @property
    def theta(self):
        "theta docs"
        return vector.compute.planar.theta.dispatch(self)

    @property
    def eta(self):
        "eta docs"
        return vector.compute.planar.eta.dispatch(self)

    @property
    def w(self):
        "w docs"
        return vector.compute.planar.w.dispatch(self)


class Lorentz(Spatial):
    @property
    def t(self):
        "t docs"
        return vector.compute.planar.t.dispatch(self)

    @property
    def tau(self):
        "tau docs"
        return vector.compute.planar.tau.dispatch(self)


class Transform2D:
    @property
    def xx(self):
        "xx docs"
        raise NotImplementedError

    @property
    def xy(self):
        "xy docs"
        raise NotImplementedError

    @property
    def yx(self):
        "yx docs"
        raise NotImplementedError

    @property
    def yy(self):
        "yy docs"
        raise NotImplementedError


class AzimuthalRotation(Transform2D):
    @property
    def angle(self):
        "angle docs"
        raise NotImplementedError


class Transform3D:
    @property
    def xx(self):
        "xx docs"
        raise NotImplementedError

    @property
    def xy(self):
        "xy docs"
        raise NotImplementedError

    @property
    def xz(self):
        "xz docs"
        raise NotImplementedError

    @property
    def yx(self):
        "yx docs"
        raise NotImplementedError

    @property
    def yy(self):
        "yy docs"
        raise NotImplementedError

    @property
    def yz(self):
        "yz docs"
        raise NotImplementedError

    @property
    def zx(self):
        "zx docs"
        raise NotImplementedError

    @property
    def zy(self):
        "zy docs"
        raise NotImplementedError

    @property
    def zz(self):
        "zz docs"
        raise NotImplementedError


class AxisAngleRotation(Transform3D):
    @property
    def phi(self):
        "phi docs"
        raise NotImplementedError

    @property
    def theta(self):
        "theta docs"
        raise NotImplementedError

    @property
    def angle(self):
        "angle docs"
        raise NotImplementedError


class EulerAngleRotation(Transform3D):
    @property
    def alpha(self):
        "alpha docs"
        raise NotImplementedError

    @property
    def beta(self):
        "beta docs"
        raise NotImplementedError

    @property
    def gamma(self):
        "gamma docs"
        raise NotImplementedError


class NauticalRotation(Transform3D):
    @property
    def yaw(self):
        "yaw docs"
        raise NotImplementedError

    @property
    def pitch(self):
        "pitch docs"
        raise NotImplementedError

    @property
    def roll(self):
        "roll docs"
        raise NotImplementedError


class Transform4D:
    @property
    def xx(self):
        "xx docs"
        raise NotImplementedError

    @property
    def xy(self):
        "xy docs"
        raise NotImplementedError

    @property
    def xz(self):
        "xz docs"
        raise NotImplementedError

    @property
    def xt(self):
        "xt docs"
        raise NotImplementedError

    @property
    def yx(self):
        "yx docs"
        raise NotImplementedError

    @property
    def yy(self):
        "yy docs"
        raise NotImplementedError

    @property
    def yz(self):
        "yz docs"
        raise NotImplementedError

    @property
    def yt(self):
        "yt docs"
        raise NotImplementedError

    @property
    def zx(self):
        "zx docs"
        raise NotImplementedError

    @property
    def zy(self):
        "zy docs"
        raise NotImplementedError

    @property
    def zz(self):
        "zz docs"
        raise NotImplementedError

    @property
    def zt(self):
        "zt docs"
        raise NotImplementedError

    @property
    def tx(self):
        "tx docs"
        raise NotImplementedError

    @property
    def ty(self):
        "ty docs"
        raise NotImplementedError

    @property
    def tz(self):
        "tz docs"
        raise NotImplementedError

    @property
    def tt(self):
        "tt docs"
        raise NotImplementedError


class LongitudinalBoost(Transform3D):
    @property
    def beta(self):
        "beta docs (derived)"
        raise NotImplementedError

    @property
    def gamma(self):
        "gamma docs (stored)"
        raise NotImplementedError


class AxisAngleBoost(Transform3D):
    @property
    def phi(self):
        "phi docs"
        raise NotImplementedError

    @property
    def theta(self):
        "theta docs"
        raise NotImplementedError

    @property
    def beta(self):
        "beta docs (derived)"
        raise NotImplementedError

    @property
    def gamma(self):
        "gamma docs (stored)"
        raise NotImplementedError
