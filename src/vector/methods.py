# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import vector.compute.lorentz
import vector.compute.planar
import vector.compute.spatial
import vector.compute.transform2d
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
    def rho2(self):
        "rho2 docs"
        return vector.compute.planar.rho2.dispatch(self)

    @property
    def phi(self):
        "phi docs"
        return vector.compute.planar.phi.dispatch(self)

    def deltaphi(self, other):
        "deltaphi docs"
        return vector.compute.planar.deltaphi.dispatch(self, other)

    def rotateZ(self, angle):
        "rotateZ docs"
        return vector.compute.planar.rotateZ.dispatch(angle, self)


class Spatial(Planar):
    @property
    def z(self):
        "z docs"
        return vector.compute.spatial.z.dispatch(self)

    @property
    def theta(self):
        "theta docs"
        return vector.compute.spatial.theta.dispatch(self)

    @property
    def eta(self):
        "eta docs"
        return vector.compute.spatial.eta.dispatch(self)

    @property
    def costheta(self):
        "costheta docs"
        return vector.compute.spatial.costheta.dispatch(self)

    @property
    def cottheta(self):
        "cottheta docs"
        return vector.compute.spatial.cottheta.dispatch(self)

    @property
    def mag(self):
        "mag docs"
        return vector.compute.spatial.mag.dispatch(self)

    @property
    def mag2(self):
        "mag2 docs"
        return vector.compute.spatial.mag2.dispatch(self)

    def cross(self, other):
        "cross docs"
        return vector.compute.spatial.cross.dispatch(self, other)

    def deltaeta(self, other):
        "deltaeta docs"
        return vector.compute.spatial.deltaeta.dispatch(self, other)

    def deltaR(self, other):
        "deltaR docs"
        return vector.compute.spatial.deltaR.dispatch(self, other)

    def deltaR2(self, other):
        "deltaR2 docs"
        return vector.compute.spatial.deltaR2.dispatch(self, other)

    def rotateX(self, angle):
        "rotateX docs"
        return vector.compute.spatial.rotateX.dispatch(angle, self)

    def rotateY(self, angle):
        "rotateY docs"
        return vector.compute.spatial.rotateY.dispatch(angle, self)

    def rotate_axis(self, axis, angle):
        "rotate_axis docs"
        return vector.compute.spatial.rotate_axis.dispatch(angle, axis, self)

    def rotate_euler(self, phi, theta, psi, order="zxz"):
        """
        rotate_euler docs

        same conventions as ROOT
        """
        return vector.compute.spatial.rotate_euler.dispatch(
            phi, theta, psi, order.lower(), self
        )

    def rotate_nautical(self, yaw, pitch, roll):
        """
        rotate_nautical docs

        transforming "from the body frame to the inertial frame"

        http://planning.cs.uiuc.edu/node102.html
        http://www.chrobotics.com/library/understanding-euler-angles
        """
        # The order of arguments is reversed because rotate_euler
        # follows ROOT's argument order: phi, theta, psi.
        return vector.compute.spatial.rotate_euler.dispatch(
            roll, pitch, yaw, "zyx", self
        )

    def rotate_quaternion(self, u, i, j, k):
        """
        rotate_quaternion docs

        same conventions as ROOT
        """
        return vector.compute.spatial.rotate_quaternion.dispatch(u, i, j, k, self)


class Lorentz(Spatial):
    @property
    def t(self):
        "t docs"
        return vector.compute.lorentz.t.dispatch(self)

    @property
    def t2(self):
        "t2 docs"
        return vector.compute.lorentz.t2.dispatch(self)

    @property
    def tau(self):
        "tau docs"
        return vector.compute.lorentz.tau.dispatch(self)

    @property
    def tau2(self):
        "tau2 docs"
        return vector.compute.lorentz.tau2.dispatch(self)


class Transform2D(vector.geometry.Transform):
    def apply(self, v):
        "apply docs"
        return vector.compute.transform2d.dispatch(v, self)

    @property
    def elements(self):
        "matrix elements docs"
        raise AssertionError

    @property
    def xx(self):
        "xx docs"
        return self.elements[0]

    @property
    def xy(self):
        "xy docs"
        return self.elements[1]

    @property
    def yx(self):
        "yx docs"
        return self.elements[2]

    @property
    def yy(self):
        "yy docs"
        return self.elements[3]


class AzimuthalRotation(Transform2D):
    @property
    def angle(self):
        "angle docs"
        raise AssertionError

    @property
    def elements(self):
        "matrix elements docs"
        return vector.compute.transform2d.from_AzimuthalRotation(self.lib, self.angle)


class Transform3D(vector.geometry.Transform):
    def apply(self, v):
        "apply docs"
        return vector.compute.transform3d.dispatch(v, self)

    @property
    def elements(self):
        "matrix elements docs"
        raise AssertionError

    @property
    def xx(self):
        "xx docs"
        return self.elements[0]

    @property
    def xy(self):
        "xy docs"
        return self.elements[1]

    @property
    def xz(self):
        "xz docs"
        return self.elements[2]

    @property
    def yx(self):
        "yx docs"
        return self.elements[3]

    @property
    def yy(self):
        "yy docs"
        return self.elements[4]

    @property
    def yz(self):
        "yz docs"
        return self.elements[5]

    @property
    def zx(self):
        "zx docs"
        return self.elements[6]

    @property
    def zy(self):
        "zy docs"
        return self.elements[7]

    @property
    def zz(self):
        "zz docs"
        return self.elements[8]


class AxisAngleRotation(Transform3D):
    @property
    def phi(self):
        "phi docs"
        raise AssertionError

    @property
    def theta(self):
        "theta docs"
        raise AssertionError

    @property
    def angle(self):
        "angle docs"
        raise AssertionError

    @property
    def elements(self):
        "matrix elements docs"
        return vector.compute.transform3d.from_AxisAngleRotation(
            self.lib, self.phi, self.theta, self.angle
        )


class EulerAngleRotation(Transform3D):
    @property
    def alpha(self):
        "alpha docs"
        raise AssertionError

    @property
    def beta(self):
        "beta docs"
        raise AssertionError

    @property
    def gamma(self):
        "gamma docs"
        raise AssertionError

    @property
    def elements(self):
        "matrix elements docs"
        return vector.compute.transform3d.from_EulerAngleRotation(
            self.lib, self.alpha, self.beta, self.gamma
        )


class NauticalRotation(Transform3D):
    @property
    def yaw(self):
        "yaw docs"
        raise AssertionError

    @property
    def pitch(self):
        "pitch docs"
        raise AssertionError

    @property
    def roll(self):
        "roll docs"
        raise AssertionError

    @property
    def elements(self):
        "matrix elements docs"
        return vector.compute.transform3d.from_NauticalRotation(
            self.lib, self.yaw, self.pitch, self.roll
        )


class Transform4D(vector.geometry.Transform):
    def apply(self, v):
        "apply docs"
        return vector.compute.transform4d.dispatch(v, self)

    @property
    def elements(self):
        "matrix elements docs"
        raise AssertionError

    @property
    def xx(self):
        "xx docs"
        return self.elements[0]

    @property
    def xy(self):
        "xy docs"
        return self.elements[1]

    @property
    def xz(self):
        "xz docs"
        return self.elements[2]

    @property
    def xt(self):
        "xt docs"
        return self.elements[3]

    @property
    def yx(self):
        "yx docs"
        return self.elements[4]

    @property
    def yy(self):
        "yy docs"
        return self.elements[5]

    @property
    def yz(self):
        "yz docs"
        return self.elements[6]

    @property
    def yt(self):
        "yt docs"
        return self.elements[7]

    @property
    def zx(self):
        "zx docs"
        return self.elements[8]

    @property
    def zy(self):
        "zy docs"
        return self.elements[9]

    @property
    def zz(self):
        "zz docs"
        return self.elements[10]

    @property
    def zt(self):
        "zt docs"
        return self.elements[11]

    @property
    def tx(self):
        "tx docs"
        return self.elements[12]

    @property
    def ty(self):
        "ty docs"
        return self.elements[13]

    @property
    def tz(self):
        "tz docs"
        return self.elements[14]

    @property
    def tt(self):
        "tt docs"
        return self.elements[15]


class LongitudinalBoost(Transform4D):
    @property
    def beta(self):
        "beta docs (derived)"
        return vector.compute.transform4d.beta(self.lib, self.gamma)

    @property
    def gamma(self):
        "gamma docs (stored)"
        raise AssertionError

    @property
    def elements(self):
        "matrix elements docs"
        return vector.compute.transform4d.from_LongitudinalBoost(
            self.lib, self.beta, self.gamma
        )


class AxisAngleBoost(Transform4D):
    @property
    def phi(self):
        "phi docs"
        raise AssertionError

    @property
    def theta(self):
        "theta docs"
        raise AssertionError

    @property
    def beta(self):
        "beta docs (derived)"
        return vector.compute.transform4d.beta(self.lib, self.gamma)

    @property
    def gamma(self):
        "gamma docs (stored)"
        raise AssertionError

    @property
    def elements(self):
        "matrix elements docs"
        return vector.compute.transform4d.from_AxisAngleBoost(
            self.lib, self.phi, self.theta, self.beta, self.gamma
        )
