# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import vector.compute.lorentz
import vector.compute.planar
import vector.compute.spatial
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

    def transform2D(self, obj):
        "transform2D docs"
        return vector.compute.planar.transform2D.dispatch(obj, self)


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

    def transform3D(self, obj):
        "transform3D docs"
        return vector.compute.spatial.transform3D.dispatch(obj, self)


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

    def transform4D(self, obj):
        "transform4D docs"
        return vector.compute.lorentz.transform4D.dispatch(obj, self)
