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
    def azimuthal(self):
        "azimuthal docs"
        raise AssertionError(repr(type(self)))

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
        """
        deltaphi docs

        (it's the signed difference, not arccos(dot))
        """
        return vector.compute.planar.deltaphi.dispatch(self, other)

    def rotateZ(self, angle):
        "rotateZ docs"
        return vector.compute.planar.rotateZ.dispatch(angle, self)

    def transform2D(self, obj):
        "transform2D docs"
        return vector.compute.planar.transform2D.dispatch(obj, self)

    def unit(self):
        "unit docs"
        return vector.compute.planar.unit.dispatch(self)

    def dot(self, other):
        "dot docs"
        return vector.compute.planar.dot.dispatch(self, other)


class Spatial(Planar):
    @property
    def longitudinal(self):
        "longitudinal docs"
        raise AssertionError(repr(type(self)))

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

    def deltaangle(self, other):
        """
        deltaangle docs

        (it's just arccos(dot))
        """
        return vector.compute.spatial.deltaangle.dispatch(self, other)

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

    def unit(self):
        "unit docs"
        return vector.compute.spatial.unit.dispatch(self)

    def dot(self, other):
        "dot docs"
        return vector.compute.spatial.dot.dispatch(self, other)


class Lorentz(Spatial):
    @property
    def temporal(self):
        "temporal docs"
        raise AssertionError(repr(type(self)))

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

    @property
    def beta(self):
        "beta docs"
        return vector.compute.lorentz.beta.dispatch(self)

    @property
    def gamma(self):
        "gamma docs"
        return vector.compute.lorentz.gamma.dispatch(self)

    @property
    def rapidity(self):
        "rapidity docs"
        return vector.compute.lorentz.rapidity.dispatch(self)

    @property
    def Et(self):
        "Et docs"
        return vector.compute.lorentz.Et.dispatch(self)

    @property
    def Et2(self):
        "Et2 docs"
        return vector.compute.lorentz.Et2.dispatch(self)

    @property
    def Mt(self):
        "Mt docs"
        return vector.compute.lorentz.Mt.dispatch(self)

    @property
    def Mt2(self):
        "Mt2 docs"
        return vector.compute.lorentz.Mt2.dispatch(self)

    def transform4D(self, obj):
        "transform4D docs"
        return vector.compute.lorentz.transform4D.dispatch(obj, self)

    def unit(self):
        "unit docs"
        return vector.compute.lorentz.unit.dispatch(self)

    def dot(self, other):
        "dot docs"
        return vector.compute.lorentz.dot.dispatch(self, other)

    def boost_p4(self, p4):
        "boost_p4 docs"
        return vector.compute.lorentz.boost_p4.dispatch(self, p4)


class Momentum:
    pass


class PlanarMomentum(Momentum):
    @property
    def pt(self):
        "pt docs"
        return self.rho

    @property
    def pt2(self):
        "pt2 docs"
        return self.rho2


class SpatialMomentum(PlanarMomentum):
    @property
    def pseudorapidity(self):
        "pseudorapidity docs"
        return self.eta

    @property
    def p(self):
        "p docs"
        return self.mag

    @property
    def p2(self):
        "p2 docs"
        return self.mag2


class LorentzMomentum(SpatialMomentum):
    @property
    def E(self):
        "E docs"
        return self.t

    @property
    def energy(self):
        "energy docs"
        return self.t

    @property
    def E2(self):
        "E2 docs"
        return self.t2

    @property
    def energy2(self):
        "energy2 docs"
        return self.t2

    @property
    def m(self):
        "m docs"
        return self.tau

    @property
    def mass(self):
        "mass docs"
        return self.tau

    @property
    def m2(self):
        "m2 docs"
        return self.tau2

    @property
    def mass2(self):
        "mass2 docs"
        return self.tau2

    @property
    def transverse_energy(self):
        "transverse_energy docs"
        return self.Et

    @property
    def transverse_energy2(self):
        "transverse_energy2 docs"
        return self.Et2

    @property
    def transverse_mass(self):
        "transverse_mass docs"
        return self.mt

    @property
    def transverse_mass2(self):
        "transverse_mass2 docs"
        return self.mt2
