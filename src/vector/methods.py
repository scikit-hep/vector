# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.


class Vector:
    pass


class Vector2D(Vector):
    def to_xy(self):
        "to_xy docs"
        from .compute.planar import x, y

        return self._wrap_result((x.dispatch(self), y.dispatch(self)), [AzimuthalXY])

    def to_rhophi(self):
        "to_rhophi docs"
        from .compute.planar import phi, rho

        return self._wrap_result(
            (rho.dispatch(self), phi.dispatch(self)), [AzimuthalRhoPhi]
        )


class Vector3D(Vector):
    def to_Vector2D(self):
        "to_Vector2D docs"
        return self.ProjectionClass2D._wrap_result(
            self.ProjectionClass2D, self.azimuthal.elements, [_aztype(self)]
        )

    def to_xy(self):
        "to_xy docs, mention projection"
        return self.to_Vector2D().to_xy()

    def to_rhophi(self):
        "to_rhophi docs, mention projection"
        return self.to_Vector2D().to_rhophi()

    def to_xyz(self):
        "to_xyz docs"
        from .compute.planar import x, y
        from .compute.spatial import z

        return self._wrap_result(
            (x.dispatch(self), y.dispatch(self), z.dispatch(self)),
            [AzimuthalXY, LongitudinalZ],
        )

    def to_xytheta(self):
        "to_xytheta docs"
        from .compute.planar import x, y
        from .compute.spatial import theta

        return self._wrap_result(
            (x.dispatch(self), y.dispatch(self), theta.dispatch(self)),
            [AzimuthalXY, LongitudinalTheta],
        )

    def to_xyeta(self):
        "to_xyeta docs"
        from .compute.planar import x, y
        from .compute.spatial import eta

        return self._wrap_result(
            (x.dispatch(self), y.dispatch(self), eta.dispatch(self)),
            [AzimuthalXY, LongitudinalEta],
        )

    def to_rhophiz(self):
        "to_rhophiz docs"
        from .compute.planar import phi, rho
        from .compute.spatial import z

        return self._wrap_result(
            (rho.dispatch(self), phi.dispatch(self), z.dispatch(self)),
            [AzimuthalRhoPhi, LongitudinalZ],
        )

    def to_rhophitheta(self):
        "to_rhophitheta docs"
        from .compute.planar import phi, rho
        from .compute.spatial import theta

        return self._wrap_result(
            (rho.dispatch(self), phi.dispatch(self), theta.dispatch(self)),
            [AzimuthalRhoPhi, LongitudinalTheta],
        )

    def to_rhophieta(self):
        "to_rhophieta docs"
        from .compute.planar import phi, rho
        from .compute.spatial import eta

        return self._wrap_result(
            (rho.dispatch(self), phi.dispatch(self), eta.dispatch(self)),
            [AzimuthalRhoPhi, LongitudinalEta],
        )


class Vector4D(Vector):
    def to_Vector3D(self):
        "to_Vector3D docs"
        return self.ProjectionClass3D._wrap_result(
            self.ProjectionClass3D,
            self.azimuthal.elements + self.longitudinal.elements,
            [_aztype(self), _ltype(self)],
        )

    def to_xyz(self):
        "to_xyz docs, mention projection"
        return self.to_Vector3D().to_xyz()

    def to_xytheta(self):
        "to_xytheta docs, mention projection"
        return self.to_Vector3D().to_xytheta()

    def to_xyeta(self):
        "to_xyeta docs, mention projection"
        return self.to_Vector3D().to_xyeta()

    def to_rhophiz(self):
        "to_rhophiz docs, mention projection"
        return self.to_Vector3D().to_rhophiz()

    def to_rhophitheta(self):
        "to_rhophitheta docs, mention projection"
        return self.to_Vector3D().to_rhophitheta()

    def to_rhophieta(self):
        "to_rhophieta docs, mention projection"
        return self.to_Vector3D().to_rhophieta()

    def to_xyzt(self):
        "to_xyzt docs"
        from .compute.lorentz import t
        from .compute.planar import x, y
        from .compute.spatial import z

        return self._wrap_result(
            (x.dispatch(self), y.dispatch(self), z.dispatch(self), t.dispatch(self)),
            [AzimuthalXY, LongitudinalZ, TemporalT],
        )

    def to_xyztau(self):
        "to_xyztau docs"
        from .compute.lorentz import tau
        from .compute.planar import x, y
        from .compute.spatial import z

        return self._wrap_result(
            (x.dispatch(self), y.dispatch(self), z.dispatch(self), tau.dispatch(self)),
            [AzimuthalXY, LongitudinalZ, TemporalTau],
        )

    def to_xythetat(self):
        "to_xythetat docs"
        from .compute.lorentz import t
        from .compute.planar import x, y
        from .compute.spatial import theta

        return self._wrap_result(
            (
                x.dispatch(self),
                y.dispatch(self),
                theta.dispatch(self),
                t.dispatch(self),
            ),
            [AzimuthalXY, LongitudinalTheta, TemporalT],
        )

    def to_xythetatau(self):
        "to_xythetatau docs"
        from .compute.lorentz import tau
        from .compute.planar import x, y
        from .compute.spatial import theta

        return self._wrap_result(
            (
                x.dispatch(self),
                y.dispatch(self),
                theta.dispatch(self),
                tau.dispatch(self),
            ),
            [AzimuthalXY, LongitudinalTheta, TemporalTau],
        )

    def to_xyetat(self):
        "to_xyetat docs"
        from .compute.lorentz import t
        from .compute.planar import x, y
        from .compute.spatial import eta

        return self._wrap_result(
            (x.dispatch(self), y.dispatch(self), eta.dispatch(self), t.dispatch(self)),
            [AzimuthalXY, LongitudinalEta, TemporalT],
        )

    def to_xyetatau(self):
        "to_xyetatau docs"
        from .compute.lorentz import tau
        from .compute.planar import x, y
        from .compute.spatial import eta

        return self._wrap_result(
            (
                x.dispatch(self),
                y.dispatch(self),
                eta.dispatch(self),
                tau.dispatch(self),
            ),
            [AzimuthalXY, LongitudinalEta, TemporalTau],
        )

    def to_rhophizt(self):
        "to_rhophizt docs"
        from .compute.lorentz import t
        from .compute.planar import phi, rho
        from .compute.spatial import z

        return self._wrap_result(
            (
                rho.dispatch(self),
                phi.dispatch(self),
                z.dispatch(self),
                t.dispatch(self),
            ),
            [AzimuthalRhoPhi, LongitudinalZ, TemporalT],
        )

    def to_rhophiztau(self):
        "to_rhophiztau docs"
        from .compute.lorentz import tau
        from .compute.planar import phi, rho
        from .compute.spatial import z

        return self._wrap_result(
            (
                rho.dispatch(self),
                phi.dispatch(self),
                z.dispatch(self),
                tau.dispatch(self),
            ),
            [AzimuthalRhoPhi, LongitudinalZ, TemporalTau],
        )

    def to_rhophithetat(self):
        "to_rhophithetat docs"
        from .compute.lorentz import t
        from .compute.planar import phi, rho
        from .compute.spatial import theta

        return self._wrap_result(
            (
                rho.dispatch(self),
                phi.dispatch(self),
                theta.dispatch(self),
                t.dispatch(self),
            ),
            [AzimuthalRhoPhi, LongitudinalTheta, TemporalT],
        )

    def to_rhophithetatau(self):
        "to_rhophithetatau docs"
        from .compute.lorentz import tau
        from .compute.planar import phi, rho
        from .compute.spatial import theta

        return self._wrap_result(
            (
                rho.dispatch(self),
                phi.dispatch(self),
                theta.dispatch(self),
                tau.dispatch(self),
            ),
            [AzimuthalRhoPhi, LongitudinalTheta, TemporalTau],
        )

    def to_rhophietat(self):
        "to_rhophietat docs"
        from .compute.lorentz import t
        from .compute.planar import phi, rho
        from .compute.spatial import eta

        return self._wrap_result(
            (
                rho.dispatch(self),
                phi.dispatch(self),
                eta.dispatch(self),
                t.dispatch(self),
            ),
            [AzimuthalRhoPhi, LongitudinalEta, TemporalT],
        )

    def to_rhophietatau(self):
        "to_rhophietatau docs"
        from .compute.lorentz import tau
        from .compute.planar import phi, rho
        from .compute.spatial import eta

        return self._wrap_result(
            (
                rho.dispatch(self),
                phi.dispatch(self),
                eta.dispatch(self),
                tau.dispatch(self),
            ),
            [AzimuthalRhoPhi, LongitudinalEta, TemporalTau],
        )


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


def _aztype(obj):
    for t in type(obj.azimuthal).__mro__:
        if t in (AzimuthalXY, AzimuthalRhoPhi):
            return t
    else:
        return None


def _ltype(obj):
    for t in type(obj.longitudinal).__mro__:
        if t in (LongitudinalZ, LongitudinalTheta, LongitudinalEta):
            return t
    else:
        return None


def _ttype(obj):
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


_repr_generic_to_momentum = {
    "x": "px",
    "y": "py",
    "rho": "pt",
    "z": "pz",
    "t": "E",
    "tau": "mass",
}


_repr_momentum_to_generic = {
    "px": "x",
    "py": "y",
    "pt": "rho",
    "pz": "z",
    "E": "t",
    "e": "t",
    "energy": "t",
    "M": "tau",
    "m": "tau",
    "mass": "tau",
}


_coordinate_order = [
    "x",
    "px",
    "y",
    "py",
    "rho",
    "pt",
    "phi",
    "z",
    "pz",
    "theta",
    "eta",
    "t",
    "E",
    "e",
    "energy",
    "tau",
    "M",
    "m",
    "mass",
]


class Planar:
    @property
    def azimuthal(self):
        "azimuthal docs"
        raise AssertionError(repr(type(self)))

    @property
    def x(self):
        "x docs"
        from .compute.planar import x

        return x.dispatch(self)

    @property
    def y(self):
        "y docs"
        from .compute.planar import y

        return y.dispatch(self)

    @property
    def rho(self):
        "rho docs"
        from .compute.planar import rho

        return rho.dispatch(self)

    @property
    def rho2(self):
        "rho2 docs"
        from .compute.planar import rho2

        return rho2.dispatch(self)

    @property
    def phi(self):
        "phi docs"
        from .compute.planar import phi

        return phi.dispatch(self)

    def deltaphi(self, other):
        """
        deltaphi docs

        (it's the signed difference, not arccos(dot))
        """
        from .compute.planar import deltaphi

        return deltaphi.dispatch(self, other)

    def rotateZ(self, angle):
        "rotateZ docs"
        from .compute.planar import rotateZ

        return rotateZ.dispatch(angle, self)

    def transform2D(self, obj):
        "transform2D docs"
        from .compute.planar import transform2D

        return transform2D.dispatch(obj, self)

    def is_parallel(self, other, tolerance=1e-5):
        "is_parallel docs (note: this 'parallel' requires same direction)"
        from .compute.planar import is_parallel

        return is_parallel.dispatch(tolerance, self, other)

    def is_antiparallel(self, other, tolerance=1e-5):
        "is_antiparallel docs"
        from .compute.planar import is_antiparallel

        return is_antiparallel.dispatch(tolerance, self, other)

    def is_perpendicular(self, other, tolerance=1e-5):
        "is_perpendicular docs"
        from .compute.planar import is_perpendicular

        return is_perpendicular.dispatch(tolerance, self, other)

    def unit(self):
        "unit docs"
        from .compute.planar import unit

        return unit.dispatch(self)

    def dot(self, other):
        "dot docs"
        from .compute.planar import dot

        return dot.dispatch(self, other)


class Spatial(Planar):
    @property
    def longitudinal(self):
        "longitudinal docs"
        raise AssertionError(repr(type(self)))

    @property
    def z(self):
        "z docs"
        from .compute.spatial import z

        return z.dispatch(self)

    @property
    def theta(self):
        "theta docs"
        from .compute.spatial import theta

        return theta.dispatch(self)

    @property
    def eta(self):
        "eta docs"
        from .compute.spatial import eta

        return eta.dispatch(self)

    @property
    def costheta(self):
        "costheta docs"
        from .compute.spatial import costheta

        return costheta.dispatch(self)

    @property
    def cottheta(self):
        "cottheta docs"
        from .compute.spatial import cottheta

        return cottheta.dispatch(self)

    @property
    def mag(self):
        "mag docs"
        from .compute.spatial import mag

        return mag.dispatch(self)

    @property
    def mag2(self):
        "mag2 docs"
        from .compute.spatial import mag2

        return mag2.dispatch(self)

    def cross(self, other):
        "cross docs"
        from .compute.spatial import cross

        return cross.dispatch(self, other)

    def deltaangle(self, other):
        """
        deltaangle docs

        (it's just arccos(dot))
        """
        from .compute.spatial import deltaangle

        return deltaangle.dispatch(self, other)

    def deltaeta(self, other):
        "deltaeta docs"
        from .compute.spatial import deltaeta

        return deltaeta.dispatch(self, other)

    def deltaR(self, other):
        "deltaR docs"
        from .compute.spatial import deltaR

        return deltaR.dispatch(self, other)

    def deltaR2(self, other):
        "deltaR2 docs"
        from .compute.spatial import deltaR2

        return deltaR2.dispatch(self, other)

    def rotateX(self, angle):
        "rotateX docs"
        from .compute.spatial import rotateX

        return rotateX.dispatch(angle, self)

    def rotateY(self, angle):
        "rotateY docs"
        from .compute.spatial import rotateY

        return rotateY.dispatch(angle, self)

    def rotate_axis(self, axis, angle):
        "rotate_axis docs"
        from .compute.spatial import rotate_axis

        return rotate_axis.dispatch(angle, axis, self)

    def rotate_euler(self, phi, theta, psi, order="zxz"):
        """
        rotate_euler docs

        same conventions as ROOT
        """
        from .compute.spatial import rotate_euler

        return rotate_euler.dispatch(phi, theta, psi, order.lower(), self)

    def rotate_nautical(self, yaw, pitch, roll):
        """
        rotate_nautical docs

        transforming "from the body frame to the inertial frame"

        http://planning.cs.uiuc.edu/node102.html
        http://www.chrobotics.com/library/understanding-euler-angles
        """
        # The order of arguments is reversed because rotate_euler
        # follows ROOT's argument order: phi, theta, psi.
        from .compute.spatial import rotate_euler

        return rotate_euler.dispatch(roll, pitch, yaw, "zyx", self)

    def rotate_quaternion(self, u, i, j, k):
        """
        rotate_quaternion docs

        same conventions as ROOT
        """
        from .compute.spatial import rotate_quaternion

        return rotate_quaternion.dispatch(u, i, j, k, self)

    def transform3D(self, obj):
        "transform3D docs"
        from .compute.spatial import transform3D

        return transform3D.dispatch(obj, self)

    def is_parallel(self, other, tolerance=1e-5):
        "is_parallel docs (note: this 'parallel' requires same direction)"
        from .compute.spatial import is_parallel

        return is_parallel.dispatch(tolerance, self, other)

    def is_antiparallel(self, other, tolerance=1e-5):
        "is_antiparallel docs"
        from .compute.spatial import is_antiparallel

        return is_antiparallel.dispatch(tolerance, self, other)

    def is_perpendicular(self, other, tolerance=1e-5):
        "is_perpendicular docs"
        from .compute.spatial import is_perpendicular

        return is_perpendicular.dispatch(tolerance, self, other)

    def unit(self):
        "unit docs"
        from .compute.spatial import unit

        return unit.dispatch(self)

    def dot(self, other):
        "dot docs"
        from .compute.spatial import dot

        return dot.dispatch(self, other)


class Lorentz(Spatial):
    @property
    def temporal(self):
        "temporal docs"
        raise AssertionError(repr(type(self)))

    @property
    def t(self):
        "t docs"
        from .compute.lorentz import t

        return t.dispatch(self)

    @property
    def t2(self):
        "t2 docs"
        from .compute.lorentz import t2

        return t2.dispatch(self)

    @property
    def tau(self):
        "tau docs"
        from .compute.lorentz import tau

        return tau.dispatch(self)

    @property
    def tau2(self):
        "tau2 docs"
        from .compute.lorentz import tau2

        return tau2.dispatch(self)

    @property
    def beta(self):
        "beta docs"
        from .compute.lorentz import beta

        return beta.dispatch(self)

    @property
    def gamma(self):
        "gamma docs"
        from .compute.lorentz import gamma

        return gamma.dispatch(self)

    @property
    def rapidity(self):
        "rapidity docs"
        from .compute.lorentz import rapidity

        return rapidity.dispatch(self)

    def transform4D(self, obj):
        "transform4D docs"
        from .compute.lorentz import transform4D

        return transform4D.dispatch(obj, self)

    def is_timelike(self, tolerance=0):
        "is_timelike docs"
        from .compute.lorentz import is_timelike

        return is_timelike.dispatch(tolerance, self)

    def is_spacelike(self, tolerance=0):
        "is_spacelike docs"
        from .compute.lorentz import is_spacelike

        return is_spacelike.dispatch(tolerance, self)

    def is_lightlike(self, tolerance=1e-5):
        "is_timelike docs"
        from .compute.lorentz import is_lightlike

        return is_lightlike.dispatch(tolerance, self)

    def to_beta3(self):
        "to_beta3 docs"
        from .compute.lorentz import to_beta3

        return to_beta3.dispatch(self)

    def unit(self):
        "unit docs"
        from .compute.lorentz import unit

        return unit.dispatch(self)

    def dot(self, other):
        "dot docs"
        from .compute.lorentz import dot

        return dot.dispatch(self, other)

    def boost_p4(self, p4):
        "boost_p4 docs"
        from .compute.lorentz import boost_p4

        return boost_p4.dispatch(self, p4)

    def boost_beta3(self, beta3):
        "boost_beta3 docs"
        from .compute.lorentz import boost_beta3

        return boost_beta3.dispatch(self, beta3)

    def boost(self, booster):
        "boost docs"
        from .compute.lorentz import boost_beta3, boost_p4

        if isinstance(booster, Vector3D):
            return boost_beta3.dispatch(self, booster)
        elif isinstance(booster, Vector4D):
            return boost_p4.dispatch(self, booster)
        else:
            raise TypeError(
                "specify a Vector3D to boost by beta (velocity with c=1) or "
                "a Vector4D to boost by a momentum 4-vector"
            )

    def boostX(self, beta=None, gamma=None):
        "boostX docs"
        from .compute.lorentz import boostX_beta, boostX_gamma

        if beta is not None and gamma is None:
            return boostX_beta.dispatch(beta, self)
        elif beta is None and gamma is not None:
            return boostX_gamma.dispatch(gamma, self)
        else:
            raise TypeError("specify 'beta' xor 'gamma', not both or neither")

    def boostY(self, beta=None, gamma=None):
        "boostY docs"
        from .compute.lorentz import boostY_beta, boostY_gamma

        if beta is not None and gamma is None:
            return boostY_beta.dispatch(beta, self)
        elif beta is None and gamma is not None:
            return boostY_gamma.dispatch(gamma, self)
        else:
            raise TypeError("specify 'beta' xor 'gamma', not both or neither")

    def boostZ(self, beta=None, gamma=None):
        "boostZ docs"
        from .compute.lorentz import boostZ_beta, boostZ_gamma

        if beta is not None and gamma is None:
            return boostZ_beta.dispatch(beta, self)
        elif beta is None and gamma is not None:
            return boostZ_gamma.dispatch(gamma, self)
        else:
            raise TypeError("specify 'beta' xor 'gamma', not both or neither")


class Momentum:
    pass


class PlanarMomentum(Momentum):
    @property
    def px(self):
        "px docs"
        return self.x

    @property
    def py(self):
        "py docs"
        return self.y

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
    def pz(self):
        "pz docs"
        return self.z

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
    def Et(self):
        "Et docs"
        from .compute.lorentz import Et

        return Et.dispatch(self)

    @property
    def transverse_energy(self):
        "transverse_energy docs"
        return self.Et

    @property
    def Et2(self):
        "Et2 docs"
        from .compute.lorentz import Et2

        return Et2.dispatch(self)

    @property
    def transverse_energy2(self):
        "transverse_energy2 docs"
        return self.Et2

    @property
    def Mt(self):
        "Mt docs"
        from .compute.lorentz import Mt

        return Mt.dispatch(self)

    @property
    def transverse_mass(self):
        "transverse_mass docs"
        return self.mt

    @property
    def Mt2(self):
        "Mt2 docs"
        from .compute.lorentz import Mt2

        return Mt2.dispatch(self)

    @property
    def transverse_mass2(self):
        "transverse_mass2 docs"
        return self.mt2
