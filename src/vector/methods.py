# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

from .protocols import (
    MomentumProtocolLorentz,
    MomentumProtocolPlanar,
    MomentumProtocolSpatial,
    VectorProtocol,
    VectorProtocolLorentz,
    VectorProtocolPlanar,
    VectorProtocolSpatial,
)


class Coordinates:
    pass


class Azimuthal(Coordinates):
    @property
    def elements(self) -> typing.Any:
        "azimuthal elements docs"
        raise AssertionError


class Longitudinal(Coordinates):
    @property
    def elements(self) -> typing.Any:
        "longitudinal elements docs"
        raise AssertionError


class Temporal(Coordinates):
    @property
    def elements(self) -> typing.Any:
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


def _aztype(obj: typing.Any) -> typing.Any:
    if hasattr(obj, "azimuthal"):
        for t in type(obj.azimuthal).__mro__:
            if t in (AzimuthalXY, AzimuthalRhoPhi):
                return t
    return None


def _ltype(obj: typing.Any) -> typing.Any:
    if hasattr(obj, "longitudinal"):
        for t in type(obj.longitudinal).__mro__:
            if t in (LongitudinalZ, LongitudinalTheta, LongitudinalEta):
                return t
    return None


def _ttype(obj: typing.Any) -> typing.Any:
    if hasattr(obj, "temporal"):
        for t in type(obj.temporal).__mro__:
            if t in (TemporalT, TemporalTau):
                return t
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
    "energy": "t",
    "M": "tau",
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
    "energy",
    "tau",
    "M",
    "mass",
]


class Vector(VectorProtocol):
    def to_xy(self) -> typing.Any:
        "to_xy docs"
        from .compute import planar

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self)),
            [AzimuthalXY, None],
            1,
        )

    def to_rhophi(self) -> typing.Any:
        "to_rhophi docs"
        from .compute import planar

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self)),
            [AzimuthalRhoPhi, None],
            1,
        )

    def to_xyz(self) -> typing.Any:
        "to_xyz docs"
        from .compute import planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.z.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord),
            [AzimuthalXY, LongitudinalZ, None],
            1,
        )

    def to_xytheta(self) -> typing.Any:
        "to_xytheta docs"
        from .compute import planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.theta.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord),
            [AzimuthalXY, LongitudinalTheta, None],
            1,
        )

    def to_xyeta(self) -> typing.Any:
        "to_xyeta docs"
        from .compute import planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.eta.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord),
            [AzimuthalXY, LongitudinalEta, None],
            1,
        )

    def to_rhophiz(self) -> typing.Any:
        "to_rhophiz docs"
        from .compute import planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.z.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord),
            [AzimuthalRhoPhi, LongitudinalZ, None],
            1,
        )

    def to_rhophitheta(self) -> typing.Any:
        "to_rhophitheta docs"
        from .compute import planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.theta.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord),
            [AzimuthalRhoPhi, LongitudinalTheta, None],
            1,
        )

    def to_rhophieta(self) -> typing.Any:
        "to_rhophieta docs"
        from .compute import planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.eta.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord),
            [AzimuthalRhoPhi, LongitudinalEta, None],
            1,
        )

    def to_xyzt(self) -> typing.Any:
        "to_xyzt docs"
        from .compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.z.dispatch(self)
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.t.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord, tcoord),
            [AzimuthalXY, LongitudinalZ, TemporalT],
            1,
        )

    def to_xyztau(self) -> typing.Any:
        "to_xyztau docs"
        from .compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.z.dispatch(self)
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.tau.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord, tcoord),
            [AzimuthalXY, LongitudinalZ, TemporalTau],
            1,
        )

    def to_xythetat(self) -> typing.Any:
        "to_xythetat docs"
        from .compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.theta.dispatch(self)
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.t.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord, tcoord),
            [AzimuthalXY, LongitudinalTheta, TemporalT],
            1,
        )

    def to_xythetatau(self) -> typing.Any:
        "to_xythetatau docs"
        from .compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.theta.dispatch(self)
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.tau.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord, tcoord),
            [AzimuthalXY, LongitudinalTheta, TemporalTau],
            1,
        )

    def to_xyetat(self) -> typing.Any:
        "to_xyetat docs"
        from .compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.eta.dispatch(self)
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.t.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord, tcoord),
            [AzimuthalXY, LongitudinalEta, TemporalT],
            1,
        )

    def to_xyetatau(self) -> typing.Any:
        "to_xyetatau docs"
        from .compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.eta.dispatch(self)
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.tau.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord, tcoord),
            [AzimuthalXY, LongitudinalEta, TemporalTau],
            1,
        )

    def to_rhophizt(self) -> typing.Any:
        "to_rhophizt docs"
        from .compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.z.dispatch(self)
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.t.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord, tcoord),
            [AzimuthalRhoPhi, LongitudinalZ, TemporalT],
            1,
        )

    def to_rhophiztau(self) -> typing.Any:
        "to_rhophiztau docs"
        from .compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.z.dispatch(self)
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.tau.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord, tcoord),
            [AzimuthalRhoPhi, LongitudinalZ, TemporalTau],
            1,
        )

    def to_rhophithetat(self) -> typing.Any:
        "to_rhophithetat docs"
        from .compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.theta.dispatch(self)
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.t.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord, tcoord),
            [AzimuthalRhoPhi, LongitudinalTheta, TemporalT],
            1,
        )

    def to_rhophithetatau(self) -> typing.Any:
        "to_rhophithetatau docs"
        from .compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.theta.dispatch(self)
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.tau.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord, tcoord),
            [AzimuthalRhoPhi, LongitudinalTheta, TemporalTau],
            1,
        )

    def to_rhophietat(self) -> typing.Any:
        "to_rhophietat docs"
        from .compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.eta.dispatch(self)
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.t.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord, tcoord),
            [AzimuthalRhoPhi, LongitudinalEta, TemporalT],
            1,
        )

    def to_rhophietatau(self) -> typing.Any:
        "to_rhophietatau docs"
        from .compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.eta.dispatch(self)
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.tau.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord, tcoord),
            [AzimuthalRhoPhi, LongitudinalEta, TemporalTau],
            1,
        )


class Vector2D(Vector, VectorProtocolPlanar):
    def to_Vector2D(self) -> typing.Any:
        "to_Vector2D docs"
        return self

    def to_Vector3D(self) -> typing.Any:
        "to_Vector3D docs"
        return self._wrap_result(
            type(self),
            self.azimuthal.elements + (0,),
            [_aztype(self), LongitudinalZ, None],
            1,
        )

    def to_Vector4D(self) -> typing.Any:
        "to_Vector4D docs"
        return self._wrap_result(
            type(self),
            self.azimuthal.elements + (0, 0),
            [_aztype(self), LongitudinalZ, TemporalT],
            1,
        )


class Vector3D(Vector, VectorProtocolSpatial):
    def to_Vector2D(self) -> typing.Any:
        "to_Vector2D docs"
        return self._wrap_result(
            type(self),
            self.azimuthal.elements,
            [_aztype(self), None],
            1,
        )

    def to_Vector3D(self) -> typing.Any:
        "to_Vector3D docs"
        return self

    def to_Vector4D(self) -> typing.Any:
        "to_Vector4D docs"
        return self._wrap_result(
            type(self),
            self.azimuthal.elements + self.longitudinal.elements + (0,),
            [_aztype(self), _ltype(self), TemporalT],
            1,
        )


class Vector4D(Vector, VectorProtocolLorentz):
    def to_Vector2D(self) -> typing.Any:
        "to_Vector2D docs"
        return self._wrap_result(
            type(self),
            self.azimuthal.elements,
            [_aztype(self), None],
            1,
        )

    def to_Vector3D(self) -> typing.Any:
        "to_Vector3D docs"
        return self._wrap_result(
            type(self),
            self.azimuthal.elements + self.longitudinal.elements,
            [_aztype(self), _ltype(self), None],
            1,
        )

    def to_Vector4D(self) -> typing.Any:
        "to_Vector4D docs"
        return self


def dim(v: typing.Any) -> typing.Any:
    if isinstance(v, Vector2D):
        return 2
    elif isinstance(v, Vector3D):
        return 3
    elif isinstance(v, Vector4D):
        return 4
    else:
        raise TypeError(f"{repr(v)} is not a vector.Vector")


def _compute_module_of(
    one: typing.Any, two: typing.Any, nontemporal: typing.Any = False
) -> typing.Any:
    if not isinstance(one, Vector):
        raise TypeError(f"{repr(one)} is not a Vector")
    if not isinstance(two, Vector):
        raise TypeError(f"{repr(two)} is not a Vector")

    if isinstance(one, Vector2D):
        import vector.compute.planar

        return vector.compute.planar

    elif isinstance(one, Vector3D):
        if isinstance(two, Vector2D):
            import vector.compute.planar

            return vector.compute.planar
        else:
            import vector.compute.spatial

            return vector.compute.spatial

    elif isinstance(one, Vector4D):
        if isinstance(two, Vector2D):
            import vector.compute.planar

            return vector.compute.planar
        elif isinstance(two, Vector3D) or nontemporal:
            import vector.compute.spatial

            return vector.compute.spatial
        else:
            import vector.compute.lorentz

            return vector.compute.lorentz


class Planar(VectorProtocolPlanar):
    @property
    def azimuthal(self) -> typing.Any:
        "azimuthal docs"
        raise AssertionError(repr(type(self)))

    @property
    def x(self) -> typing.Any:
        "x docs"
        from .compute.planar import x

        return x.dispatch(self)

    @property
    def y(self) -> typing.Any:
        "y docs"
        from .compute.planar import y

        return y.dispatch(self)

    @property
    def rho(self) -> typing.Any:
        "rho docs"
        from .compute.planar import rho

        return rho.dispatch(self)

    @property
    def rho2(self) -> typing.Any:
        "rho2 docs"
        from .compute.planar import rho2

        return rho2.dispatch(self)

    @property
    def phi(self) -> typing.Any:
        "phi docs"
        from .compute.planar import phi

        return phi.dispatch(self)

    def deltaphi(self, other: typing.Any) -> typing.Any:
        """
        deltaphi docs

        (it's the signed difference, not arccos(dot))
        """
        from .compute.planar import deltaphi

        return deltaphi.dispatch(self, other)

    def rotateZ(self, angle: typing.Any) -> typing.Any:
        "rotateZ docs"
        from .compute.planar import rotateZ

        return rotateZ.dispatch(angle, self)

    def transform2D(self, obj: typing.Any) -> typing.Any:
        "transform2D docs"
        from .compute.planar import transform2D

        return transform2D.dispatch(obj, self)

    def is_parallel(
        self, other: typing.Any, tolerance: typing.Any = 1e-5
    ) -> typing.Any:
        "is_parallel docs (note: this 'parallel' requires same direction)"
        from .compute.planar import is_parallel

        if not isinstance(other, Vector2D):
            return self.to_Vector3D().is_parallel(other, tolerance=tolerance)
        else:
            return is_parallel.dispatch(tolerance, self, other)

    def is_antiparallel(
        self, other: typing.Any, tolerance: typing.Any = 1e-5
    ) -> typing.Any:
        "is_antiparallel docs"
        from .compute.planar import is_antiparallel

        if not isinstance(other, Vector2D):
            return self.to_Vector3D().is_antiparallel(other, tolerance=tolerance)
        else:
            return is_antiparallel.dispatch(tolerance, self, other)

    def is_perpendicular(
        self, other: typing.Any, tolerance: typing.Any = 1e-5
    ) -> typing.Any:
        "is_perpendicular docs"
        from .compute.planar import is_perpendicular

        if not isinstance(other, Vector2D):
            return self.to_Vector3D().is_perpendicular(other, tolerance=tolerance)
        else:
            return is_perpendicular.dispatch(tolerance, self, other)

    def unit(self) -> typing.Any:
        "unit docs"
        from .compute.planar import unit

        return unit.dispatch(self)

    def dot(self, other: typing.Any) -> typing.Any:
        "dot docs"
        module = _compute_module_of(self, other)
        return module.dot.dispatch(self, other)

    def add(self, other: typing.Any) -> typing.Any:
        "add docs"
        module = _compute_module_of(self, other)
        return module.add.dispatch(self, other)

    def subtract(self, other: typing.Any) -> typing.Any:
        "subtract docs"
        module = _compute_module_of(self, other)
        return module.subtract.dispatch(self, other)

    def scale(self, factor: typing.Any) -> typing.Any:
        "scale docs"
        from .compute.planar import scale

        return scale.dispatch(factor, self)

    def equal(self, other: typing.Any) -> typing.Any:
        "equal docs"
        from .compute.planar import equal

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return equal.dispatch(self, other)

    def not_equal(self, other: typing.Any) -> typing.Any:
        "not_equal docs"
        from .compute.planar import not_equal

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return not_equal.dispatch(self, other)

    def isclose(
        self,
        other: typing.Any,
        rtol: typing.Any = 1e-05,
        atol: typing.Any = 1e-08,
        equal_nan: typing.Any = False,
    ) -> typing.Any:
        "isclose docs"
        from .compute.planar import isclose

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return isclose.dispatch(rtol, atol, equal_nan, self, other)


class Spatial(Planar, VectorProtocolSpatial):
    @property
    def longitudinal(self) -> typing.Any:
        "longitudinal docs"
        raise AssertionError(repr(type(self)))

    @property
    def z(self) -> typing.Any:
        "z docs"
        from .compute.spatial import z

        return z.dispatch(self)

    @property
    def theta(self) -> typing.Any:
        "theta docs"
        from .compute.spatial import theta

        return theta.dispatch(self)

    @property
    def eta(self) -> typing.Any:
        "eta docs"
        from .compute.spatial import eta

        return eta.dispatch(self)

    @property
    def costheta(self) -> typing.Any:
        "costheta docs"
        from .compute.spatial import costheta

        return costheta.dispatch(self)

    @property
    def cottheta(self) -> typing.Any:
        "cottheta docs"
        from .compute.spatial import cottheta

        return cottheta.dispatch(self)

    @property
    def mag(self) -> typing.Any:
        "mag docs"
        from .compute.spatial import mag

        return mag.dispatch(self)

    @property
    def mag2(self) -> typing.Any:
        "mag2 docs"
        from .compute.spatial import mag2

        return mag2.dispatch(self)

    def cross(self, other: typing.Any) -> typing.Any:
        "cross docs"
        from .compute.spatial import cross

        return cross.dispatch(self, other)

    def deltaangle(self, other: typing.Any) -> typing.Any:
        """
        deltaangle docs

        (it's just arccos(dot))
        """
        from .compute.spatial import deltaangle

        return deltaangle.dispatch(self, other)

    def deltaeta(self, other: typing.Any) -> typing.Any:
        "deltaeta docs"
        from .compute.spatial import deltaeta

        return deltaeta.dispatch(self, other)

    def deltaR(self, other: typing.Any) -> typing.Any:
        "deltaR docs"
        from .compute.spatial import deltaR

        return deltaR.dispatch(self, other)

    def deltaR2(self, other: typing.Any) -> typing.Any:
        "deltaR2 docs"
        from .compute.spatial import deltaR2

        return deltaR2.dispatch(self, other)

    def rotateX(self, angle: typing.Any) -> typing.Any:
        "rotateX docs"
        from .compute.spatial import rotateX

        return rotateX.dispatch(angle, self)

    def rotateY(self, angle: typing.Any) -> typing.Any:
        "rotateY docs"
        from .compute.spatial import rotateY

        return rotateY.dispatch(angle, self)

    def rotate_axis(self, axis: typing.Any, angle: typing.Any) -> typing.Any:
        "rotate_axis docs"
        from .compute.spatial import rotate_axis

        return rotate_axis.dispatch(angle, axis, self)

    def rotate_euler(
        self,
        phi: typing.Any,
        theta: typing.Any,
        psi: typing.Any,
        order: typing.Any = "zxz",
    ) -> typing.Any:
        """
        rotate_euler docs

        same conventions as ROOT
        """
        from .compute.spatial import rotate_euler

        return rotate_euler.dispatch(phi, theta, psi, order.lower(), self)

    def rotate_nautical(
        self, yaw: typing.Any, pitch: typing.Any, roll: typing.Any
    ) -> typing.Any:
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

    def rotate_quaternion(
        self, u: typing.Any, i: typing.Any, j: typing.Any, k: typing.Any
    ) -> typing.Any:
        """
        rotate_quaternion docs

        same conventions as ROOT
        """
        from .compute.spatial import rotate_quaternion

        return rotate_quaternion.dispatch(u, i, j, k, self)

    def transform3D(self, obj: typing.Any) -> typing.Any:
        "transform3D docs"
        from .compute.spatial import transform3D

        return transform3D.dispatch(obj, self)

    def is_parallel(
        self, other: typing.Any, tolerance: typing.Any = 1e-5
    ) -> typing.Any:
        "is_parallel docs (note: this 'parallel' requires same direction)"
        from .compute.spatial import is_parallel

        if isinstance(other, Vector2D):
            return is_parallel.dispatch(tolerance, self, other.to_Vector3D())
        else:
            return is_parallel.dispatch(tolerance, self, other)

    def is_antiparallel(
        self, other: typing.Any, tolerance: typing.Any = 1e-5
    ) -> typing.Any:
        "is_antiparallel docs"
        from .compute.spatial import is_antiparallel

        if isinstance(other, Vector2D):
            return is_antiparallel.dispatch(tolerance, self, other.to_Vector3D())
        else:
            return is_antiparallel.dispatch(tolerance, self, other)

    def is_perpendicular(
        self, other: typing.Any, tolerance: typing.Any = 1e-5
    ) -> typing.Any:
        "is_perpendicular docs"
        from .compute.spatial import is_perpendicular

        if isinstance(other, Vector2D):
            return is_perpendicular.dispatch(tolerance, self, other.to_Vector3D())
        else:
            return is_perpendicular.dispatch(tolerance, self, other)

    def unit(self) -> typing.Any:
        "unit docs"
        from .compute.spatial import unit

        return unit.dispatch(self)

    def dot(self, other: typing.Any) -> typing.Any:
        "dot docs"
        module = _compute_module_of(self, other)
        return module.dot.dispatch(self, other)

    def add(self, other: typing.Any) -> typing.Any:
        "add docs"
        module = _compute_module_of(self, other)
        return module.add.dispatch(self, other)

    def subtract(self, other: typing.Any) -> typing.Any:
        "subtract docs"
        module = _compute_module_of(self, other)
        return module.subtract.dispatch(self, other)

    def scale(self, factor: typing.Any) -> typing.Any:
        "scale docs"
        from .compute.spatial import scale

        return scale.dispatch(factor, self)

    def equal(self, other: typing.Any) -> typing.Any:
        "equal docs"
        from .compute.spatial import equal

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return equal.dispatch(self, other)

    def not_equal(self, other: typing.Any) -> typing.Any:
        "not_equal docs"
        from .compute.spatial import not_equal

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return not_equal.dispatch(self, other)

    def isclose(
        self,
        other: typing.Any,
        rtol: typing.Any = 1e-05,
        atol: typing.Any = 1e-08,
        equal_nan: typing.Any = False,
    ) -> typing.Any:
        "isclose docs"
        from .compute.spatial import isclose

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return isclose.dispatch(rtol, atol, equal_nan, self, other)


class Lorentz(Spatial, VectorProtocolLorentz):
    @property
    def temporal(self) -> typing.Any:
        "temporal docs"
        raise AssertionError(repr(type(self)))

    @property
    def t(self) -> typing.Any:
        "t docs"
        from .compute.lorentz import t

        return t.dispatch(self)

    @property
    def t2(self) -> typing.Any:
        "t2 docs"
        from .compute.lorentz import t2

        return t2.dispatch(self)

    @property
    def tau(self) -> typing.Any:
        "tau docs"
        from .compute.lorentz import tau

        return tau.dispatch(self)

    @property
    def tau2(self) -> typing.Any:
        "tau2 docs"
        from .compute.lorentz import tau2

        return tau2.dispatch(self)

    @property
    def beta(self) -> typing.Any:
        "beta docs"
        from .compute.lorentz import beta

        return beta.dispatch(self)

    @property
    def gamma(self) -> typing.Any:
        "gamma docs"
        from .compute.lorentz import gamma

        return gamma.dispatch(self)

    @property
    def rapidity(self) -> typing.Any:
        "rapidity docs"
        from .compute.lorentz import rapidity

        return rapidity.dispatch(self)

    def boost_p4(self, p4: typing.Any) -> typing.Any:
        "boost_p4 docs"
        from .compute.lorentz import boost_p4

        return boost_p4.dispatch(self, p4)

    def boost_beta3(self, beta3: typing.Any) -> typing.Any:
        "boost_beta3 docs"
        from .compute.lorentz import boost_beta3

        return boost_beta3.dispatch(self, beta3)

    def boost(self, booster: typing.Any) -> typing.Any:
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

    def boostX(self, beta: typing.Any = None, gamma: typing.Any = None) -> typing.Any:
        "boostX docs"
        from .compute.lorentz import boostX_beta, boostX_gamma

        if beta is not None and gamma is None:
            return boostX_beta.dispatch(beta, self)
        elif beta is None and gamma is not None:
            return boostX_gamma.dispatch(gamma, self)
        else:
            raise TypeError("specify 'beta' xor 'gamma', not both or neither")

    def boostY(self, beta: typing.Any = None, gamma: typing.Any = None) -> typing.Any:
        "boostY docs"
        from .compute.lorentz import boostY_beta, boostY_gamma

        if beta is not None and gamma is None:
            return boostY_beta.dispatch(beta, self)
        elif beta is None and gamma is not None:
            return boostY_gamma.dispatch(gamma, self)
        else:
            raise TypeError("specify 'beta' xor 'gamma', not both or neither")

    def boostZ(self, beta: typing.Any = None, gamma: typing.Any = None) -> typing.Any:
        "boostZ docs"
        from .compute.lorentz import boostZ_beta, boostZ_gamma

        if beta is not None and gamma is None:
            return boostZ_beta.dispatch(beta, self)
        elif beta is None and gamma is not None:
            return boostZ_gamma.dispatch(gamma, self)
        else:
            raise TypeError("specify 'beta' xor 'gamma', not both or neither")

    def transform4D(self, obj: typing.Any) -> typing.Any:
        "transform4D docs"
        from .compute.lorentz import transform4D

        return transform4D.dispatch(obj, self)

    def to_beta3(self) -> typing.Any:
        "to_beta3 docs"
        from .compute.lorentz import to_beta3

        return to_beta3.dispatch(self)

    def is_timelike(self, tolerance: typing.Any = 0) -> typing.Any:
        "is_timelike docs"
        from .compute.lorentz import is_timelike

        return is_timelike.dispatch(tolerance, self)

    def is_spacelike(self, tolerance: typing.Any = 0) -> typing.Any:
        "is_spacelike docs"
        from .compute.lorentz import is_spacelike

        return is_spacelike.dispatch(tolerance, self)

    def is_lightlike(self, tolerance: typing.Any = 1e-5) -> typing.Any:
        "is_timelike docs"
        from .compute.lorentz import is_lightlike

        return is_lightlike.dispatch(tolerance, self)

    def unit(self) -> typing.Any:
        "unit docs"
        from .compute.lorentz import unit

        return unit.dispatch(self)

    def dot(self, other: typing.Any) -> typing.Any:
        "dot docs"
        module = _compute_module_of(self, other)
        return module.dot.dispatch(self, other)

    def add(self, other: typing.Any) -> typing.Any:
        "add docs"
        module = _compute_module_of(self, other)
        return module.add.dispatch(self, other)

    def subtract(self, other: typing.Any) -> typing.Any:
        "subtract docs"
        module = _compute_module_of(self, other)
        return module.subtract.dispatch(self, other)

    def scale(self, factor: typing.Any) -> typing.Any:
        "scale docs"
        from .compute.lorentz import scale

        return scale.dispatch(factor, self)

    def equal(self, other: typing.Any) -> typing.Any:
        "equal docs"
        from .compute.lorentz import equal

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return equal.dispatch(self, other)

    def not_equal(self, other: typing.Any) -> typing.Any:
        "not_equal docs"
        from .compute.lorentz import not_equal

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return not_equal.dispatch(self, other)

    def isclose(
        self,
        other: typing.Any,
        rtol: typing.Any = 1e-05,
        atol: typing.Any = 1e-08,
        equal_nan: typing.Any = False,
    ) -> typing.Any:
        "isclose docs"
        from .compute.lorentz import isclose

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return isclose.dispatch(rtol, atol, equal_nan, self, other)


class Momentum:
    pass


class PlanarMomentum(Momentum, MomentumProtocolPlanar):
    @property
    def px(self) -> typing.Any:
        "px docs"
        return self.x

    @property
    def py(self) -> typing.Any:
        "py docs"
        return self.y

    @property
    def pt(self) -> typing.Any:
        "pt docs"
        return self.rho

    @property
    def pt2(self) -> typing.Any:
        "pt2 docs"
        return self.rho2


class SpatialMomentum(PlanarMomentum, MomentumProtocolSpatial):
    @property
    def pz(self) -> typing.Any:
        "pz docs"
        return self.z

    @property
    def pseudorapidity(self) -> typing.Any:
        "pseudorapidity docs"
        return self.eta

    @property
    def p(self) -> typing.Any:
        "p docs"
        return self.mag

    @property
    def p2(self) -> typing.Any:
        "p2 docs"
        return self.mag2


class LorentzMomentum(SpatialMomentum, MomentumProtocolLorentz):
    @property
    def E(self) -> typing.Any:
        "E docs"
        return self.t

    @property
    def energy(self) -> typing.Any:
        "energy docs"
        return self.t

    @property
    def E2(self) -> typing.Any:
        "E2 docs"
        return self.t2

    @property
    def energy2(self) -> typing.Any:
        "energy2 docs"
        return self.t2

    @property
    def M(self) -> typing.Any:
        "M docs"
        return self.tau

    @property
    def mass(self) -> typing.Any:
        "mass docs"
        return self.tau

    @property
    def M2(self) -> typing.Any:
        "M2 docs"
        return self.tau2

    @property
    def mass2(self) -> typing.Any:
        "mass2 docs"
        return self.tau2

    @property
    def Et(self) -> typing.Any:
        "Et docs"
        from .compute.lorentz import Et

        return Et.dispatch(self)

    @property
    def transverse_energy(self) -> typing.Any:
        "transverse_energy docs"
        return self.Et

    @property
    def Et2(self) -> typing.Any:
        "Et2 docs"
        from .compute.lorentz import Et2

        return Et2.dispatch(self)

    @property
    def transverse_energy2(self) -> typing.Any:
        "transverse_energy2 docs"
        return self.Et2

    @property
    def Mt(self) -> typing.Any:
        "Mt docs"
        from .compute.lorentz import Mt

        return Mt.dispatch(self)

    @property
    def transverse_mass(self) -> typing.Any:
        "transverse_mass docs"
        return self.Mt

    @property
    def Mt2(self) -> typing.Any:
        "Mt2 docs"
        from .compute.lorentz import Mt2

        return Mt2.dispatch(self)

    @property
    def transverse_mass2(self) -> typing.Any:
        "transverse_mass2 docs"
        return self.Mt2


def _lib_of(*objects: typing.Any) -> typing.Any:
    lib = None
    for obj in objects:
        if isinstance(obj, Vector):
            if lib is None:
                lib = obj.lib
            elif lib is not obj.lib:
                raise TypeError(
                    f"cannot use {lib} and {obj.lib} in the same calculation"
                )
    return lib


def _from_signature(
    name: typing.Any, dispatch_map: typing.Any, signature: typing.Any
) -> typing.Any:
    result = dispatch_map.get(signature)
    if result is None:
        raise TypeError(
            f"function {repr('.'.join(name.split('.')[-2:]))} has no signature {signature}"
        )
    return result


_handler_priority = [
    "vector.backends.object_",
    "vector.backends.numpy_",
    "vector.backends.awkward_",
]


def _handler_of(*objects: typing.Any) -> typing.Any:
    handler = None
    for obj in objects:
        if isinstance(obj, Vector):
            if handler is None:
                handler = obj
            elif _handler_priority.index(
                type(obj).__module__
            ) > _handler_priority.index(type(handler).__module__):
                handler = obj

    return handler


def _flavor_of(*objects: typing.Any) -> typing.Any:
    from .backends.numpy_ import VectorNumpy
    from .backends.object_ import VectorObject

    handler = None
    is_momentum = True
    for obj in objects:
        if isinstance(obj, Vector):
            if not isinstance(obj, Momentum):
                is_momentum = False
            if handler is None:
                handler = obj
            elif isinstance(obj, VectorObject):
                pass
            elif isinstance(obj, VectorNumpy):
                handler = obj

    assert handler is not None

    if is_momentum:
        return type(handler)
    else:
        return handler.GenericClass
