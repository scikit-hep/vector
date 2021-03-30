# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

from vector._typeutils import (
    BoolCollection,
    ScalarCollection,
    TransformProtocol2D,
    TransformProtocol3D,
    TransformProtocol4D,
)

Module = typing.Any  # returns a module, but we can't be specific about which one


class Coordinates:
    pass


class Azimuthal(Coordinates):
    @property
    def elements(self) -> typing.Tuple[ScalarCollection, ScalarCollection]:
        "azimuthal elements docs"
        raise AssertionError


class Longitudinal(Coordinates):
    @property
    def elements(self) -> typing.Tuple[ScalarCollection]:
        "longitudinal elements docs"
        raise AssertionError


class Temporal(Coordinates):
    @property
    def elements(self) -> typing.Tuple[ScalarCollection]:
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


SameVectorType = typing.TypeVar("SameVectorType", bound="VectorProtocol")


class VectorProtocol:
    lib: Module

    @typing.no_type_check
    def _wrap_result(
        self,
        cls: typing.Any,
        result: typing.Any,
        returns: typing.Any,
        num_vecargs: typing.Any,
    ) -> typing.Any:
        raise AssertionError

    ProjectionClass2D: typing.Type["VectorProtocolPlanar"]
    ProjectionClass3D: typing.Type["VectorProtocolSpatial"]
    ProjectionClass4D: typing.Type["VectorProtocolLorentz"]
    GenericClass: typing.Type["VectorProtocol"]

    def to_Vector2D(self) -> "VectorProtocolPlanar":
        "to_Vector2D docs"
        raise AssertionError

    def to_Vector3D(self) -> "VectorProtocolSpatial":
        "to_Vector3D docs"
        raise AssertionError

    def to_Vector4D(self) -> "VectorProtocolLorentz":
        "to_Vector4D docs"
        raise AssertionError

    def to_xy(self) -> "VectorProtocolPlanar":
        "to_xy docs"
        raise AssertionError

    def to_rhophi(self) -> "VectorProtocolPlanar":
        "to_rhophi docs"
        raise AssertionError

    def to_xyz(self) -> "VectorProtocolSpatial":
        "to_xyz docs"
        raise AssertionError

    def to_xytheta(self) -> "VectorProtocolSpatial":
        "to_xytheta docs"
        raise AssertionError

    def to_xyeta(self) -> "VectorProtocolSpatial":
        "to_xyeta docs"
        raise AssertionError

    def to_rhophiz(self) -> "VectorProtocolSpatial":
        "to_rhophiz docs"
        raise AssertionError

    def to_rhophitheta(self) -> "VectorProtocolSpatial":
        "to_rhophitheta docs"
        raise AssertionError

    def to_rhophieta(self) -> "VectorProtocolSpatial":
        "to_rhophieta docs"
        raise AssertionError

    def to_xyzt(self) -> "VectorProtocolLorentz":
        "to_xyzt docs"
        raise AssertionError

    def to_xyztau(self) -> "VectorProtocolLorentz":
        "to_xyztau docs"
        raise AssertionError

    def to_xythetat(self) -> "VectorProtocolLorentz":
        "to_xythetat docs"
        raise AssertionError

    def to_xythetatau(self) -> "VectorProtocolLorentz":
        "to_xythetatau docs"
        raise AssertionError

    def to_xyetat(self) -> "VectorProtocolLorentz":
        "to_xyetat docs"
        raise AssertionError

    def to_xyetatau(self) -> "VectorProtocolLorentz":
        "to_xyetatau docs"
        raise AssertionError

    def to_rhophizt(self) -> "VectorProtocolLorentz":
        "to_rhophizt docs"
        raise AssertionError

    def to_rhophiztau(self) -> "VectorProtocolLorentz":
        "to_rhophiztau docs"
        raise AssertionError

    def to_rhophithetat(self) -> "VectorProtocolLorentz":
        "to_rhophithetat docs"
        raise AssertionError

    def to_rhophithetatau(self) -> "VectorProtocolLorentz":
        "to_rhophithetatau docs"
        raise AssertionError

    def to_rhophietat(self) -> "VectorProtocolLorentz":
        "to_rhophietat docs"
        raise AssertionError

    def to_rhophietatau(self) -> "VectorProtocolLorentz":
        "to_rhophietatau docs"
        raise AssertionError

    def unit(self: SameVectorType) -> SameVectorType:
        "unit docs"
        raise AssertionError

    def dot(self, other: "VectorProtocol") -> ScalarCollection:
        "dot docs"
        raise AssertionError

    def add(self, other: "VectorProtocol") -> "VectorProtocol":
        "add docs"
        raise AssertionError

    def subtract(self, other: "VectorProtocol") -> "VectorProtocol":
        "subtract docs"
        raise AssertionError

    def scale(self: SameVectorType, factor: ScalarCollection) -> SameVectorType:
        "scale docs"
        raise AssertionError

    def equal(self, other: "VectorProtocol") -> BoolCollection:
        "equal docs"
        raise AssertionError

    def not_equal(self, other: "VectorProtocol") -> BoolCollection:
        "not_equal docs"
        raise AssertionError

    def isclose(
        self,
        other: "VectorProtocol",
        rtol: ScalarCollection = 1e-05,
        atol: ScalarCollection = 1e-08,
        equal_nan: BoolCollection = False,
    ) -> BoolCollection:
        "isclose docs"
        raise AssertionError


class VectorProtocolPlanar(VectorProtocol):
    @property
    def azimuthal(self) -> Azimuthal:
        "azimuthal docs"
        raise AssertionError

    @property
    def x(self) -> ScalarCollection:
        "x docs"
        raise AssertionError

    @property
    def y(self) -> ScalarCollection:
        "y docs"
        raise AssertionError

    @property
    def rho(self) -> ScalarCollection:
        "rho docs"
        raise AssertionError

    @property
    def rho2(self) -> ScalarCollection:
        "rho2 docs"
        raise AssertionError

    @property
    def phi(self) -> ScalarCollection:
        "phi docs"
        raise AssertionError

    def deltaphi(self, other: VectorProtocol) -> ScalarCollection:
        """
        deltaphi docs

        (it's the signed difference, not arccos(dot))
        """
        raise AssertionError

    def rotateZ(self: SameVectorType, angle: ScalarCollection) -> SameVectorType:
        "rotateZ docs"
        raise AssertionError

    def transform2D(self: SameVectorType, obj: TransformProtocol2D) -> SameVectorType:
        "transform2D docs"
        raise AssertionError

    def is_parallel(
        self, other: VectorProtocol, tolerance: ScalarCollection = 1e-5
    ) -> BoolCollection:
        "is_parallel docs (note: this 'parallel' requires same direction)"
        raise AssertionError

    def is_antiparallel(
        self, other: VectorProtocol, tolerance: ScalarCollection = 1e-5
    ) -> BoolCollection:
        "is_antiparallel docs"
        raise AssertionError

    def is_perpendicular(
        self, other: VectorProtocol, tolerance: ScalarCollection = 1e-5
    ) -> BoolCollection:
        "is_perpendicular docs"
        raise AssertionError


class VectorProtocolSpatial(VectorProtocolPlanar):
    @property
    def longitudinal(self) -> Longitudinal:
        "longitudinal docs"
        raise AssertionError

    @property
    def z(self) -> ScalarCollection:
        "z docs"
        raise AssertionError

    @property
    def theta(self) -> ScalarCollection:
        "theta docs"
        raise AssertionError

    @property
    def eta(self) -> ScalarCollection:
        "eta docs"
        raise AssertionError

    @property
    def costheta(self) -> ScalarCollection:
        "costheta docs"
        raise AssertionError

    @property
    def cottheta(self) -> ScalarCollection:
        "cottheta docs"
        raise AssertionError

    @property
    def mag(self) -> ScalarCollection:
        "mag docs"
        raise AssertionError

    @property
    def mag2(self) -> ScalarCollection:
        "mag2 docs"
        raise AssertionError

    def cross(self, other: "VectorProtocol") -> "VectorProtocolSpatial":
        "cross docs"
        raise AssertionError

    def deltaangle(self, other: "VectorProtocol") -> ScalarCollection:
        """
        deltaangle docs

        (it's just arccos(dot))
        """
        raise AssertionError

    def deltaeta(self, other: "VectorProtocol") -> ScalarCollection:
        "deltaeta docs"
        raise AssertionError

    def deltaR(self, other: "VectorProtocol") -> ScalarCollection:
        "deltaR docs"
        raise AssertionError

    def deltaR2(self, other: "VectorProtocol") -> ScalarCollection:
        "deltaR2 docs"
        raise AssertionError

    def rotateX(self: SameVectorType, angle: ScalarCollection) -> SameVectorType:
        "rotateX docs"
        raise AssertionError

    def rotateY(self: SameVectorType, angle: ScalarCollection) -> SameVectorType:
        "rotateY docs"
        raise AssertionError

    def rotate_axis(
        self: SameVectorType, axis: "VectorProtocol", angle: ScalarCollection
    ) -> SameVectorType:
        "rotate_axis docs"
        raise AssertionError

    def rotate_euler(
        self: SameVectorType,
        phi: ScalarCollection,
        theta: ScalarCollection,
        psi: ScalarCollection,
        order: str = "zxz",
    ) -> SameVectorType:
        """
        rotate_euler docs

        same conventions as ROOT
        """
        raise AssertionError

    def rotate_nautical(
        self: SameVectorType,
        yaw: ScalarCollection,
        pitch: ScalarCollection,
        roll: ScalarCollection,
    ) -> SameVectorType:
        """
        rotate_nautical docs

        transforming "from the body frame to the inertial frame"

        http://planning.cs.uiuc.edu/node102.html
        http://www.chrobotics.com/library/understanding-euler-angles
        """
        raise AssertionError

    def rotate_quaternion(
        self: SameVectorType,
        u: ScalarCollection,
        i: ScalarCollection,
        j: ScalarCollection,
        k: ScalarCollection,
    ) -> SameVectorType:
        """
        rotate_quaternion docs

        same conventions as ROOT
        """
        raise AssertionError

    def transform3D(self: SameVectorType, obj: TransformProtocol3D) -> SameVectorType:
        "transform3D docs"
        raise AssertionError

    def is_parallel(
        self, other: "VectorProtocol", tolerance: ScalarCollection = 1e-5
    ) -> BoolCollection:
        "is_parallel docs (note: this 'parallel' requires same direction)"
        raise AssertionError

    def is_antiparallel(
        self, other: "VectorProtocol", tolerance: ScalarCollection = 1e-5
    ) -> BoolCollection:
        "is_antiparallel docs"
        raise AssertionError

    def is_perpendicular(
        self, other: "VectorProtocol", tolerance: ScalarCollection = 1e-5
    ) -> BoolCollection:
        "is_perpendicular docs"
        raise AssertionError


class VectorProtocolLorentz(VectorProtocolSpatial):
    @property
    def temporal(self) -> Temporal:
        "temporal docs"
        raise AssertionError

    @property
    def t(self) -> ScalarCollection:
        "t docs"
        raise AssertionError

    @property
    def t2(self) -> ScalarCollection:
        "t2 docs"
        raise AssertionError

    @property
    def tau(self) -> ScalarCollection:
        "tau docs"
        raise AssertionError

    @property
    def tau2(self) -> ScalarCollection:
        "tau2 docs"
        raise AssertionError

    @property
    def beta(self) -> ScalarCollection:
        "beta docs"
        raise AssertionError

    @property
    def gamma(self) -> ScalarCollection:
        "gamma docs"
        raise AssertionError

    @property
    def rapidity(self) -> ScalarCollection:
        "rapidity docs"
        raise AssertionError

    def boost_p4(self: SameVectorType, p4: "VectorProtocolLorentz") -> SameVectorType:
        "boost_p4 docs"
        raise AssertionError

    def boost_beta3(
        self: SameVectorType, beta3: "VectorProtocolSpatial"
    ) -> SameVectorType:
        "boost_beta3 docs"
        raise AssertionError

    def boost(self: SameVectorType, booster: "VectorProtocol") -> SameVectorType:
        "boost docs"
        raise AssertionError

    def boostX(
        self: SameVectorType,
        beta: typing.Optional[ScalarCollection] = None,
        gamma: typing.Optional[ScalarCollection] = None,
    ) -> SameVectorType:
        "boostX docs"
        raise AssertionError

    def boostY(
        self: SameVectorType,
        beta: typing.Optional[ScalarCollection] = None,
        gamma: typing.Optional[ScalarCollection] = None,
    ) -> SameVectorType:
        "boostY docs"
        raise AssertionError

    def boostZ(
        self: SameVectorType,
        beta: typing.Optional[ScalarCollection] = None,
        gamma: typing.Optional[ScalarCollection] = None,
    ) -> SameVectorType:
        "boostZ docs"
        raise AssertionError

    def transform4D(self: SameVectorType, obj: TransformProtocol4D) -> SameVectorType:
        "transform4D docs"
        raise AssertionError

    def to_beta3(self) -> "VectorProtocolSpatial":
        "to_beta3 docs"
        raise AssertionError

    def is_timelike(self, tolerance: ScalarCollection = 0) -> BoolCollection:
        "is_timelike docs"
        raise AssertionError

    def is_spacelike(self, tolerance: ScalarCollection = 0) -> BoolCollection:
        "is_spacelike docs"
        raise AssertionError

    def is_lightlike(self, tolerance: ScalarCollection = 1e-5) -> BoolCollection:
        "is_timelike docs"
        raise AssertionError


class MomentumProtocolPlanar(VectorProtocolPlanar):
    @property
    def px(self) -> ScalarCollection:
        "px docs"
        raise AssertionError

    @property
    def py(self) -> ScalarCollection:
        "py docs"
        raise AssertionError

    @property
    def pt(self) -> ScalarCollection:
        "pt docs"
        raise AssertionError

    @property
    def pt2(self) -> ScalarCollection:
        "pt2 docs"
        raise AssertionError


class MomentumProtocolSpatial(VectorProtocolSpatial, MomentumProtocolPlanar):
    @property
    def pz(self) -> ScalarCollection:
        "pz docs"
        raise AssertionError

    @property
    def pseudorapidity(self) -> ScalarCollection:
        "pseudorapidity docs"
        raise AssertionError

    @property
    def p(self) -> ScalarCollection:
        "p docs"
        raise AssertionError

    @property
    def p2(self) -> ScalarCollection:
        "p2 docs"
        raise AssertionError


class MomentumProtocolLorentz(VectorProtocolLorentz, MomentumProtocolSpatial):
    @property
    def E(self) -> ScalarCollection:
        "E docs"
        raise AssertionError

    @property
    def energy(self) -> ScalarCollection:
        "energy docs"
        raise AssertionError

    @property
    def E2(self) -> ScalarCollection:
        "E2 docs"
        raise AssertionError

    @property
    def energy2(self) -> ScalarCollection:
        "energy2 docs"
        raise AssertionError

    @property
    def M(self) -> ScalarCollection:
        "M docs"
        raise AssertionError

    @property
    def mass(self) -> ScalarCollection:
        "mass docs"
        raise AssertionError

    @property
    def M2(self) -> ScalarCollection:
        "M2 docs"
        raise AssertionError

    @property
    def mass2(self) -> ScalarCollection:
        "mass2 docs"
        raise AssertionError

    @property
    def Et(self) -> ScalarCollection:
        "Et docs"
        raise AssertionError

    @property
    def transverse_energy(self) -> ScalarCollection:
        "transverse_energy docs"
        raise AssertionError

    @property
    def Et2(self) -> ScalarCollection:
        "Et2 docs"
        raise AssertionError

    @property
    def transverse_energy2(self) -> ScalarCollection:
        "transverse_energy2 docs"
        raise AssertionError

    @property
    def Mt(self) -> ScalarCollection:
        "Mt docs"
        raise AssertionError

    @property
    def transverse_mass(self) -> ScalarCollection:
        "transverse_mass docs"
        raise AssertionError

    @property
    def Mt2(self) -> ScalarCollection:
        "Mt2 docs"
        raise AssertionError

    @property
    def transverse_mass2(self) -> ScalarCollection:
        "transverse_mass2 docs"
        raise AssertionError


class Vector(VectorProtocol):
    def to_xy(self) -> VectorProtocolPlanar:
        from vector.compute import planar

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self)),  # type: ignore
            [AzimuthalXY, None],
            1,
        )

    def to_rhophi(self) -> VectorProtocolPlanar:
        from vector.compute import planar

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self)),  # type: ignore
            [AzimuthalRhoPhi, None],
            1,
        )

    def to_xyz(self) -> VectorProtocolSpatial:
        from vector.compute import planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.z.dispatch(self)  # type: ignore

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord),  # type: ignore
            [AzimuthalXY, LongitudinalZ, None],
            1,
        )

    def to_xytheta(self) -> VectorProtocolSpatial:
        from vector.compute import planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.theta.dispatch(self)  # type: ignore

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord),  # type: ignore
            [AzimuthalXY, LongitudinalTheta, None],
            1,
        )

    def to_xyeta(self) -> VectorProtocolSpatial:
        from vector.compute import planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.eta.dispatch(self)  # type: ignore

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord),  # type: ignore
            [AzimuthalXY, LongitudinalEta, None],
            1,
        )

    def to_rhophiz(self) -> VectorProtocolSpatial:
        from vector.compute import planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.z.dispatch(self)  # type: ignore

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord),  # type: ignore
            [AzimuthalRhoPhi, LongitudinalZ, None],
            1,
        )

    def to_rhophitheta(self) -> VectorProtocolSpatial:
        from vector.compute import planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.theta.dispatch(self)  # type: ignore

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord),  # type: ignore
            [AzimuthalRhoPhi, LongitudinalTheta, None],
            1,
        )

    def to_rhophieta(self) -> VectorProtocolSpatial:
        from vector.compute import planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.eta.dispatch(self)  # type: ignore

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord),  # type: ignore
            [AzimuthalRhoPhi, LongitudinalEta, None],
            1,
        )

    def to_xyzt(self) -> VectorProtocolLorentz:
        from vector.compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.z.dispatch(self)  # type: ignore
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.t.dispatch(self)  # type: ignore

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord, tcoord),  # type: ignore
            [AzimuthalXY, LongitudinalZ, TemporalT],
            1,
        )

    def to_xyztau(self) -> VectorProtocolLorentz:
        from vector.compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.z.dispatch(self)  # type: ignore
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.tau.dispatch(self)  # type: ignore

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord, tcoord),  # type: ignore
            [AzimuthalXY, LongitudinalZ, TemporalTau],
            1,
        )

    def to_xythetat(self) -> VectorProtocolLorentz:
        from vector.compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.theta.dispatch(self)  # type: ignore
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.t.dispatch(self)  # type: ignore

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord, tcoord),  # type: ignore
            [AzimuthalXY, LongitudinalTheta, TemporalT],
            1,
        )

    def to_xythetatau(self) -> VectorProtocolLorentz:
        from vector.compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.theta.dispatch(self)  # type: ignore
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.tau.dispatch(self)  # type: ignore

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord, tcoord),  # type: ignore
            [AzimuthalXY, LongitudinalTheta, TemporalTau],
            1,
        )

    def to_xyetat(self) -> VectorProtocolLorentz:
        from vector.compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.eta.dispatch(self)  # type: ignore
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.t.dispatch(self)  # type: ignore

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord, tcoord),  # type: ignore
            [AzimuthalXY, LongitudinalEta, TemporalT],
            1,
        )

    def to_xyetatau(self) -> VectorProtocolLorentz:
        from vector.compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.eta.dispatch(self)  # type: ignore
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.tau.dispatch(self)  # type: ignore

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord, tcoord),  # type: ignore
            [AzimuthalXY, LongitudinalEta, TemporalTau],
            1,
        )

    def to_rhophizt(self) -> VectorProtocolLorentz:
        from vector.compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.z.dispatch(self)  # type: ignore
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.t.dispatch(self)  # type: ignore

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord, tcoord),  # type: ignore
            [AzimuthalRhoPhi, LongitudinalZ, TemporalT],
            1,
        )

    def to_rhophiztau(self) -> VectorProtocolLorentz:
        from vector.compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.z.dispatch(self)  # type: ignore
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.tau.dispatch(self)  # type: ignore

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord, tcoord),  # type: ignore
            [AzimuthalRhoPhi, LongitudinalZ, TemporalTau],
            1,
        )

    def to_rhophithetat(self) -> VectorProtocolLorentz:
        from vector.compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.theta.dispatch(self)  # type: ignore
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.t.dispatch(self)  # type: ignore

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord, tcoord),  # type: ignore
            [AzimuthalRhoPhi, LongitudinalTheta, TemporalT],
            1,
        )

    def to_rhophithetatau(self) -> VectorProtocolLorentz:
        from vector.compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.theta.dispatch(self)  # type: ignore
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.tau.dispatch(self)  # type: ignore

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord, tcoord),  # type: ignore
            [AzimuthalRhoPhi, LongitudinalTheta, TemporalTau],
            1,
        )

    def to_rhophietat(self) -> VectorProtocolLorentz:
        from vector.compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.eta.dispatch(self)  # type: ignore
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.t.dispatch(self)  # type: ignore

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord, tcoord),  # type: ignore
            [AzimuthalRhoPhi, LongitudinalEta, TemporalT],
            1,
        )

    def to_rhophietatau(self) -> VectorProtocolLorentz:
        from vector.compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.eta.dispatch(self)  # type: ignore
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.tau.dispatch(self)  # type: ignore

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord, tcoord),  # type: ignore
            [AzimuthalRhoPhi, LongitudinalEta, TemporalTau],
            1,
        )


class Vector2D(Vector, VectorProtocolPlanar):
    def to_Vector2D(self) -> VectorProtocolPlanar:
        return self

    def to_Vector3D(self) -> VectorProtocolSpatial:
        return self._wrap_result(
            type(self),
            self.azimuthal.elements + (0,),
            [_aztype(self), LongitudinalZ, None],
            1,
        )

    def to_Vector4D(self) -> VectorProtocolLorentz:
        return self._wrap_result(
            type(self),
            self.azimuthal.elements + (0, 0),
            [_aztype(self), LongitudinalZ, TemporalT],
            1,
        )


class Vector3D(Vector, VectorProtocolSpatial):
    def to_Vector2D(self) -> VectorProtocolPlanar:
        return self._wrap_result(
            type(self),
            self.azimuthal.elements,
            [_aztype(self), None],
            1,
        )

    def to_Vector3D(self) -> VectorProtocolSpatial:
        return self

    def to_Vector4D(self) -> VectorProtocolLorentz:
        return self._wrap_result(
            type(self),
            self.azimuthal.elements + self.longitudinal.elements + (0,),
            [_aztype(self), _ltype(self), TemporalT],
            1,
        )


class Vector4D(Vector, VectorProtocolLorentz):
    def to_Vector2D(self) -> VectorProtocolPlanar:
        return self._wrap_result(
            type(self),
            self.azimuthal.elements,
            [_aztype(self), None],
            1,
        )

    def to_Vector3D(self) -> VectorProtocolSpatial:
        return self._wrap_result(
            type(self),
            self.azimuthal.elements + self.longitudinal.elements,
            [_aztype(self), _ltype(self), None],
            1,
        )

    def to_Vector4D(self) -> VectorProtocolLorentz:
        return self


class Planar(VectorProtocolPlanar):
    @property
    def x(self) -> ScalarCollection:
        from vector.compute.planar import x

        return x.dispatch(self)  # type: ignore

    @property
    def y(self) -> ScalarCollection:
        from vector.compute.planar import y

        return y.dispatch(self)  # type: ignore

    @property
    def rho(self) -> ScalarCollection:
        from vector.compute.planar import rho

        return rho.dispatch(self)  # type: ignore

    @property
    def rho2(self) -> ScalarCollection:
        from vector.compute.planar import rho2

        return rho2.dispatch(self)  # type: ignore

    @property
    def phi(self) -> ScalarCollection:
        from vector.compute.planar import phi

        return phi.dispatch(self)  # type: ignore

    def deltaphi(self, other: VectorProtocol) -> ScalarCollection:
        from vector.compute.planar import deltaphi

        return deltaphi.dispatch(self, other)  # type: ignore

    def rotateZ(self: SameVectorType, angle: ScalarCollection) -> SameVectorType:
        from vector.compute.planar import rotateZ

        return rotateZ.dispatch(angle, self)  # type: ignore

    def transform2D(self: SameVectorType, obj: TransformProtocol2D) -> SameVectorType:
        from vector.compute.planar import transform2D

        return transform2D.dispatch(obj, self)  # type: ignore

    def is_parallel(
        self, other: VectorProtocol, tolerance: ScalarCollection = 1e-5
    ) -> BoolCollection:
        from vector.compute.planar import is_parallel

        if not isinstance(other, Vector2D):
            return self.to_Vector3D().is_parallel(other, tolerance=tolerance)
        else:
            return is_parallel.dispatch(tolerance, self, other)  # type: ignore

    def is_antiparallel(
        self, other: VectorProtocol, tolerance: ScalarCollection = 1e-5
    ) -> BoolCollection:
        from vector.compute.planar import is_antiparallel

        if not isinstance(other, Vector2D):
            return self.to_Vector3D().is_antiparallel(other, tolerance=tolerance)
        else:
            return is_antiparallel.dispatch(tolerance, self, other)  # type: ignore

    def is_perpendicular(
        self, other: VectorProtocol, tolerance: ScalarCollection = 1e-5
    ) -> BoolCollection:
        from vector.compute.planar import is_perpendicular

        if not isinstance(other, Vector2D):
            return self.to_Vector3D().is_perpendicular(other, tolerance=tolerance)
        else:
            return is_perpendicular.dispatch(tolerance, self, other)  # type: ignore

    def unit(self: SameVectorType) -> SameVectorType:
        from vector.compute.planar import unit

        return unit.dispatch(self)  # type: ignore

    def dot(self, other: VectorProtocol) -> ScalarCollection:
        module = _compute_module_of(self, other)
        return module.dot.dispatch(self, other)

    def add(self, other: VectorProtocol) -> VectorProtocol:
        module = _compute_module_of(self, other)
        return module.add.dispatch(self, other)

    def subtract(self, other: VectorProtocol) -> VectorProtocol:
        module = _compute_module_of(self, other)
        return module.subtract.dispatch(self, other)

    def scale(self: SameVectorType, factor: ScalarCollection) -> SameVectorType:
        from vector.compute.planar import scale

        return scale.dispatch(factor, self)  # type: ignore

    def equal(self, other: VectorProtocol) -> BoolCollection:
        from vector.compute.planar import equal

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return equal.dispatch(self, other)  # type: ignore

    def not_equal(self, other: VectorProtocol) -> BoolCollection:
        from vector.compute.planar import not_equal

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return not_equal.dispatch(self, other)  # type: ignore

    def isclose(
        self,
        other: VectorProtocol,
        rtol: ScalarCollection = 1e-05,
        atol: ScalarCollection = 1e-08,
        equal_nan: BoolCollection = False,
    ) -> BoolCollection:
        from vector.compute.planar import isclose

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return isclose.dispatch(rtol, atol, equal_nan, self, other)  # type: ignore


class Spatial(Planar, VectorProtocolSpatial):
    @property
    def z(self) -> ScalarCollection:
        from vector.compute.spatial import z

        return z.dispatch(self)  # type: ignore

    @property
    def theta(self) -> ScalarCollection:
        from vector.compute.spatial import theta

        return theta.dispatch(self)  # type: ignore

    @property
    def eta(self) -> ScalarCollection:
        from vector.compute.spatial import eta

        return eta.dispatch(self)  # type: ignore

    @property
    def costheta(self) -> ScalarCollection:
        from vector.compute.spatial import costheta

        return costheta.dispatch(self)  # type: ignore

    @property
    def cottheta(self) -> ScalarCollection:
        from vector.compute.spatial import cottheta

        return cottheta.dispatch(self)  # type: ignore

    @property
    def mag(self) -> ScalarCollection:
        from vector.compute.spatial import mag

        return mag.dispatch(self)  # type: ignore

    @property
    def mag2(self) -> ScalarCollection:
        from vector.compute.spatial import mag2

        return mag2.dispatch(self)  # type: ignore

    def cross(self, other: VectorProtocol) -> VectorProtocolSpatial:
        from vector.compute.spatial import cross

        return cross.dispatch(self, other)  # type: ignore

    def deltaangle(self, other: VectorProtocol) -> ScalarCollection:
        from vector.compute.spatial import deltaangle

        return deltaangle.dispatch(self, other)  # type: ignore

    def deltaeta(self, other: VectorProtocol) -> ScalarCollection:
        from vector.compute.spatial import deltaeta

        return deltaeta.dispatch(self, other)  # type: ignore

    def deltaR(self, other: VectorProtocol) -> ScalarCollection:
        from vector.compute.spatial import deltaR

        return deltaR.dispatch(self, other)  # type: ignore

    def deltaR2(self, other: VectorProtocol) -> ScalarCollection:
        from vector.compute.spatial import deltaR2

        return deltaR2.dispatch(self, other)  # type: ignore

    def rotateX(self: SameVectorType, angle: ScalarCollection) -> SameVectorType:
        from vector.compute.spatial import rotateX

        return rotateX.dispatch(angle, self)  # type: ignore

    def rotateY(self: SameVectorType, angle: ScalarCollection) -> SameVectorType:
        from vector.compute.spatial import rotateY

        return rotateY.dispatch(angle, self)  # type: ignore

    def rotate_axis(
        self: SameVectorType, axis: VectorProtocol, angle: ScalarCollection
    ) -> SameVectorType:
        from vector.compute.spatial import rotate_axis

        return rotate_axis.dispatch(angle, axis, self)  # type: ignore

    def rotate_euler(
        self: SameVectorType,
        phi: ScalarCollection,
        theta: ScalarCollection,
        psi: ScalarCollection,
        order: str = "zxz",
    ) -> SameVectorType:
        from vector.compute.spatial import rotate_euler

        return rotate_euler.dispatch(phi, theta, psi, order.lower(), self)  # type: ignore

    def rotate_nautical(
        self: SameVectorType,
        yaw: ScalarCollection,
        pitch: ScalarCollection,
        roll: ScalarCollection,
    ) -> SameVectorType:
        # The order of arguments is reversed because rotate_euler
        # follows ROOT's argument order: phi, theta, psi.
        from vector.compute.spatial import rotate_euler

        return rotate_euler.dispatch(roll, pitch, yaw, "zyx", self)  # type: ignore

    def rotate_quaternion(
        self: SameVectorType,
        u: ScalarCollection,
        i: ScalarCollection,
        j: ScalarCollection,
        k: ScalarCollection,
    ) -> SameVectorType:
        from vector.compute.spatial import rotate_quaternion

        return rotate_quaternion.dispatch(u, i, j, k, self)  # type: ignore

    def transform3D(self: SameVectorType, obj: TransformProtocol3D) -> SameVectorType:
        from vector.compute.spatial import transform3D

        return transform3D.dispatch(obj, self)  # type: ignore

    def is_parallel(
        self, other: VectorProtocol, tolerance: ScalarCollection = 1e-5
    ) -> BoolCollection:
        from vector.compute.spatial import is_parallel

        if isinstance(other, Vector2D):
            return is_parallel.dispatch(tolerance, self, other.to_Vector3D())  # type: ignore
        else:
            return is_parallel.dispatch(tolerance, self, other)  # type: ignore

    def is_antiparallel(
        self, other: VectorProtocol, tolerance: ScalarCollection = 1e-5
    ) -> BoolCollection:
        from vector.compute.spatial import is_antiparallel

        if isinstance(other, Vector2D):
            return is_antiparallel.dispatch(tolerance, self, other.to_Vector3D())  # type: ignore
        else:
            return is_antiparallel.dispatch(tolerance, self, other)  # type: ignore

    def is_perpendicular(
        self, other: VectorProtocol, tolerance: ScalarCollection = 1e-5
    ) -> BoolCollection:
        from vector.compute.spatial import is_perpendicular

        if isinstance(other, Vector2D):
            return is_perpendicular.dispatch(tolerance, self, other.to_Vector3D())  # type: ignore
        else:
            return is_perpendicular.dispatch(tolerance, self, other)  # type: ignore

    def unit(self: SameVectorType) -> SameVectorType:
        from vector.compute.spatial import unit

        return unit.dispatch(self)  # type: ignore

    def dot(self, other: VectorProtocol) -> ScalarCollection:
        module = _compute_module_of(self, other)
        return module.dot.dispatch(self, other)

    def add(self, other: VectorProtocol) -> VectorProtocol:
        module = _compute_module_of(self, other)
        return module.add.dispatch(self, other)

    def subtract(self, other: VectorProtocol) -> VectorProtocol:
        module = _compute_module_of(self, other)
        return module.subtract.dispatch(self, other)

    def scale(self: SameVectorType, factor: ScalarCollection) -> SameVectorType:
        from vector.compute.spatial import scale

        return scale.dispatch(factor, self)  # type: ignore

    def equal(self, other: VectorProtocol) -> BoolCollection:
        from vector.compute.spatial import equal

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return equal.dispatch(self, other)  # type: ignore

    def not_equal(self, other: VectorProtocol) -> BoolCollection:
        from vector.compute.spatial import not_equal

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return not_equal.dispatch(self, other)  # type: ignore

    def isclose(
        self,
        other: VectorProtocol,
        rtol: ScalarCollection = 1e-05,
        atol: ScalarCollection = 1e-08,
        equal_nan: BoolCollection = False,
    ) -> BoolCollection:
        from vector.compute.spatial import isclose

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return isclose.dispatch(rtol, atol, equal_nan, self, other)  # type: ignore


class Lorentz(Spatial, VectorProtocolLorentz):
    @property
    def t(self) -> ScalarCollection:
        from vector.compute.lorentz import t

        return t.dispatch(self)  # type: ignore

    @property
    def t2(self) -> ScalarCollection:
        from vector.compute.lorentz import t2

        return t2.dispatch(self)  # type: ignore

    @property
    def tau(self) -> ScalarCollection:
        from vector.compute.lorentz import tau

        return tau.dispatch(self)  # type: ignore

    @property
    def tau2(self) -> ScalarCollection:
        from vector.compute.lorentz import tau2

        return tau2.dispatch(self)  # type: ignore

    @property
    def beta(self) -> ScalarCollection:
        from vector.compute.lorentz import beta

        return beta.dispatch(self)  # type: ignore

    @property
    def gamma(self) -> ScalarCollection:
        from vector.compute.lorentz import gamma

        return gamma.dispatch(self)  # type: ignore

    @property
    def rapidity(self) -> ScalarCollection:
        from vector.compute.lorentz import rapidity

        return rapidity.dispatch(self)  # type: ignore

    def boost_p4(self: SameVectorType, p4: VectorProtocolLorentz) -> SameVectorType:
        from vector.compute.lorentz import boost_p4

        return boost_p4.dispatch(self, p4)  # type: ignore

    def boost_beta3(
        self: SameVectorType, beta3: VectorProtocolSpatial
    ) -> SameVectorType:
        from vector.compute.lorentz import boost_beta3

        return boost_beta3.dispatch(self, beta3)  # type: ignore

    def boost(self: SameVectorType, booster: VectorProtocol) -> SameVectorType:
        from vector.compute.lorentz import boost_beta3, boost_p4

        if isinstance(booster, Vector3D):
            return boost_beta3.dispatch(self, booster)  # type: ignore
        elif isinstance(booster, Vector4D):
            return boost_p4.dispatch(self, booster)  # type: ignore
        else:
            raise TypeError(
                "specify a Vector3D to boost by beta (velocity with c=1) or "
                "a Vector4D to boost by a momentum 4-vector"
            )

    def boostX(
        self: SameVectorType,
        beta: typing.Optional[ScalarCollection] = None,
        gamma: typing.Optional[ScalarCollection] = None,
    ) -> SameVectorType:
        from vector.compute.lorentz import boostX_beta, boostX_gamma

        if beta is not None and gamma is None:
            return boostX_beta.dispatch(beta, self)  # type: ignore
        elif beta is None and gamma is not None:
            return boostX_gamma.dispatch(gamma, self)  # type: ignore
        else:
            raise TypeError("specify 'beta' xor 'gamma', not both or neither")

    def boostY(
        self: SameVectorType,
        beta: typing.Optional[ScalarCollection] = None,
        gamma: typing.Optional[ScalarCollection] = None,
    ) -> SameVectorType:
        from vector.compute.lorentz import boostY_beta, boostY_gamma

        if beta is not None and gamma is None:
            return boostY_beta.dispatch(beta, self)  # type: ignore
        elif beta is None and gamma is not None:
            return boostY_gamma.dispatch(gamma, self)  # type: ignore
        else:
            raise TypeError("specify 'beta' xor 'gamma', not both or neither")

    def boostZ(
        self: SameVectorType,
        beta: typing.Optional[ScalarCollection] = None,
        gamma: typing.Optional[ScalarCollection] = None,
    ) -> SameVectorType:
        from vector.compute.lorentz import boostZ_beta, boostZ_gamma

        if beta is not None and gamma is None:
            return boostZ_beta.dispatch(beta, self)  # type: ignore
        elif beta is None and gamma is not None:
            return boostZ_gamma.dispatch(gamma, self)  # type: ignore
        else:
            raise TypeError("specify 'beta' xor 'gamma', not both or neither")

    def transform4D(self: SameVectorType, obj: TransformProtocol4D) -> SameVectorType:
        from vector.compute.lorentz import transform4D

        return transform4D.dispatch(obj, self)  # type: ignore

    def to_beta3(self) -> VectorProtocolSpatial:
        from vector.compute.lorentz import to_beta3

        return to_beta3.dispatch(self)  # type: ignore

    def is_timelike(self, tolerance: ScalarCollection = 0) -> BoolCollection:
        from vector.compute.lorentz import is_timelike

        return is_timelike.dispatch(tolerance, self)  # type: ignore

    def is_spacelike(self, tolerance: ScalarCollection = 0) -> BoolCollection:
        from vector.compute.lorentz import is_spacelike

        return is_spacelike.dispatch(tolerance, self)  # type: ignore

    def is_lightlike(self, tolerance: ScalarCollection = 1e-5) -> BoolCollection:
        from vector.compute.lorentz import is_lightlike

        return is_lightlike.dispatch(tolerance, self)  # type: ignore

    def unit(self: SameVectorType) -> SameVectorType:
        from vector.compute.lorentz import unit

        return unit.dispatch(self)  # type: ignore

    def dot(self, other: VectorProtocol) -> ScalarCollection:
        module = _compute_module_of(self, other)
        return module.dot.dispatch(self, other)

    def add(self, other: VectorProtocol) -> VectorProtocol:
        module = _compute_module_of(self, other)
        return module.add.dispatch(self, other)

    def subtract(self, other: VectorProtocol) -> VectorProtocol:
        module = _compute_module_of(self, other)
        return module.subtract.dispatch(self, other)

    def scale(self: SameVectorType, factor: ScalarCollection) -> SameVectorType:
        from vector.compute.lorentz import scale

        return scale.dispatch(factor, self)  # type: ignore

    def equal(self, other: VectorProtocol) -> BoolCollection:
        from vector.compute.lorentz import equal

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return equal.dispatch(self, other)  # type: ignore

    def not_equal(self, other: VectorProtocol) -> BoolCollection:
        from vector.compute.lorentz import not_equal

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return not_equal.dispatch(self, other)  # type: ignore

    def isclose(
        self,
        other: VectorProtocol,
        rtol: ScalarCollection = 1e-05,
        atol: ScalarCollection = 1e-08,
        equal_nan: BoolCollection = False,
    ) -> BoolCollection:
        from vector.compute.lorentz import isclose

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return isclose.dispatch(rtol, atol, equal_nan, self, other)  # type: ignore


class Momentum:
    pass


class PlanarMomentum(Momentum, MomentumProtocolPlanar):
    @property
    def px(self) -> ScalarCollection:
        return self.x

    @property
    def py(self) -> ScalarCollection:
        return self.y

    @property
    def pt(self) -> ScalarCollection:
        return self.rho

    @property
    def pt2(self) -> ScalarCollection:
        return self.rho2


class SpatialMomentum(PlanarMomentum, MomentumProtocolSpatial):
    @property
    def pz(self) -> ScalarCollection:
        return self.z

    @property
    def pseudorapidity(self) -> ScalarCollection:
        return self.eta

    @property
    def p(self) -> ScalarCollection:
        return self.mag

    @property
    def p2(self) -> ScalarCollection:
        return self.mag2


class LorentzMomentum(SpatialMomentum, MomentumProtocolLorentz):
    @property
    def E(self) -> ScalarCollection:
        return self.t

    @property
    def energy(self) -> ScalarCollection:
        return self.t

    @property
    def E2(self) -> ScalarCollection:
        return self.t2

    @property
    def energy2(self) -> ScalarCollection:
        return self.t2

    @property
    def M(self) -> ScalarCollection:
        return self.tau

    @property
    def mass(self) -> ScalarCollection:
        return self.tau

    @property
    def M2(self) -> ScalarCollection:
        return self.tau2

    @property
    def mass2(self) -> ScalarCollection:
        return self.tau2

    @property
    def Et(self) -> ScalarCollection:
        from vector.compute.lorentz import Et

        return Et.dispatch(self)  # type: ignore

    @property
    def transverse_energy(self) -> ScalarCollection:
        return self.Et

    @property
    def Et2(self) -> ScalarCollection:
        from vector.compute.lorentz import Et2

        return Et2.dispatch(self)  # type: ignore

    @property
    def transverse_energy2(self) -> ScalarCollection:
        return self.Et2

    @property
    def Mt(self) -> ScalarCollection:
        from vector.compute.lorentz import Mt

        return Mt.dispatch(self)  # type: ignore

    @property
    def transverse_mass(self) -> ScalarCollection:
        return self.Mt

    @property
    def Mt2(self) -> ScalarCollection:
        from vector.compute.lorentz import Mt2

        return Mt2.dispatch(self)  # type: ignore

    @property
    def transverse_mass2(self) -> ScalarCollection:
        return self.Mt2


def dim(v: VectorProtocol) -> int:
    if isinstance(v, Vector2D):
        return 2
    elif isinstance(v, Vector3D):
        return 3
    elif isinstance(v, Vector4D):
        return 4
    else:
        raise TypeError(f"{repr(v)} is not a vector.Vector")


def _compute_module_of(
    one: VectorProtocol, two: VectorProtocol, nontemporal: bool = False
) -> Module:
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

    raise AssertionError(repr(one))


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


def _aztype(obj: VectorProtocolPlanar) -> typing.Type[Coordinates]:
    if hasattr(obj, "azimuthal"):
        for t in type(obj.azimuthal).__mro__:
            if t in (AzimuthalXY, AzimuthalRhoPhi):
                return t
    raise AssertionError(repr(obj))


def _ltype(obj: VectorProtocolSpatial) -> typing.Type[Coordinates]:
    if hasattr(obj, "longitudinal"):
        for t in type(obj.longitudinal).__mro__:
            if t in (LongitudinalZ, LongitudinalTheta, LongitudinalEta):
                return t
    raise AssertionError(repr(obj))


def _ttype(obj: VectorProtocolLorentz) -> typing.Type[Coordinates]:
    if hasattr(obj, "temporal"):
        for t in type(obj.temporal).__mro__:
            if t in (TemporalT, TemporalTau):
                return t
    raise AssertionError(repr(obj))


def _lib_of(*objects: VectorProtocol) -> Module:  # NumPy-like module
    lib = None
    for obj in objects:
        if isinstance(obj, Vector):
            if lib is None:
                lib = obj.lib
            elif lib is not obj.lib:
                raise TypeError(
                    f"cannot use {lib} and {obj.lib} in the same calculation"
                )

    assert lib is not None
    return lib


def _from_signature(name: str, dispatch_map: dict, signature: tuple) -> tuple:
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


def _handler_of(*objects: VectorProtocol) -> VectorProtocol:
    handler = None
    for obj in objects:
        if isinstance(obj, Vector):
            if handler is None:
                handler = obj
            elif _handler_priority.index(
                type(obj).__module__
            ) > _handler_priority.index(type(handler).__module__):
                handler = obj

    assert handler is not None
    return handler


def _flavor_of(*objects: VectorProtocol) -> typing.Type[VectorProtocol]:
    from vector.backends.numpy_ import VectorNumpy
    from vector.backends.object_ import VectorObject

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
