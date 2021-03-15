# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

import numpy

import vector.compute.lorentz
import vector.compute.planar
import vector.compute.spatial
import vector.geometry
import vector.methods
from vector.geometry import _coordinate_class_to_names


class CoordinatesObject:
    pass


class AzimuthalObject(CoordinatesObject):
    pass


class LongitudinalObject(CoordinatesObject):
    pass


class TemporalObject(CoordinatesObject):
    pass


class VectorObject:
    pass


class AzimuthalObjectXY(typing.NamedTuple):
    x: float
    y: float

    @property
    def elements(self):
        return (self.x, self.y)


AzimuthalObjectXY.__bases__ = (AzimuthalObject, vector.geometry.AzimuthalXY, tuple)


class AzimuthalObjectRhoPhi(typing.NamedTuple):
    rho: float
    phi: float

    @property
    def elements(self):
        return (self.rho, self.phi)


AzimuthalObjectRhoPhi.__bases__ = (
    AzimuthalObject,
    vector.geometry.AzimuthalRhoPhi,
    tuple,
)


class LongitudinalObjectZ(typing.NamedTuple):
    z: float

    @property
    def elements(self):
        return (self.z,)


LongitudinalObjectZ.__bases__ = (
    LongitudinalObject,
    vector.geometry.LongitudinalZ,
    tuple,
)


class LongitudinalObjectTheta(typing.NamedTuple):
    theta: float

    @property
    def elements(self):
        return (self.theta,)


LongitudinalObjectTheta.__bases__ = (
    LongitudinalObject,
    vector.geometry.LongitudinalTheta,
    tuple,
)


class LongitudinalObjectEta(typing.NamedTuple):
    eta: float

    @property
    def elements(self):
        return (self.eta,)


LongitudinalObjectEta.__bases__ = (
    LongitudinalObject,
    vector.geometry.LongitudinalEta,
    tuple,
)


class TemporalObjectT(typing.NamedTuple):
    t: float

    @property
    def elements(self):
        return (self.t,)


TemporalObjectT.__bases__ = (TemporalObject, vector.geometry.TemporalT, tuple)


class TemporalObjectTau(typing.NamedTuple):
    tau: float

    @property
    def elements(self):
        return (self.tau,)


TemporalObjectTau.__bases__ = (TemporalObject, vector.geometry.TemporalTau, tuple)


class VectorObject2D(VectorObject, vector.methods.Planar, vector.geometry.Vector2D):
    __slots__ = ("azimuthal",)

    lib = numpy

    @classmethod
    def from_xy(cls, x, y):
        return cls(AzimuthalObjectXY(x, y))

    @classmethod
    def from_rhophi(cls, rho, phi):
        return cls(AzimuthalObjectRhoPhi(rho, phi))

    def __init__(self, azimuthal):
        self.azimuthal = azimuthal

    def __repr__(self):
        aznames = _coordinate_class_to_names[vector.geometry.aztype(self)]
        out = []
        for x in aznames:
            out.append(f"{x}={getattr(self.azimuthal, x)}")
        return "vector.generic(" + ", ".join(out) + ")"

    def _wrap_result(self, result, returns):
        if isinstance(self, type):
            cls = self
        else:
            cls = type(self)

        if returns == [float] or returns == [bool]:
            return result

        elif returns == [vector.geometry.AzimuthalXY]:
            return cls(AzimuthalObjectXY(*result))

        elif returns == [vector.geometry.AzimuthalRhoPhi]:
            return cls(AzimuthalObjectRhoPhi(*result))

        else:
            raise AssertionError(repr(returns))


class MomentumObject2D(vector.methods.PlanarMomentum, VectorObject2D):
    def __repr__(self):
        aznames = _coordinate_class_to_names[vector.geometry.aztype(self)]
        out = []
        for x in aznames:
            out.append(f"{x}={getattr(self.azimuthal, x)}")
        return "vector.momentum(" + ", ".join(out) + ")"


class VectorObject3D(VectorObject, vector.methods.Spatial, vector.geometry.Vector3D):
    __slots__ = ("azimuthal", "longitudinal")

    lib = numpy
    ProjectionClass2D = VectorObject2D

    @classmethod
    def from_xyz(cls, x, y, z):
        return cls(AzimuthalObjectXY(x, y), LongitudinalObjectZ(z))

    @classmethod
    def from_xytheta(cls, x, y, theta):
        return cls(AzimuthalObjectXY(x, y), LongitudinalObjectTheta(theta))

    @classmethod
    def from_xyeta(cls, x, y, eta):
        return cls(AzimuthalObjectXY(x, y), LongitudinalObjectEta(eta))

    @classmethod
    def from_rhophiz(cls, rho, phi, z):
        return cls(AzimuthalObjectRhoPhi(rho, phi), LongitudinalObjectZ(z))

    @classmethod
    def from_rhophitheta(cls, rho, phi, theta):
        return cls(AzimuthalObjectRhoPhi(rho, phi), LongitudinalObjectTheta(theta))

    @classmethod
    def from_rhophieta(cls, rho, phi, eta):
        return cls(AzimuthalObjectRhoPhi(rho, phi), LongitudinalObjectEta(eta))

    def __init__(self, azimuthal, longitudinal):
        self.azimuthal = azimuthal
        self.longitudinal = longitudinal

    def __repr__(self):
        aznames = _coordinate_class_to_names[vector.geometry.aztype(self)]
        lnames = _coordinate_class_to_names[vector.geometry.ltype(self)]
        out = []
        for x in aznames:
            out.append(f"{x}={getattr(self.azimuthal, x)}")
        for x in lnames:
            out.append(f"{x}={getattr(self.longitudinal, x)}")
        return "vector.generic(" + ", ".join(out) + ")"

    def _wrap_result(self, result, returns):
        if isinstance(self, type):
            cls = self
        else:
            cls = type(self)

        if returns == [float] or returns == [bool]:
            return result

        elif returns == [vector.geometry.AzimuthalXY]:
            return cls(AzimuthalObjectXY(*result), self.longitudinal)

        elif returns == [vector.geometry.AzimuthalRhoPhi]:
            return cls(AzimuthalObjectRhoPhi(*result), self.longitudinal)

        elif (
            (len(returns) == 2 or (len(returns) == 3 and returns[2] is None))
            and isinstance(returns[0], type)
            and issubclass(returns[0], vector.geometry.Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], vector.geometry.Longitudinal)
        ):
            if returns[0] is vector.geometry.AzimuthalXY:
                azimuthal = AzimuthalObjectXY(result[0], result[1])
            elif returns[0] is vector.geometry.AzimuthalRhoPhi:
                azimuthal = AzimuthalObjectRhoPhi(result[0], result[1])
            else:
                raise AssertionError(repr(returns[0]))
            if returns[1] is vector.geometry.LongitudinalZ:
                longitudinal = LongitudinalObjectZ(result[2])
            elif returns[1] is vector.geometry.LongitudinalTheta:
                longitudinal = LongitudinalObjectTheta(result[2])
            elif returns[1] is vector.geometry.LongitudinalEta:
                longitudinal = LongitudinalObjectEta(result[2])
            else:
                raise AssertionError(repr(returns[1]))
            return cls(azimuthal, longitudinal)

        else:
            raise AssertionError(repr(returns))


class MomentumObject3D(vector.methods.SpatialMomentum, VectorObject3D):
    ProjectionClass2D = MomentumObject2D

    def __repr__(self):
        aznames = _coordinate_class_to_names[vector.geometry.aztype(self)]
        lnames = _coordinate_class_to_names[vector.geometry.ltype(self)]
        out = []
        for x in aznames:
            out.append(f"{x}={getattr(self.azimuthal, x)}")
        for x in lnames:
            out.append(f"{x}={getattr(self.longitudinal, x)}")
        return "vector.momentum(" + ", ".join(out) + ")"


class VectorObject4D(VectorObject, vector.methods.Lorentz, vector.geometry.Vector4D):
    __slots__ = ("azimuthal", "longitudinal", "temporal")

    lib = numpy
    ProjectionClass2D = VectorObject2D
    ProjectionClass3D = VectorObject3D

    @classmethod
    def from_xyzt(cls, x, y, z, t):
        return cls(AzimuthalObjectXY(x, y), LongitudinalObjectZ(z), TemporalObjectT(t))

    @classmethod
    def from_xyztau(cls, x, y, z, tau):
        return cls(
            AzimuthalObjectXY(x, y), LongitudinalObjectZ(z), TemporalObjectTau(tau)
        )

    @classmethod
    def from_xythetat(cls, x, y, theta, t):
        return cls(
            AzimuthalObjectXY(x, y), LongitudinalObjectTheta(theta), TemporalObjectT(t)
        )

    @classmethod
    def from_xythetatau(cls, x, y, theta, tau):
        return cls(
            AzimuthalObjectXY(x, y),
            LongitudinalObjectTheta(theta),
            TemporalObjectTau(tau),
        )

    @classmethod
    def from_xyetat(cls, x, y, eta, t):
        return cls(
            AzimuthalObjectXY(x, y), LongitudinalObjectEta(eta), TemporalObjectT(t)
        )

    @classmethod
    def from_xyetatau(cls, x, y, eta, tau):
        return cls(
            AzimuthalObjectXY(x, y), LongitudinalObjectEta(eta), TemporalObjectTau(tau)
        )

    @classmethod
    def from_rhophizt(cls, rho, phi, z, t):
        return cls(
            AzimuthalObjectRhoPhi(rho, phi), LongitudinalObjectZ(z), TemporalObjectT(t)
        )

    @classmethod
    def from_rhophiztau(cls, rho, phi, z, tau):
        return cls(
            AzimuthalObjectRhoPhi(rho, phi),
            LongitudinalObjectZ(z),
            TemporalObjectTau(tau),
        )

    @classmethod
    def from_rhophithetat(cls, rho, phi, theta, t):
        return cls(
            AzimuthalObjectRhoPhi(rho, phi),
            LongitudinalObjectTheta(theta),
            TemporalObjectT(t),
        )

    @classmethod
    def from_rhophithetatau(cls, rho, phi, theta, tau):
        return cls(
            AzimuthalObjectRhoPhi(rho, phi),
            LongitudinalObjectTheta(theta),
            TemporalObjectTau(tau),
        )

    @classmethod
    def from_rhophietat(cls, rho, phi, eta, t):
        return cls(
            AzimuthalObjectRhoPhi(rho, phi),
            LongitudinalObjectEta(eta),
            TemporalObjectT(t),
        )

    @classmethod
    def from_rhophietatau(cls, rho, phi, eta, tau):
        return cls(
            AzimuthalObjectRhoPhi(rho, phi),
            LongitudinalObjectEta(eta),
            TemporalObjectTau(tau),
        )

    def __init__(self, azimuthal, longitudinal, temporal):
        self.azimuthal = azimuthal
        self.longitudinal = longitudinal
        self.temporal = temporal

    def __repr__(self):
        aznames = _coordinate_class_to_names[vector.geometry.aztype(self)]
        lnames = _coordinate_class_to_names[vector.geometry.ltype(self)]
        tnames = _coordinate_class_to_names[vector.geometry.ttype(self)]
        out = []
        for x in aznames:
            out.append(f"{x}={getattr(self.azimuthal, x)}")
        for x in lnames:
            out.append(f"{x}={getattr(self.longitudinal, x)}")
        for x in tnames:
            out.append(f"{x}={getattr(self.temporal, x)}")
        return "vector.generic(" + ", ".join(out) + ")"

    def _wrap_result(self, result, returns):
        if isinstance(self, type):
            cls = self
        else:
            cls = type(self)

        if returns == [float] or returns == [bool]:
            return result

        elif returns == [vector.geometry.AzimuthalXY]:
            return cls(AzimuthalObjectXY(*result), self.longitudinal, self.temporal)

        elif returns == [vector.geometry.AzimuthalRhoPhi]:
            return cls(AzimuthalObjectRhoPhi(*result), self.longitudinal, self.temporal)

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], vector.geometry.Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], vector.geometry.Longitudinal)
        ):
            if returns[0] is vector.geometry.AzimuthalXY:
                azimuthal = AzimuthalObjectXY(result[0], result[1])
            elif returns[0] is vector.geometry.AzimuthalRhoPhi:
                azimuthal = AzimuthalObjectRhoPhi(result[0], result[1])
            else:
                raise AssertionError(repr(returns[0]))
            if returns[1] is vector.geometry.LongitudinalZ:
                longitudinal = LongitudinalObjectZ(result[2])
            elif returns[1] is vector.geometry.LongitudinalTheta:
                longitudinal = LongitudinalObjectTheta(result[2])
            elif returns[1] is vector.geometry.LongitudinalEta:
                longitudinal = LongitudinalObjectEta(result[2])
            else:
                raise AssertionError(repr(returns[1]))
            return cls(azimuthal, longitudinal, self.temporal)

        elif (
            len(returns) == 3
            and isinstance(returns[0], type)
            and issubclass(returns[0], vector.geometry.Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], vector.geometry.Longitudinal)
        ):
            if returns[0] is vector.geometry.AzimuthalXY:
                azimuthal = AzimuthalObjectXY(result[0], result[1])
            elif returns[0] is vector.geometry.AzimuthalRhoPhi:
                azimuthal = AzimuthalObjectRhoPhi(result[0], result[1])
            else:
                raise AssertionError(repr(returns[0]))
            if returns[1] is vector.geometry.LongitudinalZ:
                longitudinal = LongitudinalObjectZ(result[2])
            elif returns[1] is vector.geometry.LongitudinalTheta:
                longitudinal = LongitudinalObjectTheta(result[2])
            elif returns[1] is vector.geometry.LongitudinalEta:
                longitudinal = LongitudinalObjectEta(result[2])
            else:
                raise AssertionError(repr(returns[1]))
            is_4d = True
            if returns[2] is vector.geometry.TemporalT:
                temporal = TemporalObjectT(result[3])
            elif returns[2] is vector.geometry.TemporalTau:
                temporal = TemporalObjectTau(result[3])
            elif returns[2] is None:
                is_4d = False
            else:
                raise AssertionError(repr(returns[2]))
            if is_4d:
                return cls(azimuthal, longitudinal, temporal)
            else:
                return self.ProjectionClass3D(azimuthal, longitudinal)

        else:
            raise AssertionError(repr(returns))


class MomentumObject4D(vector.methods.LorentzMomentum, VectorObject4D):
    ProjectionClass2D = MomentumObject2D
    ProjectionClass3D = MomentumObject3D

    def __repr__(self):
        aznames = _coordinate_class_to_names[vector.geometry.aztype(self)]
        lnames = _coordinate_class_to_names[vector.geometry.ltype(self)]
        tnames = _coordinate_class_to_names[vector.geometry.ttype(self)]
        out = []
        for x in aznames:
            out.append(f"{x}={getattr(self.azimuthal, x)}")
        for x in lnames:
            out.append(f"{x}={getattr(self.longitudinal, x)}")
        for x in tnames:
            out.append(f"{x}={getattr(self.temporal, x)}")
        return "vector.momentum(" + ", ".join(out) + ")"


def _gather_coordinates(planar_class, spatial_class, lorentz_class, coordinates):
    azimuthal = None
    if "x" in coordinates and "y" in coordinates:
        if "rho" in coordinates or "phi" in coordinates:
            raise TypeError("specify x= and y= or rho= and phi=, but not both")
        azimuthal = AzimuthalObjectXY(coordinates.pop("x"), coordinates.pop("y"))
    elif "rho" in coordinates and "phi" in coordinates:
        if "x" in coordinates or "y" in coordinates:
            raise TypeError("specify x= and y= or rho= and phi=, but not both")
        azimuthal = AzimuthalObjectRhoPhi(
            coordinates.pop("rho"), coordinates.pop("phi")
        )

    longitudinal = None
    if "z" in coordinates:
        if "theta" in coordinates or "eta" in coordinates:
            raise TypeError("specify z= or theta= or eta=, but not more than one")
        longitudinal = LongitudinalObjectZ(coordinates.pop("z"))
    elif "theta" in coordinates:
        if "z" in coordinates or "eta" in coordinates:
            raise TypeError("specify z= or theta= or eta=, but not more than one")
        longitudinal = LongitudinalObjectTheta(coordinates.pop("theta"))
    elif "eta" in coordinates:
        if "z" in coordinates or "theta" in coordinates:
            raise TypeError("specify z= or theta= or eta=, but not more than one")
        longitudinal = LongitudinalObjectEta(coordinates.pop("eta"))

    temporal = None
    if "t" in coordinates:
        if "tau" in coordinates:
            raise TypeError("specify t= or tau=, but not more than one")
        temporal = TemporalObjectT(coordinates.pop("t"))
    elif "tau" in coordinates:
        if "t" in coordinates:
            raise TypeError("specify t= or tau=, but not more than one")
        temporal = TemporalObjectTau(coordinates.pop("tau"))

    if len(coordinates) == 0:
        if azimuthal is not None and longitudinal is None and temporal is None:
            return planar_class(azimuthal)
        if azimuthal is not None and longitudinal is not None and temporal is None:
            return spatial_class(azimuthal, longitudinal)
        if azimuthal is not None and longitudinal is not None and temporal is not None:
            return lorentz_class(azimuthal, longitudinal, temporal)

    raise TypeError(
        "unrecognized combination of coordinates, allowed combinations are:\n\n"
        "    (2D) x= y=\n"
        "    (2D) rho= phi=\n"
        "    (3D) x= y= z=\n"
        "    (3D) x= y= theta=\n"
        "    (3D) x= y= eta=\n"
        "    (3D) rho= phi= z=\n"
        "    (3D) rho= phi= theta=\n"
        "    (3D) rho= phi= eta=\n"
        "    (4D) x= y= z= t=\n"
        "    (4D) x= y= z= tau=\n"
        "    (4D) x= y= theta= t=\n"
        "    (4D) x= y= theta= tau=\n"
        "    (4D) x= y= eta= t=\n"
        "    (4D) x= y= eta= tau=\n"
        "    (4D) rho= phi= z= t=\n"
        "    (4D) rho= phi= z= tau=\n"
        "    (4D) rho= phi= theta= t=\n"
        "    (4D) rho= phi= theta= tau=\n"
        "    (4D) rho= phi= eta= t=\n"
        "    (4D) rho= phi= eta= tau="
    )


def generic(**coordinates):
    "generic docs"
    return _gather_coordinates(
        VectorObject2D, VectorObject3D, VectorObject4D, coordinates
    )


def momentum(**coordinates):
    "momentum docs"
    return _gather_coordinates(
        MomentumObject2D, MomentumObject3D, MomentumObject4D, coordinates
    )
