# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

import awkward as ak
import numpy

from vector.methods import (
    Azimuthal,
    AzimuthalRhoPhi,
    AzimuthalXY,
    Longitudinal,
    LongitudinalEta,
    LongitudinalTheta,
    LongitudinalZ,
    Lorentz,
    LorentzMomentum,
    Planar,
    PlanarMomentum,
    Spatial,
    SpatialMomentum,
    Temporal,
    TemporalT,
    TemporalTau,
    Vector2D,
    Vector3D,
    Vector4D,
)

behavior: typing.Any = {}


class CoordinatesAwkward:
    lib = numpy


class AzimuthalAwkward(CoordinatesAwkward):
    @classmethod
    def from_fields(cls, array):
        fields = ak.fields(array)
        if "x" in fields and "y" in fields:
            return AzimuthalAwkwardXY(array["x"], array["y"])
        elif "rho" in fields and "phi" in fields:
            return AzimuthalAwkwardRhoPhi(array["rho"], array["phi"])
        else:
            raise ValueError(
                f"array does not have azimuthal coordinates (x/y/rho/phi): {', '.join(fields)}"
            )


class LongitudinalAwkward(CoordinatesAwkward):
    @classmethod
    def from_fields(cls, array):
        fields = ak.fields(array)
        if "z" in fields:
            return LongitudinalAwkwardZ(array["z"])
        elif "theta" in fields:
            return LongitudinalAwkwardTheta(array["theta"])
        elif "eta" in fields:
            return LongitudinalAwkwardEta(array["eta"])
        else:
            raise ValueError(
                f"array does not have longitudinal coordinates (z/theta/eta): {', '.join(fields)}"
            )


class TemporalAwkward(CoordinatesAwkward):
    @classmethod
    def from_fields(cls, array):
        fields = ak.fields(array)
        if "t" in fields:
            return TemporalAwkwardT(array["t"])
        elif "tau" in fields:
            return TemporalAwkwardTau(array["tau"])
        else:
            raise ValueError(
                f"array does not have temporal coordinates (t/tau): {', '.join(fields)}"
            )


class AzimuthalAwkwardXY(AzimuthalAwkward, AzimuthalXY):
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def elements(self):
        return (self.x, self.y)


class AzimuthalAwkwardRhoPhi(AzimuthalAwkward, AzimuthalRhoPhi):
    __slots__ = ("rho", "phi")

    def __init__(self, rho, phi):
        self.rho = rho
        self.phi = phi

    @property
    def elements(self):
        return (self.rho, self.phi)


class LongitudinalAwkwardZ(LongitudinalAwkward, LongitudinalZ):
    __slots__ = ("z",)

    def __init__(self, z):
        self.z = z

    @property
    def elements(self):
        return (self.z,)


class LongitudinalAwkwardTheta(LongitudinalAwkward, LongitudinalTheta):
    __slots__ = ("theta",)

    def __init__(self, theta):
        self.theta = theta

    @property
    def elements(self):
        return (self.theta,)


class LongitudinalAwkwardEta(LongitudinalAwkward, LongitudinalEta):
    __slots__ = ("eta",)

    def __init__(self, eta):
        self.eta = eta

    @property
    def elements(self):
        return (self.eta,)


class TemporalAwkwardT(TemporalAwkward, TemporalT):
    __slots__ = ("t",)

    def __init__(self, t):
        self.t = t

    @property
    def elements(self):
        return (self.t,)


class TemporalAwkwardTau(TemporalAwkward, TemporalTau):
    __slots__ = ("tau",)

    def __init__(self, tau):
        self.tau = tau

    @property
    def elements(self):
        return (self.tau,)


class VectorAwkward:
    lib = numpy


class VectorAwkward2D(VectorAwkward, Planar, Vector2D):
    @property
    def azimuthal(self):
        return AzimuthalAwkward.from_fields(self)

    def _wrap_result(self, cls, result, returns):
        if returns == [float] or returns == [bool]:
            return result

        elif (
            len(returns) == 1
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
        ):
            raise NotImplementedError

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and returns[1] is None
        ):
            raise NotImplementedError

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
        ):
            raise NotImplementedError

        elif (
            len(returns) == 3
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
            and returns[2] is None
        ):
            raise NotImplementedError

        elif (
            len(returns) == 3
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
            and isinstance(returns[2], type)
            and issubclass(returns[2], Temporal)
        ):
            raise NotImplementedError

        else:
            raise AssertionError(repr(returns))


class MomentumAwkward2D(PlanarMomentum, VectorAwkward2D):
    pass


class VectorAwkward3D(VectorAwkward, Spatial, Vector3D):
    @property
    def azimuthal(self):
        return AzimuthalAwkward.from_fields(self)

    @property
    def longitudinal(self):
        return LongitudinalAwkward.from_fields(self)

    def _wrap_result(self, cls, result, returns):
        if returns == [float] or returns == [bool]:
            return result

        elif (
            len(returns) == 1
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
        ):
            raise NotImplementedError

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and returns[1] is None
        ):
            raise NotImplementedError

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
        ):
            raise NotImplementedError

        elif (
            len(returns) == 3
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
            and returns[2] is None
        ):
            raise NotImplementedError

        elif (
            len(returns) == 3
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
            and isinstance(returns[2], type)
            and issubclass(returns[2], Temporal)
        ):
            raise NotImplementedError

        else:
            raise AssertionError(repr(returns))


class MomentumAwkward3D(SpatialMomentum, VectorAwkward3D):
    pass


class VectorAwkward4D(VectorAwkward, Lorentz, Vector4D):
    @property
    def azimuthal(self):
        return AzimuthalAwkward.from_fields(self)

    @property
    def longitudinal(self):
        return LongitudinalAwkward.from_fields(self)

    @property
    def temporal(self):
        return TemporalAwkward.from_fields(self)

    def _wrap_result(self, cls, result, returns):
        if returns == [float] or returns == [bool]:
            return result

        elif (
            len(returns) == 1
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
        ):
            raise NotImplementedError

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and returns[1] is None
        ):
            raise NotImplementedError

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
        ):
            raise NotImplementedError

        elif (
            len(returns) == 3
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
            and returns[2] is None
        ):
            raise NotImplementedError

        elif (
            len(returns) == 3
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
            and isinstance(returns[2], type)
            and issubclass(returns[2], Temporal)
        ):
            raise NotImplementedError

        else:
            raise AssertionError(repr(returns))


class MomentumAwkward4D(LorentzMomentum, VectorAwkward4D):
    pass


class VectorArray2D(VectorAwkward2D, ak.Array):
    ProjectionClass2D: typing.Any
    ProjectionClass3D: typing.Any
    ProjectionClass4D: typing.Any
    GenericClass: typing.Any


behavior["*", "Vector2D"] = VectorArray2D


class VectorRecord2D(VectorAwkward2D, ak.Record):
    ProjectionClass2D: typing.Any
    ProjectionClass3D: typing.Any
    ProjectionClass4D: typing.Any
    GenericClass: typing.Any


behavior["Vector2D"] = VectorRecord2D


class VectorArray3D(VectorAwkward3D, ak.Array):
    ProjectionClass2D: typing.Any
    ProjectionClass3D: typing.Any
    ProjectionClass4D: typing.Any
    GenericClass: typing.Any


behavior["*", "Vector3D"] = VectorArray3D


class VectorRecord3D(VectorAwkward3D, ak.Record):
    ProjectionClass2D: typing.Any
    ProjectionClass3D: typing.Any
    ProjectionClass4D: typing.Any
    GenericClass: typing.Any


behavior["Vector3D"] = VectorRecord3D


class VectorArray4D(VectorAwkward4D, ak.Array):
    ProjectionClass2D: typing.Any
    ProjectionClass3D: typing.Any
    ProjectionClass4D: typing.Any
    GenericClass: typing.Any


behavior["*", "Vector4D"] = VectorArray4D


class VectorRecord4D(VectorAwkward4D, ak.Record):
    ProjectionClass2D: typing.Any
    ProjectionClass3D: typing.Any
    ProjectionClass4D: typing.Any
    GenericClass: typing.Any


behavior["Vector4D"] = VectorRecord4D


class MomentumArray2D(MomentumAwkward2D, ak.Array):
    ProjectionClass2D: typing.Any
    ProjectionClass3D: typing.Any
    ProjectionClass4D: typing.Any
    GenericClass: typing.Any


behavior["*", "Momentum2D"] = MomentumArray2D


class MomentumRecord2D(MomentumAwkward2D, ak.Record):
    ProjectionClass2D: typing.Any
    ProjectionClass3D: typing.Any
    ProjectionClass4D: typing.Any
    GenericClass: typing.Any


behavior["Momentum2D"] = MomentumRecord2D


class MomentumArray3D(MomentumAwkward3D, ak.Array):
    ProjectionClass2D: typing.Any
    ProjectionClass3D: typing.Any
    ProjectionClass4D: typing.Any
    GenericClass: typing.Any


behavior["*", "Momentum3D"] = MomentumArray3D


class MomentumRecord3D(MomentumAwkward3D, ak.Record):
    ProjectionClass2D: typing.Any
    ProjectionClass3D: typing.Any
    ProjectionClass4D: typing.Any
    GenericClass: typing.Any


behavior["Momentum3D"] = MomentumRecord3D


class MomentumArray4D(MomentumAwkward4D, ak.Array):
    ProjectionClass2D: typing.Any
    ProjectionClass3D: typing.Any
    ProjectionClass4D: typing.Any
    GenericClass: typing.Any


behavior["*", "Momentum4D"] = MomentumArray4D


class MomentumRecord4D(MomentumAwkward4D, ak.Record):
    ProjectionClass2D: typing.Any
    ProjectionClass3D: typing.Any
    ProjectionClass4D: typing.Any
    GenericClass: typing.Any


behavior["Momentum4D"] = MomentumRecord4D


_class_to_name = {
    VectorArray2D: "Vector2D",
    VectorRecord2D: "Vector2D",
    VectorArray3D: "Vector3D",
    VectorRecord3D: "Vector3D",
    VectorArray4D: "Vector4D",
    VectorRecord4D: "Vector4D",
    MomentumArray2D: "Momentum2D",
    MomentumRecord2D: "Momentum2D",
    MomentumArray3D: "Momentum3D",
    MomentumRecord3D: "Momentum3D",
    MomentumArray4D: "Momentum4D",
    MomentumRecord4D: "Momentum4D",
}

VectorArray2D.ProjectionClass2D = VectorArray2D
VectorArray2D.ProjectionClass3D = VectorArray3D
VectorArray2D.ProjectionClass4D = VectorArray4D
VectorArray2D.GenericClass = VectorArray2D

VectorRecord2D.ProjectionClass2D = VectorRecord2D
VectorRecord2D.ProjectionClass3D = VectorRecord3D
VectorRecord2D.ProjectionClass4D = VectorRecord4D
VectorRecord2D.GenericClass = VectorRecord2D

MomentumArray2D.ProjectionClass2D = MomentumArray2D
MomentumArray2D.ProjectionClass3D = MomentumArray3D
MomentumArray2D.ProjectionClass4D = MomentumArray4D
MomentumArray2D.GenericClass = VectorArray2D

MomentumRecord2D.ProjectionClass2D = MomentumRecord2D
MomentumRecord2D.ProjectionClass3D = MomentumRecord3D
MomentumRecord2D.ProjectionClass4D = MomentumRecord4D
MomentumRecord2D.GenericClass = VectorRecord2D

VectorArray3D.ProjectionClass2D = VectorArray2D
VectorArray3D.ProjectionClass3D = VectorArray3D
VectorArray3D.ProjectionClass4D = VectorArray4D
VectorArray3D.GenericClass = VectorArray3D

VectorRecord3D.ProjectionClass2D = VectorRecord2D
VectorRecord3D.ProjectionClass3D = VectorRecord3D
VectorRecord3D.ProjectionClass4D = VectorRecord4D
VectorRecord3D.GenericClass = VectorRecord3D

MomentumArray3D.ProjectionClass2D = MomentumArray2D
MomentumArray3D.ProjectionClass3D = MomentumArray3D
MomentumArray3D.ProjectionClass4D = MomentumArray4D
MomentumArray3D.GenericClass = VectorArray3D

MomentumRecord3D.ProjectionClass2D = MomentumRecord2D
MomentumRecord3D.ProjectionClass3D = MomentumRecord3D
MomentumRecord3D.ProjectionClass4D = MomentumRecord4D
MomentumRecord3D.GenericClass = VectorRecord3D

VectorArray4D.ProjectionClass2D = VectorArray2D
VectorArray4D.ProjectionClass3D = VectorArray3D
VectorArray4D.ProjectionClass4D = VectorArray4D
VectorArray4D.GenericClass = VectorArray4D

VectorRecord4D.ProjectionClass2D = VectorRecord2D
VectorRecord4D.ProjectionClass3D = VectorRecord3D
VectorRecord4D.ProjectionClass4D = VectorRecord4D
VectorRecord4D.GenericClass = VectorRecord4D

MomentumArray4D.ProjectionClass2D = MomentumArray2D
MomentumArray4D.ProjectionClass3D = MomentumArray3D
MomentumArray4D.ProjectionClass4D = MomentumArray4D
MomentumArray4D.GenericClass = VectorArray4D

MomentumRecord4D.ProjectionClass2D = MomentumRecord2D
MomentumRecord4D.ProjectionClass3D = MomentumRecord3D
MomentumRecord4D.ProjectionClass4D = MomentumRecord4D
MomentumRecord4D.GenericClass = VectorRecord4D
