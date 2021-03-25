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
    Momentum,
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


def _class_to_name(cls):
    if issubclass(cls, Momentum):
        if issubclass(cls, Vector2D):
            return "Momentum2D"
        elif issubclass(cls, Vector3D):
            return "Momentum3D"
        elif issubclass(cls, Vector4D):
            return "Momentum4D"
    else:
        if issubclass(cls, Vector2D):
            return "Vector2D"
        elif issubclass(cls, Vector3D):
            return "Vector3D"
        elif issubclass(cls, Vector4D):
            return "Vector4D"


class VectorAwkward:
    lib = numpy

    def _wrap_result(self, cls, result, returns, num_vecargs):
        if returns == [float] or returns == [bool]:
            return result

        elif (
            len(returns) == 1
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
        ):
            first = [x for x in result if isinstance(x, ak.Array)][0]
            result = [
                x if isinstance(x, ak.Array) else ak.broadcast_arrays(first, x)[1]
                for x in result
            ]

            names = []
            arrays = []
            if returns[0] is AzimuthalXY:
                names.extend(["x", "y"])
                arrays.extend([result[0], result[1]])
            elif returns[0] is AzimuthalRhoPhi:
                names.extend(["rho", "phi"])
                arrays.extend([result[0], result[1]])

            fields = ak.fields(self)
            if num_vecargs == 1:
                for name in fields:
                    if name not in ("x", "y", "rho", "phi"):
                        names.append(name)
                        arrays.append(self[name])

            if "t" in fields or "tau" in fields:
                cls = cls.ProjectionClass4D
            elif "z" in fields or "theta" in fields or "eta" in fields:
                cls = cls.ProjectionClass3D
            else:
                cls = cls.ProjectionClass2D

            return ak.zip(
                dict(zip(names, arrays)),
                depth_limit=first.layout.purelist_depth,
                with_name=_class_to_name(cls),
                behavior=self.behavior,
            )

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and returns[1] is None
        ):
            first = [x for x in result if isinstance(x, ak.Array)][0]
            result = [
                x if isinstance(x, ak.Array) else ak.broadcast_arrays(first, x)[1]
                for x in result
            ]

            names = []
            arrays = []
            if returns[0] is AzimuthalXY:
                names.extend(["x", "y"])
                arrays.extend([result[0], result[1]])
            elif returns[0] is AzimuthalRhoPhi:
                names.extend(["rho", "phi"])
                arrays.extend([result[0], result[1]])

            if num_vecargs == 1:
                for name in ak.fields(self):
                    if name not in (
                        "x",
                        "y",
                        "rho",
                        "phi",
                        "z",
                        "theta",
                        "eta",
                        "t",
                        "tau",
                    ):
                        names.append(name)
                        arrays.append(self[name])

            return ak.zip(
                dict(zip(names, arrays)),
                depth_limit=first.layout.purelist_depth,
                with_name=_class_to_name(cls.ProjectionClass2D),
                behavior=self.behavior,
            )

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
        ):
            first = [x for x in result if isinstance(x, ak.Array)][0]
            result = [
                x if isinstance(x, ak.Array) else ak.broadcast_arrays(first, x)[1]
                for x in result
            ]

            names = []
            arrays = []
            if returns[0] is AzimuthalXY:
                names.extend(["x", "y"])
                arrays.extend([result[0], result[1]])
            elif returns[0] is AzimuthalRhoPhi:
                names.extend(["rho", "phi"])
                arrays.extend([result[0], result[1]])

            if returns[1] is LongitudinalZ:
                names.append("z")
                arrays.append(result[2])
            elif returns[1] is LongitudinalTheta:
                names.append("theta")
                arrays.append(result[2])
            elif returns[1] is LongitudinalEta:
                names.append("eta")
                arrays.append(result[2])

            fields = ak.fields(self)
            if num_vecargs == 1:
                for name in fields:
                    if name not in ("x", "y", "rho", "phi", "z", "theta", "eta"):
                        names.append(name)
                        arrays.append(self[name])

            if "t" in fields or "tau" in fields:
                cls = cls.ProjectionClass4D
            else:
                cls = cls.ProjectionClass3D

            return ak.zip(
                dict(zip(names, arrays)),
                depth_limit=first.layout.purelist_depth,
                with_name=_class_to_name(cls),
                behavior=self.behavior,
            )

        elif (
            len(returns) == 3
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
            and returns[2] is None
        ):
            first = [x for x in result if isinstance(x, ak.Array)][0]
            result = [
                x if isinstance(x, ak.Array) else ak.broadcast_arrays(first, x)[1]
                for x in result
            ]

            names = []
            arrays = []
            if returns[0] is AzimuthalXY:
                names.extend(["x", "y"])
                arrays.extend([result[0], result[1]])
            elif returns[0] is AzimuthalRhoPhi:
                names.extend(["rho", "phi"])
                arrays.extend([result[0], result[1]])

            if returns[1] is LongitudinalZ:
                names.append("z")
                arrays.append(result[2])
            elif returns[1] is LongitudinalTheta:
                names.append("theta")
                arrays.append(result[2])
            elif returns[1] is LongitudinalEta:
                names.append("eta")
                arrays.append(result[2])

            if num_vecargs == 1:
                for name in ak.fields(self):
                    if name not in (
                        "x",
                        "y",
                        "rho",
                        "phi",
                        "z",
                        "theta",
                        "eta",
                        "t",
                        "tau",
                    ):
                        names.append(name)
                        arrays.append(self[name])

            return ak.zip(
                dict(zip(names, arrays)),
                depth_limit=first.layout.purelist_depth,
                with_name=_class_to_name(cls.ProjectionClass3D),
                behavior=self.behavior,
            )

        elif (
            len(returns) == 3
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
            and isinstance(returns[2], type)
            and issubclass(returns[2], Temporal)
        ):
            first = [x for x in result if isinstance(x, ak.Array)][0]
            result = [
                x if isinstance(x, ak.Array) else ak.broadcast_arrays(first, x)[1]
                for x in result
            ]

            names = []
            arrays = []
            if returns[0] is AzimuthalXY:
                names.extend(["x", "y"])
                arrays.extend([result[0], result[1]])
            elif returns[0] is AzimuthalRhoPhi:
                names.extend(["rho", "phi"])
                arrays.extend([result[0], result[1]])

            if returns[1] is LongitudinalZ:
                names.append("z")
                arrays.append(result[2])
            elif returns[1] is LongitudinalTheta:
                names.append("theta")
                arrays.append(result[2])
            elif returns[1] is LongitudinalEta:
                names.append("eta")
                arrays.append(result[2])

            if returns[2] is TemporalT:
                names.append("t")
                arrays.append(result[3])
            elif returns[2] is TemporalTau:
                names.append("tau")
                arrays.append(result[3])

            if num_vecargs == 1:
                for name in ak.fields(self):
                    if name not in (
                        "x",
                        "y",
                        "rho",
                        "phi",
                        "z",
                        "theta",
                        "eta",
                        "t",
                        "tau",
                    ):
                        names.append(name)
                        arrays.append(self[name])

            return ak.zip(
                dict(zip(names, arrays)),
                depth_limit=first.layout.purelist_depth,
                with_name=_class_to_name(cls.ProjectionClass4D),
                behavior=self.behavior,
            )

        else:
            raise AssertionError(repr(returns))


class VectorAwkward2D(VectorAwkward, Planar, Vector2D):
    @property
    def azimuthal(self):
        return AzimuthalAwkward.from_fields(self)


class MomentumAwkward2D(PlanarMomentum, VectorAwkward2D):
    pass


class VectorAwkward3D(VectorAwkward, Spatial, Vector3D):
    @property
    def azimuthal(self):
        return AzimuthalAwkward.from_fields(self)

    @property
    def longitudinal(self):
        return LongitudinalAwkward.from_fields(self)


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
