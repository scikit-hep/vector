# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

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
    SameVectorType,
    Spatial,
    SpatialMomentum,
    Temporal,
    TemporalT,
    TemporalTau,
    Vector,
    Vector2D,
    Vector3D,
    Vector4D,
    VectorProtocol,
    _aztype,
    _coordinate_class_to_names,
    _handler_of,
    _ltype,
    _repr_generic_to_momentum,
    _ttype,
)


class CoordinatesObject:
    pass


class AzimuthalObject(CoordinatesObject, Azimuthal):
    pass


class LongitudinalObject(CoordinatesObject, Longitudinal):
    pass


class TemporalObject(CoordinatesObject, Temporal):
    pass


class AzimuthalObjectXY(typing.NamedTuple):
    x: float
    y: float

    @property
    def elements(self) -> typing.Tuple[float, float]:
        return (self.x, self.y)


AzimuthalObjectXY.__bases__ = (AzimuthalObject, AzimuthalXY, tuple)


class AzimuthalObjectRhoPhi(typing.NamedTuple):
    rho: float
    phi: float

    @property
    def elements(self) -> typing.Tuple[float, float]:
        return (self.rho, self.phi)


AzimuthalObjectRhoPhi.__bases__ = (AzimuthalObject, AzimuthalRhoPhi, tuple)


class LongitudinalObjectZ(typing.NamedTuple):
    z: float

    @property
    def elements(self) -> typing.Tuple[float]:
        return (self.z,)


LongitudinalObjectZ.__bases__ = (LongitudinalObject, LongitudinalZ, tuple)


class LongitudinalObjectTheta(typing.NamedTuple):
    theta: float

    @property
    def elements(self) -> typing.Tuple[float]:
        return (self.theta,)


LongitudinalObjectTheta.__bases__ = (LongitudinalObject, LongitudinalTheta, tuple)


class LongitudinalObjectEta(typing.NamedTuple):
    eta: float

    @property
    def elements(self) -> typing.Tuple[float]:
        return (self.eta,)


LongitudinalObjectEta.__bases__ = (LongitudinalObject, LongitudinalEta, tuple)


class TemporalObjectT(typing.NamedTuple):
    t: float

    @property
    def elements(self) -> typing.Tuple[float]:
        return (self.t,)


TemporalObjectT.__bases__ = (TemporalObject, TemporalT, tuple)


class TemporalObjectTau(typing.NamedTuple):
    tau: float

    @property
    def elements(self) -> typing.Tuple[float]:
        return (self.tau,)


TemporalObjectTau.__bases__ = (TemporalObject, TemporalTau, tuple)


_coord_object_type = {
    AzimuthalXY: AzimuthalObjectXY,
    AzimuthalRhoPhi: AzimuthalObjectRhoPhi,
    LongitudinalZ: LongitudinalObjectZ,
    LongitudinalTheta: LongitudinalObjectTheta,
    LongitudinalEta: LongitudinalObjectEta,
    TemporalT: TemporalObjectT,
    TemporalTau: TemporalObjectTau,
}


def _replace_data(obj: typing.Any, result: typing.Any) -> typing.Any:
    if not isinstance(result, VectorObject):
        raise TypeError(f"can only assign a single vector to {type(obj).__name__}")

    if isinstance(result, (VectorObject2D, VectorObject3D, VectorObject4D)):
        if isinstance(obj.azimuthal, AzimuthalObjectXY):
            obj.azimuthal = AzimuthalObjectXY(result.x, result.y)
        elif isinstance(obj.azimuthal, AzimuthalObjectRhoPhi):
            obj.azimuthal = AzimuthalObjectRhoPhi(result.rho, result.phi)
        else:
            raise AssertionError(type(obj))

    if isinstance(result, (VectorObject3D, VectorObject4D)):
        if isinstance(obj.longitudinal, LongitudinalObjectZ):
            obj.longitudinal = LongitudinalObjectZ(result.z)
        elif isinstance(obj.longitudinal, LongitudinalObjectTheta):
            obj.longitudinal = LongitudinalObjectTheta(result.theta)
        elif isinstance(obj.longitudinal, LongitudinalObjectEta):
            obj.longitudinal = LongitudinalObjectEta(result.eta)
        else:
            raise AssertionError(type(obj))

    if isinstance(result, VectorObject4D):
        if isinstance(obj.temporal, TemporalObjectT):
            obj.temporal = TemporalObjectT(result.t)
        elif isinstance(obj.temporal, TemporalObjectTau):
            obj.temporal = TemporalObjectTau(result.tau)
        else:
            raise AssertionError(type(obj))

    return obj


class VectorObject(Vector):
    lib = numpy

    def __eq__(self, other: typing.Any) -> typing.Any:
        return numpy.equal(self, other)

    def __ne__(self, other: typing.Any) -> typing.Any:
        return numpy.not_equal(self, other)

    def __abs__(self) -> float:
        return numpy.absolute(self)

    def __add__(self, other: VectorProtocol) -> VectorProtocol:
        return numpy.add(self, other)

    def __radd__(self, other: VectorProtocol) -> VectorProtocol:
        return numpy.add(other, self)

    def __iadd__(self: SameVectorType, other: VectorProtocol) -> SameVectorType:
        return _replace_data(self, numpy.add(self, other))

    def __sub__(self, other: VectorProtocol) -> VectorProtocol:
        return numpy.subtract(self, other)

    def __rsub__(self, other: VectorProtocol) -> VectorProtocol:
        return numpy.subtract(other, self)

    def __isub__(self: SameVectorType, other: VectorProtocol) -> SameVectorType:
        return _replace_data(self, numpy.subtract(self, other))

    def __mul__(self, other: float) -> VectorProtocol:
        return numpy.multiply(self, other)

    def __rmul__(self, other: float) -> VectorProtocol:
        return numpy.multiply(other, self)

    def __imul__(self: SameVectorType, other: float) -> SameVectorType:
        return _replace_data(self, numpy.multiply(self, other))

    def __neg__(self: SameVectorType) -> SameVectorType:
        return numpy.negative(self)

    def __pos__(self: SameVectorType) -> SameVectorType:
        return numpy.positive(self)

    def __truediv__(self, other: float) -> VectorProtocol:
        return numpy.true_divide(self, other)

    def __rtruediv__(self, other: float) -> VectorProtocol:
        return numpy.true_divide(other, self)

    def __itruediv__(self: SameVectorType, other: float) -> VectorProtocol:
        return _replace_data(self, numpy.true_divide(self, other))

    def __pow__(self, other: float) -> float:
        return numpy.power(self, other)

    def __matmul__(self, other: VectorProtocol) -> float:
        return numpy.matmul(self, other)

    def __array_ufunc__(
        self,
        ufunc: typing.Any,
        method: typing.Any,
        *inputs: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        if not isinstance(_handler_of(*inputs), VectorObject):
            # Let a higher-precedence backend handle it.
            return NotImplemented

        outputs = kwargs.get("out", ())
        if any(not isinstance(x, VectorObject) for x in outputs):
            raise TypeError(
                "ufunc operating on VectorObjects can only use the 'out' keyword "
                "with another VectorObject"
            )

        if (
            ufunc is numpy.absolute
            and len(inputs) == 1
            and isinstance(inputs[0], Vector)
        ):
            if len(outputs) != 0:
                raise TypeError(
                    "output of 'numpy.absolute' is scalar, cannot fill a VectorObject with 'out'"
                )
            if isinstance(inputs[0], Vector2D):
                return inputs[0].rho
            elif isinstance(inputs[0], Vector3D):
                return inputs[0].mag
            elif isinstance(inputs[0], Vector4D):
                return inputs[0].tau

        elif (
            ufunc is numpy.add
            and len(inputs) == 2
            and isinstance(inputs[0], Vector)
            and isinstance(inputs[1], Vector)
        ):
            result = inputs[0].add(inputs[1])
            for output in outputs:
                _replace_data(output, result)
            return result

        elif (
            ufunc is numpy.subtract
            and len(inputs) == 2
            and isinstance(inputs[0], Vector)
            and isinstance(inputs[1], Vector)
        ):
            result = inputs[0].subtract(inputs[1])
            for output in outputs:
                _replace_data(output, result)
            return result

        elif (
            ufunc is numpy.multiply
            and len(inputs) == 2
            and isinstance(inputs[0], Vector)
            and not isinstance(inputs[1], Vector)
        ):
            result = inputs[0].scale(inputs[1])
            for output in outputs:
                _replace_data(output, result)
            return result

        elif (
            ufunc is numpy.multiply
            and len(inputs) == 2
            and not isinstance(inputs[0], Vector)
            and isinstance(inputs[1], Vector)
        ):
            result = inputs[1].scale(inputs[0])
            for output in outputs:
                _replace_data(output, result)
            return result

        elif (
            ufunc is numpy.negative
            and len(inputs) == 1
            and isinstance(inputs[0], Vector)
        ):
            result = inputs[0].scale(-1)
            for output in outputs:
                _replace_data(output, result)
            return result

        elif (
            ufunc is numpy.positive
            and len(inputs) == 1
            and isinstance(inputs[0], Vector)
        ):
            return inputs[0]

        elif (
            ufunc is numpy.true_divide
            and len(inputs) == 2
            and isinstance(inputs[0], Vector)
            and not isinstance(inputs[1], Vector)
        ):
            result = inputs[0].scale(1 / inputs[1])
            for output in outputs:
                _replace_data(output, result)
            return result

        elif (
            ufunc is numpy.power
            and len(inputs) == 2
            and isinstance(inputs[0], Vector)
            and not isinstance(inputs[1], Vector)
        ):
            result = numpy.absolute(inputs[0]) ** inputs[1]
            for output in outputs:
                _replace_data(output, result)
            return result

        elif (
            ufunc is numpy.square and len(inputs) == 1 and isinstance(inputs[0], Vector)
        ):
            if len(outputs) != 0:
                raise TypeError(
                    "output of 'numpy.square' is scalar, cannot fill a VectorObject with 'out'"
                )
            if isinstance(inputs[0], Vector2D):
                return inputs[0].rho2
            elif isinstance(inputs[0], Vector3D):
                return inputs[0].mag2
            elif isinstance(inputs[0], Vector4D):
                return inputs[0].tau2

        elif ufunc is numpy.sqrt and len(inputs) == 1 and isinstance(inputs[0], Vector):
            if len(outputs) != 0:
                raise TypeError(
                    "output of 'numpy.sqrt' is scalar, cannot fill a VectorObject with 'out'"
                )
            if isinstance(inputs[0], Vector2D):
                return inputs[0].rho2 ** 0.25
            elif isinstance(inputs[0], Vector3D):
                return inputs[0].mag2 ** 0.25
            elif isinstance(inputs[0], Vector4D):
                return inputs[0].tau2 ** 0.25

        elif ufunc is numpy.cbrt and len(inputs) == 1 and isinstance(inputs[0], Vector):
            if len(outputs) != 0:
                raise TypeError(
                    "output of 'numpy.cbrt' is scalar, cannot fill a VectorObject with 'out'"
                )
            if isinstance(inputs[0], Vector2D):
                return inputs[0].rho2 ** 0.16666666666666666
            elif isinstance(inputs[0], Vector3D):
                return inputs[0].mag2 ** 0.16666666666666666
            elif isinstance(inputs[0], Vector4D):
                return inputs[0].tau2 ** 0.16666666666666666

        elif (
            ufunc is numpy.matmul
            and len(inputs) == 2
            and isinstance(inputs[0], Vector)
            and isinstance(inputs[1], Vector)
        ):
            if len(outputs) != 0:
                raise TypeError(
                    "output of 'numpy.matmul' is scalar, cannot fill a VectorObject with 'out'"
                )
            return inputs[0].dot(inputs[1])

        elif (
            ufunc is numpy.equal
            and len(inputs) == 2
            and isinstance(inputs[0], Vector)
            and isinstance(inputs[1], Vector)
        ):
            if len(outputs) != 0:
                raise TypeError(
                    "output of 'numpy.equal' is scalar, cannot fill a VectorObject with 'out'"
                )
            return inputs[0].equal(inputs[1])

        elif (
            ufunc is numpy.not_equal
            and len(inputs) == 2
            and isinstance(inputs[0], Vector)
            and isinstance(inputs[1], Vector)
        ):
            if len(outputs) != 0:
                raise TypeError(
                    "output of 'numpy.equal' is scalar, cannot fill a VectorObject with 'out'"
                )
            return inputs[0].not_equal(inputs[1])

        else:
            return NotImplemented


class VectorObject2D(VectorObject, Planar, Vector2D):
    __slots__ = ("azimuthal",)

    azimuthal: AzimuthalObject

    @classmethod
    def from_xy(cls, x: float, y: float) -> "VectorObject2D":
        return cls(AzimuthalObjectXY(x, y))  # type: ignore

    @classmethod
    def from_rhophi(cls, rho: float, phi: float) -> "VectorObject2D":
        return cls(AzimuthalObjectRhoPhi(rho, phi))  # type: ignore

    def __init__(self, azimuthal: AzimuthalObject) -> None:
        self.azimuthal = azimuthal

    def __repr__(self) -> str:
        aznames = _coordinate_class_to_names[_aztype(self)]
        out = []
        for x in aznames:
            out.append(f"{x}={getattr(self.azimuthal, x)}")
        return "vector.obj(" + ", ".join(out) + ")"

    def __array__(self) -> numpy.ndarray:
        from vector.backends.numpy_ import VectorNumpy2D

        return VectorNumpy2D(
            self.azimuthal.elements,
            dtype=[
                (x, numpy.float64) for x in _coordinate_class_to_names[_aztype(self)]
            ],
        )

    @typing.no_type_check
    def _wrap_result(
        self,
        cls: typing.Any,
        result: typing.Any,
        returns: typing.Any,
        num_vecargs: typing.Any,
    ) -> typing.Any:
        if returns == [float] or returns == [bool]:
            return result

        elif (
            len(returns) == 1
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
        ):
            azcoords = _coord_object_type[returns[0]](result[0], result[1])
            return cls.ProjectionClass2D(azcoords)

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and returns[1] is None
        ):
            azcoords = _coord_object_type[returns[0]](result[0], result[1])
            return cls.ProjectionClass2D(azcoords)

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
        ):
            azcoords = _coord_object_type[returns[0]](result[0], result[1])
            lcoords = _coord_object_type[returns[1]](result[2])
            return cls.ProjectionClass3D(azcoords, lcoords)

        elif (
            len(returns) == 3
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
            and returns[2] is None
        ):
            azcoords = _coord_object_type[returns[0]](result[0], result[1])
            lcoords = _coord_object_type[returns[1]](result[2])
            return cls.ProjectionClass3D(azcoords, lcoords)

        elif (
            len(returns) == 3
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
            and isinstance(returns[2], type)
            and issubclass(returns[2], Temporal)
        ):
            azcoords = _coord_object_type[returns[0]](result[0], result[1])
            lcoords = _coord_object_type[returns[1]](result[2])
            tcoords = _coord_object_type[returns[2]](result[3])
            return cls.ProjectionClass4D(azcoords, lcoords, tcoords)

        else:
            raise AssertionError(repr(returns))

    @property
    def x(self) -> float:
        return super().x

    @x.setter
    def x(self, x: float) -> None:
        self.azimuthal = AzimuthalObjectXY(x, self.y)  # type: ignore

    @property
    def y(self) -> float:
        return super().y

    @y.setter
    def y(self, y: float) -> None:
        self.azimuthal = AzimuthalObjectXY(self.x, y)  # type: ignore

    @property
    def rho(self) -> float:
        return super().rho

    @rho.setter
    def rho(self, rho: float) -> None:
        self.azimuthal = AzimuthalObjectRhoPhi(rho, self.phi)  # type: ignore

    @property
    def phi(self) -> float:
        return super().phi

    @phi.setter
    def phi(self, phi: float) -> None:
        self.azimuthal = AzimuthalObjectRhoPhi(self.rho, phi)  # type: ignore


class MomentumObject2D(PlanarMomentum, VectorObject2D):
    def __repr__(self) -> str:
        aznames = _coordinate_class_to_names[_aztype(self)]
        out = []
        for x in aznames:
            y = _repr_generic_to_momentum.get(x, x)
            out.append(f"{y}={getattr(self.azimuthal, x)}")
        return "vector.obj(" + ", ".join(out) + ")"

    def __array__(self) -> numpy.ndarray:
        from vector.backends.numpy_ import MomentumNumpy2D

        return MomentumNumpy2D(
            self.azimuthal.elements,
            dtype=[
                (x, numpy.float64) for x in _coordinate_class_to_names[_aztype(self)]
            ],
        )

    @property
    def px(self) -> float:
        return super().px

    @px.setter
    def px(self, px: float) -> None:
        self.azimuthal = AzimuthalObjectXY(px, self.py)  # type: ignore

    @property
    def py(self) -> float:
        return super().py

    @py.setter
    def py(self, py: float) -> None:
        self.azimuthal = AzimuthalObjectXY(self.px, py)  # type: ignore

    @property
    def pt(self) -> float:
        return super().pt

    @pt.setter
    def pt(self, pt: float) -> None:
        self.azimuthal = AzimuthalObjectRhoPhi(pt, self.phi)  # type: ignore


class VectorObject3D(VectorObject, Spatial, Vector3D):
    __slots__ = ("azimuthal", "longitudinal")

    azimuthal: AzimuthalObject
    longitudinal: LongitudinalObject

    @classmethod
    def from_xyz(cls, x: float, y: float, z: float) -> "VectorObject3D":
        return cls(AzimuthalObjectXY(x, y), LongitudinalObjectZ(z))  # type: ignore

    @classmethod
    def from_xytheta(cls, x: float, y: float, theta: float) -> "VectorObject3D":
        return cls(AzimuthalObjectXY(x, y), LongitudinalObjectTheta(theta))  # type: ignore

    @classmethod
    def from_xyeta(cls, x: float, y: float, eta: float) -> "VectorObject3D":
        return cls(AzimuthalObjectXY(x, y), LongitudinalObjectEta(eta))  # type: ignore

    @classmethod
    def from_rhophiz(cls, rho: float, phi: float, z: float) -> "VectorObject3D":
        return cls(AzimuthalObjectRhoPhi(rho, phi), LongitudinalObjectZ(z))  # type: ignore

    @classmethod
    def from_rhophitheta(cls, rho: float, phi: float, theta: float) -> "VectorObject3D":
        return cls(AzimuthalObjectRhoPhi(rho, phi), LongitudinalObjectTheta(theta))  # type: ignore

    @classmethod
    def from_rhophieta(cls, rho: float, phi: float, eta: float) -> "VectorObject3D":
        return cls(AzimuthalObjectRhoPhi(rho, phi), LongitudinalObjectEta(eta))  # type: ignore

    def __init__(
        self, azimuthal: AzimuthalObject, longitudinal: LongitudinalObject
    ) -> None:
        self.azimuthal = azimuthal
        self.longitudinal = longitudinal

    def __repr__(self) -> str:
        aznames = _coordinate_class_to_names[_aztype(self)]
        lnames = _coordinate_class_to_names[_ltype(self)]
        out = []
        for x in aznames:
            out.append(f"{x}={getattr(self.azimuthal, x)}")
        for x in lnames:
            out.append(f"{x}={getattr(self.longitudinal, x)}")
        return "vector.obj(" + ", ".join(out) + ")"

    def __array__(self) -> numpy.ndarray:
        from vector.backends.numpy_ import VectorNumpy3D

        return VectorNumpy3D(
            self.azimuthal.elements + self.longitudinal.elements,
            dtype=[
                (x, numpy.float64)
                for x in _coordinate_class_to_names[_aztype(self)]
                + _coordinate_class_to_names[_ltype(self)]
            ],
        )

    @typing.no_type_check
    def _wrap_result(
        self,
        cls: typing.Any,
        result: typing.Any,
        returns: typing.Any,
        num_vecargs: typing.Any,
    ) -> typing.Any:
        if returns == [float] or returns == [bool]:
            return result

        elif (
            len(returns) == 1
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
        ):
            azcoords = _coord_object_type[returns[0]](result[0], result[1])
            return cls.ProjectionClass3D(azcoords, self.longitudinal)

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and returns[1] is None
        ):
            azcoords = _coord_object_type[returns[0]](result[0], result[1])
            return cls.ProjectionClass2D(azcoords)

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
        ):
            azcoords = _coord_object_type[returns[0]](result[0], result[1])
            lcoords = _coord_object_type[returns[1]](result[2])
            return cls.ProjectionClass3D(azcoords, lcoords)

        elif (
            len(returns) == 3
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
            and returns[2] is None
        ):
            azcoords = _coord_object_type[returns[0]](result[0], result[1])
            lcoords = _coord_object_type[returns[1]](result[2])
            return cls.ProjectionClass3D(azcoords, lcoords)

        elif (
            len(returns) == 3
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
            and isinstance(returns[2], type)
            and issubclass(returns[2], Temporal)
        ):
            azcoords = _coord_object_type[returns[0]](result[0], result[1])
            lcoords = _coord_object_type[returns[1]](result[2])
            tcoords = _coord_object_type[returns[2]](result[3])
            return cls.ProjectionClass4D(azcoords, lcoords, tcoords)

        else:
            raise AssertionError(repr(returns))

    @property
    def x(self) -> float:
        return super().x

    @x.setter
    def x(self, x: float) -> None:
        self.azimuthal = AzimuthalObjectXY(x, self.y)  # type: ignore

    @property
    def y(self) -> float:
        return super().y

    @y.setter
    def y(self, y: float) -> None:
        self.azimuthal = AzimuthalObjectXY(self.x, y)  # type: ignore

    @property
    def rho(self) -> float:
        return super().rho

    @rho.setter
    def rho(self, rho: float) -> None:
        self.azimuthal = AzimuthalObjectRhoPhi(rho, self.phi)  # type: ignore

    @property
    def phi(self) -> float:
        return super().phi

    @phi.setter
    def phi(self, phi: float) -> None:
        self.azimuthal = AzimuthalObjectRhoPhi(self.rho, phi)  # type: ignore

    @property
    def z(self) -> float:
        return super().z

    @z.setter
    def z(self, z: float) -> None:
        self.longitudinal = LongitudinalObjectZ(z)  # type: ignore

    @property
    def theta(self) -> float:
        return super().theta

    @theta.setter
    def theta(self, theta: float) -> None:
        self.longitudinal = LongitudinalObjectTheta(theta)  # type: ignore

    @property
    def eta(self) -> float:
        return super().eta

    @eta.setter
    def eta(self, eta: float) -> None:
        self.longitudinal = LongitudinalObjectEta(eta)  # type: ignore


class MomentumObject3D(SpatialMomentum, VectorObject3D):
    def __repr__(self) -> str:
        aznames = _coordinate_class_to_names[_aztype(self)]
        lnames = _coordinate_class_to_names[_ltype(self)]
        out = []
        for x in aznames:
            y = _repr_generic_to_momentum.get(x, x)
            out.append(f"{y}={getattr(self.azimuthal, x)}")
        for x in lnames:
            y = _repr_generic_to_momentum.get(x, x)
            out.append(f"{y}={getattr(self.longitudinal, x)}")
        return "vector.obj(" + ", ".join(out) + ")"

    def __array__(self) -> numpy.ndarray:
        from vector.backends.numpy_ import MomentumNumpy3D

        return MomentumNumpy3D(
            self.azimuthal.elements + self.longitudinal.elements,
            dtype=[
                (x, numpy.float64)
                for x in _coordinate_class_to_names[_aztype(self)]
                + _coordinate_class_to_names[_ltype(self)]
            ],
        )

    @property
    def px(self) -> float:
        return super().px

    @px.setter
    def px(self, px: float) -> None:
        self.azimuthal = AzimuthalObjectXY(px, self.py)  # type: ignore

    @property
    def py(self) -> float:
        return super().py

    @py.setter
    def py(self, py: float) -> None:
        self.azimuthal = AzimuthalObjectXY(self.px, py)  # type: ignore

    @property
    def pt(self) -> float:
        return super().pt

    @pt.setter
    def pt(self, pt: float) -> None:
        self.azimuthal = AzimuthalObjectRhoPhi(pt, self.phi)  # type: ignore

    @property
    def pz(self) -> float:
        return super().pz

    @pz.setter
    def pz(self, pz: float) -> None:
        self.longitudinal = LongitudinalObjectZ(pz)  # type: ignore


class VectorObject4D(VectorObject, Lorentz, Vector4D):
    __slots__ = ("azimuthal", "longitudinal", "temporal")

    azimuthal: AzimuthalObject
    longitudinal: LongitudinalObject
    temporal: TemporalObject

    @classmethod
    def from_xyzt(
        cls,
        x: float,
        y: float,
        z: float,
        t: float,
    ) -> "VectorObject4D":
        return cls(AzimuthalObjectXY(x, y), LongitudinalObjectZ(z), TemporalObjectT(t))  # type: ignore

    @classmethod
    def from_xyztau(
        cls,
        x: float,
        y: float,
        z: float,
        tau: float,
    ) -> "VectorObject4D":
        return cls(
            AzimuthalObjectXY(x, y), LongitudinalObjectZ(z), TemporalObjectTau(tau)  # type: ignore
        )

    @classmethod
    def from_xythetat(
        cls,
        x: float,
        y: float,
        theta: float,
        t: float,
    ) -> "VectorObject4D":
        return cls(
            AzimuthalObjectXY(x, y), LongitudinalObjectTheta(theta), TemporalObjectT(t)  # type: ignore
        )

    @classmethod
    def from_xythetatau(
        cls,
        x: float,
        y: float,
        theta: float,
        tau: float,
    ) -> "VectorObject4D":
        return cls(
            AzimuthalObjectXY(x, y),  # type: ignore
            LongitudinalObjectTheta(theta),  # type: ignore
            TemporalObjectTau(tau),  # type: ignore
        )

    @classmethod
    def from_xyetat(
        cls,
        x: float,
        y: float,
        eta: float,
        t: float,
    ) -> "VectorObject4D":
        return cls(
            AzimuthalObjectXY(x, y), LongitudinalObjectEta(eta), TemporalObjectT(t)  # type: ignore
        )

    @classmethod
    def from_xyetatau(
        cls,
        x: float,
        y: float,
        eta: float,
        tau: float,
    ) -> "VectorObject4D":
        return cls(
            AzimuthalObjectXY(x, y), LongitudinalObjectEta(eta), TemporalObjectTau(tau)  # type: ignore
        )

    @classmethod
    def from_rhophizt(
        cls,
        rho: float,
        phi: float,
        z: float,
        t: float,
    ) -> "VectorObject4D":
        return cls(
            AzimuthalObjectRhoPhi(rho, phi), LongitudinalObjectZ(z), TemporalObjectT(t)  # type: ignore
        )

    @classmethod
    def from_rhophiztau(
        cls,
        rho: float,
        phi: float,
        z: float,
        tau: float,
    ) -> "VectorObject4D":
        return cls(
            AzimuthalObjectRhoPhi(rho, phi),  # type: ignore
            LongitudinalObjectZ(z),  # type: ignore
            TemporalObjectTau(tau),  # type: ignore
        )

    @classmethod
    def from_rhophithetat(
        cls,
        rho: float,
        phi: float,
        theta: float,
        t: float,
    ) -> "VectorObject4D":
        return cls(
            AzimuthalObjectRhoPhi(rho, phi),  # type: ignore
            LongitudinalObjectTheta(theta),  # type: ignore
            TemporalObjectT(t),  # type: ignore
        )

    @classmethod
    def from_rhophithetatau(
        cls,
        rho: float,
        phi: float,
        theta: float,
        tau: float,
    ) -> "VectorObject4D":
        return cls(
            AzimuthalObjectRhoPhi(rho, phi),  # type: ignore
            LongitudinalObjectTheta(theta),  # type: ignore
            TemporalObjectTau(tau),  # type: ignore
        )

    @classmethod
    def from_rhophietat(
        cls,
        rho: float,
        phi: float,
        eta: float,
        t: float,
    ) -> "VectorObject4D":
        return cls(
            AzimuthalObjectRhoPhi(rho, phi),  # type: ignore
            LongitudinalObjectEta(eta),  # type: ignore
            TemporalObjectT(t),  # type: ignore
        )

    @classmethod
    def from_rhophietatau(
        cls,
        rho: float,
        phi: float,
        eta: float,
        tau: float,
    ) -> "VectorObject4D":
        return cls(
            AzimuthalObjectRhoPhi(rho, phi),  # type: ignore
            LongitudinalObjectEta(eta),  # type: ignore
            TemporalObjectTau(tau),  # type: ignore
        )

    def __init__(
        self,
        azimuthal: AzimuthalObject,
        longitudinal: LongitudinalObject,
        temporal: TemporalObject,
    ) -> None:
        self.azimuthal = azimuthal
        self.longitudinal = longitudinal
        self.temporal = temporal

    def __repr__(self) -> str:
        aznames = _coordinate_class_to_names[_aztype(self)]
        lnames = _coordinate_class_to_names[_ltype(self)]
        tnames = _coordinate_class_to_names[_ttype(self)]
        out = []
        for x in aznames:
            out.append(f"{x}={getattr(self.azimuthal, x)}")
        for x in lnames:
            out.append(f"{x}={getattr(self.longitudinal, x)}")
        for x in tnames:
            out.append(f"{x}={getattr(self.temporal, x)}")
        return "vector.obj(" + ", ".join(out) + ")"

    def __array__(self) -> numpy.ndarray:
        from vector.backends.numpy_ import VectorNumpy4D

        return VectorNumpy4D(
            self.azimuthal.elements
            + self.longitudinal.elements
            + self.temporal.elements,
            dtype=[
                (x, numpy.float64)
                for x in _coordinate_class_to_names[_aztype(self)]
                + _coordinate_class_to_names[_ltype(self)]
                + _coordinate_class_to_names[_ttype(self)]
            ],
        )

    @typing.no_type_check
    def _wrap_result(
        self,
        cls: typing.Any,
        result: typing.Any,
        returns: typing.Any,
        num_vecargs: typing.Any,
    ) -> typing.Any:
        if returns == [float] or returns == [bool]:
            return result

        elif (
            len(returns) == 1
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
        ):
            azcoords = _coord_object_type[returns[0]](result[0], result[1])
            return cls.ProjectionClass4D(azcoords, self.longitudinal, self.temporal)

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and returns[1] is None
        ):
            azcoords = _coord_object_type[returns[0]](result[0], result[1])
            return cls.ProjectionClass2D(azcoords)

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
        ):
            azcoords = _coord_object_type[returns[0]](result[0], result[1])
            lcoords = _coord_object_type[returns[1]](result[2])
            return cls.ProjectionClass4D(azcoords, lcoords, self.temporal)

        elif (
            len(returns) == 3
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
            and returns[2] is None
        ):
            azcoords = _coord_object_type[returns[0]](result[0], result[1])
            lcoords = _coord_object_type[returns[1]](result[2])
            return cls.ProjectionClass3D(azcoords, lcoords)

        elif (
            len(returns) == 3
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
            and isinstance(returns[2], type)
            and issubclass(returns[2], Temporal)
        ):
            azcoords = _coord_object_type[returns[0]](result[0], result[1])
            lcoords = _coord_object_type[returns[1]](result[2])
            tcoords = _coord_object_type[returns[2]](result[3])
            return cls.ProjectionClass4D(azcoords, lcoords, tcoords)

        else:
            raise AssertionError(repr(returns))

    @property
    def x(self) -> float:
        return super().x

    @x.setter
    def x(self, x: float) -> None:
        self.azimuthal = AzimuthalObjectXY(x, self.y)  # type: ignore

    @property
    def y(self) -> float:
        return super().y

    @y.setter
    def y(self, y: float) -> None:
        self.azimuthal = AzimuthalObjectXY(self.x, y)  # type: ignore

    @property
    def rho(self) -> float:
        return super().rho

    @rho.setter
    def rho(self, rho: float) -> None:
        self.azimuthal = AzimuthalObjectRhoPhi(rho, self.phi)  # type: ignore

    @property
    def phi(self) -> float:
        return super().phi

    @phi.setter
    def phi(self, phi: float) -> None:
        self.azimuthal = AzimuthalObjectRhoPhi(self.rho, phi)  # type: ignore

    @property
    def z(self) -> float:
        return super().z

    @z.setter
    def z(self, z: float) -> None:
        self.longitudinal = LongitudinalObjectZ(z)  # type: ignore

    @property
    def theta(self) -> float:
        return super().theta

    @theta.setter
    def theta(self, theta: float) -> None:
        self.longitudinal = LongitudinalObjectTheta(theta)  # type: ignore

    @property
    def eta(self) -> float:
        return super().eta

    @eta.setter
    def eta(self, eta: float) -> None:
        self.longitudinal = LongitudinalObjectEta(eta)  # type: ignore

    @property
    def t(self) -> float:
        return super().t

    @t.setter
    def t(self, t: float) -> None:
        self.temporal = TemporalObjectT(t)  # type: ignore

    @property
    def tau(self) -> float:
        return super().tau

    @tau.setter
    def tau(self, tau: float) -> None:
        self.temporal = TemporalObjectTau(tau)  # type: ignore


class MomentumObject4D(LorentzMomentum, VectorObject4D):
    def __repr__(self) -> str:
        aznames = _coordinate_class_to_names[_aztype(self)]
        lnames = _coordinate_class_to_names[_ltype(self)]
        tnames = _coordinate_class_to_names[_ttype(self)]
        out = []
        for x in aznames:
            y = _repr_generic_to_momentum.get(x, x)
            out.append(f"{y}={getattr(self.azimuthal, x)}")
        for x in lnames:
            y = _repr_generic_to_momentum.get(x, x)
            out.append(f"{y}={getattr(self.longitudinal, x)}")
        for x in tnames:
            y = _repr_generic_to_momentum.get(x, x)
            out.append(f"{y}={getattr(self.temporal, x)}")
        return "vector.obj(" + ", ".join(out) + ")"

    def __array__(self) -> numpy.ndarray:
        from vector.backends.numpy_ import MomentumNumpy4D

        return MomentumNumpy4D(
            self.azimuthal.elements
            + self.longitudinal.elements
            + self.temporal.elements,
            dtype=[
                (x, numpy.float64)
                for x in _coordinate_class_to_names[_aztype(self)]
                + _coordinate_class_to_names[_ltype(self)]
                + _coordinate_class_to_names[_ttype(self)]
            ],
        )

    @property
    def px(self) -> float:
        return super().px

    @px.setter
    def px(self, px: float) -> None:
        self.azimuthal = AzimuthalObjectXY(px, self.py)  # type: ignore

    @property
    def py(self) -> float:
        return super().py

    @py.setter
    def py(self, py: float) -> None:
        self.azimuthal = AzimuthalObjectXY(self.px, py)  # type: ignore

    @property
    def pt(self) -> float:
        return super().pt

    @pt.setter
    def pt(self, pt: float) -> None:
        self.azimuthal = AzimuthalObjectRhoPhi(pt, self.phi)  # type: ignore

    @property
    def pz(self) -> float:
        return super().pz

    @pz.setter
    def pz(self, pz: float) -> None:
        self.longitudinal = LongitudinalObjectZ(pz)  # type: ignore

    @property
    def E(self) -> float:
        return super().E

    @E.setter
    def E(self, E: float) -> None:
        self.temporal = TemporalObjectT(E)  # type: ignore

    @property
    def energy(self) -> float:
        return super().energy

    @energy.setter
    def energy(self, energy: float) -> None:
        self.temporal = TemporalObjectT(energy)  # type: ignore

    @property
    def M(self) -> float:
        return super().M

    @M.setter
    def M(self, M: float) -> None:
        self.temporal = TemporalObjectTau(M)  # type: ignore

    @property
    def mass(self) -> float:
        return super().mass

    @mass.setter
    def mass(self, mass: float) -> None:
        self.temporal = TemporalObjectTau(mass)  # type: ignore


def _gather_coordinates(
    planar_class: typing.Type[VectorObject2D],
    spatial_class: typing.Type[VectorObject3D],
    lorentz_class: typing.Type[VectorObject4D],
    coordinates: dict,
) -> typing.Any:
    azimuthal: typing.Optional[
        typing.Union[AzimuthalObjectXY, AzimuthalObjectRhoPhi]
    ] = None

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

    longitudinal: typing.Optional[
        typing.Union[
            LongitudinalObjectZ, LongitudinalObjectTheta, LongitudinalObjectEta
        ]
    ] = None

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

    temporal: typing.Optional[typing.Union[TemporalObjectT, TemporalObjectTau]] = None

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
            return planar_class(azimuthal)  # type: ignore
        if azimuthal is not None and longitudinal is not None and temporal is None:
            return spatial_class(azimuthal, longitudinal)  # type: ignore
        if azimuthal is not None and longitudinal is not None and temporal is not None:
            return lorentz_class(azimuthal, longitudinal, temporal)  # type: ignore

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


@typing.overload
def obj(*, x: float, y: float) -> VectorObject2D:
    ...


@typing.overload
def obj(*, x: float, py: float) -> MomentumObject2D:
    ...


@typing.overload
def obj(*, px: float, y: float) -> MomentumObject2D:
    ...


@typing.overload
def obj(*, px: float, py: float) -> MomentumObject2D:
    ...


@typing.overload
def obj(*, rho: float, phi: float) -> VectorObject2D:
    ...


@typing.overload
def obj(*, pt: float, phi: float) -> MomentumObject2D:
    ...


@typing.overload
def obj(*, x: float, y: float, z: float) -> VectorObject3D:
    ...


@typing.overload
def obj(*, x: float, y: float, pz: float) -> MomentumObject3D:
    ...


@typing.overload
def obj(*, x: float, py: float, z: float) -> MomentumObject3D:
    ...


@typing.overload
def obj(*, x: float, py: float, pz: float) -> MomentumObject3D:
    ...


@typing.overload
def obj(*, px: float, y: float, z: float) -> MomentumObject3D:
    ...


@typing.overload
def obj(*, px: float, y: float, pz: float) -> MomentumObject3D:
    ...


@typing.overload
def obj(*, px: float, py: float, z: float) -> MomentumObject3D:
    ...


@typing.overload
def obj(*, px: float, py: float, pz: float) -> MomentumObject3D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, z: float) -> VectorObject3D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, pz: float) -> MomentumObject3D:
    ...


@typing.overload
def obj(*, pt: float, phi: float, z: float) -> MomentumObject3D:
    ...


@typing.overload
def obj(*, pt: float, phi: float, pz: float) -> MomentumObject3D:
    ...


@typing.overload
def obj(*, x: float, y: float, theta: float) -> VectorObject3D:
    ...


@typing.overload
def obj(*, x: float, py: float, theta: float) -> MomentumObject3D:
    ...


@typing.overload
def obj(*, px: float, y: float, theta: float) -> MomentumObject3D:
    ...


@typing.overload
def obj(*, px: float, py: float, theta: float) -> MomentumObject3D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, theta: float) -> VectorObject3D:
    ...


@typing.overload
def obj(*, pt: float, phi: float, theta: float) -> MomentumObject3D:
    ...


@typing.overload
def obj(*, x: float, y: float, eta: float) -> VectorObject3D:
    ...


@typing.overload
def obj(*, x: float, py: float, eta: float) -> MomentumObject3D:
    ...


@typing.overload
def obj(*, px: float, y: float, eta: float) -> MomentumObject3D:
    ...


@typing.overload
def obj(*, px: float, py: float, eta: float) -> MomentumObject3D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, eta: float) -> VectorObject3D:
    ...


@typing.overload
def obj(*, pt: float, phi: float, eta: float) -> MomentumObject3D:
    ...


@typing.overload
def obj(*, x: float, y: float, z: float, t: float) -> VectorObject4D:
    ...


@typing.overload
def obj(*, x: float, y: float, pz: float, t: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, py: float, z: float, t: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, py: float, pz: float, t: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, y: float, z: float, t: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, y: float, pz: float, t: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, py: float, z: float, t: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, py: float, pz: float, t: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, z: float, t: float) -> VectorObject4D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, pz: float, t: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, pt: float, phi: float, z: float, t: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, pt: float, phi: float, pz: float, t: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, y: float, theta: float, t: float) -> VectorObject4D:
    ...


@typing.overload
def obj(*, x: float, py: float, theta: float, t: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, y: float, theta: float, t: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, py: float, theta: float, t: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, theta: float, t: float) -> VectorObject4D:
    ...


@typing.overload
def obj(*, pt: float, phi: float, theta: float, t: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, y: float, eta: float, t: float) -> VectorObject4D:
    ...


@typing.overload
def obj(*, x: float, py: float, eta: float, t: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, y: float, eta: float, t: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, py: float, eta: float, t: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, eta: float, t: float) -> VectorObject4D:
    ...


@typing.overload
def obj(*, pt: float, phi: float, eta: float, t: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, y: float, z: float, tau: float) -> VectorObject4D:
    ...


@typing.overload
def obj(*, x: float, y: float, pz: float, tau: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, py: float, z: float, tau: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, py: float, pz: float, tau: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, y: float, z: float, tau: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, y: float, pz: float, tau: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, py: float, z: float, tau: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, py: float, pz: float, tau: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, z: float, tau: float) -> VectorObject4D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, pz: float, tau: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, ptau: float, phi: float, z: float, tau: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, ptau: float, phi: float, pz: float, tau: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, y: float, theta: float, tau: float) -> VectorObject4D:
    ...


@typing.overload
def obj(*, x: float, py: float, theta: float, tau: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, y: float, theta: float, tau: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, py: float, theta: float, tau: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, theta: float, tau: float) -> VectorObject4D:
    ...


@typing.overload
def obj(*, ptau: float, phi: float, theta: float, tau: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, y: float, eta: float, tau: float) -> VectorObject4D:
    ...


@typing.overload
def obj(*, x: float, py: float, eta: float, tau: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, y: float, eta: float, tau: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, py: float, eta: float, tau: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, eta: float, tau: float) -> VectorObject4D:
    ...


@typing.overload
def obj(*, ptau: float, phi: float, eta: float, tau: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, y: float, z: float, E: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, y: float, pz: float, E: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, py: float, z: float, E: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, py: float, pz: float, E: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, y: float, z: float, E: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, y: float, pz: float, E: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, py: float, z: float, E: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, py: float, pz: float, E: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, z: float, E: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, pz: float, E: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, pE: float, phi: float, z: float, E: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, pE: float, phi: float, pz: float, E: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, y: float, theta: float, E: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, py: float, theta: float, E: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, y: float, theta: float, E: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, py: float, theta: float, E: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, theta: float, E: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, pE: float, phi: float, theta: float, E: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, y: float, eta: float, E: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, py: float, eta: float, E: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, y: float, eta: float, E: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, py: float, eta: float, E: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, eta: float, E: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, pE: float, phi: float, eta: float, E: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, y: float, z: float, energy: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, y: float, pz: float, energy: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, py: float, z: float, energy: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, py: float, pz: float, energy: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, y: float, z: float, energy: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, y: float, pz: float, energy: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, py: float, z: float, energy: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, py: float, pz: float, energy: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, z: float, energy: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, pz: float, energy: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, penergy: float, phi: float, z: float, energy: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, penergy: float, phi: float, pz: float, energy: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, y: float, theta: float, energy: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, py: float, theta: float, energy: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, y: float, theta: float, energy: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, py: float, theta: float, energy: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, theta: float, energy: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, penergy: float, phi: float, theta: float, energy: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, y: float, eta: float, energy: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, py: float, eta: float, energy: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, y: float, eta: float, energy: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, py: float, eta: float, energy: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, eta: float, energy: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, penergy: float, phi: float, eta: float, energy: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, y: float, z: float, M: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, y: float, pz: float, M: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, py: float, z: float, M: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, py: float, pz: float, M: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, y: float, z: float, M: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, y: float, pz: float, M: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, py: float, z: float, M: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, py: float, pz: float, M: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, z: float, M: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, pz: float, M: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, pM: float, phi: float, z: float, M: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, pM: float, phi: float, pz: float, M: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, y: float, theta: float, M: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, py: float, theta: float, M: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, y: float, theta: float, M: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, py: float, theta: float, M: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, theta: float, M: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, pM: float, phi: float, theta: float, M: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, y: float, eta: float, M: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, py: float, eta: float, M: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, y: float, eta: float, M: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, py: float, eta: float, M: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, eta: float, M: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, pM: float, phi: float, eta: float, M: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, y: float, z: float, mass: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, y: float, pz: float, mass: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, py: float, z: float, mass: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, py: float, pz: float, mass: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, y: float, z: float, mass: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, y: float, pz: float, mass: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, py: float, z: float, mass: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, py: float, pz: float, mass: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, z: float, mass: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, pz: float, mass: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, pmass: float, phi: float, z: float, mass: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, pmass: float, phi: float, pz: float, mass: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, y: float, theta: float, mass: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, py: float, theta: float, mass: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, y: float, theta: float, mass: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, py: float, theta: float, mass: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, theta: float, mass: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, pmass: float, phi: float, theta: float, mass: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, y: float, eta: float, mass: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, x: float, py: float, eta: float, mass: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, y: float, eta: float, mass: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, px: float, py: float, eta: float, mass: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, rho: float, phi: float, eta: float, mass: float) -> MomentumObject4D:
    ...


@typing.overload
def obj(*, pmass: float, phi: float, eta: float, mass: float) -> MomentumObject4D:
    ...


def obj(**coordinates: float) -> VectorObject:
    "vector.obj docs"
    is_momentum = False
    generic_coordinates = {}
    if "px" in coordinates:
        is_momentum = True
        generic_coordinates["x"] = coordinates.pop("px")
    if "py" in coordinates:
        is_momentum = True
        generic_coordinates["y"] = coordinates.pop("py")
    if "pt" in coordinates:
        is_momentum = True
        generic_coordinates["rho"] = coordinates.pop("pt")
    if "pz" in coordinates:
        is_momentum = True
        generic_coordinates["z"] = coordinates.pop("pz")
    if "E" in coordinates:
        is_momentum = True
        generic_coordinates["t"] = coordinates.pop("E")
    if "energy" in coordinates and "t" not in generic_coordinates:
        is_momentum = True
        generic_coordinates["t"] = coordinates.pop("energy")
    if "M" in coordinates:
        is_momentum = True
        generic_coordinates["tau"] = coordinates.pop("M")
    if "mass" in coordinates and "tau" not in generic_coordinates:
        is_momentum = True
        generic_coordinates["tau"] = coordinates.pop("mass")
    for x in list(coordinates):
        if x not in generic_coordinates:
            generic_coordinates[x] = coordinates.pop(x)
    if len(coordinates) != 0:
        raise TypeError(
            "duplicate coordinates (through momentum-aliases): "
            + ", ".join(repr(x) for x in coordinates)
        )
    if is_momentum:
        return _gather_coordinates(
            MomentumObject2D, MomentumObject3D, MomentumObject4D, generic_coordinates
        )
    else:
        return _gather_coordinates(
            VectorObject2D, VectorObject3D, VectorObject4D, generic_coordinates
        )


VectorObject2D.ProjectionClass2D = VectorObject2D
VectorObject2D.ProjectionClass3D = VectorObject3D
VectorObject2D.ProjectionClass4D = VectorObject4D
VectorObject2D.GenericClass = VectorObject2D

MomentumObject2D.ProjectionClass2D = MomentumObject2D
MomentumObject2D.ProjectionClass3D = MomentumObject3D
MomentumObject2D.ProjectionClass4D = MomentumObject4D
MomentumObject2D.GenericClass = VectorObject2D

VectorObject3D.ProjectionClass2D = VectorObject2D
VectorObject3D.ProjectionClass3D = VectorObject3D
VectorObject3D.ProjectionClass4D = VectorObject4D
VectorObject3D.GenericClass = VectorObject3D

MomentumObject3D.ProjectionClass2D = MomentumObject2D
MomentumObject3D.ProjectionClass3D = MomentumObject3D
MomentumObject3D.ProjectionClass4D = MomentumObject4D
MomentumObject3D.GenericClass = VectorObject3D

VectorObject4D.ProjectionClass2D = VectorObject2D
VectorObject4D.ProjectionClass3D = VectorObject3D
VectorObject4D.ProjectionClass4D = VectorObject4D
VectorObject4D.GenericClass = VectorObject4D

MomentumObject4D.ProjectionClass2D = MomentumObject2D
MomentumObject4D.ProjectionClass3D = MomentumObject3D
MomentumObject4D.ProjectionClass4D = MomentumObject4D
MomentumObject4D.GenericClass = VectorObject4D
