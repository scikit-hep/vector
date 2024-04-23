# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.
"""
Defines behaviors for Object vectors. New vectors created with the respective classes

.. code-block:: python

    vector.VectorObject2D(...)
    vector.VectorObject3D(...)

will have these behaviors built in (and will pass them to any derived objects).

The class methods can also be used to construct object type vectors -

.. code-block:: python

    vec = vector.VectorObject2D.from_xy(1, 2)

Additionally, object type vectors can also be constructed using -

.. code-block:: python

    vector.obj(...)

function.
"""

from __future__ import annotations

import numbers
import typing

import numpy

from vector._methods import (
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
    _repr_momentum_to_generic,
    _ttype,
)
from vector._typeutils import FloatArray


class CoordinatesObject:
    """Coordinates class for the Object backend."""


class AzimuthalObject(CoordinatesObject, Azimuthal):
    """Azimuthal class for the Object backend."""


class LongitudinalObject(CoordinatesObject, Longitudinal):
    """Longitudinal class for the Object backend."""


class TemporalObject(CoordinatesObject, Temporal):
    """Temporal class for the Object backend."""


class TupleXY(typing.NamedTuple):
    """``x`` and ``y`` coordinates as a ``NamedTuple``."""

    x: float
    y: float


class AzimuthalObjectXY(AzimuthalObject, AzimuthalXY, TupleXY):
    """
    Class for the ``x`` and ``y`` (azimuthal) coordinates of Object backend.
    Use the ``elements`` property to retrieve the coordinates.
    """

    @property
    def elements(self) -> tuple[float, float]:
        """
        Azimuthal coordinates (``x`` and ``y``) as a tuple.
        Each coordinate is a scalar and not a vector.

        Examples:
            >>> import vector
            >>> v = vector.VectorObject2D(x=1, y=2)
            >>> az = v.azimuthal
            >>> az.elements
            (1, 2)
        """
        return (self.x, self.y)


class TupleRhoPhi(typing.NamedTuple):
    """``rho`` and ``phi`` coordinates as a ``NamedTuple``."""

    rho: float
    phi: float


class AzimuthalObjectRhoPhi(AzimuthalObject, AzimuthalRhoPhi, TupleRhoPhi):
    """
    Class for the ``rho`` and ``phi`` (azimuthal) coordinates of Object backend.
    Use the ``elements`` property to retrieve the coordinates.
    """

    @property
    def elements(self) -> tuple[float, float]:
        """
        Azimuthal coordinates (``rho`` and ``phi``) as a tuple.
        Each coordinate is a scalar and not a vector.

        Examples:
            >>> import vector
            >>> v = vector.VectorObject2D(rho=1, phi=2)
            >>> az = v.azimuthal
            >>> az.elements
            (1, 2)
        """
        return (self.rho, self.phi)


class TupleZ(typing.NamedTuple):
    """``z`` coordinate as a ``NamedTuple``."""

    z: float


class LongitudinalObjectZ(LongitudinalObject, LongitudinalZ, TupleZ):
    """
    Class for the ``z`` (longitudinal) coordinate of Object backend.
    Use the ``elements`` property to retrieve the coordinates.
    """

    @property
    def elements(self) -> tuple[float]:
        """
        Longitudinal coordinates (``z``) as a tuple.
        Each coordinate is a scalar and not a vector.

        Examples:
            >>> import vector
            >>> v = vector.obj(rho=1, phi=2, z=3)
            >>> lc = v.longitudinal
            >>> lc.elements
            (3,)
        """
        return (self.z,)


class TupleTheta(typing.NamedTuple):
    """``theta`` coordinates as a ``NamedTuple``."""

    theta: float


class LongitudinalObjectTheta(LongitudinalObject, LongitudinalTheta, TupleTheta):
    """
    Class for the ``theta`` (longitudinal) coordinate of Object backend.
    Use the ``elements`` property to retrieve the coordinates.
    """

    @property
    def elements(self) -> tuple[float]:
        """
        Longitudinal coordinates (``theta``) as a tuple.
        Each coordinate is a scalar and not a vector.

        Examples:
            >>> import vector
            >>> v = vector.obj(rho=1, phi=2, theta=3)
            >>> lc = v.longitudinal
            >>> lc.elements
            (3,)
        """
        return (self.theta,)


class TupleEta(typing.NamedTuple):
    """``eta`` coordinate as a ``NamedTuple``."""

    eta: float


class LongitudinalObjectEta(LongitudinalObject, LongitudinalEta, TupleEta):
    """
    Class for the ``eta`` (longitudinal) coordinate of Object backend.
    Use the ``elements`` property to retrieve the coordinates.
    """

    @property
    def elements(self) -> tuple[float]:
        """
        Longitudinal coordinates (``theta``) as a tuple.
        Each coordinate is a scalar and not a vector.

        Examples:
            >>> import vector
            >>> v = vector.obj(rho=1, phi=2, eta=3)
            >>> lc = v.longitudinal
            >>> lc.elements
            (3,)
        """
        return (self.eta,)


class TupleT(typing.NamedTuple):
    """``t`` coordinate as a ``NamedTuple``."""

    t: float


class TemporalObjectT(TemporalObject, TemporalT, TupleT):
    """
    Class for the ``t`` (temporal) coordinate of Object backend.
    Use the ``elements`` property to retrieve the coordinates.
    """

    @property
    def elements(self) -> tuple[float]:
        """
        Temporal coordinates (``t``) as a tuple.
        Each coordinate is a scalar and not a vector.

        Examples:
            >>> import vector
            >>> v = vector.obj(rho=1, phi=2, theta=3, t=4)
            >>> tc = v.temporal
            >>> tc.elements
            (4,)
        """
        return (self.t,)


class TupleTau(typing.NamedTuple):
    """``tau`` coordinate as a ``NamedTuple``."""

    tau: float


class TemporalObjectTau(TemporalObject, TemporalTau, TupleTau):
    """
    Class for the ``tau`` (temporal) coordinate of Object backend.
    Use the ``elements`` property to retrieve the coordinates.
    """

    @property
    def elements(self) -> tuple[float]:
        """
        Temporal coordinates (``tau``) as a tuple.
        Each coordinate is a scalar and not a vector.

        Examples:
            >>> import vector
            >>> v = vector.obj(rho=1, phi=2, theta=3, tau=4)
            >>> tc = v.temporal
            >>> tc.elements
            (4,)
        """
        return (self.tau,)


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
    """Mixin class for Object vectors."""

    lib = numpy

    def __eq__(self, other: typing.Any) -> typing.Any:
        return numpy.equal(self, other)  # type: ignore[call-overload]

    def __ne__(self, other: typing.Any) -> typing.Any:
        return numpy.not_equal(self, other)  # type: ignore[call-overload]

    def __abs__(self) -> float:
        return numpy.absolute(self)

    def __add__(self, other: VectorProtocol) -> VectorProtocol:
        return numpy.add(self, other)  # type: ignore[call-overload]

    def __radd__(self, other: VectorProtocol) -> VectorProtocol:
        return numpy.add(other, self)  # type: ignore[call-overload]

    def __iadd__(self: SameVectorType, other: VectorProtocol) -> SameVectorType:
        return _replace_data(self, numpy.add(self, other))  # type: ignore[call-overload]

    def __sub__(self, other: VectorProtocol) -> VectorProtocol:
        return numpy.subtract(self, other)  # type: ignore[call-overload]

    def __rsub__(self, other: VectorProtocol) -> VectorProtocol:
        return numpy.subtract(other, self)  # type: ignore[call-overload]

    def __isub__(self: SameVectorType, other: VectorProtocol) -> SameVectorType:
        return _replace_data(self, numpy.subtract(self, other))  # type: ignore[call-overload]

    def __mul__(self, other: float) -> VectorProtocol:
        return numpy.multiply(self, other)  # type: ignore[call-overload]

    def __rmul__(self, other: float) -> VectorProtocol:
        return numpy.multiply(other, self)  # type: ignore[call-overload]

    def __imul__(self: SameVectorType, other: float) -> SameVectorType:
        return _replace_data(self, numpy.multiply(self, other))  # type: ignore[call-overload]

    def __neg__(self: SameVectorType) -> SameVectorType:
        return numpy.negative(self)  # type: ignore[call-overload]

    def __pos__(self: SameVectorType) -> SameVectorType:
        return numpy.positive(self)  # type: ignore[call-overload]

    def __truediv__(self, other: float) -> VectorProtocol:
        return numpy.true_divide(self, other)  # type: ignore[call-overload]

    def __rtruediv__(self, other: float) -> VectorProtocol:
        return numpy.true_divide(other, self)  # type: ignore[call-overload]

    def __itruediv__(self: SameVectorType, other: float) -> VectorProtocol:
        return _replace_data(self, numpy.true_divide(self, other))  # type: ignore[call-overload]

    def __pow__(self, other: float) -> float:
        return numpy.square(self) if other == 2 else numpy.power(self, other)  # type: ignore[call-overload]

    def __matmul__(self, other: VectorProtocol) -> float:
        return numpy.matmul(self, other)  # type: ignore[call-overload]

    def __array_ufunc__(
        self,
        ufunc: typing.Any,
        method: typing.Any,
        *inputs: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        """
        Implements NumPy's ``ufunc``s for ``VectorObject`` and its subclasses. The current
        implementation includes ``numpy.absolute``, ``numpy.add``, ``numpy.subtract``,
        ``numpy.multiply``, ``numpy.positive``, ``numpy.negative``, ``numpy.true_divide``,
        ``numpy.power``, ``numpy.square``, ``numpy.sqrt``, ``numpy.cbrt``, ``numpy.matmul``,
        ``numpy.equal``, and ``numpy.not_equal``.
        """
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
    """
    Two dimensional vector class for the object backend.

    Examples:
        >>> import vector
        >>> vec = vector.VectorObject2D(x=1, y=2)
        >>> vec.x, vec.y
        (1, 2)
        >>> vec = vector.VectorObject2D(rho=1, phi=2)
        >>> vec.rho, vec.phi
        (1, 2)
        >>> vec = vector.VectorObject2D(azimuthal=vector.backends.object.AzimuthalObjectXY(1, 2))
        >>> vec.x, vec.y
        (1, 2)

    The following class methods can also be used to
    construct 2D object type vectors -

    - :meth:`VectorObject2D.from_xy`
    - :meth:`VectorObject2D.from_rhophi`

    Additionally, the :func:`vector.obj` function can
    also be used to construct 2D object type vectors.

    For two dimensional momentum vector objects, see
    :class:`vector.backends.object.MomentumObject2D`.
    """

    __slots__ = ("azimuthal",)

    azimuthal: AzimuthalObject

    @classmethod
    def from_xy(cls, x: float, y: float) -> VectorObject2D:
        """
        Constructs a ``VectorObject2D`` from Cartesian coordinates.

        Use :class:`vector.backends.object.MomentumObject2D` to construct a vector
        with momentum properties and methods.

        Examples:
            >>> import vector
            >>> vec = vector.VectorObject2D.from_xy(1, 2)
            >>> vec
            VectorObject2D(x=1, y=2)
        """
        return cls(azimuthal=AzimuthalObjectXY(x, y))

    @classmethod
    def from_rhophi(cls, rho: float, phi: float) -> VectorObject2D:
        """
        Constructs a ``VectorObject2D`` from polar coordinates.

        Use :class:`vector.backends.object.MomentumObject2D` to construct a vector
        with momentum properties and methods.

        Examples:
            >>> import vector
            >>> vec = vector.VectorObject2D.from_rhophi(1, 2)
            >>> vec
            VectorObject2D(rho=1, phi=2)
        """
        return cls(azimuthal=AzimuthalObjectRhoPhi(rho, phi))

    def __init__(
        self, azimuthal: AzimuthalObject | None = None, **kwargs: float
    ) -> None:
        if not _is_type_safe(kwargs):
            raise TypeError("a coordinate must be of the type int or float")

        for k, v in kwargs.copy().items():
            kwargs.pop(k)
            kwargs[_repr_momentum_to_generic.get(k, k)] = v

        if not kwargs and azimuthal is not None:
            self.azimuthal = azimuthal
        elif kwargs and azimuthal is None:
            if set(kwargs) == {"x", "y"}:
                self.azimuthal = AzimuthalObjectXY(kwargs["x"], kwargs["y"])
            elif set(kwargs) == {"rho", "phi"}:
                self.azimuthal = AzimuthalObjectRhoPhi(kwargs["rho"], kwargs["phi"])
            else:
                complaint = """unrecognized combination of coordinates, allowed combinations are:\n
                    x= y=
                    rho= phi=""".replace("                    ", "    ")
                if type(self) == VectorObject2D:
                    raise TypeError(complaint)
                else:
                    raise TypeError(f"{complaint}\n\nor their momentum equivalents")
        else:
            raise TypeError("must give Azimuthal if not giving keyword arguments")

    def __repr__(self) -> str:
        aznames = _coordinate_class_to_names[_aztype(self)]
        out = [f"{x}={getattr(self.azimuthal, x)}" for x in aznames]
        return "VectorObject2D(" + ", ".join(out) + ")"

    def __array__(self) -> FloatArray:
        from vector.backends.numpy import VectorNumpy2D

        return VectorNumpy2D(
            self.azimuthal.elements,
            dtype=[
                (x, numpy.float64) for x in _coordinate_class_to_names[_aztype(self)]
            ],
        )

    def _wrap_result(
        self,
        cls: typing.Any,
        result: typing.Any,
        returns: typing.Any,
        num_vecargs: typing.Any,
    ) -> typing.Any:
        """
        Wraps the raw result of a compute function as a scalar or a vector.

        Args:
            result: Value or tuple of values from a compute function.
            returns: Signature from a ``dispatch_map``.
            num_vecargs (int): Number of vector arguments in the function
                that would be treated on an equal footing (i.e. ``add``
                has two, but ``rotate_axis`` has only one: the ``axis``
                is secondary).
        """
        if returns in ([float], [bool]):
            return result

        elif (
            (len(returns) == 1 or (len(returns) == 2 and returns[1] is None))
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
        ):
            azcoords = _coord_object_type[returns[0]](result[0], result[1])
            return cls.ProjectionClass2D(azimuthal=azcoords)

        elif (
            len(returns) == 2
            or (len(returns) == 3 and returns[2] is None)
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
        ):
            azcoords = _coord_object_type[returns[0]](result[0], result[1])
            lcoords = _coord_object_type[returns[1]](result[2])
            return cls.ProjectionClass3D(azimuthal=azcoords, longitudinal=lcoords)

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
            return cls.ProjectionClass4D(
                azimuthal=azcoords, longitudinal=lcoords, temporal=tcoords
            )

        else:
            raise AssertionError(repr(returns))

    @property
    def x(self) -> float:
        return super().x

    @x.setter
    def x(self, x: float) -> None:
        self.azimuthal = AzimuthalObjectXY(x, self.y)

    @property
    def y(self) -> float:
        return super().y

    @y.setter
    def y(self, y: float) -> None:
        self.azimuthal = AzimuthalObjectXY(self.x, y)

    @property
    def rho(self) -> float:
        return super().rho

    @rho.setter
    def rho(self, rho: float) -> None:
        self.azimuthal = AzimuthalObjectRhoPhi(rho, self.phi)

    @property
    def phi(self) -> float:
        return super().phi

    @phi.setter
    def phi(self, phi: float) -> None:
        self.azimuthal = AzimuthalObjectRhoPhi(self.rho, phi)


class MomentumObject2D(PlanarMomentum, VectorObject2D):
    """
    Two dimensional momentum vector class for the object backend.

    Examples:
        >>> import vector
        >>> vec = vector.MomentumObject2D(px=1, py=2)
        >>> vec.px, vec.py
        (1, 2)
        >>> vec = vector.MomentumObject2D(pt=1, phi=2)
        >>> vec.pt, vec.phi
        (1, 2)
        >>> vec = vector.MomentumObject2D(azimuthal=vector.backends.object.AzimuthalObjectXY(1, 2))
        >>> vec.px, vec.py
        (1, 2)

    The :func:`vector.obj` function can also be
    used to construct 2D momentum object type vectors.

    For two dimensional vector objects, see
    :class:`vector.backends.object.VectorObject2D`.
    """

    def __repr__(self) -> str:
        aznames = _coordinate_class_to_names[_aztype(self)]
        out = []
        for x in aznames:
            y = _repr_generic_to_momentum.get(x, x)
            out.append(f"{y}={getattr(self.azimuthal, x)}")
        return "MomentumObject2D(" + ", ".join(out) + ")"

    def __array__(self) -> FloatArray:
        from vector.backends.numpy import MomentumNumpy2D

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
        self.azimuthal = AzimuthalObjectXY(px, self.py)

    @property
    def py(self) -> float:
        return super().py

    @py.setter
    def py(self, py: float) -> None:
        self.azimuthal = AzimuthalObjectXY(self.px, py)

    @property
    def pt(self) -> float:
        return super().pt

    @pt.setter
    def pt(self, pt: float) -> None:
        self.azimuthal = AzimuthalObjectRhoPhi(pt, self.phi)


class VectorObject3D(VectorObject, Spatial, Vector3D):
    """
    Three dimensional vector class for the object backend.
    Use the class methods -

    Examples:
        >>> import vector
        >>> vec = vector.VectorObject3D(x=1, y=2, z=3)
        >>> vec.x, vec.y, vec.z
        (1, 2, 3)
        >>> vec = vector.VectorObject3D(rho=1, phi=2, eta=3)
        >>> vec.rho, vec.phi, vec.eta
        (1, 2, 3)
        >>> vec = vector.VectorObject3D(
        ...     azimuthal=vector.backends.object.AzimuthalObjectXY(1, 2),
        ...     longitudinal=vector.backends.object.LongitudinalObjectTheta(3)
        ... )
        >>> vec.x, vec.y, vec.theta
        (1, 2, 3)


    The following class methods can also be used to
    construct 3D object type vectors -

    - :meth:`VectorObject3D.from_xyz`
    - :meth:`VectorObject3D.from_xytheta`
    - :meth:`VectorObject3D.from_xyeta`
    - :meth:`VectorObject3D.from_rhophiz`
    - :meth:`VectorObject3D.from_rhophitheta`
    - :meth:`VectorObject3D.from_rhophieta`

    Additionally, the :func:`vector.obj` function can
    also be used to construct 3D object type vectors.

    For three dimensional momentum vector objects, see
    :class:`vector.backends.object.MomentumObject3D`.
    """

    __slots__ = ("azimuthal", "longitudinal")

    azimuthal: AzimuthalObject
    longitudinal: LongitudinalObject

    @classmethod
    def from_xyz(cls, x: float, y: float, z: float) -> VectorObject3D:
        """
        Constructs a ``VectorObject3D`` from Cartesian coordinates.

        Use :class:`vector.backends.object.MomentumObject3D` to construct a vector
        with momentum properties and methods.

        Examples:
            >>> import vector
            >>> vec = vector.VectorObject3D.from_xyz(1, 1, 1)
            >>> vec
            VectorObject3D(x=1, y=1, z=1)
        """
        return cls(
            azimuthal=AzimuthalObjectXY(x, y), longitudinal=LongitudinalObjectZ(z)
        )

    @classmethod
    def from_xytheta(cls, x: float, y: float, theta: float) -> VectorObject3D:
        r"""
        Constructs a ``VectorObject3D`` from Cartesian azimuthal coordinates and
        a polar angle $\theta$.

        Use :class:`vector.backends.object.MomentumObject3D` to construct a vector
        with momentum properties and methods.

        Examples:
            >>> import vector
            >>> vec = vector.VectorObject3D.from_xytheta(1, 1, 1)
            >>> vec
            VectorObject3D(x=1, y=1, theta=1)
        """
        return cls(
            azimuthal=AzimuthalObjectXY(x, y),
            longitudinal=LongitudinalObjectTheta(theta),
        )

    @classmethod
    def from_xyeta(cls, x: float, y: float, eta: float) -> VectorObject3D:
        r"""
        Constructs a ``VectorObject3D`` from Cartesian coordinates and a
        pseudorapidity $\eta$.

        Use :class:`vector.backends.object.MomentumObject3D` to construct a vector
        with momentum properties and methods.

        Examples:
            >>> import vector
            >>> vec = vector.VectorObject3D.from_xyeta(1, 1, 1)
            >>> vec
            VectorObject3D(x=1, y=1, eta=1)
        """
        return cls(
            azimuthal=AzimuthalObjectXY(x, y), longitudinal=LongitudinalObjectEta(eta)
        )

    @classmethod
    def from_rhophiz(cls, rho: float, phi: float, z: float) -> VectorObject3D:
        """
        Constructs a ``VectorObject3D`` from polar azimuthal coordinates and a
        Cartesian longitudinal coordinate $z$.

        Use :class:`vector.backends.object.MomentumObject3D` to construct a vector
        with momentum properties and methods.

        Examples:
            >>> import vector
            >>> vec = vector.VectorObject3D.from_rhophiz(1, 1, 1)
            >>> vec
            VectorObject3D(rho=1, phi=1, z=1)
        """
        return cls(
            azimuthal=AzimuthalObjectRhoPhi(rho, phi),
            longitudinal=LongitudinalObjectZ(z),
        )

    @classmethod
    def from_rhophitheta(cls, rho: float, phi: float, theta: float) -> VectorObject3D:
        r"""
        Constructs a ``VectorObject3D`` from polar azimuthal coordinates and a
        polar angle $\theta$.

        Use :class:`vector.backends.object.MomentumObject3D` to construct a vector
        with momentum properties and methods.

        Examples:
            >>> import vector
            >>> vec = vector.VectorObject3D.from_rhophitheta(1, 1, 1)
            >>> vec
            VectorObject3D(rho=1, phi=1, theta=1)
        """
        return cls(
            azimuthal=AzimuthalObjectRhoPhi(rho, phi),
            longitudinal=LongitudinalObjectTheta(theta),
        )

    @classmethod
    def from_rhophieta(cls, rho: float, phi: float, eta: float) -> VectorObject3D:
        r"""
        Constructs a ``VectorObject3D`` from polar azimuthal coordinates and a
        pseudorapidity $\eta$.

        Use :class:`vector.backends.object.MomentumObject3D` to construct a vector
        with momentum properties and methods.

        Examples:
            >>> import vector
            >>> vec = vector.VectorObject3D.from_rhophieta(1, 1, 1)
            >>> vec
            VectorObject3D(rho=1, phi=1, eta=1)
        """
        return cls(
            azimuthal=AzimuthalObjectRhoPhi(rho, phi),
            longitudinal=LongitudinalObjectEta(eta),
        )

    def __init__(
        self,
        azimuthal: AzimuthalObject | None = None,
        longitudinal: LongitudinalObject | None = None,
        **kwargs: float,
    ) -> None:
        if not _is_type_safe(kwargs):
            raise TypeError("a coordinate must be of the type int or float")

        for k, v in kwargs.copy().items():
            kwargs.pop(k)
            kwargs[_repr_momentum_to_generic.get(k, k)] = v

        if not kwargs and azimuthal is not None and longitudinal is not None:
            self.azimuthal = azimuthal
            self.longitudinal = longitudinal
        elif kwargs and azimuthal is None and longitudinal is None:
            if set(kwargs) == {"x", "y", "z"}:
                self.azimuthal = AzimuthalObjectXY(kwargs["x"], kwargs["y"])
                self.longitudinal = LongitudinalObjectZ(kwargs["z"])
            elif set(kwargs) == {"x", "y", "eta"}:
                self.azimuthal = AzimuthalObjectXY(kwargs["x"], kwargs["y"])
                self.longitudinal = LongitudinalObjectEta(kwargs["eta"])
            elif set(kwargs) == {"x", "y", "theta"}:
                self.azimuthal = AzimuthalObjectXY(kwargs["x"], kwargs["y"])
                self.longitudinal = LongitudinalObjectTheta(kwargs["theta"])
            elif set(kwargs) == {"rho", "phi", "z"}:
                self.azimuthal = AzimuthalObjectRhoPhi(kwargs["rho"], kwargs["phi"])
                self.longitudinal = LongitudinalObjectZ(kwargs["z"])
            elif set(kwargs) == {"rho", "phi", "eta"}:
                self.azimuthal = AzimuthalObjectRhoPhi(kwargs["rho"], kwargs["phi"])
                self.longitudinal = LongitudinalObjectEta(kwargs["eta"])
            elif set(kwargs) == {"rho", "phi", "theta"}:
                self.azimuthal = AzimuthalObjectRhoPhi(kwargs["rho"], kwargs["phi"])
                self.longitudinal = LongitudinalObjectTheta(kwargs["theta"])
            else:
                complaint = """unrecognized combination of coordinates, allowed combinations are:\n
                    x= y= z=
                    x= y= theta=
                    x= y= eta=
                    rho= phi= z=
                    rho= phi= theta=
                    rho= phi= eta=""".replace("                    ", "    ")
                if type(self) == VectorObject3D:
                    raise TypeError(complaint)
                else:
                    raise TypeError(f"{complaint}\n\nor their momentum equivalents")
        else:
            raise TypeError(
                "must give Azimuthal and Longitudinal if not giving keyword arguments"
            )

    def __repr__(self) -> str:
        aznames = _coordinate_class_to_names[_aztype(self)]
        lnames = _coordinate_class_to_names[_ltype(self)]
        out = [f"{x}={getattr(self.azimuthal, x)}" for x in aznames]
        for x in lnames:
            out.append(f"{x}={getattr(self.longitudinal, x)}")
        return "VectorObject3D(" + ", ".join(out) + ")"

    def __array__(self) -> FloatArray:
        from vector.backends.numpy import VectorNumpy3D

        return VectorNumpy3D(
            self.azimuthal.elements + self.longitudinal.elements,
            dtype=[
                (x, numpy.float64)
                for x in _coordinate_class_to_names[_aztype(self)]
                + _coordinate_class_to_names[_ltype(self)]
            ],
        )

    def _wrap_result(
        self,
        cls: typing.Any,
        result: typing.Any,
        returns: typing.Any,
        num_vecargs: typing.Any,
    ) -> typing.Any:
        """
        Wraps the raw result of a compute function as a scalar or a vector.

        Args:
            result: Value or tuple of values from a compute function.
            returns: Signature from a ``dispatch_map``.
            num_vecargs (int): Number of vector arguments in the function
                that would be treated on an equal footing (i.e. ``add``
                has two, but ``rotate_axis`` has only one: the ``axis``
                is secondary).
        """
        if returns in ([float], [bool]):
            return result

        elif (
            len(returns) == 1
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
        ):
            azcoords = _coord_object_type[returns[0]](result[0], result[1])
            return cls.ProjectionClass3D(
                azimuthal=azcoords, longitudinal=self.longitudinal
            )

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and returns[1] is None
        ):
            azcoords = _coord_object_type[returns[0]](result[0], result[1])
            return cls.ProjectionClass2D(azimuthal=azcoords)

        elif (
            len(returns) == 2
            or (len(returns) == 3 and returns[2] is None)
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
        ):
            azcoords = _coord_object_type[returns[0]](result[0], result[1])
            lcoords = _coord_object_type[returns[1]](result[2])
            return cls.ProjectionClass3D(azimuthal=azcoords, longitudinal=lcoords)

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
            return cls.ProjectionClass4D(
                azimuthal=azcoords, longitudinal=lcoords, temporal=tcoords
            )

        else:
            raise AssertionError(repr(returns))

    @property
    def x(self) -> float:
        return super().x

    @x.setter
    def x(self, x: float) -> None:
        self.azimuthal = AzimuthalObjectXY(x, self.y)

    @property
    def y(self) -> float:
        return super().y

    @y.setter
    def y(self, y: float) -> None:
        self.azimuthal = AzimuthalObjectXY(self.x, y)

    @property
    def rho(self) -> float:
        return super().rho

    @rho.setter
    def rho(self, rho: float) -> None:
        self.azimuthal = AzimuthalObjectRhoPhi(rho, self.phi)

    @property
    def phi(self) -> float:
        return super().phi

    @phi.setter
    def phi(self, phi: float) -> None:
        self.azimuthal = AzimuthalObjectRhoPhi(self.rho, phi)

    @property
    def z(self) -> float:
        return super().z

    @z.setter
    def z(self, z: float) -> None:
        self.longitudinal = LongitudinalObjectZ(z)

    @property
    def theta(self) -> float:
        return super().theta

    @theta.setter
    def theta(self, theta: float) -> None:
        self.longitudinal = LongitudinalObjectTheta(theta)

    @property
    def eta(self) -> float:
        return super().eta

    @eta.setter
    def eta(self, eta: float) -> None:
        self.longitudinal = LongitudinalObjectEta(eta)


class MomentumObject3D(SpatialMomentum, VectorObject3D):
    """
    Three dimensional momentum vector class for the object backend.

    Examples:
        >>> import vector
        >>> vec = vector.MomentumObject3D(px=1, py=2, pz=3)
        >>> vec.px, vec.py, vec.pz
        (1, 2, 3)
        >>> vec = vector.MomentumObject3D(pt=1, phi=2, pz=3)
        >>> vec.pt, vec.phi, vec.pz
        (1, 2, 3)
        >>> vec = vector.MomentumObject3D(
        ...     azimuthal=vector.backends.object.AzimuthalObjectXY(1, 2),
        ...     longitudinal=vector.backends.object.LongitudinalObjectTheta(3)
        ... )
        >>> vec.x, vec.y, vec.theta
        (1, 2, 3)

    The :func:`vector.obj` function can also be
    used to construct 3D momentum object type vectors.

    For three dimensional vector objects, see
    :class:`vector.backends.object.VectorObject3D`.
    """

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
        return "MomentumObject3D(" + ", ".join(out) + ")"

    def __array__(self) -> FloatArray:
        from vector.backends.numpy import MomentumNumpy3D

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
        self.azimuthal = AzimuthalObjectXY(px, self.py)

    @property
    def py(self) -> float:
        return super().py

    @py.setter
    def py(self, py: float) -> None:
        self.azimuthal = AzimuthalObjectXY(self.px, py)

    @property
    def pt(self) -> float:
        return super().pt

    @pt.setter
    def pt(self, pt: float) -> None:
        self.azimuthal = AzimuthalObjectRhoPhi(pt, self.phi)

    @property
    def pz(self) -> float:
        return super().pz

    @pz.setter
    def pz(self, pz: float) -> None:
        self.longitudinal = LongitudinalObjectZ(pz)


class VectorObject4D(VectorObject, Lorentz, Vector4D):
    """
    Four dimensional vector class for the object backend.
    Use the class methods -

    Examples:
        >>> import vector
        >>> vec = vector.VectorObject4D(x=1, y=2, z=3, t=4)
        >>> vec.x, vec.y, vec.z, vec.t
        (1, 2, 3, 4)
        >>> vec = vector.VectorObject4D(rho=1, phi=2, eta=3, tau=4)
        >>> vec.rho, vec.phi, vec.eta, vec.tau
        (1, 2, 3, 4)
        >>> vec = vector.VectorObject4D(
        ...     azimuthal=vector.backends.object.AzimuthalObjectXY(1, 2),
        ...     longitudinal=vector.backends.object.LongitudinalObjectTheta(3),
        ...     temporal=vector.backends.object.TemporalObjectTau(4)
        ... )
        >>> vec.x, vec.y, vec.theta, vec.tau
        (1, 2, 3, 4)

    The following class methods can also be used to
    construct 4D object type vectors -

    - :meth:`VectorObject4D.from_xyzt`
    - :meth:`VectorObject4D.from_xythetat`
    - :meth:`VectorObject4D.from_xyetat`
    - :meth:`VectorObject4D.from_rhophizt`
    - :meth:`VectorObject4D.from_rhophithetat`
    - :meth:`VectorObject4D.from_rhophietat`
    - :meth:`VectorObject4D.from_xyztau`
    - :meth:`VectorObject4D.from_xythetatau`
    - :meth:`VectorObject4D.from_xyetatau`
    - :meth:`VectorObject4D.from_rhophiztau`
    - :meth:`VectorObject4D.from_rhophithetatau`
    - :meth:`VectorObject4D.from_rhophietatau`

    Additionally, the :func:`vector.obj` function can
    also be used to construct 4D object type vectors.

    For four dimensional momentum vector objects, see
    :class:`vector.backends.object.MomentumObject4D`.
    """

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
    ) -> VectorObject4D:
        """
        Constructs a ``VectorObject3D`` from Cartesian coordinates and a time
        coordinate $t$.

        Use :class:`vector.backends.object.MomentumObject3D` to construct a vector
        with momentum properties and methods.

        Examples:
            >>> import vector
            >>> vec = vector.VectorObject4D.from_xyzt(1, 1, 1, 1)
            >>> vec
            VectorObject4D(x=1, y=1, z=1, t=1)
        """
        return cls(
            azimuthal=AzimuthalObjectXY(x, y),
            longitudinal=LongitudinalObjectZ(z),
            temporal=TemporalObjectT(t),
        )

    @classmethod
    def from_xyztau(
        cls,
        x: float,
        y: float,
        z: float,
        tau: float,
    ) -> VectorObject4D:
        r"""
        Constructs a ``VectorObject3D`` from Cartesian coordinates and a proper time
        coordinate $\tau$.

        Use :class:`vector.backends.object.MomentumObject3D` to construct a vector
        with momentum properties and methods.

        Examples:
            >>> import vector
            >>> vec = vector.VectorObject4D.from_xyztau(1, 1, 1, 1)
            >>> vec
            VectorObject4D(x=1, y=1, z=1, tau=1)
        """
        return cls(
            azimuthal=AzimuthalObjectXY(x, y),
            longitudinal=LongitudinalObjectZ(z),
            temporal=TemporalObjectTau(tau),
        )

    @classmethod
    def from_xythetat(
        cls,
        x: float,
        y: float,
        theta: float,
        t: float,
    ) -> VectorObject4D:
        r"""
        Constructs a ``VectorObject3D`` from Cartesian azimuthal coordinates, a
        polar angle $\theta$, and a time coordinate $t$.

        Use :class:`vector.backends.object.MomentumObject3D` to construct a vector
        with momentum properties and methods.

        Examples:
            >>> import vector
            >>> vec = vector.VectorObject4D.from_xythetat(1, 1, 1, 1)
            >>> vec
            VectorObject4D(x=1, y=1, theta=1, t=1)
        """
        return cls(
            azimuthal=AzimuthalObjectXY(x, y),
            longitudinal=LongitudinalObjectTheta(theta),
            temporal=TemporalObjectT(t),
        )

    @classmethod
    def from_xythetatau(
        cls,
        x: float,
        y: float,
        theta: float,
        tau: float,
    ) -> VectorObject4D:
        r"""
        Constructs a ``VectorObject3D`` from Cartesian azimuthal coordinates, a
        polar angle $\theta$, and a proper time coordinate $\tau$.

        Use :class:`vector.backends.object.MomentumObject3D` to construct a vector
        with momentum properties and methods.

        Examples:
            >>> import vector
            >>> vec = vector.VectorObject4D.from_xythetatau(1, 1, 1, 1)
            >>> vec
            VectorObject4D(x=1, y=1, theta=1, tau=1)
        """
        return cls(
            azimuthal=AzimuthalObjectXY(x, y),
            longitudinal=LongitudinalObjectTheta(theta),
            temporal=TemporalObjectTau(tau),
        )

    @classmethod
    def from_xyetat(
        cls,
        x: float,
        y: float,
        eta: float,
        t: float,
    ) -> VectorObject4D:
        r"""
        Constructs a ``VectorObject3D`` from Cartesian coordinates, a pseudorapidity
        $\eta$, and a time coordinate $t$.

        Use :class:`vector.backends.object.MomentumObject3D` to construct a vector
        with momentum properties and methods.

        Examples:
            >>> import vector
            >>> vec = vector.VectorObject4D.from_xyetat(1, 1, 1, 1)
            >>> vec
            VectorObject4D(x=1, y=1, eta=1, t=1)
        """
        return cls(
            azimuthal=AzimuthalObjectXY(x, y),
            longitudinal=LongitudinalObjectEta(eta),
            temporal=TemporalObjectT(t),
        )

    @classmethod
    def from_xyetatau(
        cls,
        x: float,
        y: float,
        eta: float,
        tau: float,
    ) -> VectorObject4D:
        r"""
        Constructs a ``VectorObject3D`` from Cartesian coordinates, a pseudorapidity
        $\eta$, and a proper time coordinate $\tau$.

        Use :class:`vector.backends.object.MomentumObject3D` to construct a vector
        with momentum properties and methods.

        Examples:
            >>> import vector
            >>> vec = vector.VectorObject4D.from_xyetatau(1, 1, 1, 1)
            >>> vec
            VectorObject4D(x=1, y=1, eta=1, tau=1)
        """
        return cls(
            azimuthal=AzimuthalObjectXY(x, y),
            longitudinal=LongitudinalObjectEta(eta),
            temporal=TemporalObjectTau(tau),
        )

    @classmethod
    def from_rhophizt(
        cls,
        rho: float,
        phi: float,
        z: float,
        t: float,
    ) -> VectorObject4D:
        """
        Constructs a ``VectorObject3D`` from polar azimuthal coordinates, a Cartesian
        longitudinal coordinate $z$, and a time coordinate $t$.

        Use :class:`vector.backends.object.MomentumObject3D` to construct a vector
        with momentum properties and methods.

        Examples:
            >>> import vector
            >>> vec = vector.VectorObject4D.from_rhophizt(1, 1, 1, 1)
            >>> vec
            VectorObject4D(rho=1, phi=1, z=1, t=1)
        """
        return cls(
            azimuthal=AzimuthalObjectRhoPhi(rho, phi),
            longitudinal=LongitudinalObjectZ(z),
            temporal=TemporalObjectT(t),
        )

    @classmethod
    def from_rhophiztau(
        cls,
        rho: float,
        phi: float,
        z: float,
        tau: float,
    ) -> VectorObject4D:
        r"""
        Constructs a ``VectorObject3D`` from polar azimuthal coordinates, a Cartesian
        longitudinal coordinate $z$, and a proper time coordinate $\tau$.

        Use :class:`vector.backends.object.MomentumObject3D` to construct a vector
        with momentum properties and methods.

        Examples:
            >>> import vector
            >>> vec = vector.VectorObject4D.from_rhophiztau(1, 1, 1, 1)
            >>> vec
            VectorObject4D(rho=1, phi=1, z=1, tau=1)
        """
        return cls(
            azimuthal=AzimuthalObjectRhoPhi(rho, phi),
            longitudinal=LongitudinalObjectZ(z),
            temporal=TemporalObjectTau(tau),
        )

    @classmethod
    def from_rhophithetat(
        cls,
        rho: float,
        phi: float,
        theta: float,
        t: float,
    ) -> VectorObject4D:
        r"""
        Constructs a ``VectorObject3D`` from polar azimuthal coordinates, a polar
        angle $\theta$, and a time coordinate $t$.

        Use :class:`vector.backends.object.MomentumObject3D` to construct a vector
        with momentum properties and methods.

        Examples:
            >>> import vector
            >>> vec = vector.VectorObject4D.from_rhophithetat(1, 1, 1, 1)
            >>> vec
            VectorObject4D(rho=1, phi=1, theta=1, t=1)
        """
        return cls(
            azimuthal=AzimuthalObjectRhoPhi(rho, phi),
            longitudinal=LongitudinalObjectTheta(theta),
            temporal=TemporalObjectT(t),
        )

    @classmethod
    def from_rhophithetatau(
        cls,
        rho: float,
        phi: float,
        theta: float,
        tau: float,
    ) -> VectorObject4D:
        r"""
        Constructs a ``VectorObject3D`` from polar azimuthal coordinates, a polar
        angle $\theta$, and a proper time coordinate $\tau$.

        Use :class:`vector.backends.object.MomentumObject3D` to construct a vector
        with momentum properties and methods.

        Examples:
            >>> import vector
            >>> vec = vector.VectorObject4D.from_rhophithetatau(1, 1, 1, 1)
            >>> vec
            VectorObject4D(rho=1, phi=1, theta=1, tau=1)
        """
        return cls(
            azimuthal=AzimuthalObjectRhoPhi(rho, phi),
            longitudinal=LongitudinalObjectTheta(theta),
            temporal=TemporalObjectTau(tau),
        )

    @classmethod
    def from_rhophietat(
        cls,
        rho: float,
        phi: float,
        eta: float,
        t: float,
    ) -> VectorObject4D:
        r"""
        Constructs a ``VectorObject3D`` from polar azimuthal coordinates, a
        pseudorapidity $\eta$, and a time coordinate $t$.

        Use :class:`vector.backends.object.MomentumObject3D` to construct a vector
        with momentum properties and methods.

        Examples:
            >>> import vector
            >>> vec = vector.VectorObject4D.from_rhophietat(1, 1, 1, 1)
            >>> vec
            VectorObject4D(rho=1, phi=1, eta=1, t=1)
        """
        return cls(
            azimuthal=AzimuthalObjectRhoPhi(rho, phi),
            longitudinal=LongitudinalObjectEta(eta),
            temporal=TemporalObjectT(t),
        )

    @classmethod
    def from_rhophietatau(
        cls,
        rho: float,
        phi: float,
        eta: float,
        tau: float,
    ) -> VectorObject4D:
        r"""
        Constructs a ``VectorObject3D`` from polar azimuthal coordinates, a
        pseudorapidity $\eta$, and a proper time coordinate $\tau$.

        Use :class:`vector.backends.object.MomentumObject3D` to construct a vector
        with momentum properties and methods.

        Examples:
            >>> import vector
            >>> vec = vector.VectorObject4D.from_rhophietatau(1, 1, 1, 1)
            >>> vec
            VectorObject4D(rho=1, phi=1, eta=1, tau=1)
        """
        return cls(
            azimuthal=AzimuthalObjectRhoPhi(rho, phi),
            longitudinal=LongitudinalObjectEta(eta),
            temporal=TemporalObjectTau(tau),
        )

    def __init__(
        self,
        azimuthal: AzimuthalObject | None = None,
        longitudinal: LongitudinalObject | None = None,
        temporal: TemporalObject | None = None,
        **kwargs: float,
    ) -> None:
        for k, v in kwargs.copy().items():
            kwargs.pop(k)
            kwargs[_repr_momentum_to_generic.get(k, k)] = v

        if (
            not kwargs
            and azimuthal is not None
            and longitudinal is not None
            and temporal is not None
        ):
            self.azimuthal = azimuthal
            self.longitudinal = longitudinal
            self.temporal = temporal
        elif kwargs and azimuthal is None and longitudinal is None and temporal is None:
            if set(kwargs) == {"x", "y", "z", "t"}:
                self.azimuthal = AzimuthalObjectXY(kwargs["x"], kwargs["y"])
                self.longitudinal = LongitudinalObjectZ(kwargs["z"])
                self.temporal = TemporalObjectT(kwargs["t"])
            elif set(kwargs) == {"x", "y", "eta", "t"}:
                self.azimuthal = AzimuthalObjectXY(kwargs["x"], kwargs["y"])
                self.longitudinal = LongitudinalObjectEta(kwargs["eta"])
                self.temporal = TemporalObjectT(kwargs["t"])
            elif set(kwargs) == {"x", "y", "theta", "t"}:
                self.azimuthal = AzimuthalObjectXY(kwargs["x"], kwargs["y"])
                self.longitudinal = LongitudinalObjectTheta(kwargs["theta"])
                self.temporal = TemporalObjectT(kwargs["t"])
            elif set(kwargs) == {"rho", "phi", "z", "t"}:
                self.azimuthal = AzimuthalObjectRhoPhi(kwargs["rho"], kwargs["phi"])
                self.longitudinal = LongitudinalObjectZ(kwargs["z"])
                self.temporal = TemporalObjectT(kwargs["t"])
            elif set(kwargs) == {"rho", "phi", "eta", "t"}:
                self.azimuthal = AzimuthalObjectRhoPhi(kwargs["rho"], kwargs["phi"])
                self.longitudinal = LongitudinalObjectEta(kwargs["eta"])
                self.temporal = TemporalObjectT(kwargs["t"])
            elif set(kwargs) == {"rho", "phi", "theta", "t"}:
                self.azimuthal = AzimuthalObjectRhoPhi(kwargs["rho"], kwargs["phi"])
                self.longitudinal = LongitudinalObjectTheta(kwargs["theta"])
                self.temporal = TemporalObjectT(kwargs["t"])
            elif set(kwargs) == {"x", "y", "z", "tau"}:
                self.azimuthal = AzimuthalObjectXY(kwargs["x"], kwargs["y"])
                self.longitudinal = LongitudinalObjectZ(kwargs["z"])
                self.temporal = TemporalObjectTau(kwargs["tau"])
            elif set(kwargs) == {"x", "y", "eta", "tau"}:
                self.azimuthal = AzimuthalObjectXY(kwargs["x"], kwargs["y"])
                self.longitudinal = LongitudinalObjectEta(kwargs["eta"])
                self.temporal = TemporalObjectTau(kwargs["tau"])
            elif set(kwargs) == {"x", "y", "theta", "tau"}:
                self.azimuthal = AzimuthalObjectXY(kwargs["x"], kwargs["y"])
                self.longitudinal = LongitudinalObjectTheta(kwargs["theta"])
                self.temporal = TemporalObjectTau(kwargs["tau"])
            elif set(kwargs) == {"rho", "phi", "z", "tau"}:
                self.azimuthal = AzimuthalObjectRhoPhi(kwargs["rho"], kwargs["phi"])
                self.longitudinal = LongitudinalObjectZ(kwargs["z"])
                self.temporal = TemporalObjectTau(kwargs["tau"])
            elif set(kwargs) == {"rho", "phi", "eta", "tau"}:
                self.azimuthal = AzimuthalObjectRhoPhi(kwargs["rho"], kwargs["phi"])
                self.longitudinal = LongitudinalObjectEta(kwargs["eta"])
                self.temporal = TemporalObjectTau(kwargs["tau"])
            elif set(kwargs) == {"rho", "phi", "theta", "tau"}:
                self.azimuthal = AzimuthalObjectRhoPhi(kwargs["rho"], kwargs["phi"])
                self.longitudinal = LongitudinalObjectTheta(kwargs["theta"])
                self.temporal = TemporalObjectTau(kwargs["tau"])
            else:
                complaint = """unrecognized combination of coordinates, allowed combinations are:\n
                    x= y= z= tau=
                    x= y= theta= t=
                    x= y= theta= tau=
                    x= y= eta= t=
                    x= y= z= t=
                    x= y= eta= tau=
                    rho= phi= z= t=
                    rho= phi= z= tau=
                    rho= phi= theta= t=
                    rho= phi= theta= tau=
                    rho= phi= eta= t=
                    rho= phi= eta= tau=""".replace("                    ", "    ")
                if type(self) == VectorObject4D:
                    raise TypeError(complaint)
                else:
                    raise TypeError(f"{complaint}\n\nor their momentum equivalents")
        else:
            raise TypeError(
                "must give Azimuthal, Longitudinal, and Temporal if not giving keyword arguments"
            )

    def __repr__(self) -> str:
        aznames = _coordinate_class_to_names[_aztype(self)]
        lnames = _coordinate_class_to_names[_ltype(self)]
        tnames = _coordinate_class_to_names[_ttype(self)]
        out = [f"{x}={getattr(self.azimuthal, x)}" for x in aznames]
        for x in lnames:
            out.append(f"{x}={getattr(self.longitudinal, x)}")
        for x in tnames:
            out.append(f"{x}={getattr(self.temporal, x)}")
        return "VectorObject4D(" + ", ".join(out) + ")"

    def __array__(self) -> FloatArray:
        from vector.backends.numpy import VectorNumpy4D

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

    def _wrap_result(
        self,
        cls: typing.Any,
        result: typing.Any,
        returns: typing.Any,
        num_vecargs: typing.Any,
    ) -> typing.Any:
        """
        Wraps the raw result of a compute function as a scalar or a vector.

        Args:
            result: Value or tuple of values from a compute function.
            returns: Signature from a ``dispatch_map``.
            num_vecargs (int): Number of vector arguments in the function
                that would be treated on an equal footing (i.e. ``add``
                has two, but ``rotate_axis`` has only one: the ``axis``
                is secondary).
        """
        if returns in ([float], [bool]):
            return result

        elif (
            len(returns) == 1
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
        ):
            azcoords = _coord_object_type[returns[0]](result[0], result[1])
            return cls.ProjectionClass4D(
                azimuthal=azcoords,
                longitudinal=self.longitudinal,
                temporal=self.temporal,
            )

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and returns[1] is None
        ):
            azcoords = _coord_object_type[returns[0]](result[0], result[1])
            return cls.ProjectionClass2D(azimuthal=azcoords)

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
        ):
            azcoords = _coord_object_type[returns[0]](result[0], result[1])
            lcoords = _coord_object_type[returns[1]](result[2])
            return cls.ProjectionClass4D(
                azimuthal=azcoords, longitudinal=lcoords, temporal=self.temporal
            )

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
            return cls.ProjectionClass3D(azimuthal=azcoords, longitudinal=lcoords)

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
            return cls.ProjectionClass4D(
                azimuthal=azcoords, longitudinal=lcoords, temporal=tcoords
            )

        else:
            raise AssertionError(repr(returns))

    @property
    def x(self) -> float:
        return super().x

    @x.setter
    def x(self, x: float) -> None:
        self.azimuthal = AzimuthalObjectXY(x, self.y)

    @property
    def y(self) -> float:
        return super().y

    @y.setter
    def y(self, y: float) -> None:
        self.azimuthal = AzimuthalObjectXY(self.x, y)

    @property
    def rho(self) -> float:
        return super().rho

    @rho.setter
    def rho(self, rho: float) -> None:
        self.azimuthal = AzimuthalObjectRhoPhi(rho, self.phi)

    @property
    def phi(self) -> float:
        return super().phi

    @phi.setter
    def phi(self, phi: float) -> None:
        self.azimuthal = AzimuthalObjectRhoPhi(self.rho, phi)

    @property
    def z(self) -> float:
        return super().z

    @z.setter
    def z(self, z: float) -> None:
        self.longitudinal = LongitudinalObjectZ(z)

    @property
    def theta(self) -> float:
        return super().theta

    @theta.setter
    def theta(self, theta: float) -> None:
        self.longitudinal = LongitudinalObjectTheta(theta)

    @property
    def eta(self) -> float:
        return super().eta

    @eta.setter
    def eta(self, eta: float) -> None:
        self.longitudinal = LongitudinalObjectEta(eta)

    @property
    def t(self) -> float:
        return super().t

    @t.setter
    def t(self, t: float) -> None:
        self.temporal = TemporalObjectT(t)

    @property
    def tau(self) -> float:
        return super().tau

    @tau.setter
    def tau(self, tau: float) -> None:
        self.temporal = TemporalObjectTau(tau)


class MomentumObject4D(LorentzMomentum, VectorObject4D):
    """
    Four dimensional momentum vector class for the object backend.

    Examples:
        >>> import vector
        >>> vec = vector.MomentumObject4D(px=1, py=2, pz=3, t=4)
        >>> vec.px, vec.py, vec.pz, vec.t
        (1, 2, 3, 4)
        >>> vec = vector.MomentumObject4D(pt=1, phi=2, pz=3, M=4)
        >>> vec.pt, vec.phi, vec.pz, vec.M
        (1, 2, 3, 4)
        >>> vec = vector.MomentumObject4D(
        ...     azimuthal=vector.backends.object.AzimuthalObjectXY(1, 2),
        ...     longitudinal=vector.backends.object.LongitudinalObjectTheta(3),
        ...     temporal=vector.backends.object.TemporalObjectTau(4)
        ... )
        >>> vec.x, vec.y, vec.theta, vec.tau
        (1, 2, 3, 4)

    The :func:`vector.obj` function can also be
    used to construct 4D momentum object type vectors.

    For four dimensional vector objects, see
    :class:`vector.backends.object.VectorObject4D`.
    """

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
        return "MomentumObject4D(" + ", ".join(out) + ")"

    def __array__(self) -> FloatArray:
        from vector.backends.numpy import MomentumNumpy4D

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
        self.azimuthal = AzimuthalObjectXY(px, self.py)

    @property
    def py(self) -> float:
        return super().py

    @py.setter
    def py(self, py: float) -> None:
        self.azimuthal = AzimuthalObjectXY(self.px, py)

    @property
    def pt(self) -> float:
        return super().pt

    @pt.setter
    def pt(self, pt: float) -> None:
        self.azimuthal = AzimuthalObjectRhoPhi(pt, self.phi)

    @property
    def pz(self) -> float:
        return super().pz

    @pz.setter
    def pz(self, pz: float) -> None:
        self.longitudinal = LongitudinalObjectZ(pz)

    @property
    def E(self) -> float:
        return super().E

    @E.setter
    def E(self, E: float) -> None:
        self.temporal = TemporalObjectT(E)

    @property
    def e(self) -> float:
        return super().e

    @e.setter
    def e(self, e: float) -> None:
        self.temporal = TemporalObjectT(e)

    @property
    def energy(self) -> float:
        return super().energy

    @energy.setter
    def energy(self, energy: float) -> None:
        self.temporal = TemporalObjectT(energy)

    @property
    def M(self) -> float:
        return super().M

    @M.setter
    def M(self, M: float) -> None:
        self.temporal = TemporalObjectTau(M)

    @property
    def m(self) -> float:
        return super().m

    @m.setter
    def m(self, m: float) -> None:
        self.temporal = TemporalObjectTau(m)

    @property
    def mass(self) -> float:
        return super().mass

    @mass.setter
    def mass(self, mass: float) -> None:
        self.temporal = TemporalObjectTau(mass)


def _is_type_safe(coordinates: dict[str, typing.Any]) -> bool:
    for _, value in coordinates.items():
        if not issubclass(type(value), numbers.Real) or isinstance(value, bool):
            return False
    return True


def _gather_coordinates(
    planar_class: type[VectorObject2D],
    spatial_class: type[VectorObject3D],
    lorentz_class: type[VectorObject4D],
    coordinates: dict[str, typing.Any],
) -> typing.Any:
    """
    Helper function for :func:`vector.backends.object.obj`.

    Constructs and returns a 2D, 3D, or 4D ``VectorObject`` or ``MomentumObject`` with
    the provided coordinates (dictionary), planar (``VectorObject2D`` or ``MomentumObject2D``),
    spatial (``VectorObject3D`` or ``MomentumObject3D``), and lorentz
    (``VectorObject4D`` or ``MomentumObject4D``) classes.
    """
    azimuthal: None | (AzimuthalObjectXY | AzimuthalObjectRhoPhi) = None

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

    longitudinal: None | (
        LongitudinalObjectZ | LongitudinalObjectTheta | LongitudinalObjectEta
    ) = None

    if "z" in coordinates:
        if "theta" in coordinates or "eta" in coordinates:
            raise TypeError("specify z= or theta= or eta=, but not more than one")
        longitudinal = LongitudinalObjectZ(coordinates.pop("z"))
    elif "theta" in coordinates:
        if "eta" in coordinates:
            raise TypeError("specify z= or theta= or eta=, but not more than one")
        longitudinal = LongitudinalObjectTheta(coordinates.pop("theta"))
    elif "eta" in coordinates:
        longitudinal = LongitudinalObjectEta(coordinates.pop("eta"))

    temporal: TemporalObjectT | TemporalObjectTau | None = None

    if "t" in coordinates:
        if "tau" in coordinates:
            raise TypeError("specify t= or tau=, but not more than one")
        temporal = TemporalObjectT(coordinates.pop("t"))
    elif "tau" in coordinates:
        temporal = TemporalObjectTau(coordinates.pop("tau"))

    if not coordinates:
        if azimuthal is not None and longitudinal is None and temporal is None:
            return planar_class(azimuthal=azimuthal)
        if azimuthal is not None and longitudinal is not None and temporal is None:
            return spatial_class(azimuthal=azimuthal, longitudinal=longitudinal)
        if azimuthal is not None and longitudinal is not None and temporal is not None:
            return lorentz_class(
                azimuthal=azimuthal, longitudinal=longitudinal, temporal=temporal
            )

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
def obj(*, x: float, y: float) -> VectorObject2D: ...


@typing.overload
def obj(*, x: float, py: float) -> MomentumObject2D: ...


@typing.overload
def obj(*, px: float, y: float) -> MomentumObject2D: ...


@typing.overload
def obj(*, px: float, py: float) -> MomentumObject2D: ...


@typing.overload
def obj(*, rho: float, phi: float) -> VectorObject2D: ...


@typing.overload
def obj(*, pt: float, phi: float) -> MomentumObject2D: ...


@typing.overload
def obj(*, x: float, y: float, z: float) -> VectorObject3D: ...


@typing.overload
def obj(*, x: float, y: float, pz: float) -> MomentumObject3D: ...


@typing.overload
def obj(*, x: float, py: float, z: float) -> MomentumObject3D: ...


@typing.overload
def obj(*, x: float, py: float, pz: float) -> MomentumObject3D: ...


@typing.overload
def obj(*, px: float, y: float, z: float) -> MomentumObject3D: ...


@typing.overload
def obj(*, px: float, y: float, pz: float) -> MomentumObject3D: ...


@typing.overload
def obj(*, px: float, py: float, z: float) -> MomentumObject3D: ...


@typing.overload
def obj(*, px: float, py: float, pz: float) -> MomentumObject3D: ...


@typing.overload
def obj(*, rho: float, phi: float, z: float) -> VectorObject3D: ...


@typing.overload
def obj(*, rho: float, phi: float, pz: float) -> MomentumObject3D: ...


@typing.overload
def obj(*, pt: float, phi: float, z: float) -> MomentumObject3D: ...


@typing.overload
def obj(*, pt: float, phi: float, pz: float) -> MomentumObject3D: ...


@typing.overload
def obj(*, x: float, y: float, theta: float) -> VectorObject3D: ...


@typing.overload
def obj(*, x: float, py: float, theta: float) -> MomentumObject3D: ...


@typing.overload
def obj(*, px: float, y: float, theta: float) -> MomentumObject3D: ...


@typing.overload
def obj(*, px: float, py: float, theta: float) -> MomentumObject3D: ...


@typing.overload
def obj(*, rho: float, phi: float, theta: float) -> VectorObject3D: ...


@typing.overload
def obj(*, pt: float, phi: float, theta: float) -> MomentumObject3D: ...


@typing.overload
def obj(*, x: float, y: float, eta: float) -> VectorObject3D: ...


@typing.overload
def obj(*, x: float, py: float, eta: float) -> MomentumObject3D: ...


@typing.overload
def obj(*, px: float, y: float, eta: float) -> MomentumObject3D: ...


@typing.overload
def obj(*, px: float, py: float, eta: float) -> MomentumObject3D: ...


@typing.overload
def obj(*, rho: float, phi: float, eta: float) -> VectorObject3D: ...


@typing.overload
def obj(*, pt: float, phi: float, eta: float) -> MomentumObject3D: ...


@typing.overload
def obj(*, x: float, y: float, z: float, t: float) -> VectorObject4D: ...


@typing.overload
def obj(*, x: float, y: float, pz: float, t: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, z: float, t: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, pz: float, t: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, z: float, t: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, pz: float, t: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, z: float, t: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, pz: float, t: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, z: float, t: float) -> VectorObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, pz: float, t: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pt: float, phi: float, z: float, t: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pt: float, phi: float, pz: float, t: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, theta: float, t: float) -> VectorObject4D: ...


@typing.overload
def obj(*, x: float, py: float, theta: float, t: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, theta: float, t: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, theta: float, t: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, theta: float, t: float) -> VectorObject4D: ...


@typing.overload
def obj(*, pt: float, phi: float, theta: float, t: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, eta: float, t: float) -> VectorObject4D: ...


@typing.overload
def obj(*, x: float, py: float, eta: float, t: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, eta: float, t: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, eta: float, t: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, eta: float, t: float) -> VectorObject4D: ...


@typing.overload
def obj(*, pt: float, phi: float, eta: float, t: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, z: float, tau: float) -> VectorObject4D: ...


@typing.overload
def obj(*, x: float, y: float, pz: float, tau: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, z: float, tau: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, pz: float, tau: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, z: float, tau: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, pz: float, tau: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, z: float, tau: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, pz: float, tau: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, z: float, tau: float) -> VectorObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, pz: float, tau: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, ptau: float, phi: float, z: float, tau: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, ptau: float, phi: float, pz: float, tau: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, theta: float, tau: float) -> VectorObject4D: ...


@typing.overload
def obj(*, x: float, py: float, theta: float, tau: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, theta: float, tau: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, theta: float, tau: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, theta: float, tau: float) -> VectorObject4D: ...


@typing.overload
def obj(*, ptau: float, phi: float, theta: float, tau: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, eta: float, tau: float) -> VectorObject4D: ...


@typing.overload
def obj(*, x: float, py: float, eta: float, tau: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, eta: float, tau: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, eta: float, tau: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, eta: float, tau: float) -> VectorObject4D: ...


@typing.overload
def obj(*, ptau: float, phi: float, eta: float, tau: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, z: float, E: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, pz: float, E: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, z: float, E: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, pz: float, E: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, z: float, E: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, pz: float, E: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, z: float, E: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, pz: float, E: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, z: float, E: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, pz: float, E: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pE: float, phi: float, z: float, E: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pE: float, phi: float, pz: float, E: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, theta: float, E: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, theta: float, E: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, theta: float, E: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, theta: float, E: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, theta: float, E: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pE: float, phi: float, theta: float, E: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, eta: float, E: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, eta: float, E: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, eta: float, E: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, eta: float, E: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, eta: float, E: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pE: float, phi: float, eta: float, E: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, z: float, e: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, pz: float, e: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, z: float, e: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, pz: float, e: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, z: float, e: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, pz: float, e: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, z: float, e: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, pz: float, e: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, z: float, e: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, pz: float, e: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pe: float, phi: float, z: float, e: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pe: float, phi: float, pz: float, e: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, theta: float, e: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, theta: float, e: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, theta: float, e: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, theta: float, e: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, theta: float, e: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pe: float, phi: float, theta: float, e: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, eta: float, e: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, eta: float, e: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, eta: float, e: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, eta: float, e: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, eta: float, e: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pe: float, phi: float, eta: float, e: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, z: float, energy: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, pz: float, energy: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, z: float, energy: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, pz: float, energy: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, z: float, energy: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, pz: float, energy: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, z: float, energy: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, pz: float, energy: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, z: float, energy: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, pz: float, energy: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pt: float, phi: float, z: float, energy: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pt: float, phi: float, pz: float, energy: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, theta: float, energy: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, theta: float, energy: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, theta: float, energy: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, theta: float, energy: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, theta: float, energy: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pt: float, phi: float, theta: float, energy: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, eta: float, energy: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, eta: float, energy: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, eta: float, energy: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, eta: float, energy: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, eta: float, energy: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pt: float, phi: float, eta: float, energy: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, z: float, M: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, pz: float, M: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, z: float, M: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, pz: float, M: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, z: float, M: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, pz: float, M: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, z: float, M: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, pz: float, M: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, z: float, M: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, pz: float, M: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pM: float, phi: float, z: float, M: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pM: float, phi: float, pz: float, M: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, theta: float, M: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, theta: float, M: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, theta: float, M: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, theta: float, M: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, theta: float, M: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pM: float, phi: float, theta: float, M: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, eta: float, M: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, eta: float, M: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, eta: float, M: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, eta: float, M: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, eta: float, M: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pM: float, phi: float, eta: float, M: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, z: float, m: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, pz: float, m: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, z: float, m: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, pz: float, m: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, z: float, m: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, pz: float, m: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, z: float, m: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, pz: float, m: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, z: float, m: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, pz: float, m: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pm: float, phi: float, z: float, m: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pm: float, phi: float, pz: float, m: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, theta: float, m: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, theta: float, m: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, theta: float, m: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, theta: float, m: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, theta: float, m: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pm: float, phi: float, theta: float, m: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, eta: float, m: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, eta: float, m: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, eta: float, m: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, eta: float, m: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, eta: float, m: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pm: float, phi: float, eta: float, m: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, z: float, mass: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, pz: float, mass: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, z: float, mass: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, pz: float, mass: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, z: float, mass: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, pz: float, mass: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, z: float, mass: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, pz: float, mass: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, z: float, mass: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, pz: float, mass: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pt: float, phi: float, z: float, mass: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pt: float, phi: float, pz: float, mass: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, theta: float, mass: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, theta: float, mass: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, theta: float, mass: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, theta: float, mass: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, theta: float, mass: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pt: float, phi: float, theta: float, mass: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, y: float, eta: float, mass: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, x: float, py: float, eta: float, mass: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, y: float, eta: float, mass: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, px: float, py: float, eta: float, mass: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, rho: float, phi: float, eta: float, mass: float) -> MomentumObject4D: ...


@typing.overload
def obj(*, pt: float, phi: float, eta: float, mass: float) -> MomentumObject4D: ...


def obj(**coordinates: float) -> VectorObject:
    """
    Constructs a single ``Object`` type vector, whose type is determined by the keyword-only
    arguments to this function.

    Allowed combinations are:

    - (2D) ``x``, ``y``
    - (2D) ``rho``, ``phi``
    - (3D) ``x``, ``y``, ``z``
    - (3D) ``x``, ``y``, ``theta``
    - (3D) ``x``, ``y``, ``eta``
    - (3D) ``rho``, ``phi``, ``z``
    - (3D) ``rho``, ``phi``, ``theta``
    - (3D) ``rho``, ``phi``, ``eta``
    - (4D) ``x``, ``y``, ``z``, ``t``
    - (4D) ``x``, ``y``, ``z``, ``tau```
    - (4D) ``x``, ``y``, ``theta``, ``t```
    - (4D) ``x``, ``y``, ``theta``, ``tau```
    - (4D) ``x``, ``y``, ``eta``, ``t```
    - (4D) ``x``, ``y``, ``eta``, ``tau```
    - (4D) ``rho``, ``phi``, ``z``, ``t```
    - (4D) ``rho``, ``phi``, ``z``, ``tau```
    - (4D) ``rho``, ``phi``, ``theta``, ``t```
    - (4D) ``rho``, ``phi``, ``theta``, ``tau```
    - (4D) ``rho``, ``phi``, ``eta``, ``t```
    - (4D) ``rho``, ``phi``, ``eta``, ``tau```

    in which

    - ``px`` may be substituted for ``x``
    - ``py`` may be substituted for ``y``
    - ``pt`` may be substituted for ``rho``
    - ``pz`` may be substituted for ``z``
    - ``E`` may be substituted for ``t``
    - ``e`` may be substituted for ``t``
    - ``energy`` may be substituted for ``t``
    - ``M`` may be substituted for ``tau``
    - ``m`` may be substituted for ``tau``
    - ``mass`` may be substituted for ``tau``

    to make the vector a momentum vector.

    Alternatively, the :class:`vector.backends.object.VectorObject2D`,
    :class:`vector.backends.object.VectorObject3D`, and
    :class:`vector.backends.object.VectorObject4D` classes (with momentum
    subclasses) have explicit constructors:

    - :meth:`vector.backends.object.VectorObject2D.from_xy`
    - :meth:`vector.backends.object.VectorObject2D.from_rhophi`

    - :meth:`vector.backends.object.VectorObject3D.from_xyz`
    - :meth:`vector.backends.object.VectorObject3D.from_xytheta`
    - :meth:`vector.backends.object.VectorObject3D.from_xyeta`
    - :meth:`vector.backends.object.VectorObject3D.from_rhophiz`
    - :meth:`vector.backends.object.VectorObject3D.from_rhophitheta`
    - :meth:`vector.backends.object.VectorObject3D.from_rhophieta`

    - :meth:`vector.backends.object.VectorObject4D.from_xyzt`
    - :meth:`vector.backends.object.VectorObject4D.from_xyztau`
    - :meth:`vector.backends.object.VectorObject4D.from_xythetat`
    - :meth:`vector.backends.object.VectorObject4D.from_xythetatau`
    - :meth:`vector.backends.object.VectorObject4D.from_xyetat`
    - :meth:`vector.backends.object.VectorObject4D.from_xyetatau`
    - :meth:`vector.backends.object.VectorObject4D.from_rhophizt`
    - :meth:`vector.backends.object.VectorObject4D.from_rhophiztau`
    - :meth:`vector.backends.object.VectorObject4D.from_rhophithetat`
    - :meth:`vector.backends.object.VectorObject4D.from_rhophithetatau`
    - :meth:`vector.backends.object.VectorObject4D.from_rhophietat`
    - :meth:`vector.backends.object.VectorObject4D.from_rhophietatau`
    """
    is_momentum = False
    generic_coordinates = {}

    if not _is_type_safe(coordinates):
        raise TypeError("a coordinate must be of the type int or float")

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
    if "e" in coordinates:
        is_momentum = True
        generic_coordinates["t"] = coordinates.pop("e")
    if "energy" in coordinates and "t" not in generic_coordinates:
        is_momentum = True
        generic_coordinates["t"] = coordinates.pop("energy")
    if "M" in coordinates:
        is_momentum = True
        generic_coordinates["tau"] = coordinates.pop("M")
    if "m" in coordinates:
        is_momentum = True
        generic_coordinates["tau"] = coordinates.pop("m")
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
VectorObject2D.MomentumClass = MomentumObject2D

MomentumObject2D.ProjectionClass2D = MomentumObject2D
MomentumObject2D.ProjectionClass3D = MomentumObject3D
MomentumObject2D.ProjectionClass4D = MomentumObject4D
MomentumObject2D.GenericClass = VectorObject2D
MomentumObject2D.MomentumClass = MomentumObject2D

VectorObject3D.ProjectionClass2D = VectorObject2D
VectorObject3D.ProjectionClass3D = VectorObject3D
VectorObject3D.ProjectionClass4D = VectorObject4D
VectorObject3D.GenericClass = VectorObject3D
VectorObject3D.MomentumClass = MomentumObject3D

MomentumObject3D.ProjectionClass2D = MomentumObject2D
MomentumObject3D.ProjectionClass3D = MomentumObject3D
MomentumObject3D.ProjectionClass4D = MomentumObject4D
MomentumObject3D.GenericClass = VectorObject3D
MomentumObject3D.MomentumClass = MomentumObject3D

VectorObject4D.ProjectionClass2D = VectorObject2D
VectorObject4D.ProjectionClass3D = VectorObject3D
VectorObject4D.ProjectionClass4D = VectorObject4D
VectorObject4D.GenericClass = VectorObject4D
VectorObject4D.MomentumClass = MomentumObject4D

MomentumObject4D.ProjectionClass2D = MomentumObject2D
MomentumObject4D.ProjectionClass3D = MomentumObject3D
MomentumObject4D.ProjectionClass4D = MomentumObject4D
MomentumObject4D.GenericClass = VectorObject4D
MomentumObject4D.MomentumClass = MomentumObject4D
