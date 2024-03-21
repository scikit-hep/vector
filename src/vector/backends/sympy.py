from __future__ import annotations

import numbers
import typing

import numpy
import sympy

import vector
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
    Spatial,
    SpatialMomentum,
    Temporal,
    TemporalT,
    TemporalTau,
    Vector,
    Vector2D,
    Vector3D,
    Vector4D,
    _aztype,
    _coordinate_class_to_names,
    _ltype,
    _repr_generic_to_momentum,
    _repr_momentum_to_generic,
    _ttype,
)
from vector._typeutils import FloatArray, ScalarCollection


class CoordinatesSympy:
    """Coordinates class for the SymPy backend."""

    lib = sympy


class AzimuthalSympy(CoordinatesSympy, Azimuthal):
    """Azimuthal class for the SymPy backend."""

    ObjectClass: type[vector.backends.object.AzimuthalObject]


class LongitudinalSympy(CoordinatesSympy, Longitudinal):
    """Longitudinal class for the SymPy backend."""

    ObjectClass: type[vector.backends.object.LongitudinalObject]


class TemporalSympy(CoordinatesSympy, Temporal):
    """Temporal class for the SymPy backend."""

    ObjectClass: type[vector.backends.object.TemporalObject]


class AzimuthalSympyXY(sympy.Array, AzimuthalSympy, AzimuthalXY):
    """
    Class for the ``rho`` and ``phi`` (azimuthal) coordinates of SymPy backend.
    Creates a structured SymPy array and returns it as an AzimuthalSympyXY object.

    Examples:
        >>> import vector
        >>> vector.backends.sympy.AzimuthalSympyXY([1, 2], [3, 4])
        AzimuthalSympyXY(x=[1, 2], y=[3, 4])
    """

    ObjectClass = vector.backends.object.AzimuthalObjectXY
    _IS_MOMENTUM = False

    def __new__(cls, x: sympy.Array, y: sympy.Array, **kwargs: typing.Any):
        obj = super().__new__(cls, [[_x, _y] for _x, _y in zip(x, y)], **kwargs)
        obj.x = sympy.Array(x)
        obj.y = sympy.Array(y)
        return obj

    def __init__(self, x: sympy.Array, y: sympy.Array, **kwargs):
        super().__init__()

    def __repr__(self):
        return f"AzimuthalSympyXY(x={self.x}, y={self.y})"

    @property
    def elements(self) -> tuple[sympy.Array, sympy.Array]:
        """
        Azimuthal coordinates (``x`` and ``y``) as a tuple.

        Each coordinate is a SymPy array of values and not a vector.

        Examples:
            >>> import vector
            >>> vec = vector.backends.sympy.AzimuthalSympyXY([1, 2], [3, 4])
            >>> vec.elements
            ([1, 2], [3, 4])
        """
        return (self.x, self.y)


class AzimuthalSympyRhoPhi(sympy.Array, AzimuthalSympy, AzimuthalRhoPhi):  # type: ignore[misc]
    """
    Class for the ``rho`` and ``phi`` (azimuthal) coordinates of SymPy backend.
    Creates a structured SymPy array and returns it as an AzimuthalSympyXY object.

    Examples:
        >>> import vector
        >>> vector.backends.sympy.AzimuthalSympyRhoPhi([1, 2], [3, 4])
        AzimuthalSympyRhoPhi(rho=[1, 2], phi=[3, 4])
    """

    ObjectClass = vector.backends.object.AzimuthalObjectRhoPhi
    _IS_MOMENTUM = False

    def __new__(cls, rho: sympy.Array, phi: sympy.Array, **kwargs: typing.Any):
        obj = super().__new__(
            cls, [[_rho, _phi] for _rho, _phi in zip(rho, phi)], **kwargs
        )
        obj.rho = sympy.Array(rho)
        obj.phi = sympy.Array(phi)
        return obj

    def __init__(self, rho: sympy.Array, phi: sympy.Array, **kwargs):
        super().__init__()

    def __repr__(self):
        return f"AzimuthalSympyRhoPhi(rho={self.rho}, phi={self.phi})"

    @property
    def elements(self) -> tuple[sympy.Array, sympy.Array]:
        """
        Azimuthal coordinates (``rho`` and ``phi``) as a tuple.

        Each coordinate is a SymPy array of values and not a vector.

        Examples:
            >>> import vector
            >>> vec = vector.backends.sympy.AzimuthalSympyRhoPhi([1, 2], [3, 4])
            >>> vec.elements
            ([1, 2], [3, 4])
        """
        return (self.rho, self.phi)


class LongitudinalSympyZ(sympy.Array, LongitudinalSympy, LongitudinalZ):  # type: ignore[misc]
    """
    Class for the ``z`` (longitudinal) coordinate of SymPy backend.
    Creates a structured SymPy array and returns it as a LongitudinalSympyZ object.

    Examples:
        >>> import vector
        >>> vector.backends.sympy.LongitudinalSympyZ([1, 2])
        LongitudinalSympyZ(z=[1, 2])
    """

    ObjectClass = vector.backends.object.LongitudinalObjectZ
    _IS_MOMENTUM = False

    def __new__(cls, z: sympy.Array, **kwargs: typing.Any):
        obj = super().__new__(cls, z, **kwargs)
        obj.z = sympy.Array(z)
        return obj

    def __init__(self, z: sympy.Array, **kwargs):
        super().__init__()

    def __repr__(self):
        return f"LongitudinalSympyZ(z={self.z})"

    @property
    def elements(self) -> tuple[sympy.Array]:
        """
        Longitudinal coordinates (``z``) as a tuple.

        Each coordinate is a SymPy array of values and not a vector.

        Examples:
            >>> import vector
            >>> vec = vector.backends.sympy.LongitudinalSympyZ([1, 2])
            >>> vec.elements
            ([1, 2],)
        """
        return (self.z,)


class LongitudinalSympyTheta(sympy.Array, LongitudinalSympy, LongitudinalTheta):  # type: ignore[misc]
    """
    Class for the ``theta`` (longitudinal) coordinate of SymPy backend.
    Creates a structured SymPy array and returns it as a LongitudinalSympyTheta object.

    Examples:
        >>> import vector
        >>> vector.backends.sympy.LongitudinalSympyTheta([1, 2])
        LongitudinalSympyTheta(theta=[1, 2])
    """

    ObjectClass = vector.backends.object.LongitudinalObjectTheta
    _IS_MOMENTUM = False

    def __new__(cls, theta: sympy.Array, **kwargs: typing.Any):
        obj = super().__new__(cls, [theta], **kwargs)
        obj.theta = sympy.Array(theta)
        return obj

    def __init__(self, theta: sympy.Array, **kwargs):
        super().__init__()

    def __repr__(self):
        return f"LongitudinalSympyTheta(theta={self.theta})"

    @property
    def elements(self) -> tuple[sympy.Array]:
        """
        Longitudinal coordinates (``theta``) as a tuple.

        Each coordinate is a SymPy array of values and not a vector.

        Examples:
            >>> import vector
            >>> vec = vector.backends.sympy.LongitudinalSympyTheta([1, 2])
            >>> vec.elements
            ([1, 2],)
        """
        return (self.theta,)


class LongitudinalSympyEta(sympy.Array, LongitudinalSympy, LongitudinalEta):  # type: ignore[misc]
    """
    Class for the ``eta`` (longitudinal) coordinate of SymPy backend.
    Creates a structured SymPy array and returns it as a LongitudinalSympyEta object.

    Examples:
        >>> import vector
        >>> vector.backends.sympy.LongitudinalSympyEta([1, 2])
        LongitudinalSympyEta(eta=[1, 2])
    """

    ObjectClass = vector.backends.object.LongitudinalObjectEta
    _IS_MOMENTUM = False

    def __new__(cls, eta: sympy.Array, **kwargs: typing.Any):
        obj = super().__new__(cls, [eta], **kwargs)
        obj.eta = sympy.Array(eta)
        return obj

    def __init__(self, eta: sympy.Array, **kwargs):
        super().__init__()

    def __repr__(self):
        return f"LongitudinalSympyEta(eta={self.eta})"

    @property
    def elements(self) -> tuple[sympy.Array]:
        """
        Longitudinal coordinates (``eta``) as a tuple.

        Each coordinate is a SymPy array of values and not a vector.

        Examples:
            >>> import vector
            >>> vec = vector.backends.sympy.LongitudinalSympyTheta([1, 2])
            >>> vec.elements
            ([1, 2],)
        """
        return (self.eta,)


class TemporalSympyT(sympy.Array, TemporalSympy, TemporalT):  # type: ignore[misc]
    """
    Class for the ``t`` (temporal) coordinate of SymPy backend.
    Creates a structured SymPy array and returns it as a TemporalSympyT object.

    Examples:
        >>> import vector
        >>> vector.backends.sympy.TemporalSympyT([1, 2])
        TemporalSympyT(t=[1, 2])
    """

    ObjectClass = vector.backends.object.TemporalObjectT
    _IS_MOMENTUM = False

    def __new__(cls, t: sympy.Array, **kwargs: typing.Any):
        obj = super().__new__(cls, [t], **kwargs)
        obj.t = sympy.Array(t)
        return obj

    def __init__(self, t: sympy.Array, **kwargs):
        super().__init__()

    def __repr__(self):
        return f"TemporalSympyT(t={self.t})"

    @property
    def elements(self) -> tuple[sympy.Array]:
        """
        Temporal coordinates (``t``) as a tuple.

        Each coordinate is a SymPy array of values and not a vector.

        Examples:
            >>> import vector
            >>> vec = vector.backends.sympy.TemporalSympyT([1, 2])
            >>> vec.elements
            ([1, 2],)
        """
        return (self.t,)


class TemporalSympyTau(sympy.Array, TemporalSympy, TemporalTau):  # type: ignore[misc]
    """
    Class for the ``tau`` (temporal) coordinate of SymPy backend.
    Creates a structured SymPy array and returns it as a TemporalSympyTau object.

    Examples:
        >>> import vector
        >>> vector.backends.sympy.TemporalSympyTau([1, 2])
        TemporalSympyTau(tau=[1, 2])
    """

    ObjectClass = vector.backends.object.TemporalObjectTau
    _IS_MOMENTUM = False

    def __new__(cls, tau: sympy.Array, **kwargs: typing.Any):
        obj = super().__new__(cls, [tau], **kwargs)
        obj.tau = sympy.Array(tau)
        return obj

    def __init__(self, tau: sympy.Array, **kwargs):
        super().__init__()

    def __repr__(self):
        return f"TemporalSympyTau(tau={self.tau})"

    @property
    def elements(self) -> tuple[sympy.Array]:
        """
        Temporal coordinates (``tau``) as a tuple.

        Each coordinate is a SymPy array of values and not a vector.

        Examples:
            >>> import vector
            >>> vec = vector.backends.sympy.TemporalSympyTau([1, 2])
            >>> vec.elements
            ([1, 2],)
        """
        return (self.tau,)


_coord_sympy_type = {
    AzimuthalXY: AzimuthalSympyXY,
    AzimuthalRhoPhi: AzimuthalSympyRhoPhi,
    LongitudinalZ: LongitudinalSympyZ,
    LongitudinalTheta: LongitudinalSympyTheta,
    LongitudinalEta: LongitudinalSympyEta,
    TemporalT: TemporalSympyT,
    TemporalTau: TemporalSympyTau,
}


def _is_type_safe(coordinates):
    if not all(isinstance(coord, list) for coord in coordinates.values()):
        raise TypeError("coordinates must be a list")
    len_init = len(coordinates[next(iter(coordinates))])
    if not all(len(x) == len_init for x in coordinates.values()):
        raise TypeError("each list should be of the same length")
    for _, value in coordinates.items():
        if not all(issubclass(type(v), numbers.Real) for v in value) or any(
            isinstance(v, bool) for v in value
        ):
            raise TypeError("coordinates must be int or float")


def _toarrays(
    result: tuple[ScalarCollection, ...] | ScalarCollection,
) -> tuple[list[int | float], ...]:
    """
    Converts a tuple of values to a tuple of
    ``sympy.tensor.array.ImmutableDenseNDimArray``s.

    Args:
        result (tuple): A tuple of values to be converted.

    Returns:
        tuple: A tuple of ``sympy.tensor.array.ImmutableDenseNDimArray``.
    """
    istuple = True
    if not isinstance(result, tuple):
        istuple = False
        result = (result,)
    result = tuple(
        x if isinstance(x, sympy.Array) else sympy.Array([x] * (len(result[0])))
        for x in result
    )
    if istuple:
        return result
    else:
        return result[0]


class VectorSympy(Vector):
    """Mixin class for Sympy vectors."""

    lib = sympy


class VectorSympy2D(sympy.Array, VectorSympy, Planar, Vector2D):  # type: ignore[misc]
    """
    Two dimensional vector class for the SymPy backend.

    Examples:
        >>> import vector
        >>> vec = vector.VectorSympy2D(x=[1, 2], y=[3, 4])
        >>> vec.x, vec.y
        ([1, 2], [3, 4])
        >>> vec = vector.VectorSympy2D(rho=[1, 2], phi=[3, 4])
        >>> vec.rho, vec.phi
        ([1, 2], [3, 4])
        >>> vec = vector.VectorObject2D(azimuthal=vector.backends.sympy.AzimuthalSympyXY([1, 2], [3, 4]))
        >>> vec.x, vec.y
        ([1, 2], [3, 4])

    For two dimensional momentum SymPy vectors, see
    :class:`vector.backends.sympy.MomentumObject2D`.
    """

    ObjectClass = vector.backends.object.VectorObject2D
    _IS_MOMENTUM = False

    __slots__ = ("azimuthal",)
    azimuthal: AzimuthalSympy

    def __new__(
        cls, azimuthal: AzimuthalSympy | None = None, **kwargs: typing.Any
    ) -> sympy.Array:
        for k, v in kwargs.copy().items():
            kwargs.pop(k)
            kwargs[_repr_momentum_to_generic.get(k, k)] = v

        if not kwargs and azimuthal is not None:
            obj = super().__new__(
                cls,
                [[x, y] for x, y in zip(azimuthal.elements[0], azimuthal.elements[1])],
            )
            obj.azimuthal = azimuthal
        elif kwargs and azimuthal is None:
            _is_type_safe(kwargs)
            if set(kwargs) == {"x", "y"}:
                obj = super().__new__(
                    cls, [[x, y] for x, y in zip(kwargs["x"], kwargs["y"])]
                )
                obj.azimuthal = AzimuthalSympyXY(kwargs["x"], kwargs["y"])
            elif set(kwargs) == {"rho", "phi"}:
                obj = super().__new__(
                    cls, [[rho, phi] for rho, phi in zip(kwargs["rho"], kwargs["phi"])]
                )
                obj.azimuthal = AzimuthalSympyRhoPhi(kwargs["rho"], kwargs["phi"])
            else:
                complaint = """unrecognized combination of coordinates, allowed combinations are:\n
                    x= y=
                    rho= phi=""".replace("                    ", "    ")
                if not cls._IS_MOMENTUM:
                    raise TypeError(complaint)
                else:
                    raise TypeError(f"{complaint}\n\nor their momentum equivalents")
        else:
            raise TypeError("must give Azimuthal if not giving keyword arguments")
        return obj

    def __init__(self, azimuthal: AzimuthalSympy | None = None, **kwargs):
        super().__init__()

    def __repr__(self) -> str:
        aznames = _coordinate_class_to_names[_aztype(self)]
        out = [f"{x}={getattr(self.azimuthal, x)}" for x in aznames]
        return "VectorSympy2D(" + ", ".join(out) + ")"

    def __array__(self) -> FloatArray:
        from vector.backends.numpy import VectorNumpy2D

        return VectorNumpy2D(
            self.azimuthal.elements,
            dtype=[
                (x, numpy.float64) for x in _coordinate_class_to_names[_aztype(self)]
            ],
        )

    @property
    def x(self) -> float:
        return super().x

    @x.setter
    def x(self, x: float) -> None:
        self.azimuthal = AzimuthalSympyXY(x, self.y)

    @property
    def y(self) -> float:
        return super().y

    @y.setter
    def y(self, y: float) -> None:
        self.azimuthal = AzimuthalSympyXY(self.x, y)

    @property
    def rho(self) -> float:
        return super().rho

    @rho.setter
    def rho(self, rho: float) -> None:
        self.azimuthal = AzimuthalSympyRhoPhi(rho, self.phi)

    @property
    def phi(self) -> float:
        return super().phi

    @phi.setter
    def phi(self, phi: float) -> None:
        self.azimuthal = AzimuthalSympyRhoPhi(self.rho, phi)

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
            result = _toarrays(result)
            azcoords = _coord_sympy_type[returns[0]](result[0], result[1])
            return cls.ProjectionClass2D(azimuthal=azcoords)

        elif (
            len(returns) == 2
            or (len(returns) == 3 and returns[2] is None)
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
        ):
            result = _toarrays(result)
            azcoords = _coord_sympy_type[returns[0]](result[0], result[1])
            lcoords = _coord_sympy_type[returns[1]](result[2])
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
            result = _toarrays(result)
            azcoords = _coord_sympy_type[returns[0]](result[0], result[1])
            lcoords = _coord_sympy_type[returns[1]](result[2])
            tcoords = _coord_sympy_type[returns[2]](result[3])
            return cls.ProjectionClass4D(
                azimuthal=azcoords, longitudinal=lcoords, temporal=tcoords
            )

        else:
            raise AssertionError(repr(returns))


class MomentumSympy2D(PlanarMomentum, VectorSympy2D):
    """
    Two dimensional momentum vector class for the SymPy backend.

    Examples:
        >>> import vector
        >>> vec = vector.MomentumSympy2D(px=[1, 2], py=[3, 4])
        >>> vec.px, vec.py
        ([1, 2], [3, 4])
        >>> vec = vector.MomentumSympy2D(pt=[1, 2], phi=[3, 4])
        >>> vec.pt, vec.phi
        ([1, 2], [3, 4])
        >>> vec = vector.MomentumSympy2D(azimuthal=vector.backends.sympy.AzimuthalSympyXY([1, 2], [3, 4]))
        >>> vec.px, vec.py
        ([1, 2], [3, 4])

    For two dimensional SymPy vectors, see
    :class:`vector.backends.object.VectorSympy2D`.
    """

    ObjectClass = vector.backends.object.MomentumObject2D
    _IS_MOMENTUM = True

    def __repr__(self) -> str:
        aznames = _coordinate_class_to_names[_aztype(self)]
        out = []
        for x in aznames:
            y = _repr_generic_to_momentum.get(x, x)
            out.append(f"{y}={getattr(self.azimuthal, x)}")
        return "MomentumSympy2D(" + ", ".join(out) + ")"

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
        self.azimuthal = AzimuthalSympyXY(px, self.py)

    @property
    def py(self) -> float:
        return super().py

    @py.setter
    def py(self, py: float) -> None:
        self.azimuthal = AzimuthalSympyXY(self.px, py)

    @property
    def pt(self) -> float:
        return super().pt

    @pt.setter
    def pt(self, pt: float) -> None:
        self.azimuthal = AzimuthalSympyRhoPhi(pt, self.phi)


class VectorSympy3D(sympy.Array, VectorSympy, Spatial, Vector3D):
    """
    Three dimensional vector class for the SymPy backend.

    Examples:
        >>> import vector
        >>> vec = vector.VectorSympy3D(x=[1, 2], y=[3, 4], z=[5, 6])
        >>> vec.x, vec.y, vec.z
        ([1, 2], [3, 4], [5, 6])
        >>> vec = vector.VectorSympy3D(rho=[1, 2], phi=[3, 4], eta=[5, 6])
        >>> vec.rho, vec.phi, vec.eta
        ([1, 2], [3, 4], [5, 6])
        >>> vec = vector.VectorSympy3D(
        ...     azimuthal=vector.backends.sympy.AzimuthalSympyXY([1, 2], [3, 4]),
        ...     longitudinal=vector.backends.sympy.LongitudinalSympyTheta([5, 6])
        ... )
        >>> vec.x, vec.y, vec.theta
        ([1, 2], [3, 4], [5, 6])

    For three dimensional momentum SymPy vectors, see
    :class:`vector.backends.object.MomentumSympy3D`.
    """

    ObjectClass = vector.backends.object.VectorObject3D
    _IS_MOMENTUM = False

    __slots__ = ("azimuthal", "longitudinal")

    azimuthal: AzimuthalSympy
    longitudinal: LongitudinalSympy

    def __new__(
        cls,
        azimuthal: AzimuthalSympy | None = None,
        longitudinal: LongitudinalSympy | None = None,
        **kwargs: float,
    ) -> None:
        for k, v in kwargs.copy().items():
            kwargs.pop(k)
            kwargs[_repr_momentum_to_generic.get(k, k)] = v

        if not kwargs and azimuthal is not None and longitudinal is not None:
            obj = super().__new__(
                cls,
                [
                    [x, y, z]
                    for x, y, z in zip(
                        azimuthal.elements[0],
                        azimuthal.elements[1],
                        longitudinal.elements[0],
                    )
                ],
            )
            obj.azimuthal = azimuthal
            obj.longitudinal = longitudinal
        elif kwargs and azimuthal is None and longitudinal is None:
            _is_type_safe(kwargs)
            if set(kwargs) == {"x", "y", "z"}:
                obj = super().__new__(
                    cls,
                    [
                        [x, y, z]
                        for x, y, z in zip(kwargs["x"], kwargs["y"], kwargs["z"])
                    ],
                )
                obj.azimuthal = AzimuthalSympyXY(kwargs["x"], kwargs["y"])
                obj.longitudinal = LongitudinalSympyZ(kwargs["z"])
            elif set(kwargs) == {"x", "y", "eta"}:
                obj = super().__new__(
                    cls,
                    [
                        [x, y, eta]
                        for x, y, eta in zip(kwargs["x"], kwargs["y"], kwargs["eta"])
                    ],
                )
                obj.azimuthal = AzimuthalSympyXY(kwargs["x"], kwargs["y"])
                obj.longitudinal = LongitudinalSympyEta(kwargs["eta"])
            elif set(kwargs) == {"x", "y", "theta"}:
                obj = super().__new__(
                    cls,
                    [
                        [x, y, theta]
                        for x, y, theta in zip(
                            kwargs["x"], kwargs["y"], kwargs["theta"]
                        )
                    ],
                )
                obj.azimuthal = AzimuthalSympyXY(kwargs["x"], kwargs["y"])
                obj.longitudinal = LongitudinalSympyTheta(kwargs["theta"])
            elif set(kwargs) == {"rho", "phi", "z"}:
                obj = super().__new__(
                    cls,
                    [
                        [rho, phi, z]
                        for rho, phi, z in zip(
                            kwargs["rho"], kwargs["phi"], kwargs["z"]
                        )
                    ],
                )
                obj.azimuthal = AzimuthalSympyRhoPhi(kwargs["rho"], kwargs["phi"])
                obj.longitudinal = LongitudinalSympyZ(kwargs["z"])
            elif set(kwargs) == {"rho", "phi", "eta"}:
                obj = super().__new__(
                    cls,
                    [
                        [rho, phi, eta]
                        for rho, phi, eta in zip(
                            kwargs["rho"], kwargs["phi"], kwargs["eta"]
                        )
                    ],
                )
                obj.azimuthal = AzimuthalSympyRhoPhi(kwargs["rho"], kwargs["phi"])
                obj.longitudinal = LongitudinalSympyEta(kwargs["eta"])
            elif set(kwargs) == {"rho", "phi", "theta"}:
                obj = super().__new__(
                    cls,
                    [
                        [rho, phi, theta]
                        for rho, phi, theta in zip(
                            kwargs["rho"], kwargs["phi"], kwargs["theta"]
                        )
                    ],
                )
                obj.azimuthal = AzimuthalSympyRhoPhi(kwargs["rho"], kwargs["phi"])
                obj.longitudinal = LongitudinalSympyTheta(kwargs["theta"])
            else:
                complaint = """unrecognized combination of coordinates, allowed combinations are:\n
                    x= y= z=
                    x= y= theta=
                    x= y= eta=
                    rho= phi= z=
                    rho= phi= theta=
                    rho= phi= eta=""".replace("                    ", "    ")
                if not cls._IS_MOMENTUM:
                    raise TypeError(complaint)
                else:
                    raise TypeError(f"{complaint}\n\nor their momentum equivalents")
        else:
            raise TypeError(
                "must give Azimuthal and Longitudinal if not giving keyword arguments"
            )
        return obj

    def __init__(
        self,
        azimuthal: AzimuthalSympy | None = None,
        longitudinal: LongitudinalSympy | None = None,
        **kwargs: float,
    ):
        super().__init__()

    def __repr__(self) -> str:
        aznames = _coordinate_class_to_names[_aztype(self)]
        lnames = _coordinate_class_to_names[_ltype(self)]
        out = [f"{x}={getattr(self.azimuthal, x)}" for x in aznames]
        for x in lnames:
            out.append(f"{x}={getattr(self.longitudinal, x)}")
        return "VectorSympy3D(" + ", ".join(out) + ")"

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
            result = _toarrays(result)
            azcoords = _coord_sympy_type[returns[0]](result[0], result[1])
            return cls.ProjectionClass3D(
                azimuthal=azcoords, longitudinal=self.longitudinal
            )

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and returns[1] is None
        ):
            result = _toarrays(result)
            azcoords = _coord_sympy_type[returns[0]](result[0], result[1])
            return cls.ProjectionClass2D(azimuthal=azcoords)

        elif (
            len(returns) == 2
            or (len(returns) == 3 and returns[2] is None)
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
        ):
            result = _toarrays(result)
            azcoords = _coord_sympy_type[returns[0]](result[0], result[1])
            lcoords = _coord_sympy_type[returns[1]](result[2])
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
            result = _toarrays(result)
            azcoords = _coord_sympy_type[returns[0]](result[0], result[1])
            lcoords = _coord_sympy_type[returns[1]](result[2])
            tcoords = _coord_sympy_type[returns[2]](result[3])
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
        self.azimuthal = AzimuthalSympyXY(x, self.y)

    @property
    def y(self) -> float:
        return super().y

    @y.setter
    def y(self, y: float) -> None:
        self.azimuthal = AzimuthalSympyXY(self.x, y)

    @property
    def rho(self) -> float:
        return super().rho

    @rho.setter
    def rho(self, rho: float) -> None:
        self.azimuthal = AzimuthalSympyRhoPhi(rho, self.phi)

    @property
    def phi(self) -> float:
        return super().phi

    @phi.setter
    def phi(self, phi: float) -> None:
        self.azimuthal = AzimuthalSympyRhoPhi(self.rho, phi)

    @property
    def z(self) -> float:
        return super().z

    @z.setter
    def z(self, z: float) -> None:
        self.longitudinal = LongitudinalSympyZ(z)

    @property
    def theta(self) -> float:
        return super().theta

    @theta.setter
    def theta(self, theta: float) -> None:
        self.longitudinal = LongitudinalSympyTheta(theta)

    @property
    def eta(self) -> float:
        return super().eta

    @eta.setter
    def eta(self, eta: float) -> None:
        self.longitudinal = LongitudinalSympyEta(eta)


class MomentumSympy3D(SpatialMomentum, VectorSympy3D):
    """
    Three dimensional momentum vector class for the SymPy backend.

    Examples:
        >>> import vector
        >>> vec = vector.MomentumSympy3D(px=[1, 2], py=[3, 4], pz=[5, 6])
        >>> vec.px, vec.py, vec.pz
        ([1, 2], [3, 4], [5, 6])
        >>> vec = vector.MomentumSympy3D(pt=[1, 2], phi=[3, 4], pz=[5, 6])
        >>> vec.pt, vec.phi, vec.pz
        ([1, 2], [3, 4], [5, 6])
        >>> vec = vector.MomentumSympy3D(
        ...     azimuthal=vector.backends.sympy.AzimuthalSympyXY([1, 2], [3, 4]),
        ...     longitudinal=vector.backends.sympy.LongitudinalSympyTheta([5, 6])
        ... )
        >>> vec.x, vec.y, vec.theta
        ([1, 2], [3, 4], [5, 6])

    For three dimensional SymPy vectors, see
    :class:`vector.backends.sympy.VectorSympy3D`.
    """

    ObjectClass = vector.backends.object.MomentumObject3D
    _IS_MOMENTUM = True

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
        return "MomentumSympy3D(" + ", ".join(out) + ")"

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
        self.azimuthal = AzimuthalSympyXY(px, self.py)

    @property
    def py(self) -> float:
        return super().py

    @py.setter
    def py(self, py: float) -> None:
        self.azimuthal = AzimuthalSympyXY(self.px, py)

    @property
    def pt(self) -> float:
        return super().pt

    @pt.setter
    def pt(self, pt: float) -> None:
        self.azimuthal = AzimuthalSympyRhoPhi(pt, self.phi)

    @property
    def pz(self) -> float:
        return super().pz

    @pz.setter
    def pz(self, pz: float) -> None:
        self.longitudinal = LongitudinalSympyZ(pz)


class VectorSympy4D(sympy.Array, VectorSympy, Lorentz, Vector4D):
    """
    Four dimensional vector class for the SymPy backend.

    Examples:
        >>> import vector
        >>> vec = vector.VectorSympy4D(x=[1, 2], y=[3, 4], z=[5, 6], t=[7, 8])
        >>> vec.x, vec.y, vec.z, vec.t
        ([1, 2], [3, 4], [5, 6], [7, 8])
        >>> vec = vector.VectorSympy4D(rho=[1, 2], phi=[3, 4], eta=[5, 6], tau=[7, 8])
        >>> vec.rho, vec.phi, vec.eta, vec.tau
        ([1, 2], [3, 4], [5, 6], [7, 8])
        >>> vec = vector.VectorSympy4D(
        ...     azimuthal=vector.backends.sympy.AzimuthalSympyXY([1, 2], [3, 4]),
        ...     longitudinal=vector.backends.sympy.LongitudinalSympyTheta([5, 6]),
        ...     temporal=vector.backends.sympy.TemporalSympyTau([7, 8])
        ... )
        >>> vec.x, vec.y, vec.theta, vec.tau
        ([1, 2], [3, 4], [5, 6], [7, 8])

    For four dimensional momentum SymPy vectors, see
    :class:`vector.backends.sympy.MomentumSympy4D`.
    """

    ObjectClass = vector.backends.object.VectorObject4D
    _IS_MOMENTUM = False

    __slots__ = ("azimuthal", "longitudinal", "temporal")

    azimuthal: AzimuthalSympy
    longitudinal: LongitudinalSympy
    temporal: TemporalSympy

    def __new__(
        cls,
        azimuthal: AzimuthalSympy | None = None,
        longitudinal: LongitudinalSympy | None = None,
        temporal: TemporalSympy | None = None,
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
            obj = super().__new__(
                cls,
                [
                    [x, y, z, t]
                    for x, y, z, t in zip(
                        azimuthal.elements[0],
                        azimuthal.elements[1],
                        longitudinal.elements[0],
                        temporal.elements[0],
                    )
                ],
            )
            obj.azimuthal = azimuthal
            obj.longitudinal = longitudinal
            obj.temporal = temporal
        elif kwargs and azimuthal is None and longitudinal is None and temporal is None:
            _is_type_safe(kwargs)
            if set(kwargs) == {"x", "y", "z", "t"}:
                obj = super().__new__(
                    cls,
                    [
                        [x, y, z, t]
                        for x, y, z, t in zip(
                            kwargs["x"], kwargs["y"], kwargs["z"], kwargs["t"]
                        )
                    ],
                )
                obj.azimuthal = AzimuthalSympyXY(kwargs["x"], kwargs["y"])
                obj.longitudinal = LongitudinalSympyZ(kwargs["z"])
                obj.temporal = TemporalSympyT(kwargs["t"])
            elif set(kwargs) == {"x", "y", "eta", "t"}:
                obj = super().__new__(
                    cls,
                    [
                        [x, y, eta, t]
                        for x, y, eta, t in zip(
                            kwargs["x"], kwargs["y"], kwargs["eta"], kwargs["t"]
                        )
                    ],
                )
                obj.azimuthal = AzimuthalSympyXY(kwargs["x"], kwargs["y"])
                obj.longitudinal = LongitudinalSympyEta(kwargs["eta"])
                obj.temporal = TemporalSympyT(kwargs["t"])
            elif set(kwargs) == {"x", "y", "theta", "t"}:
                obj = super().__new__(
                    cls,
                    [
                        [x, y, theta, t]
                        for x, y, theta, t in zip(
                            kwargs["x"], kwargs["y"], kwargs["theta"], kwargs["t"]
                        )
                    ],
                )
                obj.azimuthal = AzimuthalSympyXY(kwargs["x"], kwargs["y"])
                obj.longitudinal = LongitudinalSympyTheta(kwargs["theta"])
                obj.temporal = TemporalSympyT(kwargs["t"])
            elif set(kwargs) == {"rho", "phi", "z", "t"}:
                obj = super().__new__(
                    cls,
                    [
                        [rho, phi, z, t]
                        for rho, phi, z, t in zip(
                            kwargs["rho"], kwargs["phi"], kwargs["z"], kwargs["t"]
                        )
                    ],
                )
                obj.azimuthal = AzimuthalSympyRhoPhi(kwargs["rho"], kwargs["phi"])
                obj.longitudinal = LongitudinalSympyZ(kwargs["z"])
                obj.temporal = TemporalSympyT(kwargs["t"])
            elif set(kwargs) == {"rho", "phi", "eta", "t"}:
                obj = super().__new__(
                    cls,
                    [
                        [rho, phi, eta, t]
                        for rho, phi, eta, t in zip(
                            kwargs["rho"], kwargs["phi"], kwargs["eta"], kwargs["t"]
                        )
                    ],
                )
                obj.azimuthal = AzimuthalSympyRhoPhi(kwargs["rho"], kwargs["phi"])
                obj.longitudinal = LongitudinalSympyEta(kwargs["eta"])
                obj.temporal = TemporalSympyT(kwargs["t"])
            elif set(kwargs) == {"rho", "phi", "theta", "t"}:
                obj = super().__new__(
                    cls,
                    [
                        [rho, phi, theta, t]
                        for rho, phi, theta, t in zip(
                            kwargs["rho"], kwargs["phi"], kwargs["theta"], kwargs["t"]
                        )
                    ],
                )
                obj.azimuthal = AzimuthalSympyRhoPhi(kwargs["rho"], kwargs["phi"])
                obj.longitudinal = LongitudinalSympyTheta(kwargs["theta"])
                obj.temporal = TemporalSympyT(kwargs["t"])
            elif set(kwargs) == {"x", "y", "z", "tau"}:
                obj = super().__new__(
                    cls,
                    [
                        [x, y, z, tau]
                        for x, y, z, tau in zip(
                            kwargs["x"], kwargs["y"], kwargs["z"], kwargs["tau"]
                        )
                    ],
                )
                obj.azimuthal = AzimuthalSympyXY(kwargs["x"], kwargs["y"])
                obj.longitudinal = LongitudinalSympyZ(kwargs["z"])
                obj.temporal = TemporalSympyTau(kwargs["tau"])
            elif set(kwargs) == {"x", "y", "eta", "tau"}:
                obj = super().__new__(
                    cls,
                    [
                        [x, y, eta, tau]
                        for x, y, eta, tau in zip(
                            kwargs["x"], kwargs["y"], kwargs["eta"], kwargs["tau"]
                        )
                    ],
                )
                obj.azimuthal = AzimuthalSympyXY(kwargs["x"], kwargs["y"])
                obj.longitudinal = LongitudinalSympyEta(kwargs["eta"])
                obj.temporal = TemporalSympyTau(kwargs["tau"])
            elif set(kwargs) == {"x", "y", "theta", "tau"}:
                obj = super().__new__(
                    cls,
                    [
                        [x, y, theta, tau]
                        for x, y, theta, tau in zip(
                            kwargs["x"], kwargs["y"], kwargs["theta"], kwargs["tau"]
                        )
                    ],
                )
                obj.azimuthal = AzimuthalSympyXY(kwargs["x"], kwargs["y"])
                obj.longitudinal = LongitudinalSympyTheta(kwargs["theta"])
                obj.temporal = TemporalSympyTau(kwargs["tau"])
            elif set(kwargs) == {"rho", "phi", "z", "tau"}:
                obj = super().__new__(
                    cls,
                    [
                        [rho, phi, z, tau]
                        for rho, phi, z, tau in zip(
                            kwargs["rho"], kwargs["phi"], kwargs["z"], kwargs["tau"]
                        )
                    ],
                )
                obj.azimuthal = AzimuthalSympyRhoPhi(kwargs["rho"], kwargs["phi"])
                obj.longitudinal = LongitudinalSympyZ(kwargs["z"])
                obj.temporal = TemporalSympyTau(kwargs["tau"])
            elif set(kwargs) == {"rho", "phi", "eta", "tau"}:
                obj = super().__new__(
                    cls,
                    [
                        [rho, phi, eta, tau]
                        for rho, phi, eta, tau in zip(
                            kwargs["rho"], kwargs["phi"], kwargs["eta"], kwargs["tau"]
                        )
                    ],
                )
                obj.azimuthal = AzimuthalSympyRhoPhi(kwargs["rho"], kwargs["phi"])
                obj.longitudinal = LongitudinalSympyEta(kwargs["eta"])
                obj.temporal = TemporalSympyTau(kwargs["tau"])
            elif set(kwargs) == {"rho", "phi", "theta", "tau"}:
                obj = super().__new__(
                    cls,
                    [
                        [rho, phi, theta, tau]
                        for rho, phi, theta, tau in zip(
                            kwargs["rho"], kwargs["phi"], kwargs["theta"], kwargs["tau"]
                        )
                    ],
                )
                obj.azimuthal = AzimuthalSympyRhoPhi(kwargs["rho"], kwargs["phi"])
                obj.longitudinal = LongitudinalSympyTheta(kwargs["theta"])
                obj.temporal = TemporalSympyTau(kwargs["tau"])
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
                if not cls._IS_MOMENTUM:
                    raise TypeError(complaint)
                else:
                    raise TypeError(f"{complaint}\n\nor their momentum equivalents")
        else:
            raise TypeError(
                "must give Azimuthal, Longitudinal, and Temporal if not giving keyword arguments"
            )
        return obj

    def __init__(
        self,
        azimuthal: AzimuthalSympy | None = None,
        longitudinal: LongitudinalSympy | None = None,
        temporal: TemporalSympy | None = None,
        **kwargs: float,
    ):
        super().__init__()

    def __repr__(self) -> str:
        aznames = _coordinate_class_to_names[_aztype(self)]
        lnames = _coordinate_class_to_names[_ltype(self)]
        tnames = _coordinate_class_to_names[_ttype(self)]
        out = [f"{x}={getattr(self.azimuthal, x)}" for x in aznames]
        for x in lnames:
            out.append(f"{x}={getattr(self.longitudinal, x)}")
        for x in tnames:
            out.append(f"{x}={getattr(self.temporal, x)}")
        return "VectorSympy4D(" + ", ".join(out) + ")"

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
            result = _toarrays(result)
            azcoords = _coord_sympy_type[returns[0]](result[0], result[1])
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
            result = _toarrays(result)
            azcoords = _coord_sympy_type[returns[0]](result[0], result[1])
            return cls.ProjectionClass2D(azimuthal=azcoords)

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
        ):
            result = _toarrays(result)
            azcoords = _coord_sympy_type[returns[0]](result[0], result[1])
            lcoords = _coord_sympy_type[returns[1]](result[2])
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
            result = _toarrays(result)
            azcoords = _coord_sympy_type[returns[0]](result[0], result[1])
            lcoords = _coord_sympy_type[returns[1]](result[2])
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
            result = _toarrays(result)
            azcoords = _coord_sympy_type[returns[0]](result[0], result[1])
            lcoords = _coord_sympy_type[returns[1]](result[2])
            tcoords = _coord_sympy_type[returns[2]](result[3])
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
        self.azimuthal = AzimuthalSympyXY(x, self.y)

    @property
    def y(self) -> float:
        return super().y

    @y.setter
    def y(self, y: float) -> None:
        self.azimuthal = AzimuthalSympyXY(self.x, y)

    @property
    def rho(self) -> float:
        return super().rho

    @rho.setter
    def rho(self, rho: float) -> None:
        self.azimuthal = AzimuthalSympyRhoPhi(rho, self.phi)

    @property
    def phi(self) -> float:
        return super().phi

    @phi.setter
    def phi(self, phi: float) -> None:
        self.azimuthal = AzimuthalSympyRhoPhi(self.rho, phi)

    @property
    def z(self) -> float:
        return super().z

    @z.setter
    def z(self, z: float) -> None:
        self.longitudinal = LongitudinalSympyZ(z)

    @property
    def theta(self) -> float:
        return super().theta

    @theta.setter
    def theta(self, theta: float) -> None:
        self.longitudinal = LongitudinalSympyTheta(theta)

    @property
    def eta(self) -> float:
        return super().eta

    @eta.setter
    def eta(self, eta: float) -> None:
        self.longitudinal = LongitudinalSympyEta(eta)

    @property
    def t(self) -> float:
        return super().t

    @t.setter
    def t(self, t: float) -> None:
        self.temporal = TemporalSympyT(t)

    @property
    def tau(self) -> float:
        return super().tau

    @tau.setter
    def tau(self, tau: float) -> None:
        self.temporal = TemporalSympyTau(tau)


class MomentumSympy4D(LorentzMomentum, VectorSympy4D):
    """
    Four dimensional momentum vector class for the SymPy backend.

    Examples:
        >>> import vector
        >>> vec = vector.MomentumSympy4D(px=[1, 2], py=[3, 4], pz=[5, 6], t=[7, 8])
        >>> vec.px, vec.py, vec.pz, vec.t
        ([1, 2], [3, 4], [5, 6], [7, 8])
        >>> vec = vector.MomentumSympy4D(pt=[1, 2], phi=[3, 4], pz=[5, 6], M=[7, 8])
        >>> vec.pt, vec.phi, vec.pz, vec.M
        ([1, 2], [3, 4], [5, 6], [7, 8])
        >>> vec = vector.MomentumSympy4D(
        ...     azimuthal=vector.backends.sympy.AzimuthalSympyXY([1, 2], [3, 4]),
        ...     longitudinal=vector.backends.sympy.LongitudinalSympyTheta([5, 6]),
        ...     temporal=vector.backends.sympy.TemporalSympyTau([7, 8])
        ... )
        >>> vec.x, vec.y, vec.theta, vec.tau
        ([1, 2], [3, 4], [5, 6], [7, 8])

    For four dimensional SymPy vectors, see
    :class:`vector.backends.sympy.VectorSympy4D`.
    """

    ObjectClass = vector.backends.object.MomentumObject4D
    _IS_MOMENTUM = True

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
        return "MomentumSympy4D(" + ", ".join(out) + ")"

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
        self.azimuthal = AzimuthalSympyXY(px, self.py)

    @property
    def py(self) -> float:
        return super().py

    @py.setter
    def py(self, py: float) -> None:
        self.azimuthal = AzimuthalSympyXY(self.px, py)

    @property
    def pt(self) -> float:
        return super().pt

    @pt.setter
    def pt(self, pt: float) -> None:
        self.azimuthal = AzimuthalSympyRhoPhi(pt, self.phi)

    @property
    def pz(self) -> float:
        return super().pz

    @pz.setter
    def pz(self, pz: float) -> None:
        self.longitudinal = LongitudinalSympyZ(pz)

    @property
    def E(self) -> float:
        return super().E

    @E.setter
    def E(self, E: float) -> None:
        self.temporal = TemporalSympyT(E)

    @property
    def e(self) -> float:
        return super().e

    @e.setter
    def e(self, e: float) -> None:
        self.temporal = TemporalSympyT(e)

    @property
    def energy(self) -> float:
        return super().energy

    @energy.setter
    def energy(self, energy: float) -> None:
        self.temporal = TemporalSympyT(energy)

    @property
    def M(self) -> float:
        return super().M

    @M.setter
    def M(self, M: float) -> None:
        self.temporal = TemporalSympyTau(M)

    @property
    def m(self) -> float:
        return super().m

    @m.setter
    def m(self, m: float) -> None:
        self.temporal = TemporalSympyTau(m)

    @property
    def mass(self) -> float:
        return super().mass

    @mass.setter
    def mass(self, mass: float) -> None:
        self.temporal = TemporalSympyTau(mass)


VectorSympy2D.ProjectionClass2D = VectorSympy2D
VectorSympy2D.ProjectionClass3D = VectorSympy3D
VectorSympy2D.ProjectionClass4D = VectorSympy4D
VectorSympy2D.GenericClass = VectorSympy2D
VectorSympy2D.MomentumClass = MomentumSympy2D

MomentumSympy2D.ProjectionClass2D = MomentumSympy2D
MomentumSympy2D.ProjectionClass3D = MomentumSympy3D
MomentumSympy2D.ProjectionClass4D = MomentumSympy4D
MomentumSympy2D.GenericClass = VectorSympy2D
MomentumSympy2D.MomentumClass = MomentumSympy2D

VectorSympy3D.ProjectionClass2D = VectorSympy2D
VectorSympy3D.ProjectionClass3D = VectorSympy3D
VectorSympy3D.ProjectionClass4D = VectorSympy4D
VectorSympy3D.GenericClass = VectorSympy3D
VectorSympy3D.MomentumClass = MomentumSympy3D

MomentumSympy3D.ProjectionClass2D = MomentumSympy2D
MomentumSympy3D.ProjectionClass3D = MomentumSympy3D
MomentumSympy3D.ProjectionClass4D = MomentumSympy4D
MomentumSympy3D.GenericClass = VectorSympy3D
MomentumSympy3D.MomentumClass = MomentumSympy3D

VectorSympy4D.ProjectionClass2D = VectorSympy2D
VectorSympy4D.ProjectionClass3D = VectorSympy3D
VectorSympy4D.ProjectionClass4D = VectorSympy4D
VectorSympy4D.GenericClass = VectorSympy4D
VectorSympy4D.MomentumClass = MomentumSympy4D

MomentumSympy4D.ProjectionClass2D = MomentumSympy2D
MomentumSympy4D.ProjectionClass3D = MomentumSympy3D
MomentumSympy4D.ProjectionClass4D = MomentumSympy4D
MomentumSympy4D.GenericClass = VectorSympy4D
MomentumSympy4D.MomentumClass = MomentumSympy4D
