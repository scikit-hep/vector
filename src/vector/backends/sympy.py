from __future__ import annotations

import typing

import numpy
import sympy

import vector
from vector._lib import SympyLib
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


class CoordinatesSympy:
    """Coordinates class for the SymPy backend."""

    lib = SympyLib()


class AzimuthalSympy(CoordinatesSympy, Azimuthal):
    """Azimuthal class for the SymPy backend."""

    ObjectClass: type[vector.backends.object.AzimuthalObject]


class LongitudinalSympy(CoordinatesSympy, Longitudinal):
    """Longitudinal class for the SymPy backend."""

    ObjectClass: type[vector.backends.object.LongitudinalObject]


class TemporalSympy(CoordinatesSympy, Temporal):
    """Temporal class for the SymPy backend."""

    ObjectClass: type[vector.backends.object.TemporalObject]


class AzimuthalSympyXY(AzimuthalSympy, AzimuthalXY):
    """
    Class for the ``x`` and ``y`` (azimuthal) coordinates of SymPy backend.

    Examples:
        >>> import vector; import sympy
        >>> vector.backends.sympy.AzimuthalSympyXY(sympy.Symbol("x"), sympy.Symbol("y"))
        AzimuthalSympyXY(x=x, y=y)
    """

    def __init__(self, x: sympy.Symbol, y: sympy.Symbol):
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"AzimuthalSympyXY(x={self.x!r}, y={self.y!r})"

    @property
    def elements(self) -> tuple[sympy.Symbol, sympy.Symbol]:
        """
        Azimuthal coordinates (``x`` and ``y``) as a tuple.

        Each coordinate is a SymPy expression and not a vector.

        Examples:
            >>> import vector; import sympy
            >>> vec = vector.backends.sympy.AzimuthalSympyXY(sympy.Symbol("x"), sympy.Symbol("y"))
            >>> vec.elements
            (x, y)
        """
        return (self.x, self.y)


class AzimuthalSympyRhoPhi(AzimuthalSympy, AzimuthalRhoPhi):
    """
    Class for the ``rho`` and ``phi`` (azimuthal) coordinates of SymPy backend.

    Examples:
        >>> import vector; import sympy
        >>> vector.backends.sympy.AzimuthalSympyRhoPhi(sympy.Symbol("rho"), sympy.Symbol("phi"))
        AzimuthalSympyRhoPhi(rho=rho, phi=phi)
    """

    def __init__(self, rho: sympy.Symbol, phi: sympy.Symbol):
        self.rho = rho
        self.phi = phi

    def __repr__(self) -> str:
        return f"AzimuthalSympyRhoPhi(rho={self.rho!r}, phi={self.phi!r})"

    @property
    def elements(self) -> tuple[sympy.Symbol, sympy.Symbol]:
        """
        Azimuthal coordinates (``rho`` and ``phi``) as a tuple.

        Each coordinate is a SymPy expression and not a vector.

        Examples:
            >>> import vector
            >>> vec = vector.backends.sympy.AzimuthalSympyRhoPhi(sympy.Symbol("rho"), sympy.Symbol("phi"))
            >>> vec.elements
            (rho, phi)
        """
        return (self.rho, self.phi)


class LongitudinalSympyZ(LongitudinalSympy, LongitudinalZ):
    """
    Class for the ``z`` (longitudinal) coordinate of SymPy backend.

    Examples:
        >>> import vector; import sympy
        >>> vector.backends.sympy.LongitudinalSympyZ(sympy.Symbol("z"))
        LongitudinalSympyZ(z=z)
    """

    def __init__(self, z: sympy.Symbol):
        self.z = z

    def __repr__(self) -> str:
        return f"LongitudinalSympyZ(z={self.z!r})"

    @property
    def elements(self) -> tuple[sympy.Symbol]:
        """
        Longitudinal coordinates (``z``) as a tuple.

        Each coordinate is a SymPy expression and not a vector.

        Examples:
            >>> import vector; import sympy
            >>> vec = vector.backends.sympy.LongitudinalSympyZ(sympy.Symbol("z"))
            >>> vec.elements
            (z,)
        """
        return (self.z,)


class LongitudinalSympyTheta(LongitudinalSympy, LongitudinalTheta):
    """
    Class for the ``theta`` (longitudinal) coordinate of SymPy backend.

    Examples:
        >>> import vector; import sympy
        >>> vector.backends.sympy.LongitudinalSympyTheta(sympy.Symbol("theta"))
        LongitudinalSympyTheta(theta=theta)
    """

    def __init__(self, theta: sympy.Symbol):
        self.theta = theta

    def __repr__(self) -> str:
        return f"LongitudinalSympyTheta(theta={self.theta!r})"

    @property
    def elements(self) -> tuple[sympy.Symbol]:
        """
        Longitudinal coordinates (``theta``) as a tuple.

        Each coordinate is a SymPy expression and not a vector.

        Examples:
            >>> import vector; import sympy
            >>> vec = vector.backends.sympy.LongitudinalSympyTheta(sympy.Symbol("theta"))
            >>> vec.elements
            (theta,)
        """
        return (self.theta,)


class LongitudinalSympyEta(LongitudinalSympy, LongitudinalEta):
    """
    Class for the ``eta`` (longitudinal) coordinate of SymPy backend.

    Examples:
        >>> import vector; import sympy
        >>> vector.backends.sympy.LongitudinalSympyEta(sympy.Symbol("eta"))
        LongitudinalSympyEta(eta=eta)
    """

    def __init__(self, eta: sympy.Symbol):
        self.eta = eta

    def __repr__(self) -> str:
        return f"LongitudinalSympyEta(eta={self.eta!r})"

    @property
    def elements(self) -> tuple[sympy.Symbol]:
        """
        Longitudinal coordinates (``eta``) as a tuple.

        Each coordinate is a SymPy expression and not a vector.

        Examples:
            >>> import vector
            >>> vec = vector.backends.sympy.LongitudinalSympyEta(sympy.Symbol("eta"))
            >>> vec.elements
            (eta,)
        """
        return (self.eta,)


class TemporalSympyT(TemporalSympy, TemporalT):
    """
    Class for the ``t`` (temporal) coordinate of SymPy backend.

    Examples:
        >>> import vector
        >>> vector.backends.sympy.TemporalSympyT(sympy.Symbol("t"))
        TemporalSympyT(t=t)
    """

    def __init__(self, t: sympy.Symbol):
        self.t = t

    def __repr__(self) -> str:
        return f"TemporalSympyT(t={self.t!r})"

    @property
    def elements(self) -> tuple[sympy.Symbol]:
        """
        Temporal coordinates (``t``) as a tuple.

        Each coordinate is a SymPy expression and not a vector.

        Examples:
            >>> import vector
            >>> vec = vector.backends.sympy.TemporalSympyT(sympy.Symbol("t"))
            >>> vec.elements
            (t,)
        """
        return (self.t,)


class TemporalSympyTau(TemporalSympy, TemporalTau):
    """
    Class for the ``tau`` (temporal) coordinate of SymPy backend.

    Examples:
        >>> import vector
        >>> vector.backends.sympy.TemporalSympyTau(sympy.Symbol("tau"))
        TemporalSympyTau(tau=tau)
    """

    def __init__(self, tau: sympy.Symbol):
        self.tau = tau

    def __repr__(self) -> str:
        return f"TemporalSympyTau(tau={self.tau!r})"

    @property
    def elements(self) -> tuple[sympy.Symbol]:
        """
        Temporal coordinates (``tau``) as a tuple.

        Each coordinate is a SymPy expression and not a vector.

        Examples:
            >>> import vector
            >>> vec = vector.backends.sympy.TemporalSympyTau(sympy.Symbol("tau"))
            >>> vec.elements
            (tau,)
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


def _is_type_safe(coordinates: dict[str, typing.Any]) -> None:
    if not all(isinstance(coord, sympy.Expr) for coord in coordinates.values()):
        raise TypeError("coordinates must be a sympy expression")


def _replace_data(obj: typing.Any, result: typing.Any) -> typing.Any:
    if not isinstance(result, VectorSympy):
        raise TypeError(f"can only assign a single vector to {type(obj).__name__}")

    if isinstance(result, (VectorSympy2D, VectorSympy3D, VectorSympy4D)):
        if isinstance(obj.azimuthal, AzimuthalSympyXY):
            obj.azimuthal = AzimuthalSympyXY(result.x, result.y)
        elif isinstance(obj.azimuthal, AzimuthalSympyRhoPhi):
            obj.azimuthal = AzimuthalSympyRhoPhi(result.rho, result.phi)
        else:
            raise AssertionError(type(obj))

    if isinstance(result, (VectorSympy3D, VectorSympy4D)):
        if isinstance(obj.longitudinal, LongitudinalSympyZ):
            obj.longitudinal = LongitudinalSympyZ(result.z)
        elif isinstance(obj.longitudinal, LongitudinalSympyTheta):
            obj.longitudinal = LongitudinalSympyTheta(result.theta)
        elif isinstance(obj.longitudinal, LongitudinalSympyEta):
            obj.longitudinal = LongitudinalSympyEta(result.eta)
        else:
            raise AssertionError(type(obj))

    if isinstance(result, VectorSympy4D):
        if isinstance(obj.temporal, TemporalSympyT):
            obj.temporal = TemporalSympyT(result.t)
        elif isinstance(obj.temporal, TemporalSympyTau):
            obj.temporal = TemporalSympyTau(result.tau)
        else:
            raise AssertionError(type(obj))

    return obj


class VectorSympy(Vector):
    """Mixin class for Sympy vectors."""

    lib = SympyLib()

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
        return numpy.power(self, other)  # type: ignore[call-overload]

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
        Implements NumPy's ``ufunc``s for ``VectorSympy`` and its subclasses. The current
        implementation includes ``numpy.absolute``, ``numpy.add``, ``numpy.subtract``,
        ``numpy.multiply``, ``numpy.positive``, ``numpy.negative``, ``numpy.true_divide``,
        ``numpy.power``, ``numpy.square``, ``numpy.sqrt``, ``numpy.cbrt``, ``numpy.matmul``,
        ``numpy.equal``, and ``numpy.not_equal``.
        """
        if not isinstance(_handler_of(*inputs), VectorSympy):
            # Let a higher-precedence backend handle it.
            return NotImplemented

        outputs = kwargs.get("out", ())
        if any(not isinstance(x, VectorSympy) for x in outputs):
            raise TypeError(
                "ufunc operating on VectorSympy can only use the 'out' keyword "
                "with another VectorSympy"
            )

        if (
            ufunc is numpy.absolute
            and len(inputs) == 1
            and isinstance(inputs[0], Vector)
        ):
            if len(outputs) != 0:
                raise TypeError(
                    "output of 'numpy.absolute' is scalar, cannot fill a VectorSympy with 'out'"
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
                    "output of 'numpy.square' is scalar, cannot fill a VectorSympy with 'out'"
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
                    "output of 'numpy.sqrt' is scalar, cannot fill a VectorSympy with 'out'"
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
                    "output of 'numpy.cbrt' is scalar, cannot fill a VectorSympy with 'out'"
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
                    "output of 'numpy.matmul' is scalar, cannot fill a VectorSympy with 'out'"
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
                    "output of 'numpy.equal' is scalar, cannot fill a VectorSympy with 'out'"
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
                    "output of 'numpy.equal' is scalar, cannot fill a VectorSympy with 'out'"
                )
            return inputs[0].not_equal(inputs[1])

        else:
            return NotImplemented


class VectorSympy2D(VectorSympy, Planar, Vector2D):
    """
    Two dimensional vector class for the SymPy backend.

    Examples:
        >>> import vector; import sympy
        >>> vec = vector.VectorSympy2D(x=sympy.Symbol("x"), y=sympy.Symbol("y"))
        >>> vec.x, vec.y
        (x, y)
        >>> vec = vector.VectorSympy2D(rho=sympy.Symbol("rho"), phi=sympy.Symbol("phi"))
        >>> vec.rho, vec.phi
        (rho, phi)
        >>> vec = vector.VectorObject2D(azimuthal=vector.backends.sympy.AzimuthalSympyXY(sympy.Symbol("x"), sympy.Symbol("y")))
        >>> vec.x, vec.y
        (x, y)

    For two dimensional momentum SymPy vectors, see
    :class:`vector.backends.sympy.MomentumSympy2D`.
    """

    __slots__ = ("azimuthal",)
    azimuthal: AzimuthalSympy

    def __init__(self, azimuthal: AzimuthalSympy | None = None, **kwargs: sympy.Symbol):
        for k, v in kwargs.copy().items():
            kwargs.pop(k)
            kwargs[_repr_momentum_to_generic.get(k, k)] = v

        if not kwargs and azimuthal is not None:
            self.azimuthal = azimuthal
        elif kwargs and azimuthal is None:
            _is_type_safe(kwargs)
            if set(kwargs) == {"x", "y"}:
                self.azimuthal = AzimuthalSympyXY(kwargs["x"], kwargs["y"])
            elif set(kwargs) == {"rho", "phi"}:
                self.azimuthal = AzimuthalSympyRhoPhi(kwargs["rho"], kwargs["phi"])
            else:
                complaint = """unrecognized combination of coordinates, allowed combinations are:\n
                    x= y=
                    rho= phi=""".replace("                    ", "    ")
                if type(self) == VectorSympy2D:
                    raise TypeError(complaint)
                else:
                    raise TypeError(f"{complaint}\n\nor their momentum equivalents")
        else:
            raise TypeError("must give Azimuthal if not giving keyword arguments")

    def __repr__(self) -> str:
        aznames = _coordinate_class_to_names[_aztype(self)]
        out = [f"{x}={getattr(self.azimuthal, x)}" for x in aznames]
        return "VectorSympy2D(" + ", ".join(out) + ")"

    @property
    def x(self) -> sympy.Symbol:
        return super().x

    @x.setter
    def x(self, x: sympy.Symbol) -> None:
        self.azimuthal = AzimuthalSympyXY(x, self.y)

    @property
    def y(self) -> sympy.Symbol:
        return super().y

    @y.setter
    def y(self, y: sympy.Symbol) -> None:
        self.azimuthal = AzimuthalSympyXY(self.x, y)

    @property
    def rho(self) -> sympy.Symbol:
        return super().rho

    @rho.setter
    def rho(self, rho: sympy.Symbol) -> None:
        self.azimuthal = AzimuthalSympyRhoPhi(rho, self.phi)

    @property
    def phi(self) -> sympy.Symbol:
        return super().phi

    @phi.setter
    def phi(self, phi: sympy.Symbol) -> None:
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
        >>> import vector; import sympy
        >>> vec = vector.MomentumSympy2D(px=sympy.Symbol("px"), py=sympy.Symbol("py"))
        >>> vec.px, vec.py
        (px, py)
        >>> vec = vector.MomentumSympy2D(pt=sympy.Symbol("pt"), phi=sympy.Symbol("phi"))
        >>> vec.pt, vec.phi
        (pt, phi)
        >>> vec = vector.MomentumSympy2D(azimuthal=vector.backends.sympy.AzimuthalSympyXY(sympy.Symbol("px"), sympy.Symbol("py")))
        >>> vec.px, vec.py
        (px, py)

    For two dimensional SymPy vectors, see
    :class:`vector.backends.object.VectorSympy2D`.
    """

    def __repr__(self) -> str:
        aznames = _coordinate_class_to_names[_aztype(self)]
        out = []
        for x in aznames:
            y = _repr_generic_to_momentum.get(x, x)
            out.append(f"{y}={getattr(self.azimuthal, x)}")
        return "MomentumSympy2D(" + ", ".join(out) + ")"

    @property
    def px(self) -> sympy.Symbol:
        return super().px

    @px.setter
    def px(self, px: sympy.Symbol) -> None:
        self.azimuthal = AzimuthalSympyXY(px, self.py)

    @property
    def py(self) -> sympy.Symbol:
        return super().py

    @py.setter
    def py(self, py: sympy.Symbol) -> None:
        self.azimuthal = AzimuthalSympyXY(self.px, py)

    @property
    def pt(self) -> sympy.Symbol:
        return super().pt

    @pt.setter
    def pt(self, pt: sympy.Symbol) -> None:
        self.azimuthal = AzimuthalSympyRhoPhi(pt, self.phi)


class VectorSympy3D(VectorSympy, Spatial, Vector3D):
    """
    Three dimensional vector class for the SymPy backend.

    Examples:
        >>> import vector; import sympy
        >>> vec = vector.VectorSympy3D(x=sympy.Symbol("x"), y=sympy.Symbol("y"), z=sympy.Symbol("z"))
        >>> vec.x, vec.y, vec.z
        (x, y, z)
        >>> vec = vector.VectorSympy3D(rho=sympy.Symbol("rho"), phi=sympy.Symbol("phi"), eta=sympy.Symbol("eta"))
        >>> vec.rho, vec.phi, vec.eta
        (rho, phi, eta)
        >>> vec = vector.VectorSympy3D(
        ...     azimuthal=vector.backends.sympy.AzimuthalSympyXY(sympy.Symbol("x"), sympy.Symbol("y")),
        ...     longitudinal=vector.backends.sympy.LongitudinalSympyTheta(sympy.Symbol("theta"))
        ... )
        >>> vec.x, vec.y, vec.theta
        (x, y, theta)

    For three dimensional momentum SymPy vectors, see
    :class:`vector.backends.object.MomentumSympy3D`.
    """

    __slots__ = ("azimuthal", "longitudinal")

    azimuthal: AzimuthalSympy
    longitudinal: LongitudinalSympy

    def __init__(
        self,
        azimuthal: AzimuthalSympy | None = None,
        longitudinal: LongitudinalSympy | None = None,
        **kwargs: sympy.Symbol,
    ):
        for k, v in kwargs.copy().items():
            kwargs.pop(k)
            kwargs[_repr_momentum_to_generic.get(k, k)] = v

        if not kwargs and azimuthal is not None and longitudinal is not None:
            self.azimuthal = azimuthal
            self.longitudinal = longitudinal
        elif kwargs and azimuthal is None and longitudinal is None:
            _is_type_safe(kwargs)
            if set(kwargs) == {"x", "y", "z"}:
                self.azimuthal = AzimuthalSympyXY(kwargs["x"], kwargs["y"])
                self.longitudinal = LongitudinalSympyZ(kwargs["z"])
            elif set(kwargs) == {"x", "y", "eta"}:
                self.azimuthal = AzimuthalSympyXY(kwargs["x"], kwargs["y"])
                self.longitudinal = LongitudinalSympyEta(kwargs["eta"])
            elif set(kwargs) == {"x", "y", "theta"}:
                self.azimuthal = AzimuthalSympyXY(kwargs["x"], kwargs["y"])
                self.longitudinal = LongitudinalSympyTheta(kwargs["theta"])
            elif set(kwargs) == {"rho", "phi", "z"}:
                self.azimuthal = AzimuthalSympyRhoPhi(kwargs["rho"], kwargs["phi"])
                self.longitudinal = LongitudinalSympyZ(kwargs["z"])
            elif set(kwargs) == {"rho", "phi", "eta"}:
                self.azimuthal = AzimuthalSympyRhoPhi(kwargs["rho"], kwargs["phi"])
                self.longitudinal = LongitudinalSympyEta(kwargs["eta"])
            elif set(kwargs) == {"rho", "phi", "theta"}:
                self.azimuthal = AzimuthalSympyRhoPhi(kwargs["rho"], kwargs["phi"])
                self.longitudinal = LongitudinalSympyTheta(kwargs["theta"])
            else:
                complaint = """unrecognized combination of coordinates, allowed combinations are:\n
                    x= y= z=
                    x= y= theta=
                    x= y= eta=
                    rho= phi= z=
                    rho= phi= theta=
                    rho= phi= eta=""".replace("                    ", "    ")
                if type(self) == VectorSympy3D:
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
        return "VectorSympy3D(" + ", ".join(out) + ")"

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
            azcoords = _coord_sympy_type[returns[0]](result[0], result[1])
            lcoords = _coord_sympy_type[returns[1]](result[2])
            tcoords = _coord_sympy_type[returns[2]](result[3])
            return cls.ProjectionClass4D(
                azimuthal=azcoords, longitudinal=lcoords, temporal=tcoords
            )

        else:
            raise AssertionError(repr(returns))

    @property
    def x(self) -> sympy.Symbol:
        return super().x

    @x.setter
    def x(self, x: sympy.Symbol) -> None:
        self.azimuthal = AzimuthalSympyXY(x, self.y)

    @property
    def y(self) -> sympy.Symbol:
        return super().y

    @y.setter
    def y(self, y: sympy.Symbol) -> None:
        self.azimuthal = AzimuthalSympyXY(self.x, y)

    @property
    def rho(self) -> sympy.Symbol:
        return super().rho

    @rho.setter
    def rho(self, rho: sympy.Symbol) -> None:
        self.azimuthal = AzimuthalSympyRhoPhi(rho, self.phi)

    @property
    def phi(self) -> sympy.Symbol:
        return super().phi

    @phi.setter
    def phi(self, phi: sympy.Symbol) -> None:
        self.azimuthal = AzimuthalSympyRhoPhi(self.rho, phi)

    @property
    def z(self) -> sympy.Symbol:
        return super().z

    @z.setter
    def z(self, z: sympy.Symbol) -> None:
        self.longitudinal = LongitudinalSympyZ(z)

    @property
    def theta(self) -> sympy.Symbol:
        return super().theta

    @theta.setter
    def theta(self, theta: sympy.Symbol) -> None:
        self.longitudinal = LongitudinalSympyTheta(theta)

    @property
    def eta(self) -> sympy.Symbol:
        return super().eta

    @eta.setter
    def eta(self, eta: sympy.Symbol) -> None:
        self.longitudinal = LongitudinalSympyEta(eta)


class MomentumSympy3D(SpatialMomentum, VectorSympy3D):
    """
    Three dimensional momentum vector class for the SymPy backend.

    Examples:
        >>> import vector; import sympy
        >>> vec = vector.MomentumSympy3D(px=sympy.Symbol("px"), py=sympy.Symbol("py"), pz=sympy.Symbol("pz"))
        >>> vec.px, vec.py, vec.pz
        (px, py, pz)
        >>> vec = vector.MomentumSympy3D(pt=sympy.Symbol("pt"), phi=sympy.Symbol("phi"), pz=sympy.Symbol("pz"))
        >>> vec.pt, vec.phi, vec.pz
        (pt, phi, pz)
        >>> vec = vector.MomentumSympy3D(
        ...     azimuthal=vector.backends.sympy.AzimuthalSympyXY(sympy.Symbol("x"), sympy.Symbol("y")),
        ...     longitudinal=vector.backends.sympy.LongitudinalSympyTheta(sympy.Symbol("theta"))
        ... )
        >>> vec.x, vec.y, vec.theta
        (x, y, theta)

    For three dimensional SymPy vectors, see
    :class:`vector.backends.sympy.VectorSympy3D`.
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
        return "MomentumSympy3D(" + ", ".join(out) + ")"

    @property
    def px(self) -> sympy.Symbol:
        return super().px

    @px.setter
    def px(self, px: sympy.Symbol) -> None:
        self.azimuthal = AzimuthalSympyXY(px, self.py)

    @property
    def py(self) -> sympy.Symbol:
        return super().py

    @py.setter
    def py(self, py: sympy.Symbol) -> None:
        self.azimuthal = AzimuthalSympyXY(self.px, py)

    @property
    def pt(self) -> sympy.Symbol:
        return super().pt

    @pt.setter
    def pt(self, pt: sympy.Symbol) -> None:
        self.azimuthal = AzimuthalSympyRhoPhi(pt, self.phi)

    @property
    def pz(self) -> sympy.Symbol:
        return super().pz

    @pz.setter
    def pz(self, pz: sympy.Symbol) -> None:
        self.longitudinal = LongitudinalSympyZ(pz)


class VectorSympy4D(VectorSympy, Lorentz, Vector4D):
    """
    Four dimensional vector class for the SymPy backend.

    Examples:
        >>> import vector; import sympy
        >>> vec = vector.VectorSympy4D(x=sympy.Symbol("x"), y=sympy.Symbol("y"), z=sympy.Symbol("z"), t=sympy.Symbol("t"))
        >>> vec.x, vec.y, vec.z, vec.t
        (x, y, z, t)
        >>> vec = vector.VectorSympy4D(rho=sympy.Symbol("rho"), phi=sympy.Symbol("phi"), eta=sympy.Symbol("eta"), tau=sympy.Symbol("tau"))
        >>> vec.rho, vec.phi, vec.eta, vec.tau
        (rho, phi, eta, tau)
        >>> vec = vector.VectorSympy4D(
        ...     azimuthal=vector.backends.sympy.AzimuthalSympyXY(sympy.Symbol("x"), sympy.Symbol("y")),
        ...     longitudinal=vector.backends.sympy.LongitudinalSympyTheta(sympy.Symbol("theta")),
        ...     temporal=vector.backends.sympy.TemporalSympyTau(sympy.Symbol("tau"))
        ... )
        >>> vec.x, vec.y, vec.theta, vec.tau
        (x, y, theta, tau)

    For four dimensional momentum SymPy vectors, see
    :class:`vector.backends.sympy.MomentumSympy4D`.
    """

    __slots__ = ("azimuthal", "longitudinal", "temporal")

    azimuthal: AzimuthalSympy
    longitudinal: LongitudinalSympy
    temporal: TemporalSympy

    def __init__(
        self,
        azimuthal: AzimuthalSympy | None = None,
        longitudinal: LongitudinalSympy | None = None,
        temporal: TemporalSympy | None = None,
        **kwargs: sympy.Symbol,
    ):
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
            _is_type_safe(kwargs)
            if set(kwargs) == {"x", "y", "z", "t"}:
                self.azimuthal = AzimuthalSympyXY(kwargs["x"], kwargs["y"])
                self.longitudinal = LongitudinalSympyZ(kwargs["z"])
                self.temporal = TemporalSympyT(kwargs["t"])
            elif set(kwargs) == {"x", "y", "eta", "t"}:
                self.azimuthal = AzimuthalSympyXY(kwargs["x"], kwargs["y"])
                self.longitudinal = LongitudinalSympyEta(kwargs["eta"])
                self.temporal = TemporalSympyT(kwargs["t"])
            elif set(kwargs) == {"x", "y", "theta", "t"}:
                self.azimuthal = AzimuthalSympyXY(kwargs["x"], kwargs["y"])
                self.longitudinal = LongitudinalSympyTheta(kwargs["theta"])
                self.temporal = TemporalSympyT(kwargs["t"])
            elif set(kwargs) == {"rho", "phi", "z", "t"}:
                self.azimuthal = AzimuthalSympyRhoPhi(kwargs["rho"], kwargs["phi"])
                self.longitudinal = LongitudinalSympyZ(kwargs["z"])
                self.temporal = TemporalSympyT(kwargs["t"])
            elif set(kwargs) == {"rho", "phi", "eta", "t"}:
                self.azimuthal = AzimuthalSympyRhoPhi(kwargs["rho"], kwargs["phi"])
                self.longitudinal = LongitudinalSympyEta(kwargs["eta"])
                self.temporal = TemporalSympyT(kwargs["t"])
            elif set(kwargs) == {"rho", "phi", "theta", "t"}:
                self.azimuthal = AzimuthalSympyRhoPhi(kwargs["rho"], kwargs["phi"])
                self.longitudinal = LongitudinalSympyTheta(kwargs["theta"])
                self.temporal = TemporalSympyT(kwargs["t"])
            elif set(kwargs) == {"x", "y", "z", "tau"}:
                self.azimuthal = AzimuthalSympyXY(kwargs["x"], kwargs["y"])
                self.longitudinal = LongitudinalSympyZ(kwargs["z"])
                self.temporal = TemporalSympyTau(kwargs["tau"])
            elif set(kwargs) == {"x", "y", "eta", "tau"}:
                self.azimuthal = AzimuthalSympyXY(kwargs["x"], kwargs["y"])
                self.longitudinal = LongitudinalSympyEta(kwargs["eta"])
                self.temporal = TemporalSympyTau(kwargs["tau"])
            elif set(kwargs) == {"x", "y", "theta", "tau"}:
                self.azimuthal = AzimuthalSympyXY(kwargs["x"], kwargs["y"])
                self.longitudinal = LongitudinalSympyTheta(kwargs["theta"])
                self.temporal = TemporalSympyTau(kwargs["tau"])
            elif set(kwargs) == {"rho", "phi", "z", "tau"}:
                self.azimuthal = AzimuthalSympyRhoPhi(kwargs["rho"], kwargs["phi"])
                self.longitudinal = LongitudinalSympyZ(kwargs["z"])
                self.temporal = TemporalSympyTau(kwargs["tau"])
            elif set(kwargs) == {"rho", "phi", "eta", "tau"}:
                self.azimuthal = AzimuthalSympyRhoPhi(kwargs["rho"], kwargs["phi"])
                self.longitudinal = LongitudinalSympyEta(kwargs["eta"])
                self.temporal = TemporalSympyTau(kwargs["tau"])
            elif set(kwargs) == {"rho", "phi", "theta", "tau"}:
                self.azimuthal = AzimuthalSympyRhoPhi(kwargs["rho"], kwargs["phi"])
                self.longitudinal = LongitudinalSympyTheta(kwargs["theta"])
                self.temporal = TemporalSympyTau(kwargs["tau"])
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
                if type(self) == VectorSympy4D:
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
        return "VectorSympy4D(" + ", ".join(out) + ")"

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
            azcoords = _coord_sympy_type[returns[0]](result[0], result[1])
            return cls.ProjectionClass2D(azimuthal=azcoords)

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
        ):
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
            azcoords = _coord_sympy_type[returns[0]](result[0], result[1])
            lcoords = _coord_sympy_type[returns[1]](result[2])
            tcoords = _coord_sympy_type[returns[2]](result[3])
            return cls.ProjectionClass4D(
                azimuthal=azcoords, longitudinal=lcoords, temporal=tcoords
            )

        else:
            raise AssertionError(repr(returns))

    @property
    def x(self) -> sympy.Symbol:
        return super().x

    @x.setter
    def x(self, x: sympy.Symbol) -> None:
        self.azimuthal = AzimuthalSympyXY(x, self.y)

    @property
    def y(self) -> sympy.Symbol:
        return super().y

    @y.setter
    def y(self, y: sympy.Symbol) -> None:
        self.azimuthal = AzimuthalSympyXY(self.x, y)

    @property
    def rho(self) -> sympy.Symbol:
        return super().rho

    @rho.setter
    def rho(self, rho: sympy.Symbol) -> None:
        self.azimuthal = AzimuthalSympyRhoPhi(rho, self.phi)

    @property
    def phi(self) -> sympy.Symbol:
        return super().phi

    @phi.setter
    def phi(self, phi: sympy.Symbol) -> None:
        self.azimuthal = AzimuthalSympyRhoPhi(self.rho, phi)

    @property
    def z(self) -> sympy.Symbol:
        return super().z

    @z.setter
    def z(self, z: sympy.Symbol) -> None:
        self.longitudinal = LongitudinalSympyZ(z)

    @property
    def theta(self) -> sympy.Symbol:
        return super().theta

    @theta.setter
    def theta(self, theta: sympy.Symbol) -> None:
        self.longitudinal = LongitudinalSympyTheta(theta)

    @property
    def eta(self) -> sympy.Symbol:
        return super().eta

    @eta.setter
    def eta(self, eta: sympy.Symbol) -> None:
        self.longitudinal = LongitudinalSympyEta(eta)

    @property
    def t(self) -> sympy.Symbol:
        return super().t

    @t.setter
    def t(self, t: sympy.Symbol) -> None:
        self.temporal = TemporalSympyT(t)

    @property
    def tau(self) -> sympy.Symbol:
        return super().tau

    @tau.setter
    def tau(self, tau: sympy.Symbol) -> None:
        self.temporal = TemporalSympyTau(tau)


class MomentumSympy4D(LorentzMomentum, VectorSympy4D):
    """
    Four dimensional momentum vector class for the SymPy backend.

    Examples:
        >>> import vector; import sympy
        >>> vec = vector.MomentumSympy4D(px=sympy.Symbol("px"), py=sympy.Symbol("py"), pz=sympy.Symbol("pz"), t=sympy.Symbol("t"))
        >>> vec.px, vec.py, vec.pz, vec.t
        (px, py, pz, t)
        >>> vec = vector.MomentumSympy4D(pt=sympy.Symbol("pt"), phi=sympy.Symbol("phi"), pz=sympy.Symbol("pz"), M=sympy.Symbol("M"))
        >>> vec.pt, vec.phi, vec.pz, vec.M
        (pt, phi, pz, M)
        >>> vec = vector.MomentumSympy4D(
        ...     azimuthal=vector.backends.sympy.AzimuthalSympyXY(sympy.Symbol("x"), sympy.Symbol("y")),
        ...     longitudinal=vector.backends.sympy.LongitudinalSympyTheta(sympy.Symbol("theta")),
        ...     temporal=vector.backends.sympy.TemporalSympyTau(sympy.Symbol("tau"))
        ... )
        >>> vec.x, vec.y, vec.theta, vec.tau
        (x, y, theta, tau)

    For four dimensional SymPy vectors, see
    :class:`vector.backends.sympy.VectorSympy4D`.
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
        return "MomentumSympy4D(" + ", ".join(out) + ")"

    @property
    def px(self) -> sympy.Symbol:
        return super().px

    @px.setter
    def px(self, px: sympy.Symbol) -> None:
        self.azimuthal = AzimuthalSympyXY(px, self.py)

    @property
    def py(self) -> sympy.Symbol:
        return super().py

    @py.setter
    def py(self, py: sympy.Symbol) -> None:
        self.azimuthal = AzimuthalSympyXY(self.px, py)

    @property
    def pt(self) -> sympy.Symbol:
        return super().pt

    @pt.setter
    def pt(self, pt: sympy.Symbol) -> None:
        self.azimuthal = AzimuthalSympyRhoPhi(pt, self.phi)

    @property
    def pz(self) -> sympy.Symbol:
        return super().pz

    @pz.setter
    def pz(self, pz: sympy.Symbol) -> None:
        self.longitudinal = LongitudinalSympyZ(pz)

    @property
    def E(self) -> sympy.Symbol:
        return super().E

    @E.setter
    def E(self, E: sympy.Symbol) -> None:
        self.temporal = TemporalSympyT(E)

    @property
    def e(self) -> sympy.Symbol:
        return super().e

    @e.setter
    def e(self, e: sympy.Symbol) -> None:
        self.temporal = TemporalSympyT(e)

    @property
    def energy(self) -> sympy.Symbol:
        return super().energy

    @energy.setter
    def energy(self, energy: sympy.Symbol) -> None:
        self.temporal = TemporalSympyT(energy)

    @property
    def M(self) -> sympy.Symbol:
        return super().M

    @M.setter
    def M(self, M: sympy.Symbol) -> None:
        self.temporal = TemporalSympyTau(M)

    @property
    def m(self) -> sympy.Symbol:
        return super().m

    @m.setter
    def m(self, m: sympy.Symbol) -> None:
        self.temporal = TemporalSympyTau(m)

    @property
    def mass(self) -> sympy.Symbol:
        return super().mass

    @mass.setter
    def mass(self, mass: sympy.Symbol) -> None:
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
