# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
Defines behaviors for NumPy Array. New arrays created with the

.. code-block:: python

    vector.array(...)

function will have these behaviors built in (and will pass them to any derived
arrays).
"""

from __future__ import annotations

import collections.abc
import typing

import numpy

import vector.backends.object
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
    Momentum,
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
    VectorProtocol,
    _aztype,
    _coordinate_class_to_names,
    _coordinate_order,
    _handler_of,
    _ltype,
    _repr_momentum_to_generic,
    _ttype,
)
from vector._typeutils import BoolCollection, FloatArray, ScalarCollection

ArrayLike = ScalarCollection

T = typing.TypeVar("T", bound="VectorNumpy")
V = typing.TypeVar("V")

SameVectorNumpyType = typing.TypeVar("SameVectorNumpyType", bound="VectorNumpy")


def _reduce_sum(
    a: T,
    axis: int | None = None,
    dtype: typing.Any = None,
    out: typing.Any = None,
    keepdims: bool = False,
    initial: typing.Any = None,
    where: typing.Any = None,
) -> T:
    if where is not None:
        raise ValueError("cannot invoke reducer with `where` argument")
    if initial is not None:
        raise ValueError("cannot invoke reducer with `initial` argument")
    if out is not None:
        raise ValueError("cannot invoke reducer with `out` argument")
    if dtype is not None:
        raise ValueError("cannot invoke reducer with `dtype` argument")

    fields: dict[str, typing.Any] = {}
    if isinstance(a, Lorentz):
        fields["E"] = numpy.sum(a.t, axis=axis, keepdims=keepdims)
    if isinstance(a, Spatial):
        fields["pz"] = numpy.sum(a.z, axis=axis, keepdims=keepdims)

    assert isinstance(a, Planar)
    fields["px"] = numpy.sum(a.x, axis=axis, keepdims=keepdims)
    fields["py"] = numpy.sum(a.y, axis=axis, keepdims=keepdims)

    # Convert between representations
    if not isinstance(a, Momentum):
        fields = {_repr_momentum_to_generic[n]: v for n, v in fields.items()}

    return array(fields)


def _reduce_count_nonzero(
    a: T, axis: int | None = None, *, keepdims: bool = False
) -> ScalarCollection:
    if isinstance(a, Planar):
        is_nonzero = a.rho2 != 0
    else:
        raise AssertionError
    if isinstance(a, Spatial):
        is_nonzero = numpy.logical_or(is_nonzero, a.z != 0)
    if isinstance(a, Lorentz):
        is_nonzero = numpy.logical_or(is_nonzero, a.t2 != 0)

    return numpy.count_nonzero(is_nonzero, axis=axis, keepdims=keepdims)


def _array_from_columns(columns: dict[str, ArrayLike]) -> ArrayLike:
    """
    Converts a dictionary (or columns) of coordinates to an array.

    Args:
        columns (dict): The dictionary of coordinates
            to be converted.

    Returns:
        ``np.ndarray``: A structured array of coordinates.

    Examples:
        >>> import vector
        >>> vector.backends.numpy._array_from_columns({"x": [1, 2, 3], "y": [1, 2, 4]})
        array([(1., 1.), (2., 2.), (3., 4.)], dtype=[('x', '<f8'), ('y', '<f8')])
    """
    if len(columns) == 0:
        raise ValueError("no columns have been provided")
    names = list(columns.keys())
    names.sort(
        key=lambda x: _coordinate_order.index(x)
        if x in _coordinate_order
        else float("inf")
    )

    dtype = []
    shape: tuple[int, ...] | None = None
    for x in names:
        if hasattr(columns[x], "dtype"):
            thisdtype = (x, columns[x].dtype)
        else:
            thisdtype = (x, numpy.float64)

        if hasattr(columns[x], "shape"):
            thisshape = columns[x].shape
        elif isinstance(columns[x], collections.abc.Sized):
            thisshape = (len(columns[x]),)
        else:
            raise TypeError(f"column {x!r} has no length")

        dtype.append(thisdtype)
        if shape is None:
            shape = thisshape
        elif shape != thisshape:
            raise ValueError(f"column {x!r} has a different shape than the others")

    assert shape is not None
    array = numpy.empty(shape, dtype)
    for x in names:
        array[x] = columns[x]
    return array


def _setitem(
    array: VectorNumpy2D | VectorNumpy3D | VectorNumpy4D,
    where: typing.Any,
    what: typing.Any,
    is_momentum: bool,
) -> None:
    if isinstance(where, str):
        if is_momentum:
            where = _repr_momentum_to_generic.get(where, where)
        array.view(numpy.ndarray)[where] = what
    else:
        if not hasattr(what, "dtype") or what.dtype.names is None:
            raise TypeError(
                "right-hand side of assignment must be a structured array with "
                "the same fields as " + type(array).__name__
            )

        tofill = array[where]
        for name in what.dtype.names:
            if is_momentum:
                generic = _repr_momentum_to_generic.get(name, name)
            tofill[generic] = what[name]


def _getitem(
    array: VectorNumpy2D | VectorNumpy3D | VectorNumpy4D,
    where: typing.Any,
    is_momentum: bool,
) -> float | FloatArray:
    """
    Implementation for the ``__getitem__`` method. See :class:`GetItem` for
    more details.
    """
    if isinstance(where, str):
        if is_momentum:
            where = _repr_momentum_to_generic.get(where, where)
        return array.view(numpy.ndarray)[where]
    else:
        out = numpy.ndarray.__getitem__(array, where)
        if not isinstance(out, numpy.void):
            return out

        azimuthal, longitudinal, temporal = None, None, None
        if hasattr(array, "_azimuthal_type"):
            azimuthal = array._azimuthal_type.ObjectClass(
                *(out[x] for x in _coordinate_class_to_names[_aztype(array)])
            )
        if hasattr(array, "_longitudinal_type"):
            longitudinal = array._longitudinal_type.ObjectClass(
                *(out[x] for x in _coordinate_class_to_names[_ltype(array)])  # type: ignore[arg-type]
            )
        if hasattr(array, "_temporal_type"):
            temporal = array._temporal_type.ObjectClass(
                *(out[x] for x in _coordinate_class_to_names[_ttype(array)])  # type: ignore[arg-type]
            )
        if temporal is not None:
            return array.ObjectClass(
                azimuthal=azimuthal,
                longitudinal=longitudinal,  # type: ignore[arg-type, return-value]
                temporal=temporal,  # type: ignore[arg-type]
            )
        elif longitudinal is not None:
            return array.ObjectClass(azimuthal=azimuthal, longitudinal=longitudinal)  # type: ignore[arg-type, return-value]
        elif azimuthal is not None:
            return array.ObjectClass(azimuthal=azimuthal)  # type: ignore[return-value]
        elif issubclass(array.ObjectClass, vector.backends.object.AzimuthalObject):
            return array.ObjectClass(*tuple(out)[:2])  # type: ignore[arg-type, return-value]
        elif issubclass(array.ObjectClass, vector.backends.object.LongitudinalObject):
            coords = (
                out.view(numpy.ndarray)[0]
                if len(out) == 1  # type: ignore[arg-type]
                else out.view(numpy.ndarray)[2]
            )
            return array.ObjectClass(coords)  # type: ignore[return-value]
        else:
            coords = (
                out.view(numpy.ndarray)[0]
                if len(out) == 1  # type: ignore[arg-type]
                else out.view(numpy.ndarray)[3]
            )
            return array.ObjectClass(coords)  # type: ignore[return-value]


def _array_repr(
    array: VectorNumpy2D | VectorNumpy3D | VectorNumpy4D,
    is_momentum: bool,
) -> str:
    """
    Constructs the value for ``__repr__`` function of the provided VectorNumpy
    class.
    """
    name = type(array).__name__
    vanilla_array = array.view(numpy.ndarray)
    return name + repr(vanilla_array)[5:].replace("\n     ", "\n" + " " * len(name))


def _has(
    array: VectorNumpy2D
    | VectorNumpy3D
    | VectorNumpy4D
    | MomentumNumpy2D
    | MomentumNumpy3D
    | MomentumNumpy4D
    | CoordinatesNumpy,
    names: tuple[str, ...],
) -> bool:
    """
    Checks if a NumPy vector has the provided coordinate attributes.

    Args:
        array (NumPy vector): A NumPy Vector whose coordinate attributes
            have to be checked.
        names (tuple): Names of the attributes.

    Returns:
        bool: If the attribute exists or not.

    Examples:
        >>> from vector.backends.numpy import _has
        >>> from vector._methods import _coordinate_class_to_names
        >>> vec = vector.array([
        ...     (1.1, 2.1), (1.2, 2.2), (1.3, 2.3), (1.4, 2.4), (1.5, 2.5)
        ... ], dtype=[("x", float), ("y", float)])
        >>> _has(vec, ("x", "y"))
        True
        >>> _has(vec, _coordinate_class_to_names[vector._methods.AzimuthalXY])
        True
        >>> _has(vec, _coordinate_class_to_names[vector._methods.AzimuthalRhoPhi])
        False
    """
    dtype_names = array.dtype.names
    if dtype_names is None:
        dtype_names = ()
    return all(x in dtype_names for x in names)


def _toarrays(
    result: tuple[ScalarCollection, ...] | ScalarCollection,
) -> tuple[FloatArray, ...]:
    """
    Converts a tuple of values to a tuple of ``numpy.array``s.

    Args:
        result (tuple): A tuple of values to be converted.

    Returns:
        tuple: A tuple of ``numpy.array``.

    Examples
        >>> from vector.backends.numpy import _toarrays
        >>> _toarrays((1, 2, 3, 4))
        (array([1.]), array([2.]), array([3.]), array([4.]))
        >>> _toarrays((1, 2, (1, 2, 3)))
        (array([1.]), array([2.]), array([[1., 2., 3.]]))
        >>> _toarrays((1, 2, (1, 2, False)))
        (array([1.]), array([2.]), array([[1., 2., 0.]]))
    """
    istuple = True
    if not isinstance(result, tuple):
        istuple = False
        result = (result,)
    result = tuple(
        x if isinstance(x, numpy.ndarray) else numpy.array([x], numpy.float64)
        for x in result
    )
    if istuple:
        return result
    else:
        return result[0]


def _shape_of(result: tuple[FloatArray, ...] | ScalarCollection) -> tuple[int, ...]:
    """
    Calculates the shape of a tuple of ``numpy.array``s. The shape returned
    is the highest (numerical) value of the shapes present in the tuple.

    Args:
        result (tuple): A tuple of ``numpy.array``s.

    Returns:
        tuple: The calculated shape.

    Examples:
        >>> from vector.backends.numpy import _shape_of
        >>> import numpy as np
        >>> _shape_of((np.array([1]), np.array([2])))
        (1,)
        >>> _shape_of((np.array([1]), np.array([2, 8]), np.array([0])))
        (2,)
        >>> _shape_of((np.array([1]), np.array([2, 8])))
        (2,)
    """
    if not isinstance(result, tuple):
        result = (result,)
    shape: list[int] | None = None
    for x in result:
        if hasattr(x, "shape"):
            thisshape = list(x.shape)
        elif isinstance(x, collections.abc.Sized):
            thisshape = [len(x)]
        if shape is None or thisshape[0] > shape[0]:
            shape = thisshape

    assert shape is not None
    return tuple(shape)


def _is_type_safe(
    array: VectorNumpy2D
    | VectorNumpy3D
    | VectorNumpy4D
    | MomentumNumpy2D
    | MomentumNumpy3D
    | MomentumNumpy4D
    | CoordinatesNumpy,
) -> bool:
    for i in range(len(array.dtype)):  # type: ignore[arg-type]
        if not issubclass(
            array.dtype[i].type, (numpy.integer, numpy.floating)
        ) or issubclass(array.dtype[i].type, numpy.timedelta64):
            return False

    return True


class GetItem:
    _IS_MOMENTUM: typing.ClassVar[bool]

    @typing.overload
    def __getitem__(self, where: str) -> FloatArray: ...

    @typing.overload
    def __getitem__(self, where: typing.Any) -> float | FloatArray: ...

    def __getitem__(self, where: typing.Any) -> float | FloatArray:
        return _getitem(self, where, self.__class__._IS_MOMENTUM)  # type: ignore[arg-type]


class CoordinatesNumpy:
    """Coordinates class for the Numpy backend."""

    lib = numpy
    dtype: numpy.dtype[typing.Any]


class AzimuthalNumpy(CoordinatesNumpy, Azimuthal):
    """Azimuthal class for the NumPy backend."""

    ObjectClass: type[vector.backends.object.AzimuthalObject]


class LongitudinalNumpy(CoordinatesNumpy, Longitudinal):
    """Longitudinal class for the NumPy backend."""

    ObjectClass: type[vector.backends.object.LongitudinalObject]


class TemporalNumpy(CoordinatesNumpy, Temporal):
    """Temporal class for the NumPy backend."""

    ObjectClass: type[vector.backends.object.TemporalObject]


class AzimuthalNumpyXY(AzimuthalNumpy, AzimuthalXY, GetItem, FloatArray):  # type: ignore[misc]
    """
    Class for the ``x`` and ``y`` (azimuthal) coordinates of NumPy backend.
    Creates a structured NumPy array and returns it as an AzimuthalNumpyXY object.

    Examples:
        >>> import vector
        >>> vector.backends.numpy.AzimuthalNumpyXY([(1, 1), (2.1, 3.1)], dtype=[("x", float), ("y", float)])
        AzimuthalNumpyXY([(1. , 1. ), (2.1, 3.1)],
                         dtype=[('x', '<f8'), ('y', '<f8')])
    """

    ObjectClass = vector.backends.object.AzimuthalObjectXY
    _IS_MOMENTUM = False
    dtype: numpy.dtype[typing.Any] = numpy.dtype(
        [("x", numpy.float64), ("y", numpy.float64)]
    )

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> AzimuthalNumpyXY:
        if "dtype" in kwargs:
            AzimuthalNumpyXY.dtype = numpy.dtype(kwargs["dtype"])
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj: typing.Any) -> None:
        if not _has(self, ("x", "y")):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("x", "y")'
            )

    def __eq__(self, other: typing.Any) -> bool:
        if self.dtype != other.dtype or not isinstance(other, AzimuthalNumpyXY):
            return False
        return all(coord1 == coord2 for coord1, coord2 in zip(self, other))

    def __ne__(self, other: typing.Any) -> bool:
        if self.dtype != other.dtype or not isinstance(other, AzimuthalNumpyXY):
            return True
        return any(coord1 != coord2 for coord1, coord2 in zip(self, other))

    @property
    def elements(self) -> tuple[FloatArray, FloatArray]:
        """
        Azimuthal coordinates (``x`` and ``y``) as a tuple.

        Each coordinate is a NumPy array of values and not a vector.

        Examples:
            >>> import vector
            >>> vec = vector.backends.numpy.AzimuthalNumpyXY([(1, 1), (2.1, 3.1)], dtype=[("x", float), ("y", float)])
            >>> vec.elements
            (array([1. , 2.1]), array([1. , 3.1]))
        """
        return (self["x"], self["y"])

    @property
    def x(self) -> FloatArray:
        """The ``x`` coordinates."""
        return self["x"]

    @property
    def y(self) -> FloatArray:
        """The ``y`` coordinates."""
        return self["y"]


class AzimuthalNumpyRhoPhi(AzimuthalNumpy, AzimuthalRhoPhi, GetItem, FloatArray):  # type: ignore[misc]
    """
    Class for the ``rho`` and ``phi`` (azimuthal) coordinates of NumPy backend.
    Creates a structured NumPy array and returns it as an AzimuthalNumpyXY object.

    Examples:
        >>> import vector
        >>> vector.backends.numpy.AzimuthalNumpyRhoPhi([(1, 1), (2.1, 3.1)], dtype=[("rho", float), ("phi", float)])
        AzimuthalNumpyRhoPhi([(1. , 1. ), (2.1, 3.1)],
                             dtype=[('rho', '<f8'), ('phi', '<f8')])
    """

    ObjectClass = vector.backends.object.AzimuthalObjectRhoPhi
    _IS_MOMENTUM = False
    dtype: numpy.dtype[typing.Any] = numpy.dtype(
        [("rho", numpy.float64), ("phi", numpy.float64)]
    )

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> AzimuthalNumpyRhoPhi:
        if "dtype" in kwargs:
            AzimuthalNumpyRhoPhi.dtype = numpy.dtype(kwargs["dtype"])
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj: typing.Any) -> None:
        if not _has(self, ("rho", "phi")):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("rho", "phi")'
            )

    def __eq__(self, other: typing.Any) -> bool:
        if self.dtype != other.dtype or not isinstance(other, AzimuthalNumpyRhoPhi):
            return False
        return all(coord1 == coord2 for coord1, coord2 in zip(self, other))

    def __ne__(self, other: typing.Any) -> bool:
        if self.dtype != other.dtype or not isinstance(other, AzimuthalNumpyRhoPhi):
            return True
        return any(coord1 != coord2 for coord1, coord2 in zip(self, other))

    @property
    def elements(self) -> tuple[FloatArray, FloatArray]:
        """
        Azimuthal coordinates (``rho`` and ``phi``) as a tuple.

        Each coordinate is a NumPy array of values and not a vector.

        Examples:
            >>> import vector
            >>> vec = vector.backends.numpy.AzimuthalNumpyRhoPhi([(1, 1), (2.1, 3.1)], dtype=[("rho", float), ("phi", float)])
            >>> vec.elements
            (array([1. , 2.1]), array([1. , 3.1]))
        """
        return (self["rho"], self["phi"])

    @property
    def rho(self) -> FloatArray:
        """The ``rho`` coordinates."""
        return self["rho"]

    @property
    def phi(self) -> FloatArray:
        """The ``phi`` coordinates."""
        return self["phi"]


class LongitudinalNumpyZ(LongitudinalNumpy, LongitudinalZ, GetItem, FloatArray):  # type: ignore[misc]
    """
    Class for the ``z`` (longitudinal) coordinate of NumPy backend.
    Creates a structured NumPy array and returns it as a LongitudinalNumpyZ object.

    Examples:
        >>> import vector
        >>> vector.backends.numpy.LongitudinalNumpyZ([(1), (2.1)], dtype=[("z", float)])
        LongitudinalNumpyZ([(1. ,), (2.1,)], dtype=[('z', '<f8')])
    """

    ObjectClass = vector.backends.object.LongitudinalObjectZ
    _IS_MOMENTUM = False
    dtype: numpy.dtype[typing.Any] = numpy.dtype([("z", numpy.float64)])

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> LongitudinalNumpyZ:
        if "dtype" in kwargs:
            LongitudinalNumpyZ.dtype = numpy.dtype(kwargs["dtype"])
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj: typing.Any) -> None:
        if not _has(self, ("z",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "z"'
            )

    def __eq__(self, other: typing.Any) -> bool:
        if self.dtype != other.dtype or not isinstance(other, LongitudinalNumpyZ):
            return False
        return all(coord1 == coord2 for coord1, coord2 in zip(self, other))

    def __ne__(self, other: typing.Any) -> bool:
        if self.dtype != other.dtype or not isinstance(other, LongitudinalNumpyZ):
            return True
        return any(coord1 != coord2 for coord1, coord2 in zip(self, other))

    @property
    def elements(self) -> tuple[FloatArray]:
        """
        Longitudinal coordinates (``z``) as a tuple.

        Each coordinate is a NumPy array of values and not a vector.

        Examples:
            >>> import vector
            >>> vec = vector.backends.numpy.LongitudinalNumpyZ([(1), (2.1)], dtype=[("z", float)])
            >>> vec.elements
            (array([1. , 2.1]),)
        """
        return (self["z"],)

    @property
    def z(self) -> FloatArray:
        """The ``z`` coordinates."""
        return self["z"]


class LongitudinalNumpyTheta(LongitudinalNumpy, LongitudinalTheta, GetItem, FloatArray):  # type: ignore[misc]
    """
    Class for the ``theta`` (longitudinal) coordinate of NumPy backend.
    Creates a structured NumPy array and returns it as a LongitudinalNumpyTheta object.

    Examples:
        >>> import vector
        >>> vector.backends.numpy.LongitudinalNumpyTheta([(1), (2.1)], dtype=[("theta", float)])
        LongitudinalNumpyTheta([(1. ,), (2.1,)], dtype=[('theta', '<f8')])
    """

    ObjectClass = vector.backends.object.LongitudinalObjectTheta
    _IS_MOMENTUM = False
    dtype: numpy.dtype[typing.Any] = numpy.dtype([("theta", numpy.float64)])

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> LongitudinalNumpyTheta:
        if "dtype" in kwargs:
            LongitudinalNumpyTheta.dtype = numpy.dtype(kwargs["dtype"])
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj: typing.Any) -> None:
        if not _has(self, ("theta",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "theta"'
            )

    def __eq__(self, other: typing.Any) -> bool:
        if self.dtype != other.dtype or not isinstance(other, LongitudinalNumpyTheta):
            return False
        return all(coord1 == coord2 for coord1, coord2 in zip(self, other))

    def __ne__(self, other: typing.Any) -> bool:
        if self.dtype != other.dtype or not isinstance(other, LongitudinalNumpyTheta):
            return True
        return any(coord1 != coord2 for coord1, coord2 in zip(self, other))

    @property
    def elements(self) -> tuple[FloatArray]:
        """
        Longitudinal coordinates (``theta``) as a tuple.

        Each coordinate is a NumPy array of values and not a vector.

        Examples:
            >>> import vector
            >>> vec = vector.backends.numpy.LongitudinalNumpyTheta([(1), (2.1)], dtype=[("theta", float)])
            >>> vec.elements
            (array([1. , 2.1]),)
        """
        return (self["theta"],)

    @property
    def theta(self) -> FloatArray:
        """The ``theta`` coordinates."""
        return self["theta"]


class LongitudinalNumpyEta(LongitudinalNumpy, LongitudinalEta, GetItem, FloatArray):  # type: ignore[misc]
    """
    Class for the ``eta`` (longitudinal) coordinate of NumPy backend.
    Creates a structured NumPy array and returns it as a LongitudinalNumpyEta object.

    Examples:
        >>> import vector
        >>> vector.backends.numpy.LongitudinalNumpyEta([(1), (2.1)], dtype=[("eta", float)])
        LongitudinalNumpyEta([(1. ,), (2.1,)], dtype=[('eta', '<f8')])
    """

    ObjectClass = vector.backends.object.LongitudinalObjectEta
    _IS_MOMENTUM = False
    dtype: numpy.dtype[typing.Any] = numpy.dtype([("eta", numpy.float64)])

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> LongitudinalNumpyEta:
        if "dtype" in kwargs:
            LongitudinalNumpyEta.dtype = numpy.dtype(kwargs["dtype"])
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj: typing.Any) -> None:
        if not _has(self, ("eta",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "eta"'
            )

    def __eq__(self, other: typing.Any) -> bool:
        if self.dtype != other.dtype or not isinstance(other, LongitudinalNumpyEta):
            return False
        return all(coord1 == coord2 for coord1, coord2 in zip(self, other))

    def __ne__(self, other: typing.Any) -> bool:
        if self.dtype != other.dtype or not isinstance(other, LongitudinalNumpyEta):
            return True
        return any(coord1 != coord2 for coord1, coord2 in zip(self, other))

    @property
    def elements(self) -> tuple[FloatArray]:
        """
        Longitudinal coordinates (``eta``) as a tuple.

        Each coordinate is a NumPy array of values and not a vector.

        Examples:
            >>> import vector
            >>> vec = vector.backends.numpy.LongitudinalNumpyTheta([(1), (2.1)], dtype=[("theta", float)])
            >>> vec.elements
            (array([1. , 2.1]),)
        """
        return (self["eta"],)

    @property
    def eta(self) -> FloatArray:
        """The ``eta`` coordinates."""
        return self["eta"]


class TemporalNumpyT(TemporalNumpy, TemporalT, GetItem, FloatArray):  # type: ignore[misc]
    """
    Class for the ``t`` (temporal) coordinate of NumPy backend.
    Creates a structured NumPy array and returns it as a TemporalNumpyT object.

    Examples:
        >>> import vector
        >>> vector.backends.numpy.TemporalNumpyT([(1), (2.1)], dtype=[("t", float)])
        TemporalNumpyT([(1. ,), (2.1,)], dtype=[('t', '<f8')])
    """

    ObjectClass = vector.backends.object.TemporalObjectT
    _IS_MOMENTUM = False
    dtype: numpy.dtype[typing.Any] = numpy.dtype([("t", numpy.float64)])

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> TemporalNumpyT:
        if "dtype" in kwargs:
            TemporalNumpyT.dtype = numpy.dtype(kwargs["dtype"])
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj: typing.Any) -> None:
        if not _has(self, ("t",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "t"'
            )

    def __eq__(self, other: typing.Any) -> bool:
        if self.dtype != other.dtype or not isinstance(other, TemporalNumpyT):
            return False
        return all(coord1 == coord2 for coord1, coord2 in zip(self, other))

    def __ne__(self, other: typing.Any) -> bool:
        if self.dtype != other.dtype or not isinstance(other, TemporalNumpyT):
            return True
        return any(coord1 != coord2 for coord1, coord2 in zip(self, other))

    @property
    def elements(self) -> tuple[FloatArray]:
        """
        Temporal coordinates (``t``) as a tuple.

        Each coordinate is a NumPy array of values and not a vector.

        Examples:
            >>> import vector
            >>> vec = vector.backends.numpy.TemporalNumpyT([(1), (2.1)], dtype=[("t", float)])
            >>> vec.elements
            (array([1. , 2.1]),)
        """
        return (self["t"],)

    @property
    def t(self) -> FloatArray:
        """The ``t`` coordinates."""
        return self["t"]


class TemporalNumpyTau(TemporalNumpy, TemporalTau, GetItem, FloatArray):  # type: ignore[misc]
    """Class for the ``tau`` (temporal) coordinate of NumPy backend."""

    ObjectClass = vector.backends.object.TemporalObjectTau
    _IS_MOMENTUM = False
    dtype: numpy.dtype[typing.Any] = numpy.dtype([("tau", numpy.float64)])

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> TemporalNumpyTau:
        if "dtype" in kwargs:
            TemporalNumpyTau.dtype = numpy.dtype(kwargs["dtype"])
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj: typing.Any) -> None:
        if not _has(self, ("tau",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "tau"'
            )

    def __eq__(self, other: typing.Any) -> bool:
        if self.dtype != other.dtype or not isinstance(other, TemporalNumpyTau):
            return False
        return all(coord1 == coord2 for coord1, coord2 in zip(self, other))

    def __ne__(self, other: typing.Any) -> bool:
        if self.dtype != other.dtype or not isinstance(other, TemporalNumpyTau):
            return True
        return any(coord1 != coord2 for coord1, coord2 in zip(self, other))

    @property
    def elements(self) -> tuple[FloatArray]:
        """
        Temporal coordinates (``tau``) as a tuple.

        Each coordinate is a NumPy array of values and not a vector.

        Examples:
            >>> import vector
            >>> vec = vector.backends.numpy.TemporalNumpyTau([(1), (2.1)], dtype=[("tau", float)])
            >>> vec.elements
            (array([1. , 2.1]),)
        """
        return (self["tau"],)

    @property
    def tau(self) -> FloatArray:
        """The ``tau`` coordinates."""
        return self["tau"]


class VectorNumpy(Vector, GetItem):
    """Mixin class for NumPy vectors."""

    lib = numpy
    dtype: numpy.dtype[typing.Any]

    def allclose(
        self,
        other: VectorProtocol,
        rtol: float | FloatArray = 1e-05,
        atol: float | FloatArray = 1e-08,
        equal_nan: bool | FloatArray = False,
    ) -> BoolCollection:
        """Like ``np.ndarray.allclose``, but for VectorNumpy."""
        return self.isclose(other, rtol=rtol, atol=atol, equal_nan=equal_nan).all()

    def sum(
        self: SameVectorNumpyType,
        axis: int | None = None,
        dtype: numpy.dtype[typing.Any] | str | None = None,
        out: ArrayLike | None = None,
        keepdims: bool = False,
        initial: typing.Any = None,
        where: typing.Any = None,
    ) -> SameVectorNumpyType:
        return typing.cast(
            SameVectorNumpyType,
            # pylint: disable-next=unexpected-keyword-arg
            numpy.sum(
                self,
                axis=axis,
                dtype=dtype,
                out=out,
                keepdims=keepdims,
                initial=initial,
                where=where,
            ),  # type: ignore[call-overload]
        )

    def __eq__(self, other: typing.Any) -> typing.Any:
        return numpy.equal(self, other)  # type: ignore[call-overload]

    def __ne__(self, other: typing.Any) -> typing.Any:
        return numpy.not_equal(self, other)  # type: ignore[call-overload]

    def __reduce__(self) -> str | tuple[typing.Any, ...]:
        pickled_state = super().__reduce__()
        new_state = (*pickled_state[2], self.__dict__)
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state: typing.Any) -> None:
        self.__dict__.update(state[-1])
        super().__setstate__(state[0:-1])  # type: ignore[misc]

    def __array_ufunc__(
        self,
        ufunc: typing.Any,
        method: typing.Any,
        *inputs: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        """
        Implements NumPy's ``ufunc``s for ``VectorNumpy``. The current implementation
        includes ``numpy.absolute``, ``numpy.add``, ``numpy.subtract``, ``numpy.multiply``,
        ``numpy.positive``, ``numpy.negative``, ``numpy.true_divide``, ``numpy.power``,
        ``numpy.square``, ``numpy.sqrt``, ``numpy.cbrt``, ``numpy.matmul``, ``numpy.equal``,
        and ``numpy.not_equal``.
        """
        if not isinstance(_handler_of(*inputs), VectorNumpy):
            # Let a higher-precedence backend handle it.
            return NotImplemented

        outputs: tuple[VectorNumpy, ...] = kwargs.get("out", ())
        if any(not isinstance(x, VectorNumpy) for x in outputs):
            raise TypeError(
                "ufunc operating on VectorNumpys can only use the 'out' keyword "
                "with another VectorNumpy"
            )

        if (
            ufunc is numpy.absolute
            and len(inputs) == 1
            and isinstance(inputs[0], Vector)
        ):
            if len(outputs) != 0:
                raise TypeError(
                    "output of 'numpy.absolute' is scalar, cannot fill a VectorNumpy with 'out'"
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
                assert output.dtype.names is not None
                for name in output.dtype.names:
                    output[name] = result[name]  # type: ignore[index]
            return result

        elif (
            ufunc is numpy.subtract
            and len(inputs) == 2
            and isinstance(inputs[0], Vector)
            and isinstance(inputs[1], Vector)
        ):
            result = inputs[0].subtract(inputs[1])
            for output in outputs:
                assert output.dtype.names is not None
                for name in output.dtype.names:
                    output[name] = result[name]  # type: ignore[index]
            return result

        elif (
            ufunc is numpy.multiply
            and len(inputs) == 2
            and isinstance(inputs[0], Vector)
            and not isinstance(inputs[1], Vector)
        ):
            result = inputs[0].scale(inputs[1])
            for output in outputs:
                assert output.dtype.names is not None
                for name in output.dtype.names:
                    output[name] = result[name]  # type: ignore[index]
            return result

        elif (
            ufunc is numpy.multiply
            and len(inputs) == 2
            and not isinstance(inputs[0], Vector)
            and isinstance(inputs[1], Vector)
        ):
            result = inputs[1].scale(inputs[0])
            for output in outputs:
                assert output.dtype.names is not None
                for name in output.dtype.names:
                    output[name] = result[name]  # type: ignore[index]
            return result

        elif (
            ufunc is numpy.negative
            and len(inputs) == 1
            and isinstance(inputs[0], Vector)
        ):
            result = inputs[0].scale(-1)
            for output in outputs:
                assert output.dtype.names is not None
                for name in output.dtype.names:
                    output[name] = result[name]  # type: ignore[index]
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
                assert output.dtype.names is not None
                for name in output.dtype.names:
                    output[name] = result[name]  # type: ignore[index]
            return result

        elif (
            ufunc is numpy.power
            and len(inputs) == 2
            and isinstance(inputs[0], Vector)
            and not isinstance(inputs[1], Vector)
        ):
            result = numpy.absolute(inputs[0]) ** inputs[1]
            for output in outputs:
                assert output.dtype.names is not None
                for name in output.dtype.names:
                    output[name] = result[name]  # type: ignore[index]
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

    def __array_function__(
        self, func: typing.Any, types: typing.Any, args: typing.Any, kwargs: typing.Any
    ) -> typing.Any:
        """
        Implements NumPy's function for ``VectorNumpy`` and its subclasses. The current
        implementation includes ``numpy.isclose`` and ``numpy.allclose``.
        """
        if func is numpy.isclose:
            return type(self).isclose(*args, **kwargs)
        elif func is numpy.allclose:
            return type(self).allclose(*args, **kwargs)
        elif func is numpy.sum:
            return _reduce_sum(*args, **kwargs)
        elif func is numpy.count_nonzero:
            return _reduce_count_nonzero(*args, **kwargs)
        else:
            return NotImplemented


class VectorNumpy2D(VectorNumpy, Planar, Vector2D, FloatArray):  # type: ignore[misc]
    """
    Two dimensional vector class for the NumPy backend. This class can be directly
    used to construct two dimensional NumPy vectors. For two dimensional Momentum
    NumPy vectors see :class:`vector.backends.numpy.MomentumNumpy2D`.

    Examples:
        >>> import vector
        >>> vec = vector.VectorNumpy2D([(1.1, 2.1), (1.2, 2.2), (1.3, 2.3), (1.4, 2.4), (1.5, 2.5)],
        ...               dtype=[('x', float), ('y', float)])
        >>> vec
        VectorNumpy2D([(1.1, 2.1), (1.2, 2.2), (1.3, 2.3), (1.4, 2.4), (1.5, 2.5)],
                      dtype=[('x', '<f8'), ('y', '<f8')])
    """

    ObjectClass = vector.backends.object.VectorObject2D
    _IS_MOMENTUM = False
    _azimuthal_type: type[AzimuthalNumpyXY] | type[AzimuthalNumpyRhoPhi]

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> VectorNumpy2D:
        """Returns the object of ``VectorNumpy2D``. Behaves as ``__init__`` in this case."""
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            array = _array_from_columns(args[0])
        else:
            array = numpy.array(*args, **kwargs)
        return array.view(cls)

    def __array_finalize__(self, obj: typing.Any) -> None:
        if obj is None:
            return

        if _has(self, ("x", "y")):
            self._azimuthal_type = AzimuthalNumpyXY
        elif _has(self, ("rho", "phi")):
            self._azimuthal_type = AzimuthalNumpyRhoPhi
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("x", "y") or ("rho", "phi")'
            )

        if not _is_type_safe(self):
            raise TypeError(
                "a coordinate must be of the type numpy.integer or numpy.floating"
            )

    def __str__(self) -> str:
        return str(self.view(numpy.ndarray))

    def __repr__(self) -> str:
        return _array_repr(self, False)

    @property
    def azimuthal(self) -> AzimuthalNumpy:
        """
        Returns the azimuthal type class for the given ``VectorNumpy2D`` object.

        Examples:
            >>> import vector
            >>> vec = vector.array([
            ...     (1.1, 2.1), (1.2, 2.2), (1.3, 2.3), (1.4, 2.4), (1.5, 2.5)
            ... ], dtype=[("x", float), ("y", float)])
            >>> vec.azimuthal
            AzimuthalNumpyXY([(1.1, 2.1), (1.2, 2.2), (1.3, 2.3), (1.4, 2.4),
                              (1.5, 2.5)], dtype=[('x', '<f8'), ('y', '<f8')])
        """
        return self.view(self._azimuthal_type)  # type: ignore[return-value]

    def _wrap_result(
        self,
        cls: typing.Any,
        result: typing.Any,
        returns: typing.Any,
        num_vecargs: typing.Any,
    ) -> typing.Any:
        """
        Wraps the raw result of a compute function as an array of scalars or an
        array of vectors.

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
            or (len(returns) == 2 and returns[1] is None)
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
        ):
            result = _toarrays(result)
            dtype = [
                (name, result[i].dtype)
                for i, name in enumerate(_coordinate_class_to_names[returns[0]])
            ]

            out = numpy.empty(_shape_of(result), dtype=dtype)
            for i, name in enumerate(_coordinate_class_to_names[returns[0]]):
                out[name] = result[i]
            return out.view(cls.ProjectionClass2D)

        elif (
            (len(returns) == 2 or (len(returns) == 3 and returns[2] is None))
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
        ):
            result = _toarrays(result)
            dtype = []
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                dtype.append((name, result[i].dtype))
                i += 1
            for name in _coordinate_class_to_names[returns[1]]:
                dtype.append((name, result[i].dtype))
                i += 1
            out = numpy.empty(_shape_of(result), dtype=dtype)
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                out[name] = result[i]
                i += 1
            for name in _coordinate_class_to_names[returns[1]]:
                out[name] = result[i]
                i += 1
            return out.view(cls.ProjectionClass3D)

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
            dtype = []
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                dtype.append((name, result[i].dtype))
                i += 1
            for name in _coordinate_class_to_names[returns[1]]:
                dtype.append((name, result[i].dtype))
                i += 1
            for name in _coordinate_class_to_names[returns[2]]:
                dtype.append((name, result[i].dtype))
                i += 1
            out = numpy.empty(_shape_of(result), dtype=dtype)
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                out[name] = result[i]
                i += 1
            for name in _coordinate_class_to_names[returns[1]]:
                out[name] = result[i]
                i += 1
            for name in _coordinate_class_to_names[returns[2]]:
                out[name] = result[i]
                i += 1
            return out.view(cls.ProjectionClass4D)

        else:
            raise AssertionError(repr(returns))

    def __setitem__(self, where: typing.Any, what: typing.Any) -> None:
        return _setitem(self, where, what, False)


class MomentumNumpy2D(PlanarMomentum, VectorNumpy2D):  # type: ignore[misc]
    """
    Two dimensional momentum vector class for the NumPy backend. This class can be directly
    used to construct two dimensional NumPy momentum vectors. For two dimensional
    NumPy vectors see :class:`vector.backends.numpy.VectorNumpy2D`.

    Examples:
        >>> import vector
        >>> vec = vector.MomentumNumpy2D([(1.1, 2.1), (1.2, 2.2), (1.3, 2.3), (1.4, 2.4), (1.5, 2.5)],
        ...               dtype=[('px', float), ('py', float)])
        >>> vec
        MomentumNumpy2D([(1.1, 2.1), (1.2, 2.2), (1.3, 2.3), (1.4, 2.4), (1.5, 2.5)],
                        dtype=[('x', '<f8'), ('y', '<f8')])
    """

    ObjectClass = vector.backends.object.MomentumObject2D
    _IS_MOMENTUM = True
    dtype: numpy.dtype[typing.Any]

    def __array_finalize__(self, obj: typing.Any) -> None:
        if obj is None:
            return

        self.dtype.names = tuple(
            _repr_momentum_to_generic.get(x, x) for x in (self.dtype.names or ())
        )

        if _has(self, ("x", "y")):
            self._azimuthal_type = AzimuthalNumpyXY
        elif _has(self, ("rho", "phi")):
            self._azimuthal_type = AzimuthalNumpyRhoPhi
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("x", "y") or ("rho", "phi") or ("px", "py") or ("pt", "phi")'
            )

        if not _is_type_safe(self):
            raise TypeError(
                "a coordinate must be of the type numpy.integer or numpy.floating"
            )

    def __repr__(self) -> str:
        return _array_repr(self, True)

    def __setitem__(self, where: typing.Any, what: typing.Any) -> None:
        return _setitem(self, where, what, True)


class VectorNumpy3D(VectorNumpy, Spatial, Vector3D, FloatArray):  # type: ignore[misc]
    """
    Three dimensional vector class for the NumPy backend. This class can be directly
    used to construct three dimensional NumPy vectors. For three dimensional Momentum
    NumPy vectors see :class:`vector.backends.numpy.MomentumNumpy3D`.

    Examples:
        >>> import vector
        >>> vec = vector.VectorNumpy3D([(1.1, 2.1, 3.1), (1.2, 2.2, 3.2), (1.3, 2.3, 3.3), (1.4, 2.4, 3.4), (1.5, 2.5, 3.5)],
        ...               dtype=[('x', float), ('y', float), ('z', float)])
        >>> vec
        VectorNumpy3D([(1.1, 2.1, 3.1), (1.2, 2.2, 3.2), (1.3, 2.3, 3.3), (1.4, 2.4, 3.4),
                       (1.5, 2.5, 3.5)], dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8')])
    """

    ObjectClass = vector.backends.object.VectorObject3D
    _IS_MOMENTUM = False

    _azimuthal_type: type[AzimuthalNumpyXY] | type[AzimuthalNumpyRhoPhi]
    _longitudinal_type: (
        type[LongitudinalNumpyZ]
        | type[LongitudinalNumpyTheta]
        | type[LongitudinalNumpyEta]
    )

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> VectorNumpy3D:
        """Returns the object of ``VectorNumpy3D``. Behaves as ``__init__`` in this case."""
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            array = _array_from_columns(args[0])
        else:
            array = numpy.array(*args, **kwargs)
        return array.view(cls)

    def __array_finalize__(self, obj: typing.Any) -> None:
        if obj is None:
            return

        if _has(self, ("x", "y")):
            self._azimuthal_type = AzimuthalNumpyXY
        elif _has(self, ("rho", "phi")):
            self._azimuthal_type = AzimuthalNumpyRhoPhi
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("x", "y") or ("rho", "phi")'
            )
        if _has(self, ("z",)):
            self._longitudinal_type = LongitudinalNumpyZ
        elif _has(self, ("theta",)):
            self._longitudinal_type = LongitudinalNumpyTheta
        elif _has(self, ("eta",)):
            self._longitudinal_type = LongitudinalNumpyEta
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "z" or "theta" or "eta"'
            )

        if not _is_type_safe(self):
            raise TypeError(
                "a coordinate must be of the type numpy.integer or numpy.floating"
            )

    def __str__(self) -> str:
        return str(self.view(numpy.ndarray))

    def __repr__(self) -> str:
        return _array_repr(self, False)

    @property
    def azimuthal(self) -> AzimuthalNumpy:
        """Returns the azimuthal type class for the given ``VectorNumpy3D`` object."""
        # TODO: Add an example here - see https://github.com/scikit-hep/vector/issues/194
        return self.view(self._azimuthal_type)  # type: ignore[return-value]

    @property
    def longitudinal(self) -> LongitudinalNumpy:
        """Returns the longitudinal type class for the given ``VectorNumpy3D`` object."""
        # TODO: Add an example here - see https://github.com/scikit-hep/vector/issues/194
        return self.view(self._longitudinal_type)  # type: ignore[return-value]

    def _wrap_result(
        self,
        cls: typing.Any,
        result: typing.Any,
        returns: typing.Any,
        num_vecargs: typing.Any,
    ) -> typing.Any:
        """
        Wraps the raw result of a compute function as an array of scalars or an
        array of vectors.

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
            dtype = [
                (name, result[i].dtype)
                for i, name in enumerate(_coordinate_class_to_names[returns[0]])
            ]

            for name in _coordinate_class_to_names[_ltype(self)]:
                dtype.append((name, self.dtype[name]))
            out = numpy.empty(_shape_of(result), dtype=dtype)
            for i, name in enumerate(_coordinate_class_to_names[returns[0]]):
                out[name] = result[i]
            for name in _coordinate_class_to_names[_ltype(self)]:
                out[name] = self[name]
            return out.view(cls.ProjectionClass3D)

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and returns[1] is None
        ):
            result = _toarrays(result)
            dtype = [
                (name, result[i].dtype)
                for i, name in enumerate(_coordinate_class_to_names[returns[0]])
            ]

            out = numpy.empty(_shape_of(result), dtype=dtype)
            for i, name in enumerate(_coordinate_class_to_names[returns[0]]):
                out[name] = result[i]
            return out.view(cls.ProjectionClass2D)

        elif (
            len(returns) == 2
            or (len(returns) == 3 and returns[2] is None)
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
        ):
            result = _toarrays(result)
            dtype = []
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                dtype.append((name, result[i].dtype))
                i += 1
            for name in _coordinate_class_to_names[returns[1]]:
                dtype.append((name, result[i].dtype))
                i += 1
            out = numpy.empty(_shape_of(result), dtype=dtype)
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                out[name] = result[i]
                i += 1
            for name in _coordinate_class_to_names[returns[1]]:
                out[name] = result[i]
                i += 1
            return out.view(cls.ProjectionClass3D)

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
            dtype = []
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                dtype.append((name, result[i].dtype))
                i += 1
            for name in _coordinate_class_to_names[returns[1]]:
                dtype.append((name, result[i].dtype))
                i += 1
            for name in _coordinate_class_to_names[returns[2]]:
                dtype.append((name, result[i].dtype))
                i += 1
            out = numpy.empty(_shape_of(result), dtype=dtype)
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                out[name] = result[i]
                i += 1
            for name in _coordinate_class_to_names[returns[1]]:
                out[name] = result[i]
                i += 1
            for name in _coordinate_class_to_names[returns[2]]:
                out[name] = result[i]
                i += 1
            return out.view(cls.ProjectionClass4D)

        else:
            raise AssertionError(repr(returns))

    def __setitem__(self, where: typing.Any, what: typing.Any) -> None:
        return _setitem(self, where, what, False)


class MomentumNumpy3D(SpatialMomentum, VectorNumpy3D):  # type: ignore[misc]
    """
    Three dimensional momentum vector class for the NumPy backend. This class can be directly
    used to construct three dimensional NumPy momentum vectors. For three dimensional
    NumPy vectors see :class:`vector.backends.numpy.VectorNumpy3D`.

    Examples:
        >>> import vector
        >>> vec = vector.MomentumNumpy3D([(1.1, 2.1, 3.1), (1.2, 2.2, 3.2), (1.3, 2.3, 3.3), (1.4, 2.4, 3.4), (1.5, 2.5, 3.5)],
        ...               dtype=[('px', float), ('py', float), ('pz', float)])
        >>> vec
        MomentumNumpy3D([(1.1, 2.1, 3.1), (1.2, 2.2, 3.2), (1.3, 2.3, 3.3), (1.4, 2.4, 3.4),
                         (1.5, 2.5, 3.5)], dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8')])
    """

    ObjectClass = vector.backends.object.MomentumObject3D
    _IS_MOMENTUM = True
    dtype: numpy.dtype[typing.Any]

    def __array_finalize__(self, obj: typing.Any) -> None:
        if obj is None:
            return

        self.dtype.names = tuple(
            _repr_momentum_to_generic.get(x, x) for x in (self.dtype.names or ())
        )
        if _has(self, ("x", "y")):
            self._azimuthal_type = AzimuthalNumpyXY
        elif _has(self, ("rho", "phi")):
            self._azimuthal_type = AzimuthalNumpyRhoPhi
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("x", "y") or ("rho", "phi") or ("px", "py") or ("pt", "phi")'
            )
        if _has(self, ("z",)):
            self._longitudinal_type = LongitudinalNumpyZ
        elif _has(self, ("theta",)):
            self._longitudinal_type = LongitudinalNumpyTheta
        elif _has(self, ("eta",)):
            self._longitudinal_type = LongitudinalNumpyEta
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "z" or "theta" or "eta" or "pz"'
            )

        if not _is_type_safe(self):
            raise TypeError(
                "a coordinate must be of the type numpy.integer or numpy.floating"
            )

    def __repr__(self) -> str:
        return _array_repr(self, True)

    def __setitem__(self, where: typing.Any, what: typing.Any) -> None:
        return _setitem(self, where, what, True)


class VectorNumpy4D(VectorNumpy, Lorentz, Vector4D, FloatArray):  # type: ignore[misc]
    """
    Four dimensional vector class for the NumPy backend. This class can be directly
    used to construct four dimensional NumPy vectors. For four dimensional Momentum
    NumPy vectors see :class:`vector.backends.numpy.MomentumNumpy4D`.

    Examples:
        >>> import vector
        >>> vec = vector.VectorNumpy4D([(1.1, 2.1, 3.1, 4.1), (1.2, 2.2, 3.2, 4.2), (1.3, 2.3, 3.3, 4.3), (1.4, 2.4, 3.4, 4.4), (1.5, 2.5, 3.5, 4.5)],
        ...               dtype=[('x', float), ('y', float), ('z', float), ('t', float)])
        >>> vec
        VectorNumpy4D([(1.1, 2.1, 3.1, 4.1), (1.2, 2.2, 3.2, 4.2), (1.3, 2.3, 3.3, 4.3),
                       (1.4, 2.4, 3.4, 4.4), (1.5, 2.5, 3.5, 4.5)],
                      dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('t', '<f8')])
    """

    ObjectClass = vector.backends.object.VectorObject4D
    _IS_MOMENTUM = False

    _azimuthal_type: type[AzimuthalNumpyXY] | type[AzimuthalNumpyRhoPhi]
    _longitudinal_type: (
        type[LongitudinalNumpyZ]
        | type[LongitudinalNumpyTheta]
        | type[LongitudinalNumpyEta]
    )
    _temporal_type: type[TemporalNumpyT] | type[TemporalNumpyTau]

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> VectorNumpy4D:
        """Returns the object of ``VectorNumpy4D``. Behaves as ``__init__`` in this case."""
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            array = _array_from_columns(args[0])
        else:
            array = numpy.array(*args, **kwargs)
        return array.view(cls)

    def __array_finalize__(self, obj: typing.Any) -> None:
        if obj is None:
            return

        if _has(self, ("x", "y")):
            self._azimuthal_type = AzimuthalNumpyXY
        elif _has(self, ("rho", "phi")):
            self._azimuthal_type = AzimuthalNumpyRhoPhi
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("x", "y") or ("rho", "phi")'
            )

        if _has(self, ("z",)):
            self._longitudinal_type = LongitudinalNumpyZ
        elif _has(self, ("theta",)):
            self._longitudinal_type = LongitudinalNumpyTheta
        elif _has(self, ("eta",)):
            self._longitudinal_type = LongitudinalNumpyEta
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "z" or "theta" or "eta"'
            )

        if _has(self, ("t",)):
            self._temporal_type = TemporalNumpyT
        elif _has(self, ("tau",)):
            self._temporal_type = TemporalNumpyTau
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "t" or "tau"'
            )

        if not _is_type_safe(self):
            raise TypeError(
                "a coordinate must be of the type numpy.integer or numpy.floating"
            )

    def __str__(self) -> str:
        return str(self.view(numpy.ndarray))

    def __repr__(self) -> str:
        return _array_repr(self, False)

    @property
    def azimuthal(self) -> AzimuthalNumpy:
        """Returns the azimuthal type class for the given ``VectorNumpy4D`` object."""
        # TODO: Add an example here - see https://github.com/scikit-hep/vector/issues/194
        return self.view(self._azimuthal_type)  # type: ignore[return-value]

    @property
    def longitudinal(self) -> LongitudinalNumpy:
        """Returns the longitudinal type class for the given ``Vectornumpy4D`` object."""
        # TODO: Add an example here - see https://github.com/scikit-hep/vector/issues/194
        return self.view(self._longitudinal_type)  # type: ignore[return-value]

    @property
    def temporal(self) -> TemporalNumpy:
        """Returns the azimuthal type class for the given ``VectorNumpy4D`` object."""
        # TODO: Add an example here - see https://github.com/scikit-hep/vector/issues/194
        return self.view(self._temporal_type)  # type: ignore[return-value]

    def _wrap_result(
        self,
        cls: typing.Any,
        result: typing.Any,
        returns: typing.Any,
        num_vecargs: typing.Any,
    ) -> typing.Any:
        """
        Wraps the raw result of a compute function as an array of scalars or an
        array of vectors.

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
            dtype = [
                (name, result[i].dtype)
                for i, name in enumerate(_coordinate_class_to_names[returns[0]])
            ]

            for name in _coordinate_class_to_names[_ltype(self)]:
                dtype.append((name, self.dtype[name]))
            for name in _coordinate_class_to_names[_ttype(self)]:
                dtype.append((name, self.dtype[name]))
            out = numpy.empty(_shape_of(result), dtype=dtype)
            for i, name in enumerate(_coordinate_class_to_names[returns[0]]):
                out[name] = result[i]
            for name in _coordinate_class_to_names[_ltype(self)]:
                out[name] = self[name]
            for name in _coordinate_class_to_names[_ttype(self)]:
                out[name] = self[name]
            return out.view(cls.ProjectionClass4D)

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and returns[1] is None
        ):
            result = _toarrays(result)
            dtype = [
                (name, result[i].dtype)
                for i, name in enumerate(_coordinate_class_to_names[returns[0]])
            ]

            out = numpy.empty(_shape_of(result), dtype=dtype)
            for i, name in enumerate(_coordinate_class_to_names[returns[0]]):
                out[name] = result[i]
            return out.view(cls.ProjectionClass2D)

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
        ):
            result = _toarrays(result)
            dtype = []
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                dtype.append((name, result[i].dtype))
                i += 1
            for name in _coordinate_class_to_names[returns[1]]:
                dtype.append((name, result[i].dtype))
                i += 1
            for name in _coordinate_class_to_names[_ttype(self)]:
                dtype.append((name, self.dtype[name]))
            out = numpy.empty(_shape_of(result), dtype=dtype)
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                out[name] = result[i]
                i += 1
            for name in _coordinate_class_to_names[returns[1]]:
                out[name] = result[i]
                i += 1
            for name in _coordinate_class_to_names[_ttype(self)]:
                out[name] = self[name]
            return out.view(cls.ProjectionClass4D)

        elif (
            len(returns) == 3
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
            and returns[2] is None
        ):
            result = _toarrays(result)
            dtype = []
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                dtype.append((name, result[i].dtype))
                i += 1
            for name in _coordinate_class_to_names[returns[1]]:
                dtype.append((name, result[i].dtype))
                i += 1
            out = numpy.empty(_shape_of(result), dtype=dtype)
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                out[name] = result[i]
                i += 1
            for name in _coordinate_class_to_names[returns[1]]:
                out[name] = result[i]
                i += 1
            return out.view(cls.ProjectionClass3D)

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
            dtype = []
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                dtype.append((name, result[i].dtype))
                i += 1
            for name in _coordinate_class_to_names[returns[1]]:
                dtype.append((name, result[i].dtype))
                i += 1
            for name in _coordinate_class_to_names[returns[2]]:
                dtype.append((name, result[i].dtype))
                i += 1
            out = numpy.empty(_shape_of(result), dtype=dtype)
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                out[name] = result[i]
                i += 1
            for name in _coordinate_class_to_names[returns[1]]:
                out[name] = result[i]
                i += 1
            for name in _coordinate_class_to_names[returns[2]]:
                out[name] = result[i]
                i += 1
            return out.view(cls.ProjectionClass4D)

        else:
            raise AssertionError(repr(returns))

    def __setitem__(self, where: typing.Any, what: typing.Any) -> None:
        return _setitem(self, where, what, False)


class MomentumNumpy4D(LorentzMomentum, VectorNumpy4D):  # type: ignore[misc]
    """
    Four dimensional momentum vector class for the NumPy backend. This class can be directly
    used to construct four dimensional NumPy momentum vectors. For three dimensional
    NumPy vectors see :class:`vector.backends.numpy.VectorNumpy4D`.

    Examples:
        >>> import vector
        >>> vec = vector.MomentumNumpy4D([(1.1, 2.1, 3.1, 4.1), (1.2, 2.2, 3.2, 4.2), (1.3, 2.3, 3.3, 4.3), (1.4, 2.4, 3.4, 4.4), (1.5, 2.5, 3.5, 4.5)],
        ...               dtype=[('px', float), ('py', float), ('pz', float), ('t', float)])
        >>> vec
        MomentumNumpy4D([(1.1, 2.1, 3.1, 4.1), (1.2, 2.2, 3.2, 4.2), (1.3, 2.3, 3.3, 4.3),
                         (1.4, 2.4, 3.4, 4.4), (1.5, 2.5, 3.5, 4.5)],
                        dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('t', '<f8')])
    """

    ObjectClass = vector.backends.object.MomentumObject4D
    _IS_MOMENTUM = True
    dtype: numpy.dtype[typing.Any]

    def __array_finalize__(self, obj: typing.Any) -> None:
        if obj is None:
            return

        self.dtype.names = tuple(
            _repr_momentum_to_generic.get(x, x) for x in (self.dtype.names or ())
        )

        if _has(self, ("x", "y")):
            self._azimuthal_type = AzimuthalNumpyXY
        elif _has(self, ("rho", "phi")):
            self._azimuthal_type = AzimuthalNumpyRhoPhi
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("x", "y") or ("rho", "phi") or ("px", "py") or ("pt", "phi")'
            )

        if _has(self, ("z",)):
            self._longitudinal_type = LongitudinalNumpyZ
        elif _has(self, ("theta",)):
            self._longitudinal_type = LongitudinalNumpyTheta
        elif _has(self, ("eta",)):
            self._longitudinal_type = LongitudinalNumpyEta
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "z" or "theta" or "eta" or "pz"'
            )

        if _has(self, ("t",)):
            self._temporal_type = TemporalNumpyT
        elif _has(self, ("tau",)):
            self._temporal_type = TemporalNumpyTau
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "t" or "tau" or "E" or "e" or "energy" or "M" or "m" or "mass"'
            )

        if not _is_type_safe(self):
            raise TypeError(
                "a coordinate must be of the type numpy.integer or numpy.floating"
            )

    def __repr__(self) -> str:
        return _array_repr(self, True)

    def __setitem__(self, where: typing.Any, what: typing.Any) -> None:
        return _setitem(self, where, what, True)


def array(*args: typing.Any, **kwargs: typing.Any) -> VectorNumpy:
    """
    Constructs a NumPy array of vectors, whose type is determined by the dtype
    of the structured array or Pandas-style "columns" argument.

    All allowed signatures for ``np.array`` can be used in this function, plus
    one more:

    .. code-block:: python

        vector.array({"x": x_column, "y": y_column})

    to make an array with ``dtype=[("x", x_column.dtype), ("y", y_column.dtype)]``.

    The array must have structured ``dtype`` (i.e. ``dtype.names is not None``)
    and the following combinations of names are allowed:

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
    """
    names: tuple[str, ...]
    if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
        names = tuple(args[0].keys())
    elif "dtype" in kwargs:
        names = numpy.dtype(kwargs["dtype"]).names or ()
    elif len(args) >= 2:
        names = numpy.dtype(args[1]).names or ()
    else:
        names = ()

    cls: type[VectorNumpy]

    is_momentum = any(x in _repr_momentum_to_generic for x in names)

    if any(x in ("t", "E", "e", "energy", "tau", "M", "m", "mass") for x in names):
        cls = MomentumNumpy4D if is_momentum else VectorNumpy4D
    elif any(x in ("z", "pz", "theta", "eta") for x in names):
        cls = MomentumNumpy3D if is_momentum else VectorNumpy3D
    else:
        cls = MomentumNumpy2D if is_momentum else VectorNumpy2D

    return cls(*args, **kwargs)


VectorNumpy2D.ProjectionClass2D = VectorNumpy2D
VectorNumpy2D.ProjectionClass3D = VectorNumpy3D
VectorNumpy2D.ProjectionClass4D = VectorNumpy4D
VectorNumpy2D.GenericClass = VectorNumpy2D
VectorNumpy2D.MomentumClass = MomentumNumpy2D

MomentumNumpy2D.ProjectionClass2D = MomentumNumpy2D
MomentumNumpy2D.ProjectionClass3D = MomentumNumpy3D
MomentumNumpy2D.ProjectionClass4D = MomentumNumpy4D
MomentumNumpy2D.GenericClass = VectorNumpy2D
MomentumNumpy2D.MomentumClass = MomentumNumpy2D

VectorNumpy3D.ProjectionClass2D = VectorNumpy2D
VectorNumpy3D.ProjectionClass3D = VectorNumpy3D
VectorNumpy3D.ProjectionClass4D = VectorNumpy4D
VectorNumpy3D.GenericClass = VectorNumpy3D
VectorNumpy3D.MomentumClass = MomentumNumpy3D

MomentumNumpy3D.ProjectionClass2D = MomentumNumpy2D
MomentumNumpy3D.ProjectionClass3D = MomentumNumpy3D
MomentumNumpy3D.ProjectionClass4D = MomentumNumpy4D
MomentumNumpy3D.GenericClass = VectorNumpy3D
MomentumNumpy3D.MomentumClass = MomentumNumpy3D

VectorNumpy4D.ProjectionClass2D = VectorNumpy2D
VectorNumpy4D.ProjectionClass3D = VectorNumpy3D
VectorNumpy4D.ProjectionClass4D = VectorNumpy4D
VectorNumpy4D.GenericClass = VectorNumpy4D
VectorNumpy4D.MomentumClass = MomentumNumpy4D

MomentumNumpy4D.ProjectionClass2D = MomentumNumpy2D
MomentumNumpy4D.ProjectionClass3D = MomentumNumpy3D
MomentumNumpy4D.ProjectionClass4D = MomentumNumpy4D
MomentumNumpy4D.GenericClass = VectorNumpy4D
MomentumNumpy4D.MomentumClass = MomentumNumpy4D
