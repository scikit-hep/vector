# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import collections.abc
import typing

import numpy

import vector._backends.object_
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
    VectorProtocol,
    _aztype,
    _coordinate_class_to_names,
    _coordinate_order,
    _handler_of,
    _ltype,
    _repr_momentum_to_generic,
    _ttype,
)
from vector._typeutils import ScalarCollection

ArrayLike = ScalarCollection


def _array_from_columns(columns: typing.Dict[str, ArrayLike]) -> ArrayLike:
    if len(columns) == 0:
        raise ValueError("no columns have been provided")
    names = list(columns.keys())
    names.sort(
        key=lambda x: _coordinate_order.index(x)
        if x in _coordinate_order
        else float("inf")
    )

    dtype = []
    shape: typing.Optional[typing.Tuple[int, ...]] = None
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
            raise TypeError(f"column {repr(x)} has no length")

        dtype.append(thisdtype)
        if shape is None:
            shape = thisshape
        elif shape != thisshape:
            raise ValueError(f"column {repr(x)} has a different shape than the others")

    assert shape is not None
    array = numpy.empty(shape, dtype)
    for x in names:
        array[x] = columns[x]
    return array


def _setitem(
    array: typing.Union["VectorNumpy2D", "VectorNumpy3D", "VectorNumpy4D"],
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
    array: typing.Union["VectorNumpy2D", "VectorNumpy3D", "VectorNumpy4D"],
    where: typing.Any,
    is_momentum: bool,
) -> typing.Union[float, numpy.ndarray]:
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
            longitudinal = array._longitudinal_type.ObjectClass(  # type: ignore
                *(out[x] for x in _coordinate_class_to_names[_ltype(array)])  # type: ignore
            )
        if hasattr(array, "_temporal_type"):
            temporal = array._temporal_type.ObjectClass(  # type: ignore
                *(out[x] for x in _coordinate_class_to_names[_ttype(array)])  # type: ignore
            )
        if temporal is not None:
            return array.ObjectClass(azimuthal, longitudinal, temporal)  # type: ignore
        elif longitudinal is not None:
            return array.ObjectClass(azimuthal, longitudinal)  # type: ignore
        elif azimuthal is not None:
            return array.ObjectClass(azimuthal)  # type: ignore
        else:
            return array.ObjectClass(*out.view(numpy.ndarray))  # type: ignore


def _array_repr(
    array: typing.Union["VectorNumpy2D", "VectorNumpy3D", "VectorNumpy4D"],
    is_momentum: bool,
) -> str:
    name = type(array).__name__
    array = array.view(numpy.ndarray)
    return name + repr(array)[5:].replace("\n     ", "\n" + " " * len(name))


def _has(
    array: typing.Union[
        "VectorNumpy2D",
        "VectorNumpy3D",
        "VectorNumpy4D",
        "MomentumNumpy2D",
        "MomentumNumpy3D",
        "MomentumNumpy4D",
        "CoordinatesNumpy",
    ],
    names: typing.Tuple[str, ...],
) -> bool:
    dtype_names = array.dtype.names
    if dtype_names is None:
        dtype_names = ()
    return all(x in dtype_names for x in names)


def _toarrays(
    result: typing.Tuple[ScalarCollection, ...]
) -> typing.Tuple[numpy.ndarray, ...]:
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


def _shape_of(result: typing.Tuple[numpy.ndarray, ...]) -> typing.Tuple[int, ...]:
    if not isinstance(result, tuple):
        result = (result,)  # type: ignore
    shape = None
    for x in result:
        if hasattr(x, "shape"):
            thisshape = list(x.shape)
        elif isinstance(x, collections.abc.Sized):
            thisshape = [len(x)]
        if shape is None or thisshape[0] > shape[0]:
            shape = thisshape

    assert shape is not None
    return tuple(shape)


class GetItem:
    _IS_MOMENTUM: typing.ClassVar[bool]

    @typing.overload
    def __getitem__(self, where: str) -> numpy.ndarray:
        ...

    @typing.overload
    def __getitem__(self, where: typing.Any) -> typing.Union[float, numpy.ndarray]:
        ...

    def __getitem__(self, where: typing.Any) -> typing.Union[float, numpy.ndarray]:
        return _getitem(self, where, self.__class__._IS_MOMENTUM)  # type: ignore


class CoordinatesNumpy:
    lib = numpy
    dtype: "numpy.dtype[typing.Any]"


class AzimuthalNumpy(CoordinatesNumpy, Azimuthal):
    ObjectClass: typing.Type[vector._backends.object_.AzimuthalObject]


class LongitudinalNumpy(CoordinatesNumpy, Longitudinal):
    ObjectClass: typing.Type[vector._backends.object_.LongitudinalObject]


class TemporalNumpy(CoordinatesNumpy, Temporal):
    ObjectClass: typing.Type[vector._backends.object_.TemporalObject]


class AzimuthalNumpyXY(AzimuthalNumpy, AzimuthalXY, GetItem, numpy.ndarray):
    ObjectClass = vector._backends.object_.AzimuthalObjectXY
    _IS_MOMENTUM = False

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> "AzimuthalNumpyXY":
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj: typing.Any) -> None:
        if not _has(self, ("x", "y")):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("x", "y")'
            )

    @property
    def elements(self) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
        return (self["x"], self["y"])

    @property
    def x(self) -> numpy.ndarray:
        return self["x"]

    @property
    def y(self) -> numpy.ndarray:
        return self["y"]


class AzimuthalNumpyRhoPhi(AzimuthalNumpy, AzimuthalRhoPhi, GetItem, numpy.ndarray):
    ObjectClass = vector._backends.object_.AzimuthalObjectRhoPhi
    _IS_MOMENTUM = False

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> "AzimuthalNumpyRhoPhi":
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj: typing.Any) -> None:
        if not _has(self, ("rho", "phi")):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("rho", "phi")'
            )

    @property
    def elements(self) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
        return (self["rho"], self["phi"])

    @property
    def rho(self) -> numpy.ndarray:
        return self["rho"]

    @property
    def phi(self) -> numpy.ndarray:
        return self["phi"]


class LongitudinalNumpyZ(LongitudinalNumpy, LongitudinalZ, GetItem, numpy.ndarray):
    ObjectClass = vector._backends.object_.LongitudinalObjectZ
    _IS_MOMENTUM = False

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> "LongitudinalNumpyZ":
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj: typing.Any) -> None:
        if not _has(self, ("z",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "z"'
            )

    @property
    def elements(self) -> typing.Tuple[numpy.ndarray]:
        return (self["z"],)

    @property
    def z(self) -> numpy.ndarray:
        return self["z"]


class LongitudinalNumpyTheta(
    LongitudinalNumpy, LongitudinalTheta, GetItem, numpy.ndarray
):
    ObjectClass = vector._backends.object_.LongitudinalObjectTheta
    _IS_MOMENTUM = False

    def __new__(
        cls, *args: typing.Any, **kwargs: typing.Any
    ) -> "LongitudinalNumpyTheta":
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj: typing.Any) -> None:
        if not _has(self, ("theta",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "theta"'
            )

    @property
    def elements(self) -> typing.Tuple[numpy.ndarray]:
        return (self["theta"],)

    @property
    def theta(self) -> numpy.ndarray:
        return self["theta"]


class LongitudinalNumpyEta(LongitudinalNumpy, LongitudinalEta, GetItem, numpy.ndarray):
    ObjectClass = vector._backends.object_.LongitudinalObjectEta
    _IS_MOMENTUM = False

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> "LongitudinalNumpyEta":
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj: typing.Any) -> None:
        if not _has(self, ("eta",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "eta"'
            )

    @property
    def elements(self) -> typing.Tuple[numpy.ndarray]:
        return (self["eta"],)

    @property
    def eta(self) -> numpy.ndarray:
        return self["eta"]


class TemporalNumpyT(TemporalNumpy, TemporalT, GetItem, numpy.ndarray):
    ObjectClass = vector._backends.object_.TemporalObjectT
    _IS_MOMENTUM = False

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> "TemporalNumpyT":
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj: typing.Any) -> None:
        if not _has(self, ("t",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "t"'
            )

    @property
    def elements(self) -> typing.Tuple[numpy.ndarray]:
        return (self["t"],)

    @property
    def t(self) -> numpy.ndarray:
        return self["t"]


class TemporalNumpyTau(TemporalNumpy, TemporalTau, GetItem, numpy.ndarray):
    ObjectClass = vector._backends.object_.TemporalObjectTau
    _IS_MOMENTUM = False

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> "TemporalNumpyTau":
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj: typing.Any) -> None:
        if not _has(self, ("tau",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "tau"'
            )

    @property
    def elements(self) -> typing.Tuple[numpy.ndarray]:
        return (self["tau"],)

    @property
    def tau(self) -> numpy.ndarray:
        return self["tau"]


class VectorNumpy(Vector, GetItem):
    lib = numpy
    dtype: "numpy.dtype[typing.Any]"

    def allclose(
        self,
        other: VectorProtocol,
        rtol: typing.Union[float, numpy.ndarray] = 1e-05,
        atol: typing.Union[float, numpy.ndarray] = 1e-08,
        equal_nan: typing.Union[bool, numpy.ndarray] = False,
    ) -> numpy.ndarray:
        """
        Like ``np.ndarray.allclose``, but for VectorNumpy.
        """
        return self.isclose(other, rtol=rtol, atol=atol, equal_nan=equal_nan).all()

    def __eq__(self, other: typing.Any) -> typing.Any:
        return numpy.equal(self, other)  # type: ignore

    def __ne__(self, other: typing.Any) -> typing.Any:
        return numpy.not_equal(self, other)  # type: ignore

    def __array_ufunc__(
        self,
        ufunc: typing.Any,
        method: typing.Any,
        *inputs: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        if not isinstance(_handler_of(*inputs), VectorNumpy):
            # Let a higher-precedence backend handle it.
            return NotImplemented

        outputs: typing.Tuple["VectorNumpy", ...] = kwargs.get("out", ())
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
                    output[name] = result[name]  # type: ignore
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
                    output[name] = result[name]  # type: ignore
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
                    output[name] = result[name]  # type: ignore
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
                    output[name] = result[name]  # type: ignore
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
                    output[name] = result[name]  # type: ignore
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
                    output[name] = result[name]  # type: ignore
            return result

        elif (
            ufunc is numpy.power
            and len(inputs) == 2
            and isinstance(inputs[0], Vector)
            and not isinstance(inputs[1], Vector)
        ):
            result = numpy.absolute(inputs[0]) ** inputs[1]  # type: ignore
            for output in outputs:
                assert output.dtype.names is not None
                for name in output.dtype.names:
                    output[name] = result[name]  # type: ignore
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
        if func is numpy.isclose:
            return type(self).isclose(*args, **kwargs)
        elif func is numpy.allclose:
            return type(self).allclose(*args, **kwargs)
        else:
            return NotImplemented


class VectorNumpy2D(VectorNumpy, Planar, Vector2D, numpy.ndarray):  # type: ignore
    ObjectClass = vector._backends.object_.VectorObject2D
    _IS_MOMENTUM = False

    _azimuthal_type: typing.Union[
        typing.Type[AzimuthalNumpyXY], typing.Type[AzimuthalNumpyRhoPhi]
    ]

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> "VectorNumpy2D":
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            array = _array_from_columns(args[0])
        else:
            array = numpy.array(*args, **kwargs)
        return array.view(cls)

    def __array_finalize__(self, obj: typing.Any) -> None:
        if _has(self, ("x", "y")):
            self._azimuthal_type = AzimuthalNumpyXY
        elif _has(self, ("rho", "phi")):
            self._azimuthal_type = AzimuthalNumpyRhoPhi
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("x", "y") or ("rho", "phi")'
            )

    def __str__(self) -> str:
        return str(self.view(numpy.ndarray))

    def __repr__(self) -> str:
        return _array_repr(self, False)

    @property
    def azimuthal(self) -> AzimuthalNumpy:
        return self.view(self._azimuthal_type)  # type: ignore

    def _wrap_result(
        self,
        cls: typing.Any,
        result: typing.Any,
        returns: typing.Any,
        num_vecargs: typing.Any,
    ) -> typing.Any:
        """
        Args:
            result: Value or tuple of values from a compute function.
            returns: Signature from a ``dispatch_map``.
            num_vecargs (int): Number of vector arguments in the function
                that would be treated on an equal footing (i.e. ``add``
                has two, but ``rotate_axis`` has only one: the ``axis``
                is secondary).

        Wraps the raw result of a compute function as an array of scalars or an
        array of vectors.
        """
        if returns == [float] or returns == [bool]:
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

            out = numpy.empty(_shape_of(result), dtype=dtype)
            for i, name in enumerate(_coordinate_class_to_names[returns[0]]):
                out[name] = result[i]
            return out.view(cls.ProjectionClass2D)

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


class MomentumNumpy2D(PlanarMomentum, VectorNumpy2D):  # type: ignore
    ObjectClass = vector._backends.object_.MomentumObject2D
    _IS_MOMENTUM = True
    dtype: "numpy.dtype[typing.Any]"

    def __array_finalize__(self, obj: typing.Any) -> None:
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

    def __repr__(self) -> str:
        return _array_repr(self, True)

    def __setitem__(self, where: typing.Any, what: typing.Any) -> None:
        return _setitem(self, where, what, True)


class VectorNumpy3D(VectorNumpy, Spatial, Vector3D, numpy.ndarray):  # type: ignore
    ObjectClass = vector._backends.object_.VectorObject3D
    _IS_MOMENTUM = False

    _azimuthal_type: typing.Union[
        typing.Type[AzimuthalNumpyXY], typing.Type[AzimuthalNumpyRhoPhi]
    ]
    _longitudinal_type: typing.Union[
        typing.Type[LongitudinalNumpyZ],
        typing.Type[LongitudinalNumpyTheta],
        typing.Type[LongitudinalNumpyEta],
    ]

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> "VectorNumpy3D":
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            array = _array_from_columns(args[0])
        else:
            array = numpy.array(*args, **kwargs)
        return array.view(cls)

    def __array_finalize__(self, obj: typing.Any) -> None:
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

    def __str__(self) -> str:
        return str(self.view(numpy.ndarray))

    def __repr__(self) -> str:
        return _array_repr(self, False)

    @property
    def azimuthal(self) -> AzimuthalNumpy:
        return self.view(self._azimuthal_type)  # type: ignore

    @property
    def longitudinal(self) -> LongitudinalNumpy:
        return self.view(self._longitudinal_type)  # type: ignore

    def _wrap_result(
        self,
        cls: typing.Any,
        result: typing.Any,
        returns: typing.Any,
        num_vecargs: typing.Any,
    ) -> typing.Any:
        """
        Args:
            result: Value or tuple of values from a compute function.
            returns: Signature from a ``dispatch_map``.
            num_vecargs (int): Number of vector arguments in the function
                that would be treated on an equal footing (i.e. ``add``
                has two, but ``rotate_axis`` has only one: the ``axis``
                is secondary).

        Wraps the raw result of a compute function as an array of scalars or an
        array of vectors.
        """
        if returns == [float] or returns == [bool]:
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


class MomentumNumpy3D(SpatialMomentum, VectorNumpy3D):  # type: ignore
    ObjectClass = vector._backends.object_.MomentumObject3D
    _IS_MOMENTUM = True
    dtype: "numpy.dtype[typing.Any]"

    def __array_finalize__(self, obj: typing.Any) -> None:
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

    def __repr__(self) -> str:
        return _array_repr(self, True)

    def __setitem__(self, where: typing.Any, what: typing.Any) -> None:
        return _setitem(self, where, what, True)


class VectorNumpy4D(VectorNumpy, Lorentz, Vector4D, numpy.ndarray):  # type: ignore
    ObjectClass = vector._backends.object_.VectorObject4D
    _IS_MOMENTUM = False

    _azimuthal_type: typing.Union[
        typing.Type[AzimuthalNumpyXY], typing.Type[AzimuthalNumpyRhoPhi]
    ]
    _longitudinal_type: typing.Union[
        typing.Type[LongitudinalNumpyZ],
        typing.Type[LongitudinalNumpyTheta],
        typing.Type[LongitudinalNumpyEta],
    ]
    _temporal_type: typing.Union[
        typing.Type[TemporalNumpyT],
        typing.Type[TemporalNumpyTau],
    ]

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> "VectorNumpy4D":
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            array = _array_from_columns(args[0])
        else:
            array = numpy.array(*args, **kwargs)
        return array.view(cls)

    def __array_finalize__(self, obj: typing.Any) -> None:
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

    def __str__(self) -> str:
        return str(self.view(numpy.ndarray))

    def __repr__(self) -> str:
        return _array_repr(self, False)

    @property
    def azimuthal(self) -> AzimuthalNumpy:
        return self.view(self._azimuthal_type)  # type: ignore

    @property
    def longitudinal(self) -> LongitudinalNumpy:
        return self.view(self._longitudinal_type)  # type: ignore

    @property
    def temporal(self) -> TemporalNumpy:
        return self.view(self._temporal_type)  # type: ignore

    def _wrap_result(
        self,
        cls: typing.Any,
        result: typing.Any,
        returns: typing.Any,
        num_vecargs: typing.Any,
    ) -> typing.Any:
        """
        Args:
            result: Value or tuple of values from a compute function.
            returns: Signature from a ``dispatch_map``.
            num_vecargs (int): Number of vector arguments in the function
                that would be treated on an equal footing (i.e. ``add``
                has two, but ``rotate_axis`` has only one: the ``axis``
                is secondary).

        Wraps the raw result of a compute function as an array of scalars or an
        array of vectors.
        """
        if returns == [float] or returns == [bool]:
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


class MomentumNumpy4D(LorentzMomentum, VectorNumpy4D):  # type: ignore
    ObjectClass = vector._backends.object_.MomentumObject4D
    _IS_MOMENTUM = True
    dtype: "numpy.dtype[typing.Any]"

    def __array_finalize__(self, obj: typing.Any) -> None:
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

    No constraints are placed on the ``dtypes`` of the vector fields, though if they
    are not numbers, mathematical operations will fail. Usually, you want them to be
    ``np.integer`` or ``np.floating``.
    """

    names: typing.Tuple[str, ...]
    if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
        names = tuple(args[0].keys())
    elif "dtype" in kwargs:
        names = numpy.dtype(kwargs["dtype"]).names or ()
    elif len(args) >= 2:
        names = numpy.dtype(args[1]).names or ()
    else:
        names = ()

    cls: typing.Type[VectorNumpy]

    is_momentum = any(x in _repr_momentum_to_generic for x in names)

    if any(x in ("t", "E", "e", "energy", "tau", "M", "m", "mass") for x in names):
        cls = MomentumNumpy4D if is_momentum else VectorNumpy4D
    elif any(x in ("z", "pz", "theta", "eta") for x in names):
        cls = MomentumNumpy3D if is_momentum else VectorNumpy3D
    else:
        cls = MomentumNumpy2D if is_momentum else VectorNumpy2D

    # VectorNumpy has no constructor, so mypy flags this line
    return cls(*args, **kwargs)  # type: ignore


VectorNumpy2D.ProjectionClass2D = VectorNumpy2D
VectorNumpy2D.ProjectionClass3D = VectorNumpy3D
VectorNumpy2D.ProjectionClass4D = VectorNumpy4D
VectorNumpy2D.GenericClass = VectorNumpy2D

MomentumNumpy2D.ProjectionClass2D = MomentumNumpy2D
MomentumNumpy2D.ProjectionClass3D = MomentumNumpy3D
MomentumNumpy2D.ProjectionClass4D = MomentumNumpy4D
MomentumNumpy2D.GenericClass = VectorNumpy2D

VectorNumpy3D.ProjectionClass2D = VectorNumpy2D
VectorNumpy3D.ProjectionClass3D = VectorNumpy3D
VectorNumpy3D.ProjectionClass4D = VectorNumpy4D
VectorNumpy3D.GenericClass = VectorNumpy3D

MomentumNumpy3D.ProjectionClass2D = MomentumNumpy2D
MomentumNumpy3D.ProjectionClass3D = MomentumNumpy3D
MomentumNumpy3D.ProjectionClass4D = MomentumNumpy4D
MomentumNumpy3D.GenericClass = VectorNumpy3D

VectorNumpy4D.ProjectionClass2D = VectorNumpy2D
VectorNumpy4D.ProjectionClass3D = VectorNumpy3D
VectorNumpy4D.ProjectionClass4D = VectorNumpy4D
VectorNumpy4D.GenericClass = VectorNumpy4D

MomentumNumpy4D.ProjectionClass2D = MomentumNumpy2D
MomentumNumpy4D.ProjectionClass3D = MomentumNumpy3D
MomentumNumpy4D.ProjectionClass4D = MomentumNumpy4D
MomentumNumpy4D.GenericClass = VectorNumpy4D
