# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import collections.abc
import typing

import numpy

import vector.backends.object_
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
    Vector,
    Vector2D,
    Vector3D,
    Vector4D,
    _aztype,
    _coordinate_class_to_names,
    _coordinate_order,
    _handler_of,
    _ltype,
    _repr_momentum_to_generic,
    _ttype,
)


def _array_from_columns(columns: typing.Any) -> typing.Any:
    if len(columns) == 0:
        raise ValueError("no columns have been provided")
    names = list(columns.keys())
    names.sort(
        key=lambda x: _coordinate_order.index(x)
        if x in _coordinate_order
        else float("inf")
    )

    dtype = []
    shape = None
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

    array = numpy.empty(shape, dtype)
    for x in names:
        array[x] = columns[x]
    return array


def _setitem(
    array: typing.Any, where: typing.Any, what: typing.Any, is_momentum: typing.Any
) -> typing.Any:
    if isinstance(where, str):
        if is_momentum:
            where = _repr_momentum_to_generic.get(where, where)
        array.view(numpy.ndarray)[where] = what
    else:
        if hasattr(what, "dtype") and what.dtype.names is not None:
            tofill = array[where]
            for name in what.dtype.names:
                if is_momentum:
                    generic = _repr_momentum_to_generic.get(name, name)
                tofill[generic] = what[name]
        else:
            raise TypeError(
                "right-hand side of assignment must be a structured array with "
                "the same fields as " + type(array).__name__
            )


def _getitem(
    array: typing.Any, where: typing.Any, is_momentum: typing.Any
) -> typing.Any:
    if isinstance(where, str):
        if is_momentum:
            where = _repr_momentum_to_generic.get(where, where)
        return array.view(numpy.ndarray)[where]
    else:
        out = numpy.ndarray.__getitem__(array, where)
        if isinstance(out, numpy.void):
            azimuthal, longitudinal, temporal = None, None, None
            if hasattr(array, "_azimuthal_type"):
                azimuthal = array._azimuthal_type.ObjectClass(
                    *[out[x] for x in _coordinate_class_to_names[_aztype(array)]]
                )
            if hasattr(array, "_longitudinal_type"):
                longitudinal = array._longitudinal_type.ObjectClass(
                    *[out[x] for x in _coordinate_class_to_names[_ltype(array)]]
                )
            if hasattr(array, "_temporal_type"):
                temporal = array._temporal_type.ObjectClass(
                    *[out[x] for x in _coordinate_class_to_names[_ttype(array)]]
                )
            if temporal is not None:
                return array.ObjectClass(azimuthal, longitudinal, temporal)
            elif longitudinal is not None:
                return array.ObjectClass(azimuthal, longitudinal)
            elif azimuthal is not None:
                return array.ObjectClass(azimuthal)
            else:
                return array.ObjectClass(*out.view(numpy.ndarray))
        else:
            return out


def _array_repr(array: typing.Any, is_momentum: typing.Any) -> typing.Any:
    name = type(array).__name__
    array = array.view(numpy.ndarray)
    return name + repr(array)[5:].replace("\n     ", "\n" + " " * len(name))


def _has(array: typing.Any, names: typing.Any) -> typing.Any:
    dtype_names = array.dtype.names
    if dtype_names is None:
        dtype_names = ()
    return all(x in dtype_names for x in names)


def _toarrays(result: typing.Any) -> typing.Any:
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


def _shape_of(result: typing.Any) -> typing.Any:
    if not isinstance(result, tuple):
        result = (result,)
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


class CoordinatesNumpy:
    lib = numpy


class AzimuthalNumpy(CoordinatesNumpy):
    pass


class LongitudinalNumpy(CoordinatesNumpy):
    pass


class TemporalNumpy(CoordinatesNumpy):
    pass


class AzimuthalNumpyXY(AzimuthalNumpy, AzimuthalXY, numpy.ndarray):
    ObjectClass = vector.backends.object_.AzimuthalObjectXY

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj: typing.Any) -> typing.Any:
        if not _has(self, ("x", "y")):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("x", "y")'
            )

    @property
    def elements(self) -> typing.Any:
        return (self["x"], self["y"])

    @property
    def x(self) -> typing.Any:
        return self["x"]

    @property
    def y(self) -> typing.Any:
        return self["y"]

    def __getitem__(self, where: typing.Any) -> typing.Any:
        return _getitem(self, where, False)


class AzimuthalNumpyRhoPhi(AzimuthalNumpy, AzimuthalRhoPhi, numpy.ndarray):
    ObjectClass = vector.backends.object_.AzimuthalObjectRhoPhi

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj: typing.Any) -> typing.Any:
        if not _has(self, ("rho", "phi")):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("rho", "phi")'
            )

    @property
    def elements(self) -> typing.Any:
        return (self["rho"], self["phi"])

    @property
    def rho(self) -> typing.Any:
        return self["rho"]

    @property
    def phi(self) -> typing.Any:
        return self["phi"]

    def __getitem__(self, where: typing.Any) -> typing.Any:
        return _getitem(self, where, False)


class LongitudinalNumpyZ(LongitudinalNumpy, LongitudinalZ, numpy.ndarray):
    ObjectClass = vector.backends.object_.LongitudinalObjectZ

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj: typing.Any) -> typing.Any:
        if not _has(self, ("z",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "z"'
            )

    @property
    def elements(self) -> typing.Any:
        return (self["z"],)

    @property
    def z(self) -> typing.Any:
        return self["z"]

    def __getitem__(self, where: typing.Any) -> typing.Any:
        return _getitem(self, where, False)


class LongitudinalNumpyTheta(LongitudinalNumpy, LongitudinalTheta, numpy.ndarray):
    ObjectClass = vector.backends.object_.LongitudinalObjectTheta

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj: typing.Any) -> typing.Any:
        if not _has(self, ("theta",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "theta"'
            )

    @property
    def elements(self) -> typing.Any:
        return (self["theta"],)

    @property
    def theta(self) -> typing.Any:
        return self["theta"]

    def __getitem__(self, where: typing.Any) -> typing.Any:
        return _getitem(self, where, False)


class LongitudinalNumpyEta(LongitudinalNumpy, LongitudinalEta, numpy.ndarray):
    ObjectClass = vector.backends.object_.LongitudinalObjectEta

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj: typing.Any) -> typing.Any:
        if not _has(self, ("eta",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "eta"'
            )

    @property
    def elements(self) -> typing.Any:
        return (self["eta"],)

    @property
    def eta(self) -> typing.Any:
        return self["eta"]

    def __getitem__(self, where: typing.Any) -> typing.Any:
        return _getitem(self, where, False)


class TemporalNumpyT(TemporalNumpy, TemporalT, numpy.ndarray):
    ObjectClass = vector.backends.object_.TemporalObjectT

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj: typing.Any) -> typing.Any:
        if not _has(self, ("t",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "t"'
            )

    @property
    def elements(self) -> typing.Any:
        return (self["t"],)

    @property
    def t(self) -> typing.Any:
        return self["t"]

    def __getitem__(self, where: typing.Any) -> typing.Any:
        return _getitem(self, where, False)


class TemporalNumpyTau(TemporalNumpy, TemporalTau, numpy.ndarray):
    ObjectClass = vector.backends.object_.TemporalObjectTau

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj: typing.Any) -> typing.Any:
        if not _has(self, ("tau",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "tau"'
            )

    @property
    def elements(self) -> typing.Any:
        return (self["tau"],)

    @property
    def tau(self) -> typing.Any:
        return self["tau"]

    def __getitem__(self, where: typing.Any) -> typing.Any:
        return _getitem(self, where, False)


class VectorNumpy(Vector):
    lib = numpy

    def allclose(
        self,
        other: typing.Any,
        rtol: typing.Any = 1e-05,
        atol: typing.Any = 1e-08,
        equal_nan: typing.Any = False,
    ) -> typing.Any:
        return self.isclose(other, rtol=rtol, atol=atol, equal_nan=equal_nan).all()

    def __eq__(self, other: typing.Any) -> typing.Any:
        return numpy.equal(self, other)

    def __ne__(self, other: typing.Any) -> typing.Any:
        return numpy.not_equal(self, other)

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

        outputs = kwargs.get("out", ())
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
                for name in output.dtype.names:
                    output[name] = result[name]
            return result

        elif (
            ufunc is numpy.subtract
            and len(inputs) == 2
            and isinstance(inputs[0], Vector)
            and isinstance(inputs[1], Vector)
        ):
            result = inputs[0].subtract(inputs[1])
            for output in outputs:
                for name in output.dtype.names:
                    output[name] = result[name]
            return result

        elif (
            ufunc is numpy.multiply
            and len(inputs) == 2
            and isinstance(inputs[0], Vector)
            and not isinstance(inputs[1], Vector)
        ):
            result = inputs[0].scale(inputs[1])
            for output in outputs:
                for name in output.dtype.names:
                    output[name] = result[name]
            return result

        elif (
            ufunc is numpy.multiply
            and len(inputs) == 2
            and not isinstance(inputs[0], Vector)
            and isinstance(inputs[1], Vector)
        ):
            result = inputs[1].scale(inputs[0])
            for output in outputs:
                for name in output.dtype.names:
                    output[name] = result[name]
            return result

        elif (
            ufunc is numpy.negative
            and len(inputs) == 1
            and isinstance(inputs[0], Vector)
        ):
            result = inputs[0].scale(-1)
            for output in outputs:
                for name in output.dtype.names:
                    output[name] = result[name]
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
                for name in output.dtype.names:
                    output[name] = result[name]
            return result

        elif (
            ufunc is numpy.power
            and len(inputs) == 2
            and isinstance(inputs[0], Vector)
            and not isinstance(inputs[1], Vector)
        ):
            result = numpy.absolute(inputs[0]) ** inputs[1]
            for output in outputs:
                for name in output.dtype.names:
                    output[name] = result[name]
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


class VectorNumpy2D(VectorNumpy, Planar, Vector2D, numpy.ndarray):
    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            array = _array_from_columns(args[0])
        else:
            array = numpy.array(*args, **kwargs)
        return array.view(cls)

    def __array_finalize__(self, obj: typing.Any) -> typing.Any:
        if _has(self, ("x", "y")):
            self._azimuthal_type = AzimuthalNumpyXY
        elif _has(self, ("rho", "phi")):
            self._azimuthal_type = AzimuthalNumpyRhoPhi
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("x", "y") or ("rho", "phi")'
            )

    def __str__(self) -> typing.Any:
        return str(self.view(numpy.ndarray))

    def __repr__(self) -> typing.Any:
        return _array_repr(self, False)

    @property
    def azimuthal(self) -> typing.Any:
        return self.view(self._azimuthal_type)

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
            result = _toarrays(result)
            dtype = []
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                dtype.append((name, result[i].dtype))
                i += 1
            out = numpy.empty(_shape_of(result), dtype=dtype)
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                out[name] = result[i]
                i += 1
            return out.view(cls.ProjectionClass2D)

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and returns[1] is None
        ):
            result = _toarrays(result)
            dtype = []
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                dtype.append((name, result[i].dtype))
                i += 1
            out = numpy.empty(_shape_of(result), dtype=dtype)
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                out[name] = result[i]
                i += 1
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

    def __getitem__(self, where: typing.Any) -> typing.Any:
        return _getitem(self, where, False)

    def __setitem__(self, where: typing.Any, what: typing.Any) -> typing.Any:
        return _setitem(self, where, what, False)


class MomentumNumpy2D(PlanarMomentum, VectorNumpy2D):
    ObjectClass = vector.backends.object_.MomentumObject2D

    def __array_finalize__(self, obj: typing.Any) -> typing.Any:
        self.dtype.names = [
            _repr_momentum_to_generic.get(x, x) for x in self.dtype.names
        ]
        if _has(self, ("x", "y")):
            self._azimuthal_type = AzimuthalNumpyXY
        elif _has(self, ("rho", "phi")):
            self._azimuthal_type = AzimuthalNumpyRhoPhi
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("x", "y") or ("rho", "phi") or ("px", "py") or ("pt", "phi")'
            )

    def __repr__(self) -> typing.Any:
        return _array_repr(self, True)

    def __getitem__(self, where: typing.Any) -> typing.Any:
        return _getitem(self, where, True)

    def __setitem__(self, where: typing.Any, what: typing.Any) -> typing.Any:
        return _setitem(self, where, what, True)


class VectorNumpy3D(VectorNumpy, Spatial, Vector3D, numpy.ndarray):
    ObjectClass = vector.backends.object_.VectorObject3D

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            array = _array_from_columns(args[0])
        else:
            array = numpy.array(*args, **kwargs)
        return array.view(cls)

    def __array_finalize__(self, obj: typing.Any) -> typing.Any:
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

    def __str__(self) -> typing.Any:
        return str(self.view(numpy.ndarray))

    def __repr__(self) -> typing.Any:
        return _array_repr(self, False)

    @property
    def azimuthal(self) -> typing.Any:
        return self.view(self._azimuthal_type)

    @property
    def longitudinal(self) -> typing.Any:
        return self.view(self._longitudinal_type)

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
            result = _toarrays(result)
            dtype = []
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                dtype.append((name, result[i].dtype))
                i += 1
            for name in _coordinate_class_to_names[_ltype(self)]:
                dtype.append((name, self.dtype[name]))
            out = numpy.empty(_shape_of(result), dtype=dtype)
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                out[name] = result[i]
                i += 1
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
            dtype = []
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                dtype.append((name, result[i].dtype))
                i += 1
            out = numpy.empty(_shape_of(result), dtype=dtype)
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                out[name] = result[i]
                i += 1
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

    def __getitem__(self, where: typing.Any) -> typing.Any:
        return _getitem(self, where, False)

    def __setitem__(self, where: typing.Any, what: typing.Any) -> typing.Any:
        return _setitem(self, where, what, False)


class MomentumNumpy3D(SpatialMomentum, VectorNumpy3D):
    ObjectClass = vector.backends.object_.MomentumObject3D

    def __array_finalize__(self, obj: typing.Any) -> typing.Any:
        self.dtype.names = [
            _repr_momentum_to_generic.get(x, x) for x in self.dtype.names
        ]
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

    def __repr__(self) -> typing.Any:
        return _array_repr(self, True)

    def __getitem__(self, where: typing.Any) -> typing.Any:
        return _getitem(self, where, True)

    def __setitem__(self, where: typing.Any, what: typing.Any) -> typing.Any:
        return _setitem(self, where, what, True)


class VectorNumpy4D(VectorNumpy, Lorentz, Vector4D, numpy.ndarray):
    ObjectClass = vector.backends.object_.VectorObject4D

    def __new__(cls, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            array = _array_from_columns(args[0])
        else:
            array = numpy.array(*args, **kwargs)
        return array.view(cls)

    def __array_finalize__(self, obj: typing.Any) -> typing.Any:
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

    def __str__(self) -> typing.Any:
        return str(self.view(numpy.ndarray))

    def __repr__(self) -> typing.Any:
        return _array_repr(self, False)

    @property
    def azimuthal(self) -> typing.Any:
        return self.view(self._azimuthal_type)

    @property
    def longitudinal(self) -> typing.Any:
        return self.view(self._longitudinal_type)

    @property
    def temporal(self) -> typing.Any:
        return self.view(self._temporal_type)

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
            result = _toarrays(result)
            dtype = []
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                dtype.append((name, result[i].dtype))
                i += 1
            for name in _coordinate_class_to_names[_ltype(self)]:
                dtype.append((name, self.dtype[name]))
            for name in _coordinate_class_to_names[_ttype(self)]:
                dtype.append((name, self.dtype[name]))
            out = numpy.empty(_shape_of(result), dtype=dtype)
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                out[name] = result[i]
                i += 1
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
            dtype = []
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                dtype.append((name, result[i].dtype))
                i += 1
            out = numpy.empty(_shape_of(result), dtype=dtype)
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                out[name] = result[i]
                i += 1
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

    def __getitem__(self, where: typing.Any) -> typing.Any:
        return _getitem(self, where, False)

    def __setitem__(self, where: typing.Any, what: typing.Any) -> typing.Any:
        return _setitem(self, where, what, False)


class MomentumNumpy4D(LorentzMomentum, VectorNumpy4D):
    ObjectClass = vector.backends.object_.MomentumObject4D

    def __array_finalize__(self, obj: typing.Any) -> typing.Any:
        self.dtype.names = [
            _repr_momentum_to_generic.get(x, x) for x in self.dtype.names
        ]
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
                'field "t" or "tau" or "E" or "energy" or "M" or "mass"'
            )

    def __repr__(self) -> typing.Any:
        return _array_repr(self, True)

    def __getitem__(self, where: typing.Any) -> typing.Any:
        return _getitem(self, where, True)

    def __setitem__(self, where: typing.Any, what: typing.Any) -> typing.Any:
        return _setitem(self, where, what, True)


def array(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    "vector.array docs"
    names = None
    if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
        names = tuple(args[0].keys())
    elif "dtype" in kwargs:
        names = numpy.dtype(kwargs["dtype"]).names
    elif len(args) >= 2:
        names = numpy.dtype(args[1]).names
    if names is None:
        names = ()

    is_momentum = any(x in _repr_momentum_to_generic for x in names)
    if any(x in ("t", "E", "energy", "tau", "M", "mass") for x in names):
        if is_momentum:
            cls = MomentumNumpy4D
        else:
            cls = VectorNumpy4D
    elif any(x in ("z", "pz", "theta", "eta") for x in names):
        if is_momentum:
            cls = MomentumNumpy3D
        else:
            cls = VectorNumpy3D
    else:
        if is_momentum:
            cls = MomentumNumpy2D
        else:
            cls = VectorNumpy2D

    return cls(*args, **kwargs)


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
