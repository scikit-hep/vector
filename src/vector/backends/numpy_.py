# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import collections.abc

import numpy

import vector.backends.object_
from vector.methods import (
    Azimuthal,
    AzimuthalRhoPhi,
    AzimuthalXY,
    Coordinates,
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
    _handler,
    _ltype,
    _repr_generic_to_momentum,
    _repr_momentum_to_generic,
    _ttype,
)


def _array_from_columns(columns):
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


def _setitem(array, where, what, is_momentum):
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


def _getitem(array, where, is_momentum):
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


def _array_repr(array, is_momentum):
    name = type(array).__name__
    array = array.view(numpy.ndarray)
    if is_momentum:
        array = array.view(
            [
                (_repr_generic_to_momentum.get(x, x), array.dtype[x])
                for x in array.dtype.names
            ]
        )
    return name + repr(array)[5:].replace("\n     ", "\n" + " " * len(name))


def _has(array, names):
    dtype_names = array.dtype.names
    if dtype_names is None:
        dtype_names = ()
    return all(x in dtype_names for x in names)


def _toarrays(result):
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


def _shape_of(result):
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
    return tuple(shape)


class CoordinatesNumpy:
    lib = numpy


class AzimuthalNumpy(CoordinatesNumpy):
    pass


class LongitudinalNumpy(CoordinatesNumpy):
    pass


class TemporalNumpy(CoordinatesNumpy):
    pass


class VectorNumpy:
    def allclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False):
        return self.isclose(other, rtol=rtol, atol=atol, equal_nan=equal_nan).all()

    def __eq__(self, other):
        return numpy.equal(self, other)

    def __ne__(self, other):
        return numpy.not_equal(self, other)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if not isinstance(_handler(inputs), VectorNumpy):
            # Let the array-of-vectors object handle it.
            return NotImplemented

        if isinstance(self, Vector2D):
            from vector.compute.planar import add, dot, equal, not_equal
            from vector.compute.planar import rho as absolute
            from vector.compute.planar import rho2 as absolute2
            from vector.compute.planar import scale, subtract
        elif isinstance(self, Vector3D):
            from vector.compute.spatial import add, dot, equal
            from vector.compute.spatial import mag as absolute
            from vector.compute.spatial import mag2 as absolute2
            from vector.compute.spatial import not_equal, scale, subtract
        elif isinstance(self, Vector4D):
            from vector.compute.lorentz import (
                add,
                dot,
                equal,
                not_equal,
                scale,
                subtract,
            )
            from vector.compute.lorentz import tau as absolute
            from vector.compute.lorentz import tau2 as absolute2

        outputs = kwargs.get("out", ())
        if any(not isinstance(x, VectorNumpy) for x in outputs):
            raise TypeError(
                "ufunc operating on VectorNumpys can only fill another VectorNumpy "
                "with 'out' keyword"
            )

        if ufunc is numpy.absolute and len(inputs) == 1:
            result = absolute.dispatch(inputs[0])
            for output in outputs:
                for name in output.dtype.names:
                    output[name] = result[name]
            return result

        elif ufunc is numpy.add and len(inputs) == 2:
            result = add.dispatch(inputs[0], inputs[1])
            for output in outputs:
                for name in output.dtype.names:
                    output[name] = result[name]
            return result

        elif ufunc is numpy.subtract and len(inputs) == 2:
            result = subtract.dispatch(inputs[0], inputs[1])
            for output in outputs:
                for name in output.dtype.names:
                    output[name] = result[name]
            return result

        elif (
            ufunc is numpy.multiply
            and not isinstance(inputs[0], (Vector, Coordinates))
            and len(inputs) == 2
        ):
            result = scale.dispatch(inputs[0], inputs[1])
            for output in outputs:
                for name in output.dtype.names:
                    output[name] = result[name]
            return result

        elif (
            ufunc is numpy.multiply
            and not isinstance(inputs[1], (Vector, Coordinates))
            and len(inputs) == 2
        ):
            result = scale.dispatch(inputs[1], inputs[0])
            for output in outputs:
                for name in output.dtype.names:
                    output[name] = result[name]
            return result

        elif ufunc is numpy.negative and len(inputs) == 1:
            result = scale.dispatch(-1, inputs[0])
            for output in outputs:
                for name in output.dtype.names:
                    output[name] = result[name]
            return result

        elif ufunc is numpy.positive and len(inputs) == 1:
            result = inputs[0]
            for output in outputs:
                for name in output.dtype.names:
                    output[name] = result[name]
            return result

        elif (
            ufunc is numpy.true_divide
            and not isinstance(inputs[1], (Vector, Coordinates))
            and len(inputs) == 2
        ):
            result = scale.dispatch(1 / inputs[1], inputs[0])
            for output in outputs:
                for name in output.dtype.names:
                    output[name] = result[name]
            return result

        elif (
            ufunc is numpy.power
            and not isinstance(inputs[1], (Vector, Coordinates))
            and len(inputs) == 2
        ):
            result = absolute.dispatch(inputs[0]) ** inputs[1]
            for output in outputs:
                for name in output.dtype.names:
                    output[name] = result[name]
            return result

        elif ufunc is numpy.square and len(inputs) == 1:
            result = absolute2.dispatch(inputs[0])
            for output in outputs:
                for name in output.dtype.names:
                    output[name] = result[name]
            return result

        elif ufunc is numpy.sqrt and len(inputs) == 1:
            result = numpy.sqrt(absolute.dispatch(inputs[0]))
            for output in outputs:
                for name in output.dtype.names:
                    output[name] = result[name]
            return result

        elif ufunc is numpy.cbrt and len(inputs) == 1:
            result = numpy.cbrt(absolute.dispatch(inputs[0]))
            for output in outputs:
                for name in output.dtype.names:
                    output[name] = result[name]
            return result

        elif ufunc is numpy.matmul and len(inputs) == 2:
            result = dot.dispatch(inputs[0], inputs[1])
            for output in outputs:
                for name in output.dtype.names:
                    output[name] = result[name]
            return result

        elif ufunc is numpy.equal and len(inputs) == 2:
            result = equal.dispatch(inputs[0], inputs[1])
            for output in outputs:
                for name in output.dtype.names:
                    output[name] = result[name]
            return result

        elif ufunc is numpy.not_equal and len(inputs) == 2:
            result = not_equal.dispatch(inputs[0], inputs[1])
            for output in outputs:
                for name in output.dtype.names:
                    output[name] = result[name]
            return result

        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if func is numpy.isclose:
            return type(self).isclose(*args, **kwargs)
        elif func is numpy.allclose:
            return type(self).allclose(*args, **kwargs)
        else:
            return NotImplemented


class AzimuthalNumpyXY(AzimuthalNumpy, AzimuthalXY, numpy.ndarray):
    ObjectClass = vector.backends.object_.AzimuthalObjectXY

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not _has(self, ("x", "y")):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("x", "y")'
            )

    @property
    def elements(self):
        return (self["x"], self["y"])

    @property
    def x(self):
        return self["x"]

    @property
    def y(self):
        return self["y"]

    def __getitem__(self, where):
        return _getitem(self, where, False)


class AzimuthalNumpyRhoPhi(AzimuthalNumpy, AzimuthalRhoPhi, numpy.ndarray):
    ObjectClass = vector.backends.object_.AzimuthalObjectRhoPhi

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not _has(self, ("rho", "phi")):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("rho", "phi")'
            )

    @property
    def elements(self):
        return (self["rho"], self["phi"])

    @property
    def rho(self):
        return self["rho"]

    @property
    def phi(self):
        return self["phi"]

    def __getitem__(self, where):
        return _getitem(self, where, False)


class LongitudinalNumpyZ(LongitudinalNumpy, LongitudinalZ, numpy.ndarray):
    ObjectClass = vector.backends.object_.LongitudinalObjectZ

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not _has(self, ("z",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "z"'
            )

    @property
    def elements(self):
        return (self["z"],)

    @property
    def z(self):
        return self["z"]

    def __getitem__(self, where):
        return _getitem(self, where, False)


class LongitudinalNumpyTheta(LongitudinalNumpy, LongitudinalTheta, numpy.ndarray):
    ObjectClass = vector.backends.object_.LongitudinalObjectTheta

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not _has(self, ("theta",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "theta"'
            )

    @property
    def elements(self):
        return (self["theta"],)

    @property
    def theta(self):
        return self["theta"]

    def __getitem__(self, where):
        return _getitem(self, where, False)


class LongitudinalNumpyEta(LongitudinalNumpy, LongitudinalEta, numpy.ndarray):
    ObjectClass = vector.backends.object_.LongitudinalObjectEta

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not _has(self, ("eta",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "eta"'
            )

    @property
    def elements(self):
        return (self["eta"],)

    @property
    def eta(self):
        return self["eta"]

    def __getitem__(self, where):
        return _getitem(self, where, False)


class TemporalNumpyT(TemporalNumpy, TemporalT, numpy.ndarray):
    ObjectClass = vector.backends.object_.TemporalObjectT

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not _has(self, ("t",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "t"'
            )

    @property
    def elements(self):
        return (self["t"],)

    @property
    def t(self):
        return self["t"]

    def __getitem__(self, where):
        return _getitem(self, where, False)


class TemporalNumpyTau(TemporalNumpy, TemporalTau, numpy.ndarray):
    ObjectClass = vector.backends.object_.TemporalObjectTau

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not _has(self, ("tau",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "tau"'
            )

    @property
    def elements(self):
        return (self["tau"],)

    @property
    def tau(self):
        return self["tau"]

    def __getitem__(self, where):
        return _getitem(self, where, False)


class VectorNumpy2D(VectorNumpy, Planar, Vector2D, numpy.ndarray):
    lib = numpy
    ObjectClass = vector.backends.object_.VectorObject2D

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            array = _array_from_columns(args[0])
        else:
            array = numpy.array(*args, **kwargs)
        return array.view(cls)

    def __array_finalize__(self, obj):
        if _has(self, ("x", "y")):
            self._azimuthal_type = AzimuthalNumpyXY
        elif _has(self, ("rho", "phi")):
            self._azimuthal_type = AzimuthalNumpyRhoPhi
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("x", "y") or ("rho", "phi")'
            )

    def __str__(self):
        return str(self.view(numpy.ndarray))

    def __repr__(self):
        return _array_repr(self, False)

    @property
    def azimuthal(self):
        return self.view(self._azimuthal_type)

    def _wrap_result(self, result, returns):
        if returns == [float] or returns == [bool]:
            return result

        elif returns == [AzimuthalXY] or returns == [AzimuthalRhoPhi]:
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
            return out.view(type(self))

        else:
            raise AssertionError(repr(returns))

    def __getitem__(self, where):
        return _getitem(self, where, False)

    def __setitem__(self, where, what):
        return _setitem(self, where, what, False)


class MomentumNumpy2D(PlanarMomentum, VectorNumpy2D):
    ObjectClass = vector.backends.object_.MomentumObject2D

    def __array_finalize__(self, obj):
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

    def __repr__(self):
        return _array_repr(self, True)

    def __getitem__(self, where):
        return _getitem(self, where, True)

    def __setitem__(self, where, what):
        return _setitem(self, where, what, True)


class VectorNumpy3D(VectorNumpy, Spatial, Vector3D, numpy.ndarray):
    lib = numpy
    ObjectClass = vector.backends.object_.VectorObject3D
    ProjectionClass2D = VectorNumpy2D

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            array = _array_from_columns(args[0])
        else:
            array = numpy.array(*args, **kwargs)
        return array.view(cls)

    def __array_finalize__(self, obj):
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

    def __str__(self):
        return str(self.view(numpy.ndarray))

    def __repr__(self):
        return _array_repr(self, False)

    @property
    def azimuthal(self):
        return self.view(self._azimuthal_type)

    @property
    def longitudinal(self):
        return self.view(self._longitudinal_type)

    def _wrap_result(self, result, returns):
        if returns == [float] or returns == [bool]:
            return result

        elif returns == [AzimuthalXY] or returns == [AzimuthalRhoPhi]:
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
            return out.view(type(self))

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
            return out.view(type(self))

        else:
            raise AssertionError(repr(returns))

    def __getitem__(self, where):
        return _getitem(self, where, False)

    def __setitem__(self, where, what):
        return _setitem(self, where, what, False)


class MomentumNumpy3D(SpatialMomentum, VectorNumpy3D):
    ObjectClass = vector.backends.object_.MomentumObject3D
    ProjectionClass2D = MomentumNumpy2D

    def __array_finalize__(self, obj):
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

    def __repr__(self):
        return _array_repr(self, True)

    def __getitem__(self, where):
        return _getitem(self, where, True)

    def __setitem__(self, where, what):
        return _setitem(self, where, what, True)


class VectorNumpy4D(VectorNumpy, Lorentz, Vector4D, numpy.ndarray):
    lib = numpy
    ObjectClass = vector.backends.object_.VectorObject4D
    ProjectionClass2D = VectorNumpy2D
    ProjectionClass3D = VectorNumpy3D

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            array = _array_from_columns(args[0])
        else:
            array = numpy.array(*args, **kwargs)
        return array.view(cls)

    def __array_finalize__(self, obj):
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

    def __str__(self):
        return str(self.view(numpy.ndarray))

    def __repr__(self):
        return _array_repr(self, False)

    @property
    def azimuthal(self):
        return self.view(self._azimuthal_type)

    @property
    def longitudinal(self):
        return self.view(self._longitudinal_type)

    @property
    def temporal(self):
        return self.view(self._temporal_type)

    def _wrap_result(self, result, returns):
        if returns == [float] or returns == [bool]:
            return result

        elif returns == [AzimuthalXY] or returns == [AzimuthalRhoPhi]:
            result = _toarrays(result)
            dtype = []
            for i, name in enumerate(_coordinate_class_to_names[returns[0]]):
                dtype.append((name, result[i].dtype))
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
            return out.view(type(self))

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
            return out.view(type(self))

        elif (
            len(returns) == 3
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
            is_4d = False
            if isinstance(returns[2], type) and issubclass(returns[2], Temporal):
                is_4d = True
                for name in _coordinate_class_to_names[returns[2]]:
                    dtype.append((name, result[i].dtype))
                    i += 1
            elif returns[2] is not None:
                raise AssertionError(repr(type(returns[2])))
            out = numpy.empty(_shape_of(result), dtype=dtype)
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                out[name] = result[i]
                i += 1
            for name in _coordinate_class_to_names[returns[1]]:
                out[name] = result[i]
                i += 1
            if is_4d:
                for name in _coordinate_class_to_names[returns[2]]:
                    out[name] = result[i]
                    i += 1
                return out.view(type(self))
            else:
                return out.view(self.ProjectionClass3D)

        else:
            raise AssertionError(repr(returns))

    def __getitem__(self, where):
        return _getitem(self, where, False)

    def __setitem__(self, where, what):
        return _setitem(self, where, what, False)


class MomentumNumpy4D(LorentzMomentum, VectorNumpy4D):
    ObjectClass = vector.backends.object_.MomentumObject4D
    ProjectionClass2D = MomentumNumpy2D
    ProjectionClass3D = MomentumNumpy3D

    def __array_finalize__(self, obj):
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
                'field "t" or "tau" or "E" or "e" or "energy" or "M" or "m" or "mass"'
            )

    def __repr__(self):
        return _array_repr(self, True)

    def __getitem__(self, where):
        return _getitem(self, where, True)

    def __setitem__(self, where, what):
        return _setitem(self, where, what, True)


def array(*args, **kwargs):
    "array docs"
    names = None
    if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
        names = args[0].keys()
    elif "dtype" in kwargs:
        names = numpy.dtype(kwargs["dtype"]).names
    elif len(args) >= 2:
        names = numpy.dtype(args[1]).names
    if names is None:
        names = ()

    is_momentum = any(x in _repr_momentum_to_generic for x in names)
    if any(x in ("t", "E", "e", "energy", "tau", "M", "m", "mass") for x in names):
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
