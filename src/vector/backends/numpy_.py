# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy

import vector.backends.object_
import vector.geometry
import vector.methods
from vector.geometry import _coordinate_class_to_names


def _getitem(array, where):
    if isinstance(where, str):
        return array.view(numpy.ndarray)[where]
    else:
        out = numpy.ndarray.__getitem__(array, where)
        if isinstance(out, numpy.void):
            azimuthal, longitudinal, temporal = None, None, None
            if hasattr(array, "_azimuthal_type"):
                azimuthal = array._azimuthal_type.object_type(*out.view(numpy.ndarray))
            if hasattr(array, "_longitudinal_type"):
                longitudinal = array._longitudinal_type.object_type(
                    *out.view(numpy.ndarray)
                )
            if hasattr(array, "_temporal_type"):
                temporal = array._temporal_type.object_type(*out.view(numpy.ndarray))
            if temporal is not None:
                return array.object_type(azimuthal, longitudinal, temporal)
            elif longitudinal is not None:
                return array.object_type(azimuthal, longitudinal)
            elif azimuthal is not None:
                return array.object_type(azimuthal)
            else:
                return array.object_type(*out.view(numpy.ndarray))
        else:
            return out


def _array_repr(array):
    name = type(array).__name__
    return name + repr(array.view(numpy.ndarray))[5:].replace(
        "\n     ", "\n" + " " * len(name)
    )


def _has(array, names):
    dtype_names = array.dtype.names
    if dtype_names is None:
        dtype_names = ()
    return all(x in names for x in dtype_names)


class CoordinatesNumpy:
    lib = numpy


class AzimuthalNumpy(CoordinatesNumpy):
    pass


class LongitudinalNumpy(CoordinatesNumpy):
    pass


class TemporalNumpy(CoordinatesNumpy):
    pass


class AzimuthalNumpyXY(numpy.ndarray, AzimuthalNumpy, vector.geometry.AzimuthalXY):
    object_type = vector.backends.object_.AzimuthalObjectXY

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
        return _getitem(self, where)


class AzimuthalNumpyRhoPhi(
    numpy.ndarray, AzimuthalNumpy, vector.geometry.AzimuthalRhoPhi
):
    object_type = vector.backends.object_.AzimuthalObjectRhoPhi

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
        return _getitem(self, where)


class LongitudinalNumpyZ(
    numpy.ndarray, LongitudinalNumpy, vector.geometry.LongitudinalZ
):
    object_type = vector.backends.object_.LongitudinalObjectZ

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
        return _getitem(self, where)


class LongitudinalNumpyTheta(
    numpy.ndarray, LongitudinalNumpy, vector.geometry.LongitudinalTheta
):
    object_type = vector.backends.object_.LongitudinalObjectTheta

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
        return _getitem(self, where)


class LongitudinalNumpyEta(
    numpy.ndarray, LongitudinalNumpy, vector.geometry.LongitudinalEta
):
    object_type = vector.backends.object_.LongitudinalObjectEta

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
        return _getitem(self, where)


class TemporalNumpyT(numpy.ndarray, TemporalNumpy, vector.geometry.TemporalT):
    object_type = vector.backends.object_.TemporalObjectT

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
        return _getitem(self, where)


class TemporalNumpyTau(numpy.ndarray, TemporalNumpy, vector.geometry.TemporalTau):
    object_type = vector.backends.object_.TemporalObjectTau

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
        return _getitem(self, where)


class PlanarNumpy(numpy.ndarray, vector.methods.Planar):
    lib = numpy

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

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
        return _array_repr(self)

    @property
    def azimuthal(self):
        return self.view(self._azimuthal_type)


class PlanarVectorNumpy(vector.geometry.PlanarVector, PlanarNumpy):
    object_type = vector.backends.object_.PlanarVectorObject

    def __getitem__(self, where):
        return _getitem(self, where)

    def _wrap_result(self, result, returns):
        if returns == [float]:
            return result
        elif returns == [vector.geometry.AzimuthalXY] or returns == [
            vector.geometry.AzimuthalRhoPhi
        ]:
            dtype = []
            for i, name in enumerate(_coordinate_class_to_names[returns[0]]):
                dtype.append((name, result[i].dtype))
            out = numpy.empty(result[0].shape, dtype=dtype)
            for i, name in enumerate(_coordinate_class_to_names[returns[0]]):
                out[name] = result[i]
            return out.view(PlanarVectorNumpy)
        else:
            raise AssertionError(repr(returns))


class PlanarPointNumpy(vector.geometry.PlanarPoint, PlanarNumpy):
    object_type = vector.backends.object_.PlanarPointObject

    def __getitem__(self, where):
        return _getitem(self, where)

    def _wrap_result(self, result, returns):
        if returns == [float]:
            return result
        elif returns == [vector.geometry.AzimuthalXY] or returns == [
            vector.geometry.AzimuthalRhoPhi
        ]:
            dtype = []
            for i, name in enumerate(_coordinate_class_to_names[returns[0]]):
                dtype.append((name, result[i].dtype))
            out = numpy.empty(result[0].shape, dtype=dtype)
            for i, name in enumerate(_coordinate_class_to_names[returns[0]]):
                out[name] = result[i]
            return out.view(PlanarPointNumpy)
        else:
            raise AssertionError(repr(returns))


class SpatialNumpy(numpy.ndarray, vector.methods.Spatial):
    lib = numpy

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

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
        return _array_repr(self)

    @property
    def azimuthal(self):
        return self.view(self._azimuthal_type)

    @property
    def longitudinal(self):
        return self.view(self._longitudinal_type)


class SpatialVectorNumpy(vector.geometry.SpatialVector, SpatialNumpy):
    object_type = vector.backends.object_.SpatialVectorObject

    def __getitem__(self, where):
        return _getitem(self, where)

    def _wrap_result(self, result, returns):
        if returns == [float]:
            return result
        elif returns == [vector.geometry.AzimuthalXY] or returns == [
            vector.geometry.AzimuthalRhoPhi
        ]:
            dtype = []
            for i, name in enumerate(_coordinate_class_to_names[returns[0]]):
                dtype.append((name, result[i].dtype))
            for name in _coordinate_class_to_names[vector.geometry.ltype(self)]:
                dtype.append((name, self.dtype[name]))
            out = numpy.empty(result[0].shape, dtype=dtype)
            for i, name in enumerate(_coordinate_class_to_names[returns[0]]):
                out[name] = result[i]
            return out.view(SpatialVectorNumpy)
        else:
            raise AssertionError(repr(returns))


class SpatialPointNumpy(vector.geometry.SpatialPoint, SpatialNumpy):
    object_type = vector.backends.object_.SpatialPointObject

    def __getitem__(self, where):
        return _getitem(self, where)

    def _wrap_result(self, result, returns):
        if returns == [float]:
            return result
        elif returns == [vector.geometry.AzimuthalXY] or returns == [
            vector.geometry.AzimuthalRhoPhi
        ]:
            dtype = []
            for i, name in enumerate(_coordinate_class_to_names[returns[0]]):
                dtype.append((name, result[i].dtype))
            for name in _coordinate_class_to_names[vector.geometry.ltype(self)]:
                dtype.append((name, self.dtype[name]))
            out = numpy.empty(result[0].shape, dtype=dtype)
            for i, name in enumerate(_coordinate_class_to_names[returns[0]]):
                out[name] = result[i]
            return out.view(SpatialPointNumpy)
        else:
            raise AssertionError(repr(returns))


class LorentzNumpy(numpy.ndarray, vector.methods.Lorentz):
    lib = numpy

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

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
        return _array_repr(self)

    @property
    def azimuthal(self):
        return self.view(self._azimuthal_type)

    @property
    def longitudinal(self):
        return self.view(self._longitudinal_type)

    @property
    def temporal(self):
        return self.view(self._temporal_type)


class LorentzVectorNumpy(vector.geometry.LorentzVector, LorentzNumpy):
    object_type = vector.backends.object_.LorentzVectorObject

    def __getitem__(self, where):
        return _getitem(self, where)

    def _wrap_result(self, result, returns):
        if returns == [float]:
            return result
        elif returns == [vector.geometry.AzimuthalXY] or returns == [
            vector.geometry.AzimuthalRhoPhi
        ]:
            dtype = []
            for i, name in enumerate(_coordinate_class_to_names[returns[0]]):
                dtype.append((name, result[i].dtype))
            for name in _coordinate_class_to_names[vector.geometry.ltype(self)]:
                dtype.append((name, self.dtype[name]))
            for name in _coordinate_class_to_names[vector.geometry.ttype(self)]:
                dtype.append((name, self.dtype[name]))
            out = numpy.empty(result[0].shape, dtype=dtype)
            for i, name in enumerate(_coordinate_class_to_names[returns[0]]):
                out[name] = result[i]
            return out.view(LorentzPointNumpy)
        else:
            raise AssertionError(repr(returns))


class LorentzPointNumpy(vector.geometry.LorentzPoint, LorentzNumpy):
    object_type = vector.backends.object_.LorentzPointObject

    def __getitem__(self, where):
        return _getitem(self, where)

    def _wrap_result(self, result, returns):
        if returns == [float]:
            return result
        elif returns == [vector.geometry.AzimuthalXY] or returns == [
            vector.geometry.AzimuthalRhoPhi
        ]:
            dtype = []
            for i, name in enumerate(_coordinate_class_to_names[returns[0]]):
                dtype.append((name, result[i].dtype))
            for name in _coordinate_class_to_names[vector.geometry.ltype(self)]:
                dtype.append((name, self.dtype[name]))
            for name in _coordinate_class_to_names[vector.geometry.ttype(self)]:
                dtype.append((name, self.dtype[name]))
            out = numpy.empty(result[0].shape, dtype=dtype)
            for i, name in enumerate(_coordinate_class_to_names[returns[0]]):
                out[name] = result[i]
            return out.view(LorentzPointNumpy)
        else:
            raise AssertionError(repr(returns))


class TransformNumpy:
    lib = numpy


class Transform2DNumpy(numpy.ndarray, TransformNumpy, vector.methods.Transform2D):
    object_type = vector.backends.object_.Transform2DObject

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not _has(self, ("xx", "xy", "yx", "yy")):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype with fields "
                '("xx", "xy", "yx", "yy")'
            )

    def __getitem__(self, where):
        return _getitem(self, where)

    @property
    def elements(self):
        return tuple(self[x] for x in ("xx", "xy", "yx", "yy"))

    def apply(self, v):
        x, y = vector.methods.Transform2D.apply(self, v)
        out = numpy.empty(x.shape, dtype=[("x", x.dtype), ("y", y.dtype)])
        out["x"], out["y"] = x, y
        return out.view(PlanarVectorNumpy)


class AzimuthalRotationNumpy(
    numpy.ndarray, TransformNumpy, vector.methods.AzimuthalRotation
):
    object_type = vector.backends.object_.AzimuthalRotationObject

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not _has(self, ("angle",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "angle"'
            )

    @property
    def angle(self):
        return self["angle"].view(numpy.ndarray)

    def __getitem__(self, where):
        return _getitem(self, where)

    def apply(self, v):
        x, y = vector.methods.Transform2D.apply(self, v)
        out = numpy.empty(x.shape, dtype=[("x", x.dtype), ("y", y.dtype)])
        out["x"], out["y"] = x, y
        return out.view(PlanarVectorNumpy)


class Transform3DNumpy(numpy.ndarray, TransformNumpy, vector.methods.Transform3D):
    object_type = vector.backends.object_.Transform3DObject

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not _has(self, ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz")):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype with fields "
                '("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz")'
            )

    def __getitem__(self, where):
        return _getitem(self, where)

    @property
    def elements(self):
        return tuple(
            self[x] for x in ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz")
        )

    def apply(self, v):
        x, y, z = vector.methods.Transform3D.apply(self, v)
        out = numpy.empty(
            x.shape, dtype=[("x", x.dtype), ("y", y.dtype), ("z", z.dtype)]
        )
        out["x"], out["y"], out["z"] = x, y, z
        return out.view(SpatialVectorNumpy)


class AxisAngleRotationNumpy(
    numpy.ndarray, TransformNumpy, vector.methods.AxisAngleRotation
):
    object_type = vector.backends.object_.AxisAngleRotationObject

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not _has(self, ("phi", "theta", "angle")):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("phi", "theta", "angle")'
            )

    @property
    def phi(self):
        return self["phi"].view(numpy.ndarray)

    @property
    def theta(self):
        return self["theta"].view(numpy.ndarray)

    @property
    def angle(self):
        return self["angle"].view(numpy.ndarray)

    def __getitem__(self, where):
        return _getitem(self, where)

    def apply(self, v):
        x, y, z = vector.methods.Transform3D.apply(self, v)
        out = numpy.empty(
            x.shape, dtype=[("x", x.dtype), ("y", y.dtype), ("z", z.dtype)]
        )
        out["x"], out["y"], out["z"] = x, y, z
        return out.view(SpatialVectorNumpy)


class EulerAngleRotationNumpy(
    numpy.ndarray, TransformNumpy, vector.methods.EulerAngleRotation
):
    object_type = vector.backends.object_.EulerAngleRotationObject

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not _has(self, ("alpha", "beta", "gamma")):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("alpha", "beta", "gamma")'
            )

    @property
    def alpha(self):
        return self["alpha"].view(numpy.ndarray)

    @property
    def beta(self):
        return self["beta"].view(numpy.ndarray)

    @property
    def gamma(self):
        return self["gamma"].view(numpy.ndarray)

    def __getitem__(self, where):
        return _getitem(self, where)

    def apply(self, v):
        x, y, z = vector.methods.Transform3D.apply(self, v)
        out = numpy.empty(
            x.shape, dtype=[("x", x.dtype), ("y", y.dtype), ("z", z.dtype)]
        )
        out["x"], out["y"], out["z"] = x, y, z
        return out.view(SpatialVectorNumpy)


class NauticalRotationNumpy(
    numpy.ndarray, TransformNumpy, vector.methods.NauticalRotation
):
    object_type = vector.backends.object_.NauticalRotationObject

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not _has(self, ("yaw", "pitch", "roll")):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("yaw", "pitch", "roll")'
            )

    @property
    def yaw(self):
        return self["yaw"].view(numpy.ndarray)

    @property
    def pitch(self):
        return self["pitch"].view(numpy.ndarray)

    @property
    def roll(self):
        return self["roll"].view(numpy.ndarray)

    def __getitem__(self, where):
        return _getitem(self, where)

    def apply(self, v):
        x, y, z = vector.methods.Transform3D.apply(self, v)
        out = numpy.empty(
            x.shape, dtype=[("x", x.dtype), ("y", y.dtype), ("z", z.dtype)]
        )
        out["x"], out["y"], out["z"] = x, y, z
        return out.view(SpatialVectorNumpy)


class Transform4DNumpy(numpy.ndarray, TransformNumpy, vector.methods.Transform4D):
    object_type = vector.backends.object_.Transform4DObject

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not _has(
            self,
            (
                "xx",
                "xy",
                "xz",
                "xt",
                "yx",
                "yy",
                "yz",
                "yt",
                "zx",
                "zy",
                "zz",
                "zt",
                "tx",
                "ty",
                "tz",
                "tt",
            ),
        ):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype with fields "
                '("xx", "xy", "xz", "xt", "yx", "yy", "yz", "yt", "zx", "zy", "zz", "zt", "tx", "ty", "tz", "tt")'
            )

    def __getitem__(self, where):
        return _getitem(self, where)

    @property
    def elements(self):
        return tuple(
            self[x]
            for x in (
                "xx",
                "xy",
                "xz",
                "xt",
                "yx",
                "yy",
                "yz",
                "yt",
                "zx",
                "zy",
                "zz",
                "zt",
                "tx",
                "ty",
                "tz",
                "tt",
            )
        )

    def apply(self, v):
        x, y, z, t = vector.methods.Transform4D.apply(self, v)
        out = numpy.empty(
            x.shape,
            dtype=[("x", x.dtype), ("y", y.dtype), ("z", z.dtype), ("t", t.dtype)],
        )
        out["x"], out["y"], out["z"], out["t"] = x, y, z, t
        return out.view(LorentzVectorNumpy)


class LongitudinalBoostNumpy(
    numpy.ndarray, TransformNumpy, vector.methods.LongitudinalBoost
):
    object_type = vector.backends.object_.LongitudinalBoostObject

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not _has(self, ("gamma",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "gamma"'
            )

    @property
    def gamma(self):
        return self["gamma"].view(numpy.ndarray)

    def __getitem__(self, where):
        return _getitem(self, where)

    def apply(self, v):
        x, y, z, t = vector.methods.Transform4D.apply(self, v)
        out = numpy.empty(
            x.shape,
            dtype=[("x", x.dtype), ("y", y.dtype), ("z", z.dtype), ("t", t.dtype)],
        )
        out["x"], out["y"], out["z"], out["t"] = x, y, z, t
        return out.view(LorentzVectorNumpy)


class AxisAngleBoostNumpy(numpy.ndarray, TransformNumpy, vector.methods.AxisAngleBoost):
    object_type = vector.backends.object_.AxisAngleBoostObject

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not _has(self, ("phi", "theta", "gamma")):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("phi", "theta", "gamma")'
            )

    @property
    def phi(self):
        return self["phi"].view(numpy.ndarray)

    @property
    def theta(self):
        return self["theta"].view(numpy.ndarray)

    @property
    def gamma(self):
        return self["gamma"].view(numpy.ndarray)

    def __getitem__(self, where):
        return _getitem(self, where)

    def apply(self, v):
        x, y, z, t = vector.methods.Transform4D.apply(self, v)
        out = numpy.empty(
            x.shape,
            dtype=[("x", x.dtype), ("y", y.dtype), ("z", z.dtype), ("t", t.dtype)],
        )
        out["x"], out["y"], out["z"], out["t"] = x, y, z, t
        return out.view(LorentzVectorNumpy)
