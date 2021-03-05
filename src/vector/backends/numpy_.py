# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy

import vector.backends.object_
import vector.geometry
import vector.methods


def getitem(array, where, object_type):
    if isinstance(where, str):
        return array.view(numpy.ndarray)[where]
    else:
        out = numpy.ndarray.__getitem__(array, where)
        if isinstance(out, numpy.void):
            return object_type(*out)
        else:
            return out


def has(array, names):
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
    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not has(self, ("x", "y")):
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
        return getitem(self, where, vector.backends.object_.AzimuthalObjectXY)


class AzimuthalNumpyRhoPhi(
    numpy.ndarray, AzimuthalNumpy, vector.geometry.AzimuthalRhoPhi
):
    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not has(self, ("rho", "phi")):
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
        return getitem(self, where, vector.backends.object_.AzimuthalObjectRhoPhi)


class LongitudinalNumpyZ(
    numpy.ndarray, LongitudinalNumpy, vector.geometry.LongitudinalZ
):
    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not has(self, ("z",)):
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
        return getitem(self, where, vector.backends.object_.LongitudinalObjectZ)


class LongitudinalNumpyTheta(
    numpy.ndarray, LongitudinalNumpy, vector.geometry.LongitudinalTheta
):
    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not has(self, ("theta",)):
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
        return getitem(self, where, vector.backends.object_.LongitudinalObjectTheta)


class LongitudinalNumpyEta(
    numpy.ndarray, LongitudinalNumpy, vector.geometry.LongitudinalEta
):
    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not has(self, ("eta",)):
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
        return getitem(self, where, vector.backends.object_.LongitudinalObjectEta)


class LongitudinalNumpyW(
    numpy.ndarray, LongitudinalNumpy, vector.geometry.LongitudinalW
):
    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not has(self, ("w",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "w"'
            )

    @property
    def elements(self):
        return (self["w"],)

    @property
    def w(self):
        return self["w"]

    def __getitem__(self, where):
        return getitem(self, where, vector.backends.object_.LongitudinalObjectW)


class TemporalNumpyT(numpy.ndarray, TemporalNumpy, vector.geometry.TemporalT):
    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not has(self, ("t",)):
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
        return getitem(self, where, vector.backends.object_.TemporalObjectT)


class TemporalNumpyTau(numpy.ndarray, TemporalNumpy, vector.geometry.TemporalTau):
    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not has(self, ("tau",)):
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
        return getitem(self, where, vector.backends.object_.TemporalObjectTau)


class PlanarNumpy(numpy.ndarray, vector.methods.Planar):
    lib = numpy

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if has(self, ("x", "y")):
            self._azimuthal_type = AzimuthalNumpyXY
        elif has(self, ("rho", "phi")):
            self._azimuthal_type = AzimuthalNumpyRhoPhi
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("x", "y") or ("rho", "phi")'
            )

    @property
    def azimuthal(self):
        return self.view(self._azimuthal_type)


class PlanarVectorNumpy(vector.geometry.PlanarVector, PlanarNumpy):
    def __getitem__(self, where):
        return getitem(self, where, vector.backends.object_.PlanarVectorObject)


class PlanarPointNumpy(vector.geometry.PlanarPoint, PlanarNumpy):
    def __getitem__(self, where):
        return getitem(self, where, vector.backends.object_.PlanarPointObject)


class SpatialNumpy(numpy.ndarray, vector.methods.Spatial):
    lib = numpy

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if has(self, ("x", "y")):
            self._azimuthal_type = AzimuthalNumpyXY
        elif has(self, ("rho", "phi")):
            self._azimuthal_type = AzimuthalNumpyRhoPhi
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("x", "y") or ("rho", "phi")'
            )
        if has(self, ("z",)):
            self._longitudinal_type = LongitudinalNumpyZ
        elif has(self, ("theta",)):
            self._longitudinal_type = LongitudinalNumpyTheta
        elif has(self, ("eta",)):
            self._longitudinal_type = LongitudinalNumpyEta
        elif has(self, ("w",)):
            self._longitudinal_type = LongitudinalNumpyW
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "z" or "theta" or "eta" or "w"'
            )


class SpatialVectorNumpy(vector.geometry.SpatialVector, SpatialNumpy):
    def __getitem__(self, where):
        return getitem(self, where, vector.backends.object_.SpatialVectorObject)


class SpatialPointNumpy(vector.geometry.SpatialPoint, SpatialNumpy):
    def __getitem__(self, where):
        return getitem(self, where, vector.backends.object_.SpatialPointObject)


class LorentzNumpy(numpy.ndarray, vector.methods.Lorentz):
    lib = numpy

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if has(self, ("x", "y")):
            self._azimuthal_type = AzimuthalNumpyXY
        elif has(self, ("rho", "phi")):
            self._azimuthal_type = AzimuthalNumpyRhoPhi
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("x", "y") or ("rho", "phi")'
            )
        if has(self, ("z",)):
            self._longitudinal_type = LongitudinalNumpyZ
        elif has(self, ("theta",)):
            self._longitudinal_type = LongitudinalNumpyTheta
        elif has(self, ("eta",)):
            self._longitudinal_type = LongitudinalNumpyEta
        elif has(self, ("w",)):
            self._longitudinal_type = LongitudinalNumpyW
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "z" or "theta" or "eta" or "w"'
            )
        if has(self, ("t",)):
            self._temporal_type = TemporalNumpyT
        elif has(self, ("tau",)):
            self._temporal_type = TemporalNumpyTau
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "t" or "tau"'
            )


class LorentzVectorNumpy(vector.geometry.LorentzVector, LorentzNumpy):
    def __getitem__(self, where):
        return getitem(self, where, vector.backends.object_.LorentzVectorObject)


class LorentzPointNumpy(vector.geometry.LorentzPoint, LorentzNumpy):
    def __getitem__(self, where):
        return getitem(self, where, vector.backends.object_.LorentzPointObject)


class TransformNumpy:
    lib = numpy


class Transform2DNumpy(numpy.ndarray, TransformNumpy, vector.methods.Transform2D):
    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not has(self, ("xx", "xy", "yx", "yy")):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype with fields "
                '("xx", "xy", "yx", "yy")'
            )

    def __getitem__(self, where):
        return getitem(self, where, vector.backends.object_.Transform2DObject)

    @property
    def elements(self):
        return tuple(self[x] for x in ("xx", "xy", "yx", "yy"))

    def apply(self, v):
        x, y = vector.methods.Transform2D.apply(self, v)
        out = numpy.empty(x.shape, dtype=[("x", x.dtype), ("y", y.dtype)])
        out["x"], out["y"] = x, y
        return out.view(PlanarVectorNumpy)


class AzimuthalRotationNumpy(numpy.ndarray, TransformNumpy, vector.methods.AzimuthalRotation):
    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not has(self, ("angle",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "angle"'
            )

    @property
    def angle(self):
        return self["angle"].view(numpy.ndarray)

    def __getitem__(self, where):
        return getitem(self, where, vector.backends.object_.AzimuthalRotationObject)

    def apply(self, v):
        x, y = vector.methods.Transform2D.apply(self, v)
        out = numpy.empty(x.shape, dtype=[("x", x.dtype), ("y", y.dtype)])
        out["x"], out["y"] = x, y
        return out.view(PlanarVectorNumpy)


class Transform3DNumpy(numpy.ndarray, TransformNumpy, vector.methods.Transform3D):
    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not has(self, ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz")):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype with fields "
                '("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz")'
            )

    def __getitem__(self, where):
        return getitem(self, where, vector.backends.object_.Transform3DObject)

    @property
    def elements(self):
        return tuple(self[x] for x in ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"))

    def apply(self, v):
        x, y, z = vector.methods.Transform3D.apply(self, v)
        out = numpy.empty(x.shape, dtype=[("x", x.dtype), ("y", y.dtype), ("z", z.dtype)])
        out["x"], out["y"], out["z"] = x, y, z
        return out.view(SpatialVectorNumpy)


class AxisAngleRotationNumpy(numpy.ndarray, TransformNumpy, vector.methods.AxisAngleRotation):
    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not has(self, ("phi", "theta", "angle")):
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
        return getitem(self, where, vector.backends.object_.AxisAngleRotationObject)

    def apply(self, v):
        x, y, z = vector.methods.Transform3D.apply(self, v)
        out = numpy.empty(x.shape, dtype=[("x", x.dtype), ("y", y.dtype), ("z", z.dtype)])
        out["x"], out["y"], out["z"] = x, y, z
        return out.view(SpatialVectorNumpy)


class EulerAngleRotationNumpy(numpy.ndarray, TransformNumpy, vector.methods.EulerAngleRotation):
    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not has(self, ("alpha", "beta", "gamma")):
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
        return getitem(self, where, vector.backends.object_.EulerAngleRotationObject)

    def apply(self, v):
        x, y, z = vector.methods.Transform3D.apply(self, v)
        out = numpy.empty(x.shape, dtype=[("x", x.dtype), ("y", y.dtype), ("z", z.dtype)])
        out["x"], out["y"], out["z"] = x, y, z
        return out.view(SpatialVectorNumpy)


class NauticalRotationNumpy(numpy.ndarray, TransformNumpy, vector.methods.NauticalRotation):
    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not has(self, ("yaw", "pitch", "roll")):
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
        return getitem(self, where, vector.backends.object_.NauticalRotationObject)

    def apply(self, v):
        x, y, z = vector.methods.Transform3D.apply(self, v)
        out = numpy.empty(x.shape, dtype=[("x", x.dtype), ("y", y.dtype), ("z", z.dtype)])
        out["x"], out["y"], out["z"] = x, y, z
        return out.view(SpatialVectorNumpy)


class Transform4DNumpy(numpy.ndarray, TransformNumpy, vector.methods.Transform4D):
    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not has(self, ("xx", "xy", "xz", "xt", "yx", "yy", "yz", "yt", "zx", "zy", "zz", "zt", "tx", "ty", "tz", "tt")):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype with fields "
                '("xx", "xy", "xz", "xt", "yx", "yy", "yz", "yt", "zx", "zy", "zz", "zt", "tx", "ty", "tz", "tt")'
            )

    def __getitem__(self, where):
        return getitem(self, where, vector.backends.object_.Transform4DObject)

    @property
    def elements(self):
        return tuple(self[x] for x in ("xx", "xy", "xz", "xt", "yx", "yy", "yz", "yt", "zx", "zy", "zz", "zt", "tx", "ty", "tz", "tt"))

    def apply(self, v):
        x, y, z, t = vector.methods.Transform4D.apply(self, v)
        out = numpy.empty(x.shape, dtype=[("x", x.dtype), ("y", y.dtype), ("z", z.dtype), ("t", t.dtype)])
        out["x"], out["y"], out["z"], out["t"] = x, y, z, t
        return out.view(LorentzVectorNumpy)


class LongitudinalBoostNumpy(numpy.ndarray, TransformNumpy, vector.methods.LongitudinalBoost):
    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not has(self, ("gamma",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "gamma"'
            )

    @property
    def gamma(self):
        return self["gamma"].view(numpy.ndarray)

    def __getitem__(self, where):
        return getitem(self, where, vector.backends.object_.LongitudinalBoostObject)

    def apply(self, v):
        x, y, z, t = vector.methods.Transform4D.apply(self, v)
        out = numpy.empty(x.shape, dtype=[("x", x.dtype), ("y", y.dtype), ("z", z.dtype), ("t", t.dtype)])
        out["x"], out["y"], out["z"], out["t"] = x, y, z, t
        return out.view(LorentzVectorNumpy)


class AxisAngleBoostNumpy(numpy.ndarray, TransformNumpy, vector.methods.AxisAngleBoost):
    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not has(self, ("phi", "theta", "gamma")):
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
        return getitem(self, where, vector.backends.object_.AxisAngleBoostObject)

    def apply(self, v):
        x, y, z, t = vector.methods.Transform4D.apply(self, v)
        out = numpy.empty(x.shape, dtype=[("x", x.dtype), ("y", y.dtype), ("z", z.dtype), ("t", t.dtype)])
        out["x"], out["y"], out["z"], out["t"] = x, y, z, t
        return out.view(LorentzVectorNumpy)
