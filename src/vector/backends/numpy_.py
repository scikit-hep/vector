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


class AzimuthalNumpy:
    pass


class LongitudinalNumpy:
    pass


class TemporalNumpy:
    pass


class AzimuthalNumpyXY(numpy.ndarray, AzimuthalNumpy, vector.geometry.AzimuthalXY):
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
    @property
    def elements(self):
        return (self["w"],)

    @property
    def w(self):
        return self["w"]

    def __getitem__(self, where):
        return getitem(self, where, vector.backends.object_.LongitudinalObjectW)


class TemporalNumpyT(numpy.ndarray, TemporalNumpy, vector.geometry.TemporalT):
    @property
    def elements(self):
        return (self["t"],)

    @property
    def t(self):
        return self["t"]

    def __getitem__(self, where):
        return getitem(self, where, vector.backends.object_.TemporalObjectT)


class TemporalNumpyTau(numpy.ndarray, TemporalNumpy, vector.geometry.TemporalTau):
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
        if self.dtype.names == ("x", "y"):
            self._azimuthal_type = AzimuthalNumpyXY
        elif self.dtype.names == ("rho", "phi"):
            self._azimuthal_type = AzimuthalNumpyRhoPhi
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype with fields "
                '("x", "y") or ("rho", "phi")'
            )

    @property
    def azimuthal(self):
        return self.view(self._azimuthal_type)


class PlanarVectorNumpy(vector.geometry.PlanarVector, PlanarNumpy):
    def __getitem__(self, where):
        return getitem(self, where, vector.backends.object_.PlanarVectorObject)


class SpatialNumpy(numpy.ndarray, vector.methods.Spatial):
    lib = numpy

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)


class SpatialVectorNumpy(vector.geometry.SpatialVector, SpatialNumpy):
    def __getitem__(self, where):
        return getitem(self, where, vector.backends.object_.SpatialVectorObject)


class LorentzNumpy(numpy.ndarray, vector.methods.Lorentz):
    lib = numpy

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        raise NotImplementedError


class LorentzVectorNumpy(vector.geometry.LorentzVector, LorentzNumpy):
    def __getitem__(self, where):
        return getitem(self, where, vector.backends.object_.LorentzVectorObject)


class TransformNumpy:
    lib = numpy


class Transform2DNumpy(numpy.ndarray, TransformNumpy, vector.methods.Transform2D):
    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if self.dtype.names != ("xx", "xy", "yx", "yy"):
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
        out["x"] = x
        out["y"] = y
        return out.view(PlanarVectorNumpy)
