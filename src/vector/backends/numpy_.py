# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy

import vector.geometry
import vector.methods


class AzimuthalNumpy:
    pass


class LongitudinalNumpy:
    pass


class TemporalNumpy:
    pass


class AzimuthalNumpyXY(numpy.ndarray, AzimuthalNumpy, vector.geometry.AzimuthalXY):
    @property
    def elements(self):
        return (self["x"].view(numpy.ndarray), self["y"].view(numpy.ndarray))

    @property
    def x(self):
        return self["x"].view(numpy.ndarray)

    @property
    def y(self):
        return self["y"].view(numpy.ndarray)


class AzimuthalNumpyRhoPhi(
    numpy.ndarray, AzimuthalNumpy, vector.geometry.AzimuthalRhoPhi
):
    @property
    def elements(self):
        return (self["rho"].view(numpy.ndarray), self["phi"].view(numpy.ndarray))

    @property
    def rho(self):
        return self["rho"].view(numpy.ndarray)

    @property
    def phi(self):
        return self["phi"].view(numpy.ndarray)


class LongitudinalNumpyZ(
    numpy.ndarray, LongitudinalNumpy, vector.geometry.LongitudinalZ
):
    @property
    def elements(self):
        return (self["z"].view(numpy.ndarray),)

    @property
    def z(self):
        return self["z"].view(numpy.ndarray)


class LongitudinalNumpyTheta(
    numpy.ndarray, LongitudinalNumpy, vector.geometry.LongitudinalTheta
):
    @property
    def elements(self):
        return (self["theta"].view(numpy.ndarray),)

    @property
    def theta(self):
        return self["theta"].view(numpy.ndarray)


class LongitudinalNumpyEta(
    numpy.ndarray, LongitudinalNumpy, vector.geometry.LongitudinalEta
):
    @property
    def elements(self):
        return (self["eta"].view(numpy.ndarray),)

    @property
    def eta(self):
        return self["eta"].view(numpy.ndarray)


class LongitudinalNumpyW(
    numpy.ndarray, LongitudinalNumpy, vector.geometry.LongitudinalW
):
    @property
    def elements(self):
        return (self["w"].view(numpy.ndarray),)

    @property
    def w(self):
        return self["w"].view(numpy.ndarray)


class TemporalNumpyT(numpy.ndarray, TemporalNumpy, vector.geometry.TemporalT):
    @property
    def elements(self):
        return (self["t"].view(numpy.ndarray),)

    @property
    def t(self):
        return self["t"].view(numpy.ndarray)


class TemporalNumpyTau(numpy.ndarray, TemporalNumpy, vector.geometry.TemporalTau):
    @property
    def elements(self):
        return (self["tau"].view(numpy.ndarray),)

    @property
    def tau(self):
        return self["tau"].view(numpy.ndarray)


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
    pass


class SpatialNumpy(numpy.ndarray, vector.methods.Spatial):
    pass


class SpatialVectorNumpy(vector.geometry.SpatialVector, SpatialNumpy):
    pass


class LorentzNumpy(numpy.ndarray, vector.methods.Lorentz):
    pass


class LorentzVectorNumpy(vector.geometry.LorentzVector, LorentzNumpy):
    pass
