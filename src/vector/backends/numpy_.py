# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy

import vector.compute


class AzimuthalNumpy:
    pass


class LongitudinalNumpy:
    pass


class TemporalNumpy:
    pass


class AzimuthalNumpyXY(numpy.ndarray, AzimuthalNumpy, vector.geometry.AzimuthalXY):
    def __iter__(self):
        yield self["x"].view(numpy.ndarray)
        yield self["y"].view(numpy.ndarray)

    @property
    def x(self):
        return self["x"].view(numpy.ndarray)

    @property
    def y(self):
        return self["y"].view(numpy.ndarray)


class AzimuthalNumpyRhoPhi(
    numpy.ndarray, AzimuthalNumpy, vector.geometry.AzimuthalRhoPhi
):
    def __iter__(self):
        yield self["rho"].view(numpy.ndarray)
        yield self["phi"].view(numpy.ndarray)

    @property
    def rho(self):
        return self["rho"].view(numpy.ndarray)

    @property
    def phi(self):
        return self["phi"].view(numpy.ndarray)


class LongitudinalNumpyZ(
    numpy.ndarray, LongitudinalNumpy, vector.geometry.LongitudinalZ
):
    def __iter__(self):
        yield self["z"].view(numpy.ndarray)

    @property
    def z(self):
        return self["z"].view(numpy.ndarray)


class LongitudinalNumpyTheta(
    numpy.ndarray, LongitudinalNumpy, vector.geometry.LongitudinalTheta
):
    def __iter__(self):
        yield self["theta"].view(numpy.ndarray)

    @property
    def theta(self):
        return self["theta"].view(numpy.ndarray)


class LongitudinalNumpyEta(
    numpy.ndarray, LongitudinalNumpy, vector.geometry.LongitudinalEta
):
    def __iter__(self):
        yield self["eta"].view(numpy.ndarray)

    @property
    def eta(self):
        return self["eta"].view(numpy.ndarray)


class LongitudinalNumpyW(
    numpy.ndarray, LongitudinalNumpy, vector.geometry.LongitudinalW
):
    def __iter__(self):
        yield self["w"].view(numpy.ndarray)

    @property
    def w(self):
        return self["w"].view(numpy.ndarray)


class TemporalNumpyT(numpy.ndarray, TemporalNumpy, vector.geometry.TemporalT):
    def __iter__(self):
        yield self["t"].view(numpy.ndarray)

    @property
    def t(self):
        return self["t"].view(numpy.ndarray)


class TemporalNumpyTau(numpy.ndarray, TemporalNumpy, vector.geometry.TemporalTau):
    def __iter__(self):
        yield self["tau"].view(numpy.ndarray)

    @property
    def tau(self):
        return self["tau"].view(numpy.ndarray)


class PlanarVectorNumpy(numpy.ndarray, vector.geometry.Planar, vector.geometry.Vector):
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

    @property
    def x(self):
        return vector.compute.planar.x.dispatch(numpy, self)

    @property
    def y(self):
        return vector.compute.planar.y.dispatch(numpy, self)

    @property
    def rho(self):
        return vector.compute.planar.rho.dispatch(numpy, self)

    @property
    def phi(self):
        return vector.compute.planar.phi.dispatch(numpy, self)

    @property
    def rho2(self):
        return vector.compute.planar.rho2.dispatch(numpy, self)


class SpatialVectorNumpy(
    numpy.ndarray, vector.geometry.Spatial, vector.geometry.Vector
):
    pass


class LorentzVectorNumpy(
    numpy.ndarray, vector.geometry.Lorentz, vector.geometry.Vector
):
    pass
