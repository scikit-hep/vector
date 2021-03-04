# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

import numpy

import vector.geometry
import vector.methods


class AzimuthalObject:
    pass


class LongitudinalObject:
    pass


class TemporalObject:
    pass


class AzimuthalObjectXY(typing.NamedTuple):
    x: float
    y: float


AzimuthalObjectXY.__bases__ = (AzimuthalObject, vector.geometry.AzimuthalXY, tuple)


class AzimuthalObjectRhoPhi(typing.NamedTuple):
    rho: float
    phi: float


AzimuthalObjectRhoPhi.__bases__ = (
    AzimuthalObject,
    vector.geometry.AzimuthalRhoPhi,
    tuple,
)


class LongitudinalObjectZ(typing.NamedTuple):
    z: float


LongitudinalObjectZ.__bases__ = (
    LongitudinalObject,
    vector.geometry.LongitudinalZ,
    tuple,
)


class LongitudinalObjectTheta(typing.NamedTuple):
    theta: float


LongitudinalObjectTheta.__bases__ = (
    LongitudinalObject,
    vector.geometry.LongitudinalTheta,
    tuple,
)


class LongitudinalObjectEta(typing.NamedTuple):
    eta: float


LongitudinalObjectEta.__bases__ = (
    LongitudinalObject,
    vector.geometry.LongitudinalEta,
    tuple,
)


class LongitudinalObjectW(typing.NamedTuple):
    w: float


LongitudinalObjectW.__bases__ = (
    LongitudinalObject,
    vector.geometry.LongitudinalW,
    tuple,
)


class TemporalObjectT(typing.NamedTuple):
    t: float


TemporalObjectT.__bases__ = (TemporalObject, vector.geometry.TemporalT, tuple)


class TemporalObjectTau(typing.NamedTuple):
    tau: float


TemporalObjectTau.__bases__ = (TemporalObject, vector.geometry.TemporalTau, tuple)


class PlanarObject(vector.methods.Planar):
    __slots__ = ("azimuthal",)

    lib = numpy

    def __init__(self, azimuthal):
        self.azimuthal = azimuthal


class PlanarVectorObject(vector.geometry.PlanarVector, PlanarObject):
    pass


class SpatialObject(vector.methods.Spatial):
    __slots__ = ("azimuthal", "longitudinal")

    def __init__(self, azimuthal, longitudinal):
        self.azimuthal = azimuthal
        self.longitudinal = longitudinal


class SpatialVectorObject(vector.geometry.SpatialVector, SpatialObject):
    pass


class LorentzObject(vector.methods.Lorentz):
    __slots__ = ("azimuthal", "longitudinal", "temporal")

    def __init__(self, azimuthal, longitudinal, temporal):
        self.azimuthal = azimuthal
        self.longitudinal = longitudinal
        self.temporal = temporal


class LorentzVectorObject(vector.geometry.LorentzVector, LorentzObject):
    pass