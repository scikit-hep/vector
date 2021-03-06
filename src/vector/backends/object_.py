# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

import numpy

import vector.geometry
import vector.methods


class CoordinatesObject:
    pass


class AzimuthalObject(CoordinatesObject):
    pass


class LongitudinalObject(CoordinatesObject):
    pass


class TemporalObject(CoordinatesObject):
    pass


class AzimuthalObjectXY(typing.NamedTuple):
    x: float
    y: float

    @property
    def elements(self):
        return (self.x, self.y)


AzimuthalObjectXY.__bases__ = (AzimuthalObject, vector.geometry.AzimuthalXY, tuple)


class AzimuthalObjectRhoPhi(typing.NamedTuple):
    rho: float
    phi: float

    @property
    def elements(self):
        return (self.rho, self.phi)


AzimuthalObjectRhoPhi.__bases__ = (
    AzimuthalObject,
    vector.geometry.AzimuthalRhoPhi,
    tuple,
)


class LongitudinalObjectZ(typing.NamedTuple):
    z: float

    @property
    def elements(self):
        return (self.z,)


LongitudinalObjectZ.__bases__ = (
    LongitudinalObject,
    vector.geometry.LongitudinalZ,
    tuple,
)


class LongitudinalObjectTheta(typing.NamedTuple):
    theta: float

    @property
    def elements(self):
        return (self.theta,)


LongitudinalObjectTheta.__bases__ = (
    LongitudinalObject,
    vector.geometry.LongitudinalTheta,
    tuple,
)


class LongitudinalObjectEta(typing.NamedTuple):
    eta: float

    @property
    def elements(self):
        return (self.eta,)


LongitudinalObjectEta.__bases__ = (
    LongitudinalObject,
    vector.geometry.LongitudinalEta,
    tuple,
)


class TemporalObjectT(typing.NamedTuple):
    t: float

    @property
    def elements(self):
        return (self.t,)


TemporalObjectT.__bases__ = (TemporalObject, vector.geometry.TemporalT, tuple)


class TemporalObjectTau(typing.NamedTuple):
    tau: float

    @property
    def elements(self):
        return (self.tau,)


TemporalObjectTau.__bases__ = (TemporalObject, vector.geometry.TemporalTau, tuple)


class PlanarObject(vector.methods.Planar):
    __slots__ = ("azimuthal",)

    lib = numpy

    def __init__(self, azimuthal):
        self.azimuthal = azimuthal

    def __repr__(self):
        return f"{type(self).__name__}({self.azimuthal})"


class PlanarVectorObject(vector.geometry.PlanarVector, PlanarObject):
    pass


class PlanarPointObject(vector.geometry.PlanarPoint, PlanarObject):
    pass


class SpatialObject(vector.methods.Spatial):
    __slots__ = ("azimuthal", "longitudinal")

    lib = numpy

    def __init__(self, azimuthal, longitudinal):
        self.azimuthal = azimuthal
        self.longitudinal = longitudinal

    def __repr__(self):
        return f"{type(self).__name__}({self.azimuthal}, {self.longitudinal})"


class SpatialVectorObject(vector.geometry.SpatialVector, SpatialObject):
    pass


class SpatialPointObject(vector.geometry.SpatialPoint, SpatialObject):
    pass


class LorentzObject(vector.methods.Lorentz):
    __slots__ = ("azimuthal", "longitudinal", "temporal")

    lib = numpy

    def __init__(self, azimuthal, longitudinal, temporal):
        self.azimuthal = azimuthal
        self.longitudinal = longitudinal
        self.temporal = temporal

    def __repr__(self):
        return f"{type(self).__name__}({self.azimuthal}, {self.longitudinal}, {self.temporal})"


class LorentzVectorObject(vector.geometry.LorentzVector, LorentzObject):
    pass


class LorentzPointObject(vector.geometry.LorentzPoint, LorentzObject):
    pass


class TransformObject:
    lib = numpy


class Transform2DObject(typing.NamedTuple):
    xx: float
    xy: float
    yx: float
    yy: float

    @property
    def elements(self):
        return self

    def apply(self, v):
        x, y = vector.methods.Transform2D.apply(self, v)
        return PlanarVectorObject(AzimuthalObjectXY(x, y))


Transform2DObject.__bases__ = (TransformObject, vector.methods.Transform2D, tuple)


class AzimuthalRotationObject(typing.NamedTuple):
    angle: float

    def apply(self, v):
        x, y = vector.methods.AzimuthalRotation.apply(self, v)
        return PlanarVectorObject(AzimuthalObjectXY(x, y))


AzimuthalRotationObject.__bases__ = (
    TransformObject,
    vector.methods.AzimuthalRotation,
    tuple,
)


class Transform3DObject(typing.NamedTuple):
    xx: float
    xy: float
    xz: float
    yx: float
    yy: float
    yz: float
    zx: float
    zy: float
    zz: float

    @property
    def elements(self):
        return self

    def apply(self, v):
        x, y, z = vector.methods.Transform3D.apply(self, v)
        return SpatialVectorObject(AzimuthalObjectXY(x, y), LongitudinalObjectZ(z))


Transform3DObject.__bases__ = (TransformObject, vector.methods.Transform3D, tuple)


class AxisAngleRotationObject(typing.NamedTuple):
    phi: float
    theta: float
    angle: float

    def apply(self, v):
        x, y, z = vector.methods.AxisAngleRotation.apply(self, v)
        return SpatialVectorObject(AzimuthalObjectXY(x, y), LongitudinalObjectZ(z))


AxisAngleRotationObject.__bases__ = (
    TransformObject,
    vector.methods.AxisAngleRotation,
    tuple,
)


class EulerAngleRotationObject(typing.NamedTuple):
    alpha: float
    beta: float
    gamma: float

    def apply(self, v):
        x, y, z = vector.methods.EulerAngleRotation.apply(self, v)
        return SpatialVectorObject(AzimuthalObjectXY(x, y), LongitudinalObjectZ(z))


EulerAngleRotationObject.__bases__ = (
    TransformObject,
    vector.methods.EulerAngleRotation,
    tuple,
)


class NauticalRotationObject(typing.NamedTuple):
    yaw: float
    pitch: float
    roll: float

    def apply(self, v):
        x, y, z = vector.methods.NauticalRotation.apply(self, v)
        return SpatialVectorObject(AzimuthalObjectXY(x, y), LongitudinalObjectZ(z))


NauticalRotationObject.__bases__ = (
    TransformObject,
    vector.methods.NauticalRotation,
    tuple,
)


class Transform4DObject(typing.NamedTuple):
    xx: float
    xy: float
    xz: float
    xt: float
    yx: float
    yy: float
    yz: float
    yt: float
    zx: float
    zy: float
    zz: float
    zt: float
    tx: float
    ty: float
    tz: float
    tt: float

    @property
    def elements(self):
        return self

    def apply(self, v):
        x, y, z, t = vector.methods.Transform4D.apply(self, v)
        return SpatialVectorObject(
            AzimuthalObjectXY(x, y), LongitudinalObjectZ(z), TemporalObjectT(t)
        )


Transform4DObject.__bases__ = (TransformObject, vector.methods.Transform4D, tuple)


class LongitudinalBoostObject(typing.NamedTuple):
    gamma: float

    def apply(self, v):
        x, y, z, t = vector.methods.LongitudinalBoost.apply(self, v)
        return SpatialVectorObject(
            AzimuthalObjectXY(x, y), LongitudinalObjectZ(z), TemporalObjectT(t)
        )


LongitudinalBoostObject.__bases__ = (
    TransformObject,
    vector.methods.LongitudinalBoost,
    tuple,
)


class AxisAngleBoostObject(typing.NamedTuple):
    gamma: float

    def apply(self, v):
        x, y, z, t = vector.methods.AxisAngleBoost.apply(self, v)
        return SpatialVectorObject(
            AzimuthalObjectXY(x, y), LongitudinalObjectZ(z), TemporalObjectT(t)
        )


AxisAngleBoostObject.__bases__ = (
    TransformObject,
    vector.methods.AxisAngleBoost,
    tuple,
)
