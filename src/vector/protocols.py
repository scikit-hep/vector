# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing


class VectorProtocol:
    lib: typing.Any

    _wrap_result: typing.Any

    ProjectionClass2D: typing.Any
    ProjectionClass3D: typing.Any
    ProjectionClass4D: typing.Any
    GenericClass: typing.Any

    to_Vector2D: typing.Any
    to_Vector3D: typing.Any
    to_Vector4D: typing.Any

    to_xy: typing.Any
    to_rhophi: typing.Any
    to_xyz: typing.Any
    to_xytheta: typing.Any
    to_xyeta: typing.Any
    to_rhophiz: typing.Any
    to_rhophitheta: typing.Any
    to_rhophieta: typing.Any
    to_xyzt: typing.Any
    to_xyztau: typing.Any
    to_xythetat: typing.Any
    to_xythetatau: typing.Any
    to_xyetat: typing.Any
    to_xyetatau: typing.Any
    to_rhophizt: typing.Any
    to_rhophiztau: typing.Any
    to_rhophithetat: typing.Any
    to_rhophithetatau: typing.Any
    to_rhophietat: typing.Any
    to_rhophietatau: typing.Any

    unit: typing.Any
    dot: typing.Any
    add: typing.Any
    subtract: typing.Any
    scale: typing.Any
    equal: typing.Any
    not_equal: typing.Any
    isclose: typing.Any


class VectorProtocolPlanar(VectorProtocol):
    azimuthal: typing.Any

    x: typing.Any
    y: typing.Any
    rho: typing.Any
    rho2: typing.Any
    phi: typing.Any
    deltaphi: typing.Any
    rotateZ: typing.Any
    transform2D: typing.Any
    is_parallel: typing.Any
    is_antiparallel: typing.Any
    is_perpendicular: typing.Any


class VectorProtocolSpatial(VectorProtocolPlanar):
    longitudinal: typing.Any

    z: typing.Any
    theta: typing.Any
    eta: typing.Any
    costheta: typing.Any
    cottheta: typing.Any
    mag: typing.Any
    mag2: typing.Any
    cross: typing.Any
    deltaangle: typing.Any
    deltaeta: typing.Any
    deltaR: typing.Any
    deltaR2: typing.Any
    rotateX: typing.Any
    rotateY: typing.Any
    rotate_axis: typing.Any
    rotate_euler: typing.Any
    rotate_nautical: typing.Any
    rotate_quaternion: typing.Any
    transform3D: typing.Any
    is_parallel: typing.Any
    is_antiparallel: typing.Any
    is_perpendicular: typing.Any


class VectorProtocolLorentz(VectorProtocolSpatial):
    temporal: typing.Any

    t: typing.Any
    t2: typing.Any
    tau: typing.Any
    tau2: typing.Any
    beta: typing.Any
    gamma: typing.Any
    rapidity: typing.Any
    boost_p4: typing.Any
    boost_beta3: typing.Any
    boost: typing.Any
    boostX: typing.Any
    boostY: typing.Any
    boostZ: typing.Any
    transform4D: typing.Any
    to_beta3: typing.Any
    is_timelike: typing.Any
    is_spacelike: typing.Any
    is_lightlike: typing.Any


class MomentumProtocolPlanar(VectorProtocolPlanar):
    px: typing.Any
    py: typing.Any
    pt: typing.Any
    pt2: typing.Any


class MomentumProtocolSpatial(VectorProtocolSpatial, MomentumProtocolPlanar):
    pz: typing.Any
    pseudorapidity: typing.Any
    p: typing.Any
    p2: typing.Any


class MomentumProtocolLorentz(VectorProtocolLorentz, MomentumProtocolSpatial):
    E: typing.Any
    energy: typing.Any
    E2: typing.Any
    energy2: typing.Any
    M: typing.Any
    mass: typing.Any
    M2: typing.Any
    mass2: typing.Any
    Et: typing.Any
    transverse_energy: typing.Any
    Et2: typing.Any
    transverse_energy2: typing.Any
    Mt: typing.Any
    transverse_mass: typing.Any
    Mt2: typing.Any
    transverse_mass2: typing.Any
