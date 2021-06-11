# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

# type: ignore

"""
Implements VectorObjects in Numba. Every function should be made usable in Numba.
"""

import operator

import numba
import numpy

import vector
from vector._backends.numba_ import numba_modules
from vector._backends.object_ import (
    AzimuthalObject,
    AzimuthalObjectRhoPhi,
    AzimuthalObjectXY,
    LongitudinalObject,
    LongitudinalObjectEta,
    LongitudinalObjectTheta,
    LongitudinalObjectZ,
    MomentumObject2D,
    MomentumObject3D,
    MomentumObject4D,
    TemporalObject,
    TemporalObjectT,
    TemporalObjectTau,
    VectorObject2D,
    VectorObject3D,
    VectorObject4D,
    _coord_object_type,
)
from vector._methods import (
    AzimuthalRhoPhi,
    AzimuthalXY,
    LongitudinalEta,
    LongitudinalTheta,
    LongitudinalZ,
    Momentum,
    TemporalT,
    TemporalTau,
    _from_signature,
)


@numba.extending.overload(numpy.nan_to_num)  # FIXME: This needs to go into Numba!
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    if isinstance(x, numba.types.Array):

        def nan_to_num_impl(x, copy=True, nan=0.0, posinf=None, neginf=None):
            out = numpy.copy(x).reshape(-1) if copy else x.reshape(-1)
            for i in range(len(out)):
                if numpy.isnan(out[i]):
                    out[i] = nan
                if posinf is not None and numpy.isinf(out[i]) and out[i] > 0:
                    out[i] = posinf
                if neginf is not None and numpy.isinf(out[i]) and out[i] < 0:
                    out[i] = neginf
            return out.reshape(x.shape)

    else:

        def nan_to_num_impl(x, copy=True, nan=0.0, posinf=None, neginf=None):
            if numpy.isnan(x):
                return nan
            if posinf is not None and numpy.isinf(x) and x > 0:
                return posinf
            if neginf is not None and numpy.isinf(x) and x < 0:
                return neginf
            return x

    return nan_to_num_impl


@numba.extending.overload(numpy.isclose)  # FIXME: This needs to go into Numba!
def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    if isinstance(a, numba.types.Array) and isinstance(b, numba.types.Array):

        def isclose_impl(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
            a, b = numpy.broadcast_arrays(a, b)
            x = a.reshape(-1)
            y = b.astype(numpy.float64).reshape(-1)
            out = numpy.zeros(len(x), numpy.bool_)
            for i in range(len(out)):
                if numpy.isnan(x[i]) and numpy.isnan(y[i]):
                    out[i] = equal_nan
                elif numpy.isinf(x[i]) and numpy.isinf(y[i]):
                    out[i] = (x[i] > 0) == (y[i] > 0)
                else:
                    out[i] = abs(x[i] - y[i]) <= atol + rtol * abs(y[i])
            return out.reshape(a.shape)

    else:

        def isclose_impl(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
            x = a
            y = numpy.float64(b)
            if numpy.isnan(x) and numpy.isnan(y):
                return equal_nan
            elif numpy.isinf(x) and numpy.isinf(y):
                return (x > 0) == (y > 0)
            else:
                return abs(x - y) <= atol + rtol * abs(y)

    return isclose_impl


# Since CoordinateObjects are NamedTuples, we get their types wrapped for free.


def is_azimuthaltype(typ):
    return isinstance(
        typ, (numba.types.NamedTuple, numba.types.NamedUniTuple)
    ) and issubclass(typ.instance_class, AzimuthalObject)


def is_longitudinaltype(typ):
    return isinstance(
        typ, (numba.types.NamedTuple, numba.types.NamedUniTuple)
    ) and issubclass(typ.instance_class, LongitudinalObject)


def is_temporaltype(typ):
    return isinstance(
        typ, (numba.types.NamedTuple, numba.types.NamedUniTuple)
    ) and issubclass(typ.instance_class, TemporalObject)


def numba_aztype(typ):
    for t in typ.azimuthaltype.instance_class.__mro__:
        if t in (AzimuthalXY, AzimuthalRhoPhi):
            return t
    raise AssertionError


def numba_ltype(typ):
    for t in typ.longitudinaltype.instance_class.__mro__:
        if t in (LongitudinalZ, LongitudinalTheta, LongitudinalEta):
            return t
    raise AssertionError


def numba_ttype(typ):
    for t in typ.temporaltype.instance_class.__mro__:
        if t in (TemporalT, TemporalTau):
            return t
    raise AssertionError


# VectorObject2D (and momentum) ###############################################


class VectorObject2DType(numba.types.Type):
    instance_class = VectorObject2D

    def __init__(self, azimuthaltype):
        super().__init__(name=f"VectorObject2DType({azimuthaltype})")
        self.azimuthaltype = azimuthaltype


class MomentumObject2DType(VectorObject2DType):
    instance_class = MomentumObject2D

    def __init__(self, azimuthaltype):
        super().__init__(azimuthaltype)
        self.name = f"MomentumObject2DType({azimuthaltype})"


@numba.extending.register_model(VectorObject2DType)
@numba.extending.register_model(MomentumObject2DType)
class VectorObject2DModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("azimuthal", fe_type.azimuthaltype),
        ]
        super().__init__(dmm, fe_type, members)


@numba.extending.typeof_impl.register(VectorObject2D)
def VectorObject2D_typeof(val, c):
    if isinstance(val, MomentumObject2D):
        return MomentumObject2DType(numba.typeof(val.azimuthal))
    else:
        return VectorObject2DType(numba.typeof(val.azimuthal))


numba.extending.make_attribute_wrapper(VectorObject2DType, "azimuthal", "azimuthal")
numba.extending.make_attribute_wrapper(MomentumObject2DType, "azimuthal", "azimuthal")


@numba.extending.type_callable(VectorObject2D)
def VectorObject2D_constructor_typer(context):
    def typer(azimuthaltype):
        if is_azimuthaltype(azimuthaltype):
            return VectorObject2DType(azimuthaltype)
        else:
            raise numba.TypingError(
                "VectorObject2D constructor requires an AzimuthalObject as its argument"
            )

    return typer


@numba.extending.type_callable(MomentumObject2D)
def MomentumObject2D_constructor_typer(context):
    def typer(azimuthaltype):
        if is_azimuthaltype(azimuthaltype):
            return MomentumObject2DType(azimuthaltype)
        else:
            raise numba.TypingError(
                "MomentumObject2D constructor requires an AzimuthalObject as its argument"
            )

    return typer


@numba.extending.lower_builtin(VectorObject2D, numba.types.Type)
@numba.extending.lower_builtin(MomentumObject2D, numba.types.Type)
def VectorObject2D_constructor_impl(context, builder, sig, args):
    typ = sig.return_type
    proxyout = numba.core.cgutils.create_struct_proxy(typ)(context, builder)
    proxyout.azimuthal = args[0]
    return proxyout._getvalue()


@numba.extending.unbox(VectorObject2DType)
def VectorObject2D_unbox(typ, obj, c):
    azimuthal_obj = c.pyapi.object_getattr_string(obj, "azimuthal")
    proxyout = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    proxyout.azimuthal = c.pyapi.to_native_value(typ.azimuthaltype, azimuthal_obj).value
    c.pyapi.decref(azimuthal_obj)
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)


@numba.extending.box(VectorObject2DType)
def VectorObject2D_box(typ, val, c):
    if isinstance(typ, MomentumObject2DType):
        cls_obj = c.pyapi.unserialize(c.pyapi.serialize_object(MomentumObject2D))
    else:
        cls_obj = c.pyapi.unserialize(c.pyapi.serialize_object(VectorObject2D))
    proxyin = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    azimuthal_obj = c.pyapi.from_native_value(typ.azimuthaltype, proxyin.azimuthal)
    output_obj = c.pyapi.call_function_objargs(cls_obj, (azimuthal_obj,))
    c.pyapi.decref(cls_obj)
    c.pyapi.decref(azimuthal_obj)
    return output_obj


# VectorObject3D (and momentum) ###############################################


class VectorObject3DType(numba.types.Type):
    instance_class = VectorObject3D

    def __init__(self, azimuthaltype, longitudinaltype):
        super().__init__(
            name=f"VectorObject3DType({azimuthaltype}, {longitudinaltype})"
        )
        self.azimuthaltype = azimuthaltype
        self.longitudinaltype = longitudinaltype


class MomentumObject3DType(VectorObject3DType):
    instance_class = MomentumObject3D

    def __init__(self, azimuthaltype, longitudinaltype):
        super().__init__(azimuthaltype, longitudinaltype)
        self.name = f"MomentumObject3DType({azimuthaltype}, {longitudinaltype})"


@numba.extending.register_model(VectorObject3DType)
@numba.extending.register_model(MomentumObject3DType)
class VectorObject3DModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("azimuthal", fe_type.azimuthaltype),
            ("longitudinal", fe_type.longitudinaltype),
        ]
        super().__init__(dmm, fe_type, members)


@numba.extending.typeof_impl.register(VectorObject3D)
def VectorObject3D_typeof(val, c):
    if isinstance(val, MomentumObject3D):
        return MomentumObject3DType(
            numba.typeof(val.azimuthal), numba.typeof(val.longitudinal)
        )
    else:
        return VectorObject3DType(
            numba.typeof(val.azimuthal), numba.typeof(val.longitudinal)
        )


numba.extending.make_attribute_wrapper(VectorObject3DType, "azimuthal", "azimuthal")
numba.extending.make_attribute_wrapper(
    VectorObject3DType, "longitudinal", "longitudinal"
)
numba.extending.make_attribute_wrapper(MomentumObject3DType, "azimuthal", "azimuthal")
numba.extending.make_attribute_wrapper(
    MomentumObject3DType, "longitudinal", "longitudinal"
)


@numba.extending.type_callable(VectorObject3D)
def VectorObject3D_constructor_typer(context):
    def typer(azimuthaltype, longitudinaltype):
        if is_azimuthaltype(azimuthaltype) and is_longitudinaltype(longitudinaltype):
            return VectorObject3DType(azimuthaltype, longitudinaltype)
        else:
            raise numba.TypingError(
                "VectorObject3D constructor requires an AzimuthalObject and a "
                "LongitudinalObject as its arguments"
            )

    return typer


@numba.extending.type_callable(MomentumObject3D)
def MomentumObject3D_constructor_typer(context):
    def typer(azimuthaltype, longitudinaltype):
        if is_azimuthaltype(azimuthaltype) and is_longitudinaltype(longitudinaltype):
            return MomentumObject3DType(azimuthaltype, longitudinaltype)
        else:
            raise numba.TypingError(
                "MomentumObject3D constructor requires an AzimuthalObject and a "
                "LongitudinalObject as its arguments"
            )

    return typer


@numba.extending.lower_builtin(VectorObject3D, numba.types.Type, numba.types.Type)
@numba.extending.lower_builtin(MomentumObject3D, numba.types.Type, numba.types.Type)
def VectorObject3D_constructor_impl(context, builder, sig, args):
    typ = sig.return_type
    proxyout = numba.core.cgutils.create_struct_proxy(typ)(context, builder)
    proxyout.azimuthal = args[0]
    proxyout.longitudinal = args[1]
    return proxyout._getvalue()


@numba.extending.unbox(VectorObject3DType)
def VectorObject3D_unbox(typ, obj, c):
    azimuthal_obj = c.pyapi.object_getattr_string(obj, "azimuthal")
    longitudinal_obj = c.pyapi.object_getattr_string(obj, "longitudinal")
    proxyout = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    proxyout.azimuthal = c.pyapi.to_native_value(typ.azimuthaltype, azimuthal_obj).value
    proxyout.longitudinal = c.pyapi.to_native_value(
        typ.longitudinaltype, longitudinal_obj
    ).value
    c.pyapi.decref(azimuthal_obj)
    c.pyapi.decref(longitudinal_obj)
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)


@numba.extending.box(VectorObject3DType)
def VectorObject3D_box(typ, val, c):
    if isinstance(typ, MomentumObject3DType):
        cls_obj = c.pyapi.unserialize(c.pyapi.serialize_object(MomentumObject3D))
    else:
        cls_obj = c.pyapi.unserialize(c.pyapi.serialize_object(VectorObject3D))
    proxyin = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    azimuthal_obj = c.pyapi.from_native_value(typ.azimuthaltype, proxyin.azimuthal)
    longitudinal_obj = c.pyapi.from_native_value(
        typ.longitudinaltype, proxyin.longitudinal
    )
    output_obj = c.pyapi.call_function_objargs(
        cls_obj, (azimuthal_obj, longitudinal_obj)
    )
    c.pyapi.decref(cls_obj)
    c.pyapi.decref(azimuthal_obj)
    c.pyapi.decref(longitudinal_obj)
    return output_obj


# VectorObject4D (and momentum) ###############################################


class VectorObject4DType(numba.types.Type):
    instance_class = VectorObject4D

    def __init__(self, azimuthaltype, longitudinaltype, temporaltype):
        super().__init__(
            name=f"VectorObject4DType({azimuthaltype}, {longitudinaltype}, {temporaltype})"
        )
        self.azimuthaltype = azimuthaltype
        self.longitudinaltype = longitudinaltype
        self.temporaltype = temporaltype


class MomentumObject4DType(VectorObject4DType):
    instance_class = MomentumObject4D

    def __init__(self, azimuthaltype, longitudinaltype, temporaltype):
        super().__init__(azimuthaltype, longitudinaltype, temporaltype)
        self.name = (
            f"MomentumObject4DType({azimuthaltype}, {longitudinaltype}, {temporaltype})"
        )


@numba.extending.register_model(VectorObject4DType)
@numba.extending.register_model(MomentumObject4DType)
class VectorObject4DModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("azimuthal", fe_type.azimuthaltype),
            ("longitudinal", fe_type.longitudinaltype),
            ("temporal", fe_type.temporaltype),
        ]
        super().__init__(dmm, fe_type, members)


@numba.extending.typeof_impl.register(VectorObject4D)
def VectorObject4D_typeof(val, c):
    if isinstance(val, MomentumObject4D):
        return MomentumObject4DType(
            numba.typeof(val.azimuthal),
            numba.typeof(val.longitudinal),
            numba.typeof(val.temporal),
        )
    else:
        return VectorObject4DType(
            numba.typeof(val.azimuthal),
            numba.typeof(val.longitudinal),
            numba.typeof(val.temporal),
        )


numba.extending.make_attribute_wrapper(VectorObject4DType, "azimuthal", "azimuthal")
numba.extending.make_attribute_wrapper(
    VectorObject4DType, "longitudinal", "longitudinal"
)
numba.extending.make_attribute_wrapper(VectorObject4DType, "temporal", "temporal")
numba.extending.make_attribute_wrapper(MomentumObject4DType, "azimuthal", "azimuthal")
numba.extending.make_attribute_wrapper(
    MomentumObject4DType, "longitudinal", "longitudinal"
)
numba.extending.make_attribute_wrapper(MomentumObject4DType, "temporal", "temporal")


@numba.extending.type_callable(VectorObject4D)
def VectorObject4D_constructor_typer(context):
    def typer(azimuthaltype, longitudinaltype, temporaltype):
        if (
            is_azimuthaltype(azimuthaltype)
            and is_temporaltype(temporaltype)
            and is_temporaltype(temporaltype)
        ):
            return VectorObject4DType(azimuthaltype, longitudinaltype, temporaltype)
        else:
            raise numba.TypingError(
                "VectorObject4D constructor requires an AzimuthalObject and a "
                "LongitudinalObject and a TemporalObject as its arguments"
            )

    return typer


@numba.extending.type_callable(MomentumObject4D)
def MomentumObject4D_constructor_typer(context):
    def typer(azimuthaltype, longitudinaltype, temporaltype):
        if (
            is_azimuthaltype(azimuthaltype)
            and is_temporaltype(temporaltype)
            and is_temporaltype(temporaltype)
        ):
            return MomentumObject4DType(azimuthaltype, longitudinaltype, temporaltype)
        else:
            raise numba.TypingError(
                "MomentumObject4D constructor requires an AzimuthalObject and a "
                "LongitudinalObject and a TemporalObject as its arguments"
            )

    return typer


@numba.extending.lower_builtin(
    VectorObject4D, numba.types.Type, numba.types.Type, numba.types.Type
)
@numba.extending.lower_builtin(
    MomentumObject4D, numba.types.Type, numba.types.Type, numba.types.Type
)
def VectorObject4D_constructor_impl(context, builder, sig, args):
    typ = sig.return_type
    proxyout = numba.core.cgutils.create_struct_proxy(typ)(context, builder)
    proxyout.azimuthal = args[0]
    proxyout.longitudinal = args[1]
    proxyout.temporal = args[2]
    return proxyout._getvalue()


@numba.extending.unbox(VectorObject4DType)
def VectorObject4D_unbox(typ, obj, c):
    azimuthal_obj = c.pyapi.object_getattr_string(obj, "azimuthal")
    longitudinal_obj = c.pyapi.object_getattr_string(obj, "longitudinal")
    temporal_obj = c.pyapi.object_getattr_string(obj, "temporal")
    proxyout = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    proxyout.azimuthal = c.pyapi.to_native_value(typ.azimuthaltype, azimuthal_obj).value
    proxyout.longitudinal = c.pyapi.to_native_value(
        typ.longitudinaltype, longitudinal_obj
    ).value
    proxyout.temporal = c.pyapi.to_native_value(typ.temporaltype, temporal_obj).value
    c.pyapi.decref(azimuthal_obj)
    c.pyapi.decref(longitudinal_obj)
    c.pyapi.decref(temporal_obj)
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)


@numba.extending.box(VectorObject4DType)
def VectorObject4D_box(typ, val, c):
    if isinstance(typ, MomentumObject4DType):
        cls_obj = c.pyapi.unserialize(c.pyapi.serialize_object(MomentumObject4D))
    else:
        cls_obj = c.pyapi.unserialize(c.pyapi.serialize_object(VectorObject4D))
    proxyin = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    azimuthal_obj = c.pyapi.from_native_value(typ.azimuthaltype, proxyin.azimuthal)
    longitudinal_obj = c.pyapi.from_native_value(
        typ.longitudinaltype, proxyin.longitudinal
    )
    temporal_obj = c.pyapi.from_native_value(typ.temporaltype, proxyin.temporal)
    output_obj = c.pyapi.call_function_objargs(
        cls_obj, (azimuthal_obj, longitudinal_obj, temporal_obj)
    )
    c.pyapi.decref(cls_obj)
    c.pyapi.decref(azimuthal_obj)
    c.pyapi.decref(longitudinal_obj)
    c.pyapi.decref(temporal_obj)
    return output_obj


# vector.obj factory function #################################################


@numba.jit(nopython=True)
def vector_obj_Azimuthal_xy(x, px, y, py, rho, pt, phi):
    return AzimuthalObjectXY(x, y)


@numba.jit(nopython=True)
def vector_obj_Azimuthal_xpy(x, px, y, py, rho, pt, phi):
    return AzimuthalObjectXY(x, py)


@numba.jit(nopython=True)
def vector_obj_Azimuthal_pxy(x, px, y, py, rho, pt, phi):
    return AzimuthalObjectXY(px, y)


@numba.jit(nopython=True)
def vector_obj_Azimuthal_pxpy(x, px, y, py, rho, pt, phi):
    return AzimuthalObjectXY(px, py)


@numba.jit(nopython=True)
def vector_obj_Azimuthal_rhophi(x, px, y, py, rho, pt, phi):
    return AzimuthalObjectRhoPhi(rho, phi)


@numba.jit(nopython=True)
def vector_obj_Azimuthal_ptphi(x, px, y, py, rho, pt, phi):
    return AzimuthalObjectRhoPhi(pt, phi)


@numba.jit(nopython=True)
def vector_obj_Longitudinal_z(z, pz, theta, eta):
    return LongitudinalObjectZ(z)


@numba.jit(nopython=True)
def vector_obj_Longitudinal_pz(z, pz, theta, eta):
    return LongitudinalObjectZ(pz)


@numba.jit(nopython=True)
def vector_obj_Longitudinal_theta(z, pz, theta, eta):
    return LongitudinalObjectTheta(theta)


@numba.jit(nopython=True)
def vector_obj_Longitudinal_eta(z, pz, theta, eta):
    return LongitudinalObjectEta(eta)


@numba.jit(nopython=True)
def vector_obj_Temporal_t(t, E, e, energy, tau, M, m, mass):
    return TemporalObjectT(t)


@numba.jit(nopython=True)
def vector_obj_Temporal_E(t, E, e, energy, tau, M, m, mass):
    return TemporalObjectT(E)


@numba.jit(nopython=True)
def vector_obj_Temporal_e(t, E, e, energy, tau, M, m, mass):
    return TemporalObjectT(e)


@numba.jit(nopython=True)
def vector_obj_Temporal_energy(t, E, e, energy, tau, M, m, mass):
    return TemporalObjectT(energy)


@numba.jit(nopython=True)
def vector_obj_Temporal_tau(t, E, e, energy, tau, M, m, mass):
    return TemporalObjectTau(tau)


@numba.jit(nopython=True)
def vector_obj_Temporal_M(t, E, e, energy, tau, M, m, mass):
    return TemporalObjectTau(M)


@numba.jit(nopython=True)
def vector_obj_Temporal_m(t, E, e, energy, tau, M, m, mass):
    return TemporalObjectTau(m)


@numba.jit(nopython=True)
def vector_obj_Temporal_mass(t, E, e, energy, tau, M, m, mass):
    return TemporalObjectTau(mass)


@numba.extending.overload(vector.obj)
def vector_obj(
    unrecognized_argument=None,
    x=None,
    px=None,
    y=None,
    py=None,
    rho=None,
    pt=None,
    phi=None,
    z=None,
    pz=None,
    theta=None,
    eta=None,
    t=None,
    E=None,
    e=None,
    energy=None,
    tau=None,
    M=None,
    m=None,
    mass=None,
):
    if unrecognized_argument is not None:
        raise numba.TypingError(
            "only keyword arguments are allowed in vector.obj; no positional arguments"
        )

    has_x = x is not None
    has_px = px is not None
    has_y = y is not None
    has_py = py is not None
    has_rho = rho is not None
    has_pt = pt is not None
    has_phi = phi is not None
    has_z = z is not None
    has_pz = pz is not None
    has_theta = theta is not None
    has_eta = eta is not None
    has_t = t is not None
    has_E = E is not None
    has_e = e is not None
    has_energy = energy is not None
    has_tau = tau is not None
    has_M = M is not None
    has_m = m is not None
    has_mass = mass is not None

    is_momentum = False
    azimuthal = None
    longitudinal = None
    temporal = None

    if (has_x and has_y) and not (has_rho or has_pt or has_phi):
        if has_px or has_py:
            raise numba.TypingError(
                "duplicate coordinates (through momentum-aliases): x/px or y/py"
            )
        azimuthal = vector_obj_Azimuthal_xy
    elif (has_x and has_py) and not (has_rho or has_pt or has_phi):
        is_momentum = True
        if has_px or has_y:
            raise numba.TypingError(
                "duplicate coordinates (through momentum-aliases): x/px or y/py"
            )
        azimuthal = vector_obj_Azimuthal_xpy
    elif (has_px and has_y) and not (has_rho or has_pt or has_phi):
        is_momentum = True
        if has_x or has_py:
            raise numba.TypingError(
                "duplicate coordinates (through momentum-aliases): x/px or y/py"
            )
        azimuthal = vector_obj_Azimuthal_pxy
    elif (has_px and has_py) and not (has_rho or has_pt or has_phi):
        is_momentum = True
        if has_x or has_y:
            raise numba.TypingError(
                "duplicate coordinates (through momentum-aliases): x/px or y/py"
            )
        azimuthal = vector_obj_Azimuthal_pxpy
    elif (has_rho and has_phi) and not (has_x or has_px or has_y or has_py):
        if has_pt:
            raise numba.TypingError(
                "duplicate coordinates (through momentum-aliases): rho/pt"
            )
        azimuthal = vector_obj_Azimuthal_rhophi
    elif (has_pt and has_phi) and not (has_x or has_px or has_y or has_py):
        is_momentum = True
        if has_rho:
            raise numba.TypingError(
                "duplicate coordinates (through momentum-aliases): rho/pt"
            )
        azimuthal = vector_obj_Azimuthal_ptphi

    if has_z and not (has_theta or has_eta):
        if has_pz:
            raise numba.TypingError(
                "duplicate coordinates (through momentum-aliases): z/pz"
            )
        longitudinal = vector_obj_Longitudinal_z
    elif has_pz and not (has_theta or has_eta):
        is_momentum = True
        if has_z:
            raise numba.TypingError(
                "duplicate coordinates (through momentum-aliases): z/pz"
            )
        longitudinal = vector_obj_Longitudinal_pz
    elif has_theta and not (has_z or has_eta):
        longitudinal = vector_obj_Longitudinal_theta
    elif has_eta and not (has_z or has_theta):
        longitudinal = vector_obj_Longitudinal_eta

    if has_t and not (has_tau or has_M or has_m or has_mass):
        if has_E or has_e or has_energy:
            raise numba.TypingError(
                "duplicate coordinates (through momentum-aliases): t/E/e/energy"
            )
        temporal = vector_obj_Temporal_t
    elif has_E and not (has_tau or has_M or has_m or has_mass):
        is_momentum = True
        if has_t or has_e or has_energy:
            raise numba.TypingError(
                "duplicate coordinates (through momentum-aliases): t/E/e/energy"
            )
        temporal = vector_obj_Temporal_E
    elif has_e and not (has_tau or has_M or has_m or has_mass):
        is_momentum = True
        if has_t or has_E or has_energy:
            raise numba.TypingError(
                "duplicate coordinates (through momentum-aliases): t/E/e/energy"
            )
        temporal = vector_obj_Temporal_e
    elif has_energy and not (has_tau or has_M or has_m or has_mass):
        is_momentum = True
        if has_t or has_E or has_e:
            raise numba.TypingError(
                "duplicate coordinates (through momentum-aliases): t/E/e/energy"
            )
        temporal = vector_obj_Temporal_energy
    elif has_tau and not (has_t or has_E or has_e or has_energy):
        if has_M or has_m or has_mass:
            raise numba.TypingError(
                "duplicate coordinates (through momentum-aliases): tau/M/m/mass"
            )
        temporal = vector_obj_Temporal_tau
    elif has_M and not (has_t or has_E or has_e or has_energy):
        is_momentum = True
        if has_tau or has_m or has_mass:
            raise numba.TypingError(
                "duplicate coordinates (through momentum-aliases): tau/M/m/mass"
            )
        temporal = vector_obj_Temporal_M
    elif has_m and not (has_t or has_E or has_e or has_energy):
        is_momentum = True
        if has_tau or has_M or has_mass:
            raise numba.TypingError(
                "duplicate coordinates (through momentum-aliases): tau/M/m/mass"
            )
        temporal = vector_obj_Temporal_m
    elif has_mass and not (has_t or has_E or has_e or has_energy):
        is_momentum = True
        if has_tau or has_M or has_m:
            raise numba.TypingError(
                "duplicate coordinates (through momentum-aliases): tau/M/m/mass"
            )
        temporal = vector_obj_Temporal_mass

    if azimuthal is not None and longitudinal is not None and temporal is not None:
        if is_momentum:

            def vector_obj_impl(
                unrecognized_argument=None,
                x=None,
                px=None,
                y=None,
                py=None,
                rho=None,
                pt=None,
                phi=None,
                z=None,
                pz=None,
                theta=None,
                eta=None,
                t=None,
                E=None,
                e=None,
                energy=None,
                tau=None,
                M=None,
                m=None,
                mass=None,
            ):
                return MomentumObject4D(
                    azimuthal(x, px, y, py, rho, pt, phi),
                    longitudinal(z, pz, theta, eta),
                    temporal(t, E, e, energy, tau, M, m, mass),
                )

        else:

            def vector_obj_impl(
                unrecognized_argument=None,
                x=None,
                px=None,
                y=None,
                py=None,
                rho=None,
                pt=None,
                phi=None,
                z=None,
                pz=None,
                theta=None,
                eta=None,
                t=None,
                E=None,
                e=None,
                energy=None,
                tau=None,
                M=None,
                m=None,
                mass=None,
            ):
                return VectorObject4D(
                    azimuthal(x, px, y, py, rho, pt, phi),
                    longitudinal(z, pz, theta, eta),
                    temporal(t, E, e, energy, tau, M, m, mass),
                )

    elif azimuthal is not None and longitudinal is not None:
        if is_momentum:

            def vector_obj_impl(
                unrecognized_argument=None,
                x=None,
                px=None,
                y=None,
                py=None,
                rho=None,
                pt=None,
                phi=None,
                z=None,
                pz=None,
                theta=None,
                eta=None,
                t=None,
                E=None,
                e=None,
                energy=None,
                tau=None,
                M=None,
                m=None,
                mass=None,
            ):
                return MomentumObject3D(
                    azimuthal(x, px, y, py, rho, pt, phi),
                    longitudinal(z, pz, theta, eta),
                )

        else:

            def vector_obj_impl(
                unrecognized_argument=None,
                x=None,
                px=None,
                y=None,
                py=None,
                rho=None,
                pt=None,
                phi=None,
                z=None,
                pz=None,
                theta=None,
                eta=None,
                t=None,
                E=None,
                e=None,
                energy=None,
                tau=None,
                M=None,
                m=None,
                mass=None,
            ):
                return VectorObject3D(
                    azimuthal(x, px, y, py, rho, pt, phi),
                    longitudinal(z, pz, theta, eta),
                )

    elif azimuthal is not None:
        if is_momentum:

            def vector_obj_impl(
                unrecognized_argument=None,
                x=None,
                px=None,
                y=None,
                py=None,
                rho=None,
                pt=None,
                phi=None,
                z=None,
                pz=None,
                theta=None,
                eta=None,
                t=None,
                E=None,
                e=None,
                energy=None,
                tau=None,
                M=None,
                m=None,
                mass=None,
            ):
                return MomentumObject2D(azimuthal(x, px, y, py, rho, pt, phi))

        else:

            def vector_obj_impl(
                unrecognized_argument=None,
                x=None,
                px=None,
                y=None,
                py=None,
                rho=None,
                pt=None,
                phi=None,
                z=None,
                pz=None,
                theta=None,
                eta=None,
                t=None,
                E=None,
                e=None,
                energy=None,
                tau=None,
                M=None,
                m=None,
                mass=None,
            ):
                return VectorObject2D(azimuthal(x, px, y, py, rho, pt, phi))

    else:
        raise numba.TypingError(
            "unrecognized combination of coordinates, allowed combinations are:\n\n"
            "    (2D) x= y=\n"
            "    (2D) rho= phi=\n"
            "    (3D) x= y= z=\n"
            "    (3D) x= y= theta=\n"
            "    (3D) x= y= eta=\n"
            "    (3D) rho= phi= z=\n"
            "    (3D) rho= phi= theta=\n"
            "    (3D) rho= phi= eta=\n"
            "    (4D) x= y= z= t=\n"
            "    (4D) x= y= z= tau=\n"
            "    (4D) x= y= theta= t=\n"
            "    (4D) x= y= theta= tau=\n"
            "    (4D) x= y= eta= t=\n"
            "    (4D) x= y= eta= tau=\n"
            "    (4D) rho= phi= z= t=\n"
            "    (4D) rho= phi= z= tau=\n"
            "    (4D) rho= phi= theta= t=\n"
            "    (4D) rho= phi= theta= tau=\n"
            "    (4D) rho= phi= eta= t=\n"
            "    (4D) rho= phi= eta= tau="
        )

    return vector_obj_impl


# properties and methods ######################################################


@numba.extending.overload_method(VectorObject2DType, "to_Vector2D")
def VectorObject2D_to_Vector2D(v):
    def VectorObject2D_to_Vector2D_impl(v):
        return v

    return VectorObject2D_to_Vector2D_impl


@numba.extending.overload_method(VectorObject2DType, "to_Vector3D")
def VectorObject2D_to_Vector3D(v):
    if issubclass(v.instance_class, Momentum):

        def VectorObject2D_to_Vector3D_impl(v):
            return MomentumObject3D(v.azimuthal, LongitudinalObjectZ(0.0))

    else:

        def VectorObject2D_to_Vector3D_impl(v):
            return VectorObject3D(v.azimuthal, LongitudinalObjectZ(0.0))

    return VectorObject2D_to_Vector3D_impl


@numba.extending.overload_method(VectorObject2DType, "to_Vector4D")
def VectorObject2D_to_Vector4D(v):
    if issubclass(v.instance_class, Momentum):

        def VectorObject2D_to_Vector4D_impl(v):
            return MomentumObject4D(
                v.azimuthal, LongitudinalObjectZ(0.0), TemporalObjectT(0.0)
            )

    else:

        def VectorObject2D_to_Vector4D_impl(v):
            return VectorObject4D(
                v.azimuthal, LongitudinalObjectZ(0.0), TemporalObjectT(0.0)
            )

    return VectorObject2D_to_Vector4D_impl


@numba.extending.overload_method(VectorObject3DType, "to_Vector2D")
def VectorObject3D_to_Vector2D(v):
    if issubclass(v.instance_class, Momentum):

        def VectorObject3D_to_Vector2D_impl(v):
            return MomentumObject2D(v.azimuthal)

    else:

        def VectorObject3D_to_Vector2D_impl(v):
            return VectorObject2D(v.azimuthal)

    return VectorObject3D_to_Vector2D_impl


@numba.extending.overload_method(VectorObject3DType, "to_Vector3D")
def VectorObject3D_to_Vector3D(v):
    def VectorObject3D_to_Vector3D_impl(v):
        return v

    return VectorObject3D_to_Vector3D_impl


@numba.extending.overload_method(VectorObject3DType, "to_Vector4D")
def VectorObject3D_to_Vector4D(v):
    if issubclass(v.instance_class, Momentum):

        def VectorObject3D_to_Vector4D_impl(v):
            return MomentumObject4D(v.azimuthal, v.longitudinal, TemporalObjectT(0.0))

    else:

        def VectorObject3D_to_Vector4D_impl(v):
            return VectorObject4D(v.azimuthal, v.longitudinal, TemporalObjectT(0.0))

    return VectorObject3D_to_Vector4D_impl


@numba.extending.overload_method(VectorObject4DType, "to_Vector2D")
def VectorObject4D_to_Vector2D(v):
    if issubclass(v.instance_class, Momentum):

        def VectorObject4D_to_Vector2D_impl(v):
            return MomentumObject2D(v.azimuthal)

    else:

        def VectorObject4D_to_Vector2D_impl(v):
            return VectorObject2D(v.azimuthal)

    return VectorObject4D_to_Vector2D_impl


@numba.extending.overload_method(VectorObject4DType, "to_Vector3D")
def VectorObject4D_to_Vector3D(v):
    if issubclass(v.instance_class, Momentum):

        def VectorObject4D_to_Vector3D_impl(v):
            return MomentumObject3D(v.azimuthal, v.longitudinal)

    else:

        def VectorObject4D_to_Vector3D_impl(v):
            return VectorObject3D(v.azimuthal, v.longitudinal)

    return VectorObject4D_to_Vector3D_impl


@numba.extending.overload_method(VectorObject4DType, "to_Vector4D")
def VectorObject4D_to_Vector4D(v):
    def VectorObject4D_to_Vector4D_impl(v):
        return v

    return VectorObject4D_to_Vector4D_impl


@numba.jit(nopython=True)
def make_AzimuthalObjectXY(v):
    return AzimuthalObjectXY(v.x, v.y)


@numba.jit(nopython=True)
def make_AzimuthalObjectRhoPhi(v):
    return AzimuthalObjectRhoPhi(v.rho, v.phi)


@numba.jit(nopython=True)
def make_LongitudinalObjectZ(v):
    return LongitudinalObjectZ(v.z)


@numba.jit(nopython=True)
def make_LongitudinalObjectZ_zero(v):
    return LongitudinalObjectZ(0.0)


@numba.jit(nopython=True)
def make_LongitudinalObjectTheta(v):
    return LongitudinalObjectTheta(v.theta)


@numba.jit(nopython=True)
def make_LongitudinalObjectTheta_zero(v):
    return LongitudinalObjectTheta(0.0)


@numba.jit(nopython=True)
def make_LongitudinalObjectEta(v):
    return LongitudinalObjectEta(v.eta)


@numba.jit(nopython=True)
def make_LongitudinalObjectEta_zero(v):
    return LongitudinalObjectEta(0.0)


@numba.jit(nopython=True)
def make_TemporalObjectT(v):
    return TemporalObjectT(v.t)


@numba.jit(nopython=True)
def make_TemporalObjectT_zero(v):
    return TemporalObjectT(0.0)


@numba.jit(nopython=True)
def make_TemporalObjectTau(v):
    return TemporalObjectTau(v.tau)


@numba.jit(nopython=True)
def make_TemporalObjectTau_zero(v):
    return TemporalObjectTau(0.0)


make_coordobject = {
    (VectorObject2DType, AzimuthalObjectXY): make_AzimuthalObjectXY,
    (VectorObject3DType, AzimuthalObjectXY): make_AzimuthalObjectXY,
    (VectorObject4DType, AzimuthalObjectXY): make_AzimuthalObjectXY,
    (VectorObject2DType, AzimuthalObjectRhoPhi): make_AzimuthalObjectRhoPhi,
    (VectorObject3DType, AzimuthalObjectRhoPhi): make_AzimuthalObjectRhoPhi,
    (VectorObject4DType, AzimuthalObjectRhoPhi): make_AzimuthalObjectRhoPhi,
    (VectorObject2DType, LongitudinalObjectZ): make_LongitudinalObjectZ_zero,
    (VectorObject3DType, LongitudinalObjectZ): make_LongitudinalObjectZ,
    (VectorObject4DType, LongitudinalObjectZ): make_LongitudinalObjectZ,
    (VectorObject2DType, LongitudinalObjectTheta): make_LongitudinalObjectTheta_zero,
    (VectorObject3DType, LongitudinalObjectTheta): make_LongitudinalObjectTheta,
    (VectorObject4DType, LongitudinalObjectTheta): make_LongitudinalObjectTheta,
    (VectorObject2DType, LongitudinalObjectEta): make_LongitudinalObjectEta_zero,
    (VectorObject3DType, LongitudinalObjectEta): make_LongitudinalObjectEta,
    (VectorObject4DType, LongitudinalObjectEta): make_LongitudinalObjectEta,
    (VectorObject2DType, TemporalObjectT): make_TemporalObjectT_zero,
    (VectorObject3DType, TemporalObjectT): make_TemporalObjectT_zero,
    (VectorObject4DType, TemporalObjectT): make_TemporalObjectT,
    (VectorObject2DType, TemporalObjectTau): make_TemporalObjectTau_zero,
    (VectorObject3DType, TemporalObjectTau): make_TemporalObjectTau_zero,
    (VectorObject4DType, TemporalObjectTau): make_TemporalObjectTau,
}


def add_coordinate_change(vectortype, azcoordtype, lcoordtype, tcoordtype):
    methodname = "to_"
    if azcoordtype is AzimuthalObjectXY:
        methodname += "xy"
    elif azcoordtype is AzimuthalObjectRhoPhi:
        methodname += "rhophi"
    if lcoordtype is LongitudinalObjectZ:
        methodname += "z"
    if lcoordtype is LongitudinalObjectTheta:
        methodname += "theta"
    if lcoordtype is LongitudinalObjectEta:
        methodname += "eta"
    if tcoordtype is TemporalObjectT:
        methodname += "t"
    if tcoordtype is TemporalObjectTau:
        methodname += "tau"

    @numba.extending.overload_method(vectortype, methodname)
    def overloader(v):
        if lcoordtype is None and tcoordtype is None:
            azcoords = make_coordobject[vectortype, azcoordtype]

            if issubclass(v.instance_class, Momentum):

                def overloader_impl(v):
                    return MomentumObject2D(azcoords(v))

            else:

                def overloader_impl(v):
                    return VectorObject2D(azcoords(v))

        elif tcoordtype is None:
            azcoords = make_coordobject[vectortype, azcoordtype]
            lcoords = make_coordobject[vectortype, lcoordtype]

            if issubclass(v.instance_class, Momentum):

                def overloader_impl(v):
                    return MomentumObject3D(azcoords(v), lcoords(v))

            else:

                def overloader_impl(v):
                    return VectorObject3D(azcoords(v), lcoords(v))

        else:
            azcoords = make_coordobject[vectortype, azcoordtype]
            lcoords = make_coordobject[vectortype, lcoordtype]
            tcoords = make_coordobject[vectortype, tcoordtype]

            if issubclass(v.instance_class, Momentum):

                def overloader_impl(v):
                    return MomentumObject4D(azcoords(v), lcoords(v), tcoords(v))

            else:

                def overloader_impl(v):
                    return VectorObject4D(azcoords(v), lcoords(v), tcoords(v))

        return overloader_impl


for vectortype in (VectorObject2DType, VectorObject3DType, VectorObject4DType):
    for azcoordtype in (AzimuthalObjectXY, AzimuthalObjectRhoPhi):
        for lcoordtype in (
            None,
            LongitudinalObjectZ,
            LongitudinalObjectTheta,
            LongitudinalObjectEta,
        ):
            for tcoordtype in (None, TemporalObjectT, TemporalObjectTau):
                only_tcoordtype = lcoordtype is None and tcoordtype is not None
                if not only_tcoordtype:
                    add_coordinate_change(
                        vectortype, azcoordtype, lcoordtype, tcoordtype
                    )


@numba.jit(nopython=True)
def azimuthalxy_coord1(v):
    return v.azimuthal.x


@numba.jit(nopython=True)
def azimuthalxy_coord2(v):
    return v.azimuthal.y


@numba.jit(nopython=True)
def azimuthalrhophi_coord1(v):
    return v.azimuthal.rho


@numba.jit(nopython=True)
def azimuthalrhophi_coord2(v):
    return v.azimuthal.phi


@numba.jit(nopython=True)
def longitudinalz_coord1(v):
    return v.longitudinal.z


@numba.jit(nopython=True)
def longitudinaltheta_coord1(v):
    return v.longitudinal.theta


@numba.jit(nopython=True)
def longitudinaleta_coord1(v):
    return v.longitudinal.eta


@numba.jit(nopython=True)
def temporalt_coord1(v):
    return v.temporal.t


@numba.jit(nopython=True)
def temporaltau_coord1(v):
    return v.temporal.tau


getcoord1 = {
    AzimuthalXY: azimuthalxy_coord1,
    AzimuthalRhoPhi: azimuthalrhophi_coord1,
    LongitudinalZ: longitudinalz_coord1,
    LongitudinalTheta: longitudinaltheta_coord1,
    LongitudinalEta: longitudinaleta_coord1,
    TemporalT: temporalt_coord1,
    TemporalTau: temporaltau_coord1,
}

getcoord2 = {
    AzimuthalXY: azimuthalxy_coord2,
    AzimuthalRhoPhi: azimuthalrhophi_coord2,
}

planar_properties = ["x", "y", "rho", "rho2", "phi"]
spatial_properties = ["z", "theta", "eta", "costheta", "cottheta", "mag", "mag2"]
lorentz_properties = ["t", "t2", "tau", "tau2", "beta", "gamma", "rapidity"]
lorentz_momentum_properties = ["Et", "Et2", "Mt", "Mt2"]


def add_planar_property(vectortype, propertyname):
    @numba.extending.overload_attribute(vectortype, propertyname)
    def overloader(v):
        function, *returns = _from_signature(
            propertyname, numba_modules["planar"][propertyname], (numba_aztype(v),)
        )
        coord1 = getcoord1[numba_aztype(v)]
        coord2 = getcoord2[numba_aztype(v)]

        def overloader_impl(v):
            return function(numpy, coord1(v), coord2(v))

        return overloader_impl


def add_spatial_property(vectortype, propertyname):
    @numba.extending.overload_attribute(vectortype, propertyname)
    def overloader(v):
        function, *returns = _from_signature(
            propertyname,
            numba_modules["spatial"][propertyname],
            (numba_aztype(v), numba_ltype(v)),
        )
        coord1 = getcoord1[numba_aztype(v)]
        coord2 = getcoord2[numba_aztype(v)]
        coord3 = getcoord1[numba_ltype(v)]

        def overloader_impl(v):
            return function(numpy, coord1(v), coord2(v), coord3(v))

        return overloader_impl


def add_lorentz_property(vectortype, propertyname):
    @numba.extending.overload_attribute(vectortype, propertyname)
    def overloader(v):
        function, *returns = _from_signature(
            propertyname,
            numba_modules["lorentz"][propertyname],
            (numba_aztype(v), numba_ltype(v), numba_ttype(v)),
        )
        coord1 = getcoord1[numba_aztype(v)]
        coord2 = getcoord2[numba_aztype(v)]
        coord3 = getcoord1[numba_ltype(v)]
        coord4 = getcoord1[numba_ttype(v)]

        def overloader_impl(v):
            return function(numpy, coord1(v), coord2(v), coord3(v), coord4(v))

        return overloader_impl


for propertyname in planar_properties:
    for vectortype in (VectorObject2DType, VectorObject3DType, VectorObject4DType):
        add_planar_property(vectortype, propertyname)

for propertyname in spatial_properties:
    for vectortype in (VectorObject3DType, VectorObject4DType):
        add_spatial_property(vectortype, propertyname)

for propertyname in lorentz_properties:
    for vectortype in (VectorObject4DType,):
        add_lorentz_property(vectortype, propertyname)

for propertyname in lorentz_momentum_properties:
    for vectortype in (MomentumObject4DType,):
        add_lorentz_property(vectortype, propertyname)

planar_binary_methods = ["deltaphi"]
spatial_binary_methods = ["deltaangle", "deltaeta", "deltaR", "deltaR2"]
general_binary_methods = ["dot", "add", "subtract", "equal", "not_equal"]


def dimension_of(v):
    if isinstance(v, VectorObject2DType):
        return 2
    elif isinstance(v, VectorObject3DType):
        return 3
    elif isinstance(v, VectorObject4DType):
        return 4


def flavor_of(v1, v2):
    if issubclass(v1.instance_class, Momentum) and issubclass(
        v2.instance_class, Momentum
    ):
        return v1.instance_class
    else:
        return v1.instance_class.GenericClass


def add_binary_method(vectortype, gn, methodname):
    @numba.extending.overload_method(vectortype, methodname)
    def overloader(v1, v2):
        groupname = gn

        min_dimension = min(dimension_of(v1), dimension_of(v2))

        if (methodname == "equal" or methodname == "not_equal") and dimension_of(
            v1
        ) != dimension_of(v2):
            raise numba.TypingError(
                f"{type(v1).__name__} and {type(v2).__name__} do not have the same dimension"
            )

        if min_dimension == 2:
            if groupname is None:
                groupname = "planar"
            coord11 = getcoord1[numba_aztype(v1)]
            coord12 = getcoord2[numba_aztype(v1)]
            coord21 = getcoord1[numba_aztype(v2)]
            coord22 = getcoord2[numba_aztype(v2)]

        elif min_dimension == 3:
            if groupname is None:
                groupname = "spatial"
            coord11 = getcoord1[numba_aztype(v1)]
            coord12 = getcoord2[numba_aztype(v1)]
            coord13 = getcoord1[numba_ltype(v1)]
            coord21 = getcoord1[numba_aztype(v2)]
            coord22 = getcoord2[numba_aztype(v2)]
            coord23 = getcoord1[numba_ltype(v2)]

        elif min_dimension == 4:
            if groupname is None:
                groupname = "lorentz"
            coord11 = getcoord1[numba_aztype(v1)]
            coord12 = getcoord2[numba_aztype(v1)]
            coord13 = getcoord1[numba_ltype(v1)]
            coord14 = getcoord1[numba_ttype(v1)]
            coord21 = getcoord1[numba_aztype(v2)]
            coord22 = getcoord2[numba_aztype(v2)]
            coord23 = getcoord1[numba_ltype(v2)]
            coord24 = getcoord1[numba_ttype(v2)]

        if groupname == "planar":
            signature = (numba_aztype(v1), numba_aztype(v2))

        elif groupname == "spatial":
            signature = (
                numba_aztype(v1),
                numba_ltype(v1),
                numba_aztype(v2),
                numba_ltype(v2),
            )

        elif groupname == "lorentz":
            signature = (
                numba_aztype(v1),
                numba_ltype(v1),
                numba_ttype(v1),
                numba_aztype(v2),
                numba_ltype(v2),
                numba_ttype(v2),
            )

        function, *returns = _from_signature(
            groupname + "." + methodname,
            numba_modules[groupname][methodname],
            signature,
        )

        if returns == [bool] or returns == [float]:
            if groupname == "planar":

                def overloader_impl(v1, v2):
                    return function(
                        numpy, coord11(v1), coord12(v1), coord21(v2), coord22(v2)
                    )

            elif groupname == "spatial":

                def overloader_impl(v1, v2):
                    return function(
                        numpy,
                        coord11(v1),
                        coord12(v1),
                        coord13(v1),
                        coord21(v2),
                        coord22(v2),
                        coord23(v2),
                    )

            elif groupname == "lorentz":

                def overloader_impl(v1, v2):
                    return function(
                        numpy,
                        coord11(v1),
                        coord12(v1),
                        coord13(v1),
                        coord14(v1),
                        coord21(v2),
                        coord22(v2),
                        coord23(v2),
                        coord24(v2),
                    )

        else:
            if groupname == "planar":
                instance_class = flavor_of(v1, v2).ProjectionClass2D
                azcoords = _coord_object_type[returns[0]]

                def overloader_impl(v1, v2):
                    out1, out2 = function(
                        numpy, coord11(v1), coord12(v1), coord21(v2), coord22(v2)
                    )
                    return instance_class(azcoords(out1, out2))

            elif groupname == "spatial":
                instance_class = flavor_of(v1, v2).ProjectionClass3D
                azcoords = _coord_object_type[returns[0]]
                lcoords = _coord_object_type[returns[1]]

                def overloader_impl(v1, v2):
                    out1, out2, out3 = function(
                        numpy,
                        coord11(v1),
                        coord12(v1),
                        coord13(v1),
                        coord21(v2),
                        coord22(v2),
                        coord23(v2),
                    )
                    return instance_class(azcoords(out1, out2), lcoords(out3))

            elif groupname == "lorentz":
                instance_class = flavor_of(v1, v2).ProjectionClass4D
                azcoords = _coord_object_type[returns[0]]
                lcoords = _coord_object_type[returns[1]]
                tcoords = _coord_object_type[returns[2]]

                def overloader_impl(v1, v2):
                    out1, out2, out3, out4 = function(
                        numpy,
                        coord11(v1),
                        coord12(v1),
                        coord13(v1),
                        coord14(v1),
                        coord21(v2),
                        coord22(v2),
                        coord23(v2),
                        coord24(v2),
                    )
                    return instance_class(
                        azcoords(out1, out2), lcoords(out3), tcoords(out4)
                    )

        return overloader_impl


for methodname in planar_binary_methods:
    for vectortype in (VectorObject2DType, VectorObject3DType, VectorObject4DType):
        add_binary_method(vectortype, "planar", methodname)

for methodname in spatial_binary_methods:
    for vectortype in (VectorObject3DType, VectorObject4DType):
        add_binary_method(vectortype, "spatial", methodname)

for methodname in general_binary_methods:
    add_binary_method(VectorObject2DType, None, methodname)

for methodname in general_binary_methods:
    add_binary_method(VectorObject3DType, None, methodname)

for methodname in general_binary_methods:
    add_binary_method(VectorObject4DType, None, methodname)


tolerance_methods = ["is_parallel", "is_antiparallel", "is_perpendicular"]


def add_tolerance_method(vectortype, methodname):
    @numba.extending.overload_method(vectortype, methodname)
    def overloader(v1, v2, tolerance=1e-5):
        if isinstance(v1, VectorObject2DType) and not isinstance(
            v2, VectorObject2DType
        ):

            def overloader_impl(v1, v2, tolerance=1e-5):
                return v1.to_Vector3D().is_parallel(v2, tolerance=tolerance)

            return overloader_impl

        if not isinstance(v1, VectorObject2DType) and isinstance(
            v2, VectorObject2DType
        ):

            def overloader_impl(v1, v2, tolerance=1e-5):
                return v1.is_parallel(v2.to_Vector3D(), tolerance=tolerance)

            return overloader_impl

        if issubclass(vectortype, VectorObject2DType):
            groupname = "planar"
            signature = (numba_aztype(v1), numba_aztype(v2))
            coord11 = getcoord1[numba_aztype(v1)]
            coord12 = getcoord2[numba_aztype(v1)]
            coord21 = getcoord1[numba_aztype(v2)]
            coord22 = getcoord2[numba_aztype(v2)]

        elif issubclass(vectortype, (VectorObject3DType, VectorObject4DType)):
            groupname = "spatial"
            signature = (
                numba_aztype(v1),
                numba_ltype(v1),
                numba_aztype(v2),
                numba_ltype(v2),
            )
            coord11 = getcoord1[numba_aztype(v1)]
            coord12 = getcoord2[numba_aztype(v1)]
            coord13 = getcoord1[numba_ltype(v1)]
            coord21 = getcoord1[numba_aztype(v2)]
            coord22 = getcoord2[numba_aztype(v2)]
            coord23 = getcoord1[numba_ltype(v2)]

        function, *returns = _from_signature(
            groupname + "." + methodname,
            numba_modules[groupname][methodname],
            signature,
        )

        if issubclass(vectortype, VectorObject2DType):

            def overloader_impl(v1, v2, tolerance=1e-5):
                return function(
                    numpy,
                    tolerance,
                    coord11(v1),
                    coord12(v1),
                    coord21(v2),
                    coord22(v2),
                )

        elif issubclass(vectortype, (VectorObject3DType, VectorObject4DType)):

            def overloader_impl(v1, v2, tolerance=1e-5):
                return function(
                    numpy,
                    tolerance,
                    coord11(v1),
                    coord12(v1),
                    coord13(v1),
                    coord21(v2),
                    coord22(v2),
                    coord23(v2),
                )

        return overloader_impl


for methodname in tolerance_methods:
    add_tolerance_method(VectorObject2DType, methodname)

for methodname in tolerance_methods:
    add_tolerance_method(VectorObject3DType, methodname)

for methodname in tolerance_methods:
    add_tolerance_method(VectorObject4DType, methodname)


def add_isclose_method(vectortype):
    @numba.extending.overload_method(vectortype, "isclose")
    def overloader(v1, v2, rtol=1e-05, atol=1e-08, equal_nan=False):
        if isinstance(v1, VectorObject2DType) and isinstance(v2, VectorObject2DType):
            groupname = "planar"
            signature = (numba_aztype(v1), numba_aztype(v2))
            coord11 = getcoord1[numba_aztype(v1)]
            coord12 = getcoord2[numba_aztype(v1)]
            coord21 = getcoord1[numba_aztype(v2)]
            coord22 = getcoord2[numba_aztype(v2)]

        elif isinstance(v1, VectorObject3DType) and isinstance(v2, VectorObject3DType):
            groupname = "spatial"
            signature = (
                numba_aztype(v1),
                numba_ltype(v1),
                numba_aztype(v2),
                numba_ltype(v2),
            )
            coord11 = getcoord1[numba_aztype(v1)]
            coord12 = getcoord2[numba_aztype(v1)]
            coord13 = getcoord1[numba_ltype(v1)]
            coord21 = getcoord1[numba_aztype(v2)]
            coord22 = getcoord2[numba_aztype(v2)]
            coord23 = getcoord1[numba_ltype(v2)]

        elif isinstance(v1, VectorObject4DType) and isinstance(v2, VectorObject4DType):
            groupname = "lorentz"
            signature = (
                numba_aztype(v1),
                numba_ltype(v1),
                numba_ttype(v1),
                numba_aztype(v2),
                numba_ltype(v2),
                numba_ttype(v2),
            )
            coord11 = getcoord1[numba_aztype(v1)]
            coord12 = getcoord2[numba_aztype(v1)]
            coord13 = getcoord1[numba_ltype(v1)]
            coord14 = getcoord1[numba_ttype(v1)]
            coord21 = getcoord1[numba_aztype(v2)]
            coord22 = getcoord2[numba_aztype(v2)]
            coord23 = getcoord1[numba_ltype(v2)]
            coord24 = getcoord1[numba_ttype(v2)]

        else:
            raise numba.TypingError(
                f"{type(v1).__name__} and {type(v2).__name__} do not have the same dimension"
            )

        function, *returns = _from_signature(
            groupname + ".isclose",
            numba_modules[groupname]["isclose"],
            signature,
        )

        if isinstance(v1, VectorObject2DType) and isinstance(v2, VectorObject2DType):

            def overloader_impl(v1, v2, rtol=1e-05, atol=1e-08, equal_nan=False):
                return function(
                    numpy,
                    rtol,
                    atol,
                    equal_nan,
                    coord11(v1),
                    coord12(v1),
                    coord21(v2),
                    coord22(v2),
                )

        elif isinstance(v1, VectorObject3DType) and isinstance(v2, VectorObject3DType):

            def overloader_impl(v1, v2, rtol=1e-05, atol=1e-08, equal_nan=False):
                return function(
                    numpy,
                    rtol,
                    atol,
                    equal_nan,
                    coord11(v1),
                    coord12(v1),
                    coord13(v1),
                    coord21(v2),
                    coord22(v2),
                    coord23(v2),
                )

        elif isinstance(v1, VectorObject4DType) and isinstance(v2, VectorObject4DType):

            def overloader_impl(v1, v2, rtol=1e-05, atol=1e-08, equal_nan=False):
                return function(
                    numpy,
                    rtol,
                    atol,
                    equal_nan,
                    coord11(v1),
                    coord12(v1),
                    coord13(v1),
                    coord14(v1),
                    coord21(v2),
                    coord22(v2),
                    coord23(v2),
                    coord24(v2),
                )

        return overloader_impl


add_isclose_method(VectorObject2DType)
add_isclose_method(VectorObject3DType)
add_isclose_method(VectorObject4DType)


# the rest are special in one way or another ##################################


def add_rotateZ(vectortype):
    @numba.extending.overload_method(vectortype, "rotateZ")
    def overloader(v, angle):
        if isinstance(angle, (numba.types.Integer, numba.types.Float)):
            function, *returns = _from_signature(
                "", numba_modules["planar"]["rotateZ"], (numba_aztype(v),)
            )

            instance_class = v.instance_class
            coord1 = getcoord1[numba_aztype(v)]
            coord2 = getcoord2[numba_aztype(v)]
            azcoords = _coord_object_type[returns[0]]

            if issubclass(vectortype, VectorObject2DType):

                def overloader_impl(v, angle):
                    out1, out2 = function(numpy, angle, coord1(v), coord2(v))
                    return instance_class(azcoords(out1, out2))

            elif issubclass(vectortype, VectorObject3DType):

                def overloader_impl(v, angle):
                    out1, out2 = function(numpy, angle, coord1(v), coord2(v))
                    return instance_class(azcoords(out1, out2), v.longitudinal)

            elif issubclass(vectortype, VectorObject4DType):

                def overloader_impl(v, angle):
                    out1, out2 = function(numpy, angle, coord1(v), coord2(v))
                    return instance_class(
                        azcoords(out1, out2), v.longitudinal, v.temporal
                    )

            return overloader_impl

        else:
            raise numba.TypingError(
                "'angle' must be an integer or a floating-point number"
            )


for vectortype in (VectorObject2DType, VectorObject3DType, VectorObject4DType):
    add_rotateZ(vectortype)


def add_transform2D(vectortype):
    @numba.extending.overload_method(vectortype, "transform2D")
    def overloader(v, obj):
        function, *returns = _from_signature(
            "", numba_modules["planar"]["transform2D"], (numba_aztype(v),)
        )

        instance_class = v.instance_class
        coord1 = getcoord1[numba_aztype(v)]
        coord2 = getcoord2[numba_aztype(v)]
        azcoords = _coord_object_type[returns[0]]

        if issubclass(vectortype, VectorObject2DType):

            def overloader_impl(v, obj):
                out1, out2 = function(
                    numpy,
                    obj["xx"],
                    obj["xy"],
                    obj["yx"],
                    obj["yy"],
                    coord1(v),
                    coord2(v),
                )
                return instance_class(azcoords(out1, out2))

        elif issubclass(vectortype, VectorObject3DType):

            def overloader_impl(v, obj):
                out1, out2 = function(
                    numpy,
                    obj["xx"],
                    obj["xy"],
                    obj["yx"],
                    obj["yy"],
                    coord1(v),
                    coord2(v),
                )
                return instance_class(azcoords(out1, out2), v.longitudinal)

        elif issubclass(vectortype, VectorObject4DType):

            def overloader_impl(v, obj):
                out1, out2 = function(
                    numpy,
                    obj["xx"],
                    obj["xy"],
                    obj["yx"],
                    obj["yy"],
                    coord1(v),
                    coord2(v),
                )
                return instance_class(azcoords(out1, out2), v.longitudinal, v.temporal)

        return overloader_impl


for vectortype in (VectorObject2DType, VectorObject3DType, VectorObject4DType):
    add_transform2D(vectortype)


@numba.extending.overload_method(VectorObject2DType, "unit")
def VectorObject2DType_unit(v):
    function, *returns = _from_signature(
        "", numba_modules["planar"]["unit"], (numba_aztype(v),)
    )

    instance_class = v.instance_class
    coord1 = getcoord1[numba_aztype(v)]
    coord2 = getcoord2[numba_aztype(v)]
    azcoords = _coord_object_type[returns[0]]

    def VectorObject2DType_unit_impl(v):
        out1, out2 = function(numpy, coord1(v), coord2(v))
        return instance_class(azcoords(out1, out2))

    return VectorObject2DType_unit_impl


@numba.extending.overload_method(VectorObject3DType, "unit")
def VectorObject3DType_unit(v):
    function, *returns = _from_signature(
        "", numba_modules["spatial"]["unit"], (numba_aztype(v), numba_ltype(v))
    )

    instance_class = v.instance_class
    coord1 = getcoord1[numba_aztype(v)]
    coord2 = getcoord2[numba_aztype(v)]
    coord3 = getcoord1[numba_ltype(v)]
    azcoords = _coord_object_type[returns[0]]
    lcoords = _coord_object_type[returns[1]]

    def VectorObject3DType_unit_impl(v):
        out1, out2, out3 = function(numpy, coord1(v), coord2(v), coord3(v))
        return instance_class(azcoords(out1, out2), lcoords(out3))

    return VectorObject3DType_unit_impl


@numba.extending.overload_method(VectorObject4DType, "unit")
def VectorObject4DType_unit(v):
    function, *returns = _from_signature(
        "",
        numba_modules["lorentz"]["unit"],
        (numba_aztype(v), numba_ltype(v), numba_ttype(v)),
    )

    instance_class = v.instance_class
    coord1 = getcoord1[numba_aztype(v)]
    coord2 = getcoord2[numba_aztype(v)]
    coord3 = getcoord1[numba_ltype(v)]
    coord4 = getcoord1[numba_ttype(v)]
    azcoords = _coord_object_type[returns[0]]
    lcoords = _coord_object_type[returns[1]]
    tcoords = _coord_object_type[returns[2]]

    def VectorObject4DType_unit_impl(v):
        out1, out2, out3, out4 = function(
            numpy, coord1(v), coord2(v), coord3(v), coord4(v)
        )
        return instance_class(azcoords(out1, out2), lcoords(out3), tcoords(out4))

    return VectorObject4DType_unit_impl


@numba.extending.overload_method(VectorObject2DType, "scale")
def VectorObject2DType_scale(v, factor):
    if isinstance(factor, (numba.types.Integer, numba.types.Float)):
        function, *returns = _from_signature(
            "", numba_modules["planar"]["scale"], (numba_aztype(v),)
        )

        instance_class = v.instance_class
        coord1 = getcoord1[numba_aztype(v)]
        coord2 = getcoord2[numba_aztype(v)]
        azcoords = _coord_object_type[returns[0]]

        def VectorObject2DType_scale_impl(v, factor):
            out1, out2 = function(numpy, factor, coord1(v), coord2(v))
            return instance_class(azcoords(out1, out2))

        return VectorObject2DType_scale_impl

    else:
        raise numba.TypingError(
            "'factor' must be an integer or a floating-point number"
        )


@numba.extending.overload_method(VectorObject3DType, "scale")
def VectorObject3DType_scale(v, factor):
    if isinstance(factor, (numba.types.Integer, numba.types.Float)):
        function, *returns = _from_signature(
            "", numba_modules["spatial"]["scale"], (numba_aztype(v), numba_ltype(v))
        )

        instance_class = v.instance_class
        coord1 = getcoord1[numba_aztype(v)]
        coord2 = getcoord2[numba_aztype(v)]
        coord3 = getcoord1[numba_ltype(v)]
        azcoords = _coord_object_type[returns[0]]
        lcoords = _coord_object_type[returns[1]]

        def VectorObject3DType_scale_impl(v, factor):
            out1, out2, out3 = function(numpy, factor, coord1(v), coord2(v), coord3(v))
            return instance_class(azcoords(out1, out2), lcoords(out3))

        return VectorObject3DType_scale_impl

    else:
        raise numba.TypingError(
            "'factor' must be an integer or a floating-point number"
        )


@numba.extending.overload_method(VectorObject4DType, "scale")
def VectorObject4DType_scale(v, factor):
    if isinstance(factor, (numba.types.Integer, numba.types.Float)):
        function, *returns = _from_signature(
            "",
            numba_modules["lorentz"]["scale"],
            (numba_aztype(v), numba_ltype(v), numba_ttype(v)),
        )

        instance_class = v.instance_class
        coord1 = getcoord1[numba_aztype(v)]
        coord2 = getcoord2[numba_aztype(v)]
        coord3 = getcoord1[numba_ltype(v)]
        coord4 = getcoord1[numba_ttype(v)]
        azcoords = _coord_object_type[returns[0]]
        lcoords = _coord_object_type[returns[1]]
        tcoords = _coord_object_type[returns[2]]

        def VectorObject4DType_scale_impl(v, factor):
            out1, out2, out3, out4 = function(
                numpy, factor, coord1(v), coord2(v), coord3(v), coord4(v)
            )
            return instance_class(azcoords(out1, out2), lcoords(out3), tcoords(out4))

        return VectorObject4DType_scale_impl

    else:
        raise numba.TypingError(
            "'factor' must be an integer or a floating-point number"
        )


@numba.extending.overload_method(VectorObject3DType, "cross")
@numba.extending.overload_method(VectorObject4DType, "cross")
def VectorObject34DType_cross(v1, v2):
    if isinstance(v1, VectorObject3DType) and isinstance(v2, VectorObject3DType):
        function, *returns = _from_signature(
            "",
            numba_modules["spatial"]["cross"],
            (numba_aztype(v1), numba_ltype(v1), numba_aztype(v2), numba_ltype(v2)),
        )

        instance_class = flavor_of(v1, v2).ProjectionClass3D
        coord11 = getcoord1[numba_aztype(v1)]
        coord12 = getcoord2[numba_aztype(v1)]
        coord13 = getcoord1[numba_ltype(v1)]
        coord21 = getcoord1[numba_aztype(v2)]
        coord22 = getcoord2[numba_aztype(v2)]
        coord23 = getcoord1[numba_ltype(v2)]
        azcoords = _coord_object_type[returns[0]]
        lcoords = _coord_object_type[returns[1]]

        def VectorObject34DType_cross_impl(v1, v2):
            out1, out2, out3 = function(
                numpy,
                coord11(v1),
                coord12(v1),
                coord13(v1),
                coord21(v2),
                coord22(v2),
                coord23(v2),
            )
            return instance_class(azcoords(out1, out2), lcoords(out3))

    elif isinstance(v1, VectorObject3DType) and isinstance(v2, VectorObject4DType):

        def VectorObject34DType_cross_impl(v1, v2):
            return v1.cross(v2.to_Vector3D())

    elif isinstance(v1, VectorObject4DType) and isinstance(v2, VectorObject3DType):

        def VectorObject34DType_cross_impl(v1, v2):
            return v1.to_Vector3D().cross(v2)

    elif isinstance(v1, VectorObject4DType) and isinstance(v2, VectorObject4DType):

        def VectorObject34DType_cross_impl(v1, v2):
            return v1.to_Vector3D().cross(v2.to_Vector3D())

    else:
        raise numba.TypingError(
            "both vectors in 'cross' must be 3D or 4D (result is 3D)"
        )

    return VectorObject34DType_cross_impl


@numba.extending.overload_method(VectorObject3DType, "rotateX")
@numba.extending.overload_method(VectorObject4DType, "rotateX")
def VectorObject34DType_rotateX(v, angle):
    if isinstance(angle, (numba.types.Integer, numba.types.Float)):
        function, *returns = _from_signature(
            "", numba_modules["spatial"]["rotateX"], (numba_aztype(v), numba_ltype(v))
        )

        instance_class = v.instance_class
        coord1 = getcoord1[numba_aztype(v)]
        coord2 = getcoord2[numba_aztype(v)]
        coord3 = getcoord1[numba_ltype(v)]
        azcoords = _coord_object_type[returns[0]]
        lcoords = _coord_object_type[returns[1]]

        if isinstance(v, VectorObject3DType):

            def VectorObject34DType_rotateX_impl(v, angle):
                out1, out2, out3 = function(
                    numpy, angle, coord1(v), coord2(v), coord3(v)
                )
                return instance_class(azcoords(out1, out2), lcoords(out3))

        else:

            def VectorObject34DType_rotateX_impl(v, angle):
                out1, out2, out3 = function(
                    numpy, angle, coord1(v), coord2(v), coord3(v)
                )
                return instance_class(azcoords(out1, out2), lcoords(out3), v.temporal)

        return VectorObject34DType_rotateX_impl

    else:
        raise numba.TypingError("'angle' must be an integer or a floating-point number")


@numba.extending.overload_method(VectorObject3DType, "rotateY")
@numba.extending.overload_method(VectorObject4DType, "rotateY")
def VectorObject34DType_rotateY(v, angle):
    if isinstance(angle, (numba.types.Integer, numba.types.Float)):
        function, *returns = _from_signature(
            "", numba_modules["spatial"]["rotateY"], (numba_aztype(v), numba_ltype(v))
        )

        instance_class = v.instance_class
        coord1 = getcoord1[numba_aztype(v)]
        coord2 = getcoord2[numba_aztype(v)]
        coord3 = getcoord1[numba_ltype(v)]
        azcoords = _coord_object_type[returns[0]]
        lcoords = _coord_object_type[returns[1]]

        if isinstance(v, VectorObject3DType):

            def VectorObject34DType_rotateY_impl(v, angle):
                out1, out2, out3 = function(
                    numpy, angle, coord1(v), coord2(v), coord3(v)
                )
                return instance_class(azcoords(out1, out2), lcoords(out3))

        else:

            def VectorObject34DType_rotateY_impl(v, angle):
                out1, out2, out3 = function(
                    numpy, angle, coord1(v), coord2(v), coord3(v)
                )
                return instance_class(azcoords(out1, out2), lcoords(out3), v.temporal)

        return VectorObject34DType_rotateY_impl

    else:
        raise numba.TypingError("'angle' must be an integer or a floating-point number")


@numba.extending.overload_method(VectorObject3DType, "rotate_axis")
@numba.extending.overload_method(VectorObject4DType, "rotate_axis")
def VectorObject34DType_rotate_axis(v, axis, angle):
    if isinstance(angle, (numba.types.Integer, numba.types.Float)):
        function, *returns = _from_signature(
            "",
            numba_modules["spatial"]["rotate_axis"],
            (numba_aztype(axis), numba_ltype(axis), numba_aztype(v), numba_ltype(v)),
        )

        instance_class = v.instance_class
        coord11 = getcoord1[numba_aztype(axis)]
        coord12 = getcoord2[numba_aztype(axis)]
        coord13 = getcoord1[numba_ltype(axis)]
        coord21 = getcoord1[numba_aztype(v)]
        coord22 = getcoord2[numba_aztype(v)]
        coord23 = getcoord1[numba_ltype(v)]
        azcoords = _coord_object_type[returns[0]]
        lcoords = _coord_object_type[returns[1]]

        if isinstance(v, VectorObject3DType):

            def VectorObject34DType_rotate_axis(v, axis, angle):
                out1, out2, out3 = function(
                    numpy,
                    angle,
                    coord11(axis),
                    coord12(axis),
                    coord13(axis),
                    coord21(v),
                    coord22(v),
                    coord23(v),
                )
                return instance_class(azcoords(out1, out2), lcoords(out3))

        else:

            def VectorObject34DType_rotate_axis(v, axis, angle):
                out1, out2, out3 = function(
                    numpy,
                    angle,
                    coord11(axis),
                    coord12(axis),
                    coord13(axis),
                    coord21(v),
                    coord22(v),
                    coord23(v),
                )
                return instance_class(azcoords(out1, out2), lcoords(out3), v.temporal)

        return VectorObject34DType_rotate_axis

    else:
        raise numba.TypingError("'angle' must be an integer or a floating-point number")


@numba.extending.overload_method(VectorObject3DType, "rotate_euler")
@numba.extending.overload_method(VectorObject4DType, "rotate_euler")
def VectorObject34DType_rotate_euler(v, phi, theta, psi, order="zxz"):
    if (
        isinstance(phi, (numba.types.Integer, numba.types.Float))
        and isinstance(theta, (numba.types.Integer, numba.types.Float))
        and isinstance(psi, (numba.types.Integer, numba.types.Float))
    ):
        if isinstance(order, str):
            pass
        elif isinstance(order, numba.types.StringLiteral):
            order = order.literal_value
        elif isinstance(order, (numba.types.UnicodeType, numba.types.Bytes)):
            raise numba.TypingError(
                "'order' argument for 'rotate_euler' must be a compile-time string"
            )

        function, *returns = _from_signature(
            "",
            numba_modules["spatial"]["rotate_euler"],
            (numba_aztype(v), numba_ltype(v), order),
        )

        instance_class = v.instance_class
        coord1 = getcoord1[numba_aztype(v)]
        coord2 = getcoord2[numba_aztype(v)]
        coord3 = getcoord1[numba_ltype(v)]
        azcoords = _coord_object_type[returns[0]]
        lcoords = _coord_object_type[returns[1]]

        if isinstance(v, VectorObject3DType):

            def VectorObject34DType_rotate_axis_impl(v, phi, theta, psi, order="zxz"):
                out1, out2, out3 = function(
                    numpy, phi, theta, psi, coord1(v), coord2(v), coord3(v)
                )
                return instance_class(azcoords(out1, out2), lcoords(out3))

        else:

            def VectorObject34DType_rotate_axis_impl(v, phi, theta, psi, order="zxz"):
                out1, out2, out3 = function(
                    numpy, phi, theta, psi, coord1(v), coord2(v), coord3(v)
                )
                return instance_class(azcoords(out1, out2), lcoords(out3), v.temporal)

        return VectorObject34DType_rotate_axis_impl

    else:
        raise numba.TypingError(
            "'phi', 'theta', and 'psi' must be integers or floating-point numbers"
        )


@numba.extending.overload_method(VectorObject3DType, "rotate_nautical")
@numba.extending.overload_method(VectorObject4DType, "rotate_nautical")
def VectorObject34DType_rotate_nautical(v, yaw, pitch, roll):
    if (
        isinstance(yaw, (numba.types.Integer, numba.types.Float))
        and isinstance(pitch, (numba.types.Integer, numba.types.Float))
        and isinstance(roll, (numba.types.Integer, numba.types.Float))
    ):
        function, *returns = _from_signature(
            "",
            numba_modules["spatial"]["rotate_euler"],
            (numba_aztype(v), numba_ltype(v), "zyx"),
        )

        instance_class = v.instance_class
        coord1 = getcoord1[numba_aztype(v)]
        coord2 = getcoord2[numba_aztype(v)]
        coord3 = getcoord1[numba_ltype(v)]
        azcoords = _coord_object_type[returns[0]]
        lcoords = _coord_object_type[returns[1]]

        if isinstance(v, VectorObject3DType):

            def VectorObject34DType_rotate_nautical_impl(v, yaw, pitch, roll):
                out1, out2, out3 = function(
                    numpy, roll, pitch, yaw, coord1(v), coord2(v), coord3(v)
                )
                return instance_class(azcoords(out1, out2), lcoords(out3))

        else:

            def VectorObject34DType_rotate_nautical_impl(v, yaw, pitch, roll):
                out1, out2, out3 = function(
                    numpy, roll, pitch, yaw, coord1(v), coord2(v), coord3(v)
                )
                return instance_class(azcoords(out1, out2), lcoords(out3), v.temporal)

        return VectorObject34DType_rotate_nautical_impl

    else:
        raise numba.TypingError(
            "'yaw', 'pitch', and 'roll' must be integers or floating-point numbers"
        )


@numba.extending.overload_method(VectorObject3DType, "rotate_quaternion")
@numba.extending.overload_method(VectorObject4DType, "rotate_quaternion")
def VectorObject34DType_rotate_quaternion(v, u, i, j, k):
    if (
        isinstance(u, (numba.types.Integer, numba.types.Float))
        and isinstance(i, (numba.types.Integer, numba.types.Float))
        and isinstance(j, (numba.types.Integer, numba.types.Float))
        and isinstance(k, (numba.types.Integer, numba.types.Float))
    ):
        function, *returns = _from_signature(
            "",
            numba_modules["spatial"]["rotate_quaternion"],
            (numba_aztype(v), numba_ltype(v)),
        )

        instance_class = v.instance_class
        coord1 = getcoord1[numba_aztype(v)]
        coord2 = getcoord2[numba_aztype(v)]
        coord3 = getcoord1[numba_ltype(v)]
        azcoords = _coord_object_type[returns[0]]
        lcoords = _coord_object_type[returns[1]]

        if isinstance(v, VectorObject3DType):

            def VectorObject34DType_rotate_quaternion_impl(v, u, i, j, k):
                out1, out2, out3 = function(
                    numpy, u, i, j, k, coord1(v), coord2(v), coord3(v)
                )
                return instance_class(azcoords(out1, out2), lcoords(out3))

        else:

            def VectorObject34DType_rotate_quaternion_impl(v, u, i, j, k):
                out1, out2, out3 = function(
                    numpy, u, i, j, k, coord1(v), coord2(v), coord3(v)
                )
                return instance_class(azcoords(out1, out2), lcoords(out3), v.temporal)

        return VectorObject34DType_rotate_quaternion_impl

    else:
        raise numba.TypingError(
            "'u', 'i', 'j', and 'k' must be integers or floating-point numbers"
        )


@numba.extending.overload_method(VectorObject3DType, "transform3D")
@numba.extending.overload_method(VectorObject4DType, "transform3D")
def VectorObject34DType_transform3D(v, obj):
    function, *returns = _from_signature(
        "",
        numba_modules["spatial"]["transform3D"],
        (numba_aztype(v), numba_ltype(v)),
    )

    instance_class = v.instance_class
    coord1 = getcoord1[numba_aztype(v)]
    coord2 = getcoord2[numba_aztype(v)]
    coord3 = getcoord1[numba_ltype(v)]
    azcoords = _coord_object_type[returns[0]]
    lcoords = _coord_object_type[returns[1]]

    if isinstance(v, VectorObject3DType):

        def VectorObject34DType_transform3D_impl(v, obj):
            out1, out2, out3 = function(
                numpy,
                obj["xx"],
                obj["xy"],
                obj["xz"],
                obj["yx"],
                obj["yy"],
                obj["yz"],
                obj["zx"],
                obj["zy"],
                obj["zz"],
                coord1(v),
                coord2(v),
                coord3(v),
            )
            return instance_class(azcoords(out1, out2), lcoords(out3))

    else:

        def VectorObject34DType_transform3D_impl(v, obj):
            out1, out2, out3 = function(
                numpy,
                obj["xx"],
                obj["xy"],
                obj["xz"],
                obj["yx"],
                obj["yy"],
                obj["yz"],
                obj["zx"],
                obj["zy"],
                obj["zz"],
                coord1(v),
                coord2(v),
                coord3(v),
            )
            return instance_class(azcoords(out1, out2), lcoords(out3), v.temporal)

    return VectorObject34DType_transform3D_impl


@numba.extending.overload_method(VectorObject4DType, "boost_p4")
def VectorObject4DType_boost_p4(v, p4):
    function, *returns = _from_signature(
        "",
        numba_modules["lorentz"]["boost_p4"],
        (
            numba_aztype(v),
            numba_ltype(v),
            numba_ttype(v),
            numba_aztype(p4),
            numba_ltype(p4),
            numba_ttype(p4),
        ),
    )

    instance_class = v.instance_class
    coord11 = getcoord1[numba_aztype(v)]
    coord12 = getcoord2[numba_aztype(v)]
    coord13 = getcoord1[numba_ltype(v)]
    coord14 = getcoord1[numba_ttype(v)]
    coord21 = getcoord1[numba_aztype(p4)]
    coord22 = getcoord2[numba_aztype(p4)]
    coord23 = getcoord1[numba_ltype(p4)]
    coord24 = getcoord1[numba_ttype(p4)]
    azcoords = _coord_object_type[returns[0]]
    lcoords = _coord_object_type[returns[1]]
    tcoords = _coord_object_type[returns[2]]

    def VectorObject4DType_boost_p4_impl(v, p4):
        out1, out2, out3, out4 = function(
            numpy,
            coord11(v),
            coord12(v),
            coord13(v),
            coord14(v),
            coord21(p4),
            coord22(p4),
            coord23(p4),
            coord24(p4),
        )
        return instance_class(azcoords(out1, out2), lcoords(out3), tcoords(out4))

    return VectorObject4DType_boost_p4_impl


@numba.extending.overload_method(VectorObject4DType, "boost_beta3")
def VectorObject4DType_boost_beta3(v, beta3):
    function, *returns = _from_signature(
        "",
        numba_modules["lorentz"]["boost_beta3"],
        (
            numba_aztype(v),
            numba_ltype(v),
            numba_ttype(v),
            numba_aztype(beta3),
            numba_ltype(beta3),
        ),
    )

    instance_class = v.instance_class
    coord11 = getcoord1[numba_aztype(v)]
    coord12 = getcoord2[numba_aztype(v)]
    coord13 = getcoord1[numba_ltype(v)]
    coord14 = getcoord1[numba_ttype(v)]
    coord21 = getcoord1[numba_aztype(beta3)]
    coord22 = getcoord2[numba_aztype(beta3)]
    coord23 = getcoord1[numba_ltype(beta3)]
    azcoords = _coord_object_type[returns[0]]
    lcoords = _coord_object_type[returns[1]]
    tcoords = _coord_object_type[returns[2]]

    def VectorObject4DType_boost_beta3_impl(v, beta3):
        out1, out2, out3, out4 = function(
            numpy,
            coord11(v),
            coord12(v),
            coord13(v),
            coord14(v),
            coord21(beta3),
            coord22(beta3),
            coord23(beta3),
        )
        return instance_class(azcoords(out1, out2), lcoords(out3), tcoords(out4))

    return VectorObject4DType_boost_beta3_impl


@numba.extending.overload_method(VectorObject4DType, "boost")
def VectorObject4DType_boost(v, booster):
    if isinstance(booster, VectorObject3DType):

        def VectorObject4DType_boost_impl(v, booster):
            return v.boost_beta3(booster)

    elif isinstance(booster, VectorObject4DType):

        def VectorObject4DType_boost_impl(v, booster):
            return v.boost_p4(booster)

    else:
        raise numba.TypingError(
            "specify a Vector3D to boost by beta (velocity with c=1) or "
            "a Vector4D to boost by a momentum 4-vector"
        )

    return VectorObject4DType_boost_impl


def add_boostXYZ(methodname):
    @numba.extending.overload_method(VectorObject4DType, methodname)
    def VectorObject4DType_boostXYZ(v, beta=None, gamma=None):
        if not isinstance(
            beta, (type(None), numba.types.NoneType, numba.types.Omitted)
        ) and isinstance(
            gamma, (type(None), numba.types.NoneType, numba.types.Omitted)
        ):
            if not isinstance(beta, (numba.types.Integer, numba.types.Float)):
                raise numba.TypingError(
                    "'beta' must be an integer or a floating-point number"
                )
            function, *returns = _from_signature(
                "",
                numba_modules["lorentz"][methodname + "_beta"],
                (numba_aztype(v), numba_ltype(v), numba_ttype(v)),
            )

        elif isinstance(
            beta, (type(None), numba.types.NoneType, numba.types.Omitted)
        ) and not isinstance(
            gamma, (type(None), numba.types.NoneType, numba.types.Omitted)
        ):
            if not isinstance(gamma, (numba.types.Integer, numba.types.Float)):
                raise numba.TypingError(
                    "'gamma' must be an integer or a floating-point number"
                )
            function, *returns = _from_signature(
                "",
                numba_modules["lorentz"][methodname + "_gamma"],
                (numba_aztype(v), numba_ltype(v), numba_ttype(v)),
            )

        else:
            raise numba.TypingError("specify 'beta' xor 'gamma', not both or neither")

        instance_class = v.instance_class
        coord1 = getcoord1[numba_aztype(v)]
        coord2 = getcoord2[numba_aztype(v)]
        coord3 = getcoord1[numba_ltype(v)]
        coord4 = getcoord1[numba_ttype(v)]
        azcoords = _coord_object_type[returns[0]]
        lcoords = _coord_object_type[returns[1]]
        tcoords = _coord_object_type[returns[2]]

        if not isinstance(
            beta, (type(None), numba.types.NoneType, numba.types.Omitted)
        ):

            def VectorObject4DType_boostXYZ_impl(v, beta=None, gamma=None):
                out1, out2, out3, out4 = function(
                    numpy,
                    beta,
                    coord1(v),
                    coord2(v),
                    coord3(v),
                    coord4(v),
                )
                return instance_class(
                    azcoords(out1, out2), lcoords(out3), tcoords(out4)
                )

        else:

            def VectorObject4DType_boostXYZ_impl(v, beta=None, gamma=None):
                out1, out2, out3, out4 = function(
                    numpy,
                    gamma,
                    coord1(v),
                    coord2(v),
                    coord3(v),
                    coord4(v),
                )
                return instance_class(
                    azcoords(out1, out2), lcoords(out3), tcoords(out4)
                )

        return VectorObject4DType_boostXYZ_impl


for methodname in "boostX", "boostY", "boostZ":
    add_boostXYZ(methodname)


@numba.extending.overload_method(VectorObject4DType, "to_beta3")
def VectorObject4DType_to_beta3(v):
    function, *returns = _from_signature(
        "",
        numba_modules["lorentz"]["to_beta3"],
        (numba_aztype(v), numba_ltype(v), numba_ttype(v)),
    )

    instance_class = v.instance_class.ProjectionClass3D
    coord1 = getcoord1[numba_aztype(v)]
    coord2 = getcoord2[numba_aztype(v)]
    coord3 = getcoord1[numba_ltype(v)]
    coord4 = getcoord1[numba_ttype(v)]
    azcoords = _coord_object_type[returns[0]]
    lcoords = _coord_object_type[returns[1]]

    def VectorObject4DType_to_beta3_impl(v):
        out1, out2, out3 = function(
            numpy,
            coord1(v),
            coord2(v),
            coord3(v),
            coord4(v),
        )
        return instance_class(azcoords(out1, out2), lcoords(out3))

    return VectorObject4DType_to_beta3_impl


@numba.extending.overload_method(VectorObject4DType, "transform4D")
def VectorObject4DType_transform4D(v, obj):
    function, *returns = _from_signature(
        "",
        numba_modules["lorentz"]["transform4D"],
        (numba_aztype(v), numba_ltype(v), numba_ttype(v)),
    )

    instance_class = v.instance_class
    coord1 = getcoord1[numba_aztype(v)]
    coord2 = getcoord2[numba_aztype(v)]
    coord3 = getcoord1[numba_ltype(v)]
    coord4 = getcoord1[numba_ttype(v)]
    azcoords = _coord_object_type[returns[0]]
    lcoords = _coord_object_type[returns[1]]
    tcoords = _coord_object_type[returns[2]]

    def VectorObject4DType_transform4D_impl(v, obj):
        out1, out2, out3, out4 = function(
            numpy,
            obj["xx"],
            obj["xy"],
            obj["xz"],
            obj["xt"],
            obj["yx"],
            obj["yy"],
            obj["yz"],
            obj["yt"],
            obj["zx"],
            obj["zy"],
            obj["zz"],
            obj["zt"],
            obj["tx"],
            obj["ty"],
            obj["tz"],
            obj["tt"],
            coord1(v),
            coord2(v),
            coord3(v),
            coord4(v),
        )
        return instance_class(azcoords(out1, out2), lcoords(out3), tcoords(out4))

    return VectorObject4DType_transform4D_impl


@numba.extending.overload_method(VectorObject4DType, "is_timelike")
def VectorObject4DType_is_timelike(v, tolerance=0):
    function, *returns = _from_signature(
        "",
        numba_modules["lorentz"]["is_timelike"],
        (numba_aztype(v), numba_ltype(v), numba_ttype(v)),
    )

    coord1 = getcoord1[numba_aztype(v)]
    coord2 = getcoord2[numba_aztype(v)]
    coord3 = getcoord1[numba_ltype(v)]
    coord4 = getcoord1[numba_ttype(v)]

    def VectorObject4DType_is_timelike_impl(v, tolerance=0):
        return function(
            numpy,
            tolerance,
            coord1(v),
            coord2(v),
            coord3(v),
            coord4(v),
        )

    return VectorObject4DType_is_timelike_impl


@numba.extending.overload_method(VectorObject4DType, "is_spacelike")
def VectorObject4DType_is_spacelike(v, tolerance=0):
    function, *returns = _from_signature(
        "",
        numba_modules["lorentz"]["is_spacelike"],
        (numba_aztype(v), numba_ltype(v), numba_ttype(v)),
    )

    coord1 = getcoord1[numba_aztype(v)]
    coord2 = getcoord2[numba_aztype(v)]
    coord3 = getcoord1[numba_ltype(v)]
    coord4 = getcoord1[numba_ttype(v)]

    def VectorObject4DType_is_spacelike_impl(v, tolerance=0):
        return function(
            numpy,
            tolerance,
            coord1(v),
            coord2(v),
            coord3(v),
            coord4(v),
        )

    return VectorObject4DType_is_spacelike_impl


@numba.extending.overload_method(VectorObject4DType, "is_lightlike")
def VectorObject4DType_is_lightlike(v, tolerance=1e-5):
    function, *returns = _from_signature(
        "",
        numba_modules["lorentz"]["is_lightlike"],
        (numba_aztype(v), numba_ltype(v), numba_ttype(v)),
    )

    coord1 = getcoord1[numba_aztype(v)]
    coord2 = getcoord2[numba_aztype(v)]
    coord3 = getcoord1[numba_ltype(v)]
    coord4 = getcoord1[numba_ttype(v)]

    def VectorObject4DType_is_lightlike_impl(v, tolerance=1e-5):
        return function(
            numpy,
            tolerance,
            coord1(v),
            coord2(v),
            coord3(v),
            coord4(v),
        )

    return VectorObject4DType_is_lightlike_impl


# momentum aliases ############################################################


@numba.extending.overload_attribute(MomentumObject2DType, "px")
@numba.extending.overload_attribute(MomentumObject3DType, "px")
@numba.extending.overload_attribute(MomentumObject4DType, "px")
def MomentumObject234DType_px(v):
    def MomentumObject234DType_px_impl(v):
        return v.x

    return MomentumObject234DType_px_impl


@numba.extending.overload_attribute(MomentumObject2DType, "py")
@numba.extending.overload_attribute(MomentumObject3DType, "py")
@numba.extending.overload_attribute(MomentumObject4DType, "py")
def MomentumObject234DType_py(v):
    def MomentumObject234DType_py_impl(v):
        return v.y

    return MomentumObject234DType_py_impl


@numba.extending.overload_attribute(MomentumObject2DType, "pt")
@numba.extending.overload_attribute(MomentumObject3DType, "pt")
@numba.extending.overload_attribute(MomentumObject4DType, "pt")
def MomentumObject234DType_pt(v):
    def MomentumObject234DType_pt_impl(v):
        return v.rho

    return MomentumObject234DType_pt_impl


@numba.extending.overload_attribute(MomentumObject2DType, "pt2")
@numba.extending.overload_attribute(MomentumObject3DType, "pt2")
@numba.extending.overload_attribute(MomentumObject4DType, "pt2")
def MomentumObject234DType_pt2(v):
    def MomentumObject234DType_pt2_impl(v):
        return v.rho2

    return MomentumObject234DType_pt2_impl


@numba.extending.overload_attribute(MomentumObject3DType, "pz")
@numba.extending.overload_attribute(MomentumObject4DType, "pz")
def MomentumObject34DType_pz(v):
    def MomentumObject34DType_pz_impl(v):
        return v.z

    return MomentumObject34DType_pz_impl


@numba.extending.overload_attribute(MomentumObject3DType, "pseudorapidity")
@numba.extending.overload_attribute(MomentumObject4DType, "pseudorapidity")
def MomentumObject34DType_pseudorapidity(v):
    def MomentumObject34DType_pseudorapidity_impl(v):
        return v.eta

    return MomentumObject34DType_pseudorapidity_impl


@numba.extending.overload_attribute(MomentumObject3DType, "p")
@numba.extending.overload_attribute(MomentumObject4DType, "p")
def MomentumObject34DType_p(v):
    def MomentumObject34DType_p_impl(v):
        return v.mag

    return MomentumObject34DType_p_impl


@numba.extending.overload_attribute(MomentumObject3DType, "p2")
@numba.extending.overload_attribute(MomentumObject4DType, "p2")
def MomentumObject34DType_p2(v):
    def MomentumObject34DType_p2_impl(v):
        return v.mag2

    return MomentumObject34DType_p2_impl


@numba.extending.overload_attribute(MomentumObject4DType, "E")
def MomentumObject4DType_E(v):
    def MomentumObject4DType_E_impl(v):
        return v.t

    return MomentumObject4DType_E_impl


@numba.extending.overload_attribute(MomentumObject4DType, "energy")
def MomentumObject4DType_energy(v):
    def MomentumObject4DType_energy_impl(v):
        return v.t

    return MomentumObject4DType_energy_impl


@numba.extending.overload_attribute(MomentumObject4DType, "E2")
def MomentumObject4DType_E2(v):
    def MomentumObject4DType_E2_impl(v):
        return v.t2

    return MomentumObject4DType_E2_impl


@numba.extending.overload_attribute(MomentumObject4DType, "energy2")
def MomentumObject4DType_energy2(v):
    def MomentumObject4DType_energy2_impl(v):
        return v.t2

    return MomentumObject4DType_energy2_impl


@numba.extending.overload_attribute(MomentumObject4DType, "M")
def MomentumObject4DType_M(v):
    def MomentumObject4DType_M_impl(v):
        return v.tau

    return MomentumObject4DType_M_impl


@numba.extending.overload_attribute(MomentumObject4DType, "mass")
def MomentumObject4DType_mass(v):
    def MomentumObject4DType_mass_impl(v):
        return v.tau

    return MomentumObject4DType_mass_impl


@numba.extending.overload_attribute(MomentumObject4DType, "M2")
def MomentumObject4DType_M2(v):
    def MomentumObject4DType_M2_impl(v):
        return v.tau2

    return MomentumObject4DType_M2_impl


@numba.extending.overload_attribute(MomentumObject4DType, "mass2")
def MomentumObject4DType_mass2(v):
    def MomentumObject4DType_mass2_impl(v):
        return v.tau2

    return MomentumObject4DType_mass2_impl


@numba.extending.overload_attribute(MomentumObject4DType, "transverse_energy")
def MomentumObject4DType_transverse_energy(v):
    def MomentumObject4DType_transverse_energy_impl(v):
        return v.Et

    return MomentumObject4DType_transverse_energy_impl


@numba.extending.overload_attribute(MomentumObject4DType, "transverse_energy2")
def MomentumObject4DType_transverse_energy2(v):
    def MomentumObject4DType_transverse_energy2_impl(v):
        return v.Et2

    return MomentumObject4DType_transverse_energy2_impl


@numba.extending.overload_attribute(MomentumObject4DType, "transverse_mass")
def MomentumObject4DType_transverse_mass(v):
    def MomentumObject4DType_transverse_mass_impl(v):
        return v.Mt

    return MomentumObject4DType_transverse_mass_impl


@numba.extending.overload_attribute(MomentumObject4DType, "transverse_mass2")
def MomentumObject4DType_transverse_mass2(v):
    def MomentumObject4DType_transverse_mass2_impl(v):
        return v.Mt2

    return MomentumObject4DType_transverse_mass2_impl


# NumPy functions in Numba ####################################################


@numba.extending.overload(numpy.absolute)
def numpy_absolute(v):
    if isinstance(v, VectorObject2DType):

        def numpy_absolute_impl(v):
            return v.rho

        return numpy_absolute_impl

    elif isinstance(v, VectorObject3DType):

        def numpy_absolute_impl(v):
            return v.mag

        return numpy_absolute_impl

    elif isinstance(v, VectorObject4DType):

        def numpy_absolute_impl(v):
            return v.tau

        return numpy_absolute_impl


@numba.extending.overload(numpy.square)
def numpy_square(v):
    if isinstance(v, VectorObject2DType):

        def numpy_square_impl(v):
            return v.rho2

        return numpy_square_impl

    elif isinstance(v, VectorObject3DType):

        def numpy_square_impl(v):
            return v.mag2

        return numpy_square_impl

    elif isinstance(v, VectorObject4DType):

        def numpy_square_impl(v):
            return v.tau2

        return numpy_square_impl


@numba.extending.overload(numpy.sqrt)
def numpy_sqrt(v):
    if isinstance(v, VectorObject2DType):

        def numpy_sqrt_impl(v):
            return v.rho2 ** 0.25

        return numpy_sqrt_impl

    elif isinstance(v, VectorObject3DType):

        def numpy_sqrt_impl(v):
            return v.mag2 ** 0.25

        return numpy_sqrt_impl

    elif isinstance(v, VectorObject4DType):

        def numpy_sqrt_impl(v):
            return v.tau2 ** 0.25

        return numpy_sqrt_impl


@numba.extending.overload(numpy.cbrt)
def numpy_cbrt(v):
    if isinstance(v, VectorObject2DType):

        def numpy_cbrt_impl(v):
            return v.rho2 ** 0.16666666666666666

        return numpy_cbrt_impl

    elif isinstance(v, VectorObject3DType):

        def numpy_cbrt_impl(v):
            return v.mag2 ** 0.16666666666666666

        return numpy_cbrt_impl

    elif isinstance(v, VectorObject4DType):

        def numpy_cbrt_impl(v):
            return v.tau2 ** 0.16666666666666666

        return numpy_cbrt_impl


@numba.extending.overload(numpy.power)
def numpy_power(v, expo):
    if isinstance(v, VectorObject2DType) and isinstance(
        expo, (numba.types.Integer, numba.types.Float)
    ):

        def numpy_power_impl(v, expo):
            if expo == 2:
                return v.rho2
            else:
                return v.rho ** expo

        return numpy_power_impl

    elif isinstance(v, VectorObject3DType) and isinstance(
        expo, (numba.types.Integer, numba.types.Float)
    ):

        def numpy_power_impl(v, expo):
            if expo == 2:
                return v.mag2
            else:
                return v.mag ** expo

        return numpy_power_impl

    elif isinstance(v, VectorObject4DType) and isinstance(
        expo, (numba.types.Integer, numba.types.Float)
    ):

        def numpy_power_impl(v, expo):
            if expo == 2:
                return v.tau2
            else:
                return v.tau ** expo

        return numpy_power_impl


@numba.extending.overload(numpy.add)
def numpy_add(v1, v2):
    if isinstance(
        v1, (VectorObject2DType, VectorObject3DType, VectorObject4DType)
    ) and isinstance(v2, (VectorObject2DType, VectorObject3DType, VectorObject4DType)):

        def numpy_add_impl(v1, v2):
            return v1.add(v2)

        return numpy_add_impl


@numba.extending.overload(numpy.subtract)
def numpy_subtract(v1, v2):
    if isinstance(
        v1, (VectorObject2DType, VectorObject3DType, VectorObject4DType)
    ) and isinstance(v2, (VectorObject2DType, VectorObject3DType, VectorObject4DType)):

        def numpy_subtract_impl(v1, v2):
            return v1.subtract(v2)

        return numpy_subtract_impl


@numba.extending.overload(numpy.multiply)
def numpy_multiply(a, b):
    if isinstance(
        a, (VectorObject2DType, VectorObject3DType, VectorObject4DType)
    ) and isinstance(b, (numba.types.Integer, numba.types.Float)):

        def numpy_multiply_impl(a, b):
            return a.scale(b)

        return numpy_multiply_impl

    elif isinstance(a, (numba.types.Integer, numba.types.Float)) and isinstance(
        b, (VectorObject2DType, VectorObject3DType, VectorObject4DType)
    ):

        def numpy_multiply_impl(a, b):
            return b.scale(a)

        return numpy_multiply_impl


@numba.extending.overload(numpy.true_divide)
def numpy_true_divide(a, b):
    if isinstance(
        a, (VectorObject2DType, VectorObject3DType, VectorObject4DType)
    ) and isinstance(b, (numba.types.Integer, numba.types.Float)):

        def numpy_true_divide_impl(a, b):
            return a.scale(1 / b)

        return numpy_true_divide_impl


@numba.extending.overload(numpy.negative)
def numpy_negative(v):
    if isinstance(v, (VectorObject2DType, VectorObject3DType, VectorObject4DType)):

        def numpy_negative_impl(v):
            return v.scale(-1)

        return numpy_negative_impl


@numba.extending.overload(numpy.positive)
def numpy_positive(v):
    if isinstance(v, (VectorObject2DType, VectorObject3DType, VectorObject4DType)):

        def numpy_positive_impl(v):
            return v

        return numpy_positive_impl


@numba.extending.overload(numpy.matmul)
def numpy_matmul(v1, v2):
    if isinstance(
        v1, (VectorObject2DType, VectorObject3DType, VectorObject4DType)
    ) and isinstance(v2, (VectorObject2DType, VectorObject3DType, VectorObject4DType)):

        def numpy_matmul_impl(v1, v2):
            return v1.dot(v2)

        return numpy_matmul_impl


# unary operator overloading ##################################################


@numba.extending.overload(abs)
def operator_abs(v):
    if isinstance(v, VectorObject2DType):

        def operator_abs_impl(v):
            return v.rho

        return operator_abs_impl

    elif isinstance(v, VectorObject3DType):

        def operator_abs_impl(v):
            return v.mag

        return operator_abs_impl

    elif isinstance(v, VectorObject4DType):

        def operator_abs_impl(v):
            return v.tau

        return operator_abs_impl


@numba.extending.overload(operator.neg)
def operator_neg(v):
    if isinstance(v, (VectorObject2DType, VectorObject3DType, VectorObject4DType)):

        def operator_neg_impl(v):
            return v.scale(-1)

        return operator_neg_impl


@numba.extending.overload(operator.pos)
def operator_pos(v):
    if isinstance(v, (VectorObject2DType, VectorObject3DType, VectorObject4DType)):

        def operator_pos_impl(v):
            return v

        return operator_pos_impl


@numba.extending.overload(bool)
def operator_truth(v):
    if isinstance(v, VectorObject2DType):

        def operator_truth_impl(v):
            return v.rho2 != 0

        return operator_truth_impl

    elif isinstance(v, VectorObject3DType):

        def operator_truth_impl(v):
            return v.mag2 != 0

        return operator_truth_impl

    elif isinstance(v, VectorObject4DType) and issubclass(
        v.temporaltype.instance_class, TemporalT
    ):

        def operator_truth_impl(v):
            return v.mag2 != 0 or v.t != 0

        return operator_truth_impl

    elif isinstance(v, VectorObject4DType) and issubclass(
        v.temporaltype.instance_class, TemporalTau
    ):

        def operator_truth_impl(v):
            return v.mag2 != 0 or v.tau != 0

        return operator_truth_impl


# binary operator overloading #################################################


@numba.extending.overload(operator.eq)
def operator_eq(v1, v2):
    if isinstance(
        v1, (VectorObject2DType, VectorObject3DType, VectorObject4DType)
    ) and isinstance(v2, (VectorObject2DType, VectorObject3DType, VectorObject4DType)):

        def operator_eq_impl(v1, v2):
            return v1.equal(v2)

        return operator_eq_impl


@numba.extending.overload(operator.ne)
def operator_ne(v1, v2):
    if isinstance(
        v1, (VectorObject2DType, VectorObject3DType, VectorObject4DType)
    ) and isinstance(v2, (VectorObject2DType, VectorObject3DType, VectorObject4DType)):

        def operator_ne_impl(v1, v2):
            return v1.not_equal(v2)

        return operator_ne_impl


@numba.extending.overload(operator.add)
def operator_add(v1, v2):
    if isinstance(
        v1, (VectorObject2DType, VectorObject3DType, VectorObject4DType)
    ) and isinstance(v2, (VectorObject2DType, VectorObject3DType, VectorObject4DType)):

        def operator_add_impl(v1, v2):
            return v1.add(v2)

        return operator_add_impl


@numba.extending.overload(operator.sub)
def operator_sub(v1, v2):
    if isinstance(
        v1, (VectorObject2DType, VectorObject3DType, VectorObject4DType)
    ) and isinstance(v2, (VectorObject2DType, VectorObject3DType, VectorObject4DType)):

        def operator_sub_impl(v1, v2):
            return v1.subtract(v2)

        return operator_sub_impl


@numba.extending.overload(operator.mul)
def operator_mul(a, b):
    if isinstance(
        a, (VectorObject2DType, VectorObject3DType, VectorObject4DType)
    ) and not isinstance(
        b, (VectorObject2DType, VectorObject3DType, VectorObject4DType)
    ):

        def operator_mul_impl(a, b):
            return a.scale(b)

        return operator_mul_impl

    elif not isinstance(
        a, (VectorObject2DType, VectorObject3DType, VectorObject4DType)
    ) and isinstance(b, (VectorObject2DType, VectorObject3DType, VectorObject4DType)):

        def operator_mul_impl(a, b):
            return b.scale(a)

        return operator_mul_impl


@numba.extending.overload(operator.truediv)
def operator_truediv(a, b):
    if isinstance(
        a, (VectorObject2DType, VectorObject3DType, VectorObject4DType)
    ) and not isinstance(
        b, (VectorObject2DType, VectorObject3DType, VectorObject4DType)
    ):

        def operator_truediv_impl(a, b):
            return a.scale(1.0 / b)

        return operator_truediv_impl


@numba.extending.overload(operator.pow)
def operator_pow(a, b):
    if (
        isinstance(a, VectorObject2DType)
        and isinstance(b, numba.types.Literal)
        and b.literal_value == 2
    ):

        def operator_pow_impl(a, b):
            return a.rho2

        return operator_pow_impl

    elif (
        isinstance(a, VectorObject3DType)
        and isinstance(b, numba.types.Literal)
        and b.literal_value == 2
    ):

        def operator_pow_impl(a, b):
            return a.mag2

        return operator_pow_impl

    elif (
        isinstance(a, VectorObject4DType)
        and isinstance(b, numba.types.Literal)
        and b.literal_value == 2
    ):

        def operator_pow_impl(a, b):
            return a.tau2

        return operator_pow_impl

    elif isinstance(
        a, (VectorObject2DType, VectorObject3DType, VectorObject4DType)
    ) and not isinstance(
        b, (VectorObject2DType, VectorObject3DType, VectorObject4DType)
    ):

        def operator_pow_impl(a, b):
            return abs(a) ** b

        return operator_pow_impl


if hasattr(operator, "matmul"):

    @numba.extending.overload(operator.matmul)
    def operator_matmul(v1, v2):
        if isinstance(
            v1, (VectorObject2DType, VectorObject3DType, VectorObject4DType)
        ) and isinstance(
            v2, (VectorObject2DType, VectorObject3DType, VectorObject4DType)
        ):

            def operator_matmul_impl(v1, v2):
                return v1.dot(v2)

            return operator_matmul_impl


# helper functions for Awkward Numba backend ##################################


@numba.jit(nopython=True)
def _awkward_numba_xy(record):
    return AzimuthalObjectXY(record["x"], record["y"])


@numba.jit(nopython=True)
def _awkward_numba_pxy(record):
    return AzimuthalObjectXY(record["px"], record["y"])


@numba.jit(nopython=True)
def _awkward_numba_xpy(record):
    return AzimuthalObjectXY(record["x"], record["py"])


@numba.jit(nopython=True)
def _awkward_numba_pxpy(record):
    return AzimuthalObjectXY(record["px"], record["py"])


@numba.jit(nopython=True)
def _awkward_numba_rhophi(record):
    return AzimuthalObjectRhoPhi(record["rho"], record["phi"])


@numba.jit(nopython=True)
def _awkward_numba_ptphi(record):
    return AzimuthalObjectRhoPhi(record["pt"], record["phi"])


@numba.jit(nopython=True)
def _awkward_numba_z(record):
    return LongitudinalObjectZ(record["z"])


@numba.jit(nopython=True)
def _awkward_numba_pz(record):
    return LongitudinalObjectZ(record["pz"])


@numba.jit(nopython=True)
def _awkward_numba_theta(record):
    return LongitudinalObjectTheta(record["theta"])


@numba.jit(nopython=True)
def _awkward_numba_eta(record):
    return LongitudinalObjectEta(record["eta"])


@numba.jit(nopython=True)
def _awkward_numba_t(record):
    return TemporalObjectT(record["t"])


@numba.jit(nopython=True)
def _awkward_numba_E(record):
    return TemporalObjectT(record["E"])


@numba.jit(nopython=True)
def _awkward_numba_e(record):
    return TemporalObjectT(record["e"])


@numba.jit(nopython=True)
def _awkward_numba_energy(record):
    return TemporalObjectT(record["energy"])


@numba.jit(nopython=True)
def _awkward_numba_tau(record):
    return TemporalObjectTau(record["tau"])


@numba.jit(nopython=True)
def _awkward_numba_M(record):
    return TemporalObjectTau(record["M"])


@numba.jit(nopython=True)
def _awkward_numba_m(record):
    return TemporalObjectTau(record["m"])


@numba.jit(nopython=True)
def _awkward_numba_mass(record):
    return TemporalObjectTau(record["mass"])
