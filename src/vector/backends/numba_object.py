# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numba
import numba.core.typing
import numba.core.typing.ctypes_utils
import numpy

import vector
from vector.backends.numba_ import numba_modules
from vector.backends.object_ import (
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
)
from vector.methods import AzimuthalXY, _from_signature

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


class VectorObject2DType(numba.types.Type):
    def __init__(self, azimuthaltype):
        super().__init__(name=f"VectorObject2DType({azimuthaltype})")
        self.azimuthaltype = azimuthaltype


@numba.extending.register_model(VectorObject2DType)
class VectorObject2DModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("azimuthal", fe_type.azimuthaltype),
        ]
        super().__init__(dmm, fe_type, members)


@numba.extending.typeof_impl.register(VectorObject2D)
def VectorObject2D_typeof(val, c):
    return VectorObject2DType(numba.typeof(val.azimuthal))


numba.extending.make_attribute_wrapper(VectorObject2DType, "azimuthal", "azimuthal")


@numba.extending.type_callable(VectorObject2D)
def VectorObject2D_constructor_typer(context):
    def typer(azimuthaltype):
        if is_azimuthaltype(azimuthaltype):
            return VectorObject2DType(azimuthaltype)

    return typer


@numba.extending.lower_builtin(VectorObject2D, numba.types.Type)
def VectorObject2D_constructor_impl(context, builder, sig, args):
    typ = sig.return_type
    proxyout = numba.core.cgutils.create_struct_proxy(typ)(context, builder)
    proxyout.azimuthal = args[0]
    return proxyout._getvalue()


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
    VectorObject2D_obj = c.pyapi.unserialize(c.pyapi.serialize_object(VectorObject2D))
    proxyin = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    azimuthal_obj = c.pyapi.from_native_value(typ.azimuthaltype, proxyin.azimuthal)
    output_obj = c.pyapi.call_function_objargs(VectorObject2D_obj, (azimuthal_obj,))
    c.pyapi.decref(VectorObject2D_obj)
    c.pyapi.decref(azimuthal_obj)
    return output_obj


@numba.extending.overload_attribute(VectorObject2DType, "x")
def VectorObject2D_x(v):
    function, *returns = _from_signature(
        "", numba_modules["planar"]["x"], (AzimuthalXY,)
    )

    def VectorObject2D_x_impl(v):
        return function(numpy, v.azimuthal.x, v.azimuthal.y)

    return VectorObject2D_x_impl


@numba.extending.overload_method(VectorObject2DType, "deltaphi")
def VectorObject2D_deltaphi(v1, v2):
    if isinstance(v2, VectorObject2DType):
        function, *returns = _from_signature(
            "", numba_modules["planar"]["deltaphi"], (AzimuthalXY, AzimuthalXY)
        )

        def VectorObject2D_deltaphi_impl(v1, v2):
            return function(
                numpy, v1.azimuthal.x, v1.azimuthal.y, v2.azimuthal.x, v2.azimuthal.y
            )

        return VectorObject2D_deltaphi_impl


@numba.extending.overload_method(VectorObject2DType, "rotateZ")
def VectorObject2D_rotateZ(v, angle):
    if isinstance(angle, (numba.types.Integer, numba.types.Float)):
        function, *returns = _from_signature(
            "", numba_modules["planar"]["rotateZ"], (AzimuthalXY,)
        )

        def VectorObject2D_rotateZ_impl(v, angle):
            x, y = function(numpy, angle, v.azimuthal.x, v.azimuthal.y)
            return VectorObject2D(AzimuthalObjectXY(x, y))

        return VectorObject2D_rotateZ_impl
