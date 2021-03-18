# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numba
import numba.core.typing
import numba.core.typing.ctypes_utils
import numpy

from vector.backends.numba_ import numba_modules
from vector.backends.object_ import (
    AzimuthalObject,
    AzimuthalObjectXY,
    LongitudinalObject,
    TemporalObject,
    VectorObject2D,
)
from vector.methods import AzimuthalXY, _from_signature

# Since CoordinateObjects are NamedTuples, we get their types wrapped for free.


def is_azimuthaltype(typ):
    return isinstance(typ, numba.types.NamedUniTuple) and issubclass(
        typ.instance_class, AzimuthalObject
    )


def is_longitudinaltype(typ):
    return isinstance(typ, numba.types.NamedUniTuple) and issubclass(
        typ.instance_class, LongitudinalObject
    )


def is_temporaltype(typ):
    return isinstance(typ, numba.types.NamedUniTuple) and issubclass(
        typ.instance_class, TemporalObject
    )


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
@numba.extending.lower_builtin(VectorObject2D, numba.types.Type)
def VectorObject2D_constructor(context, builder, sig, args):
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
