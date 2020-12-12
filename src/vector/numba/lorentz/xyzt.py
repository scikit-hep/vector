# -*- coding: utf-8 -*-
# Copyright (c) 2019-2020, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import absolute_import, division, print_function

import operator

import numba

from vector import core
from vector.single.lorentz.xyzt import LorentzXYZTFree


# Typing phase - Teaching Numba about our types


class LorentzXYZTType(numba.types.Type):
    def __init__(self):
        # Type names have to be unique identifiers; they determine whether Numba
        # will recompile a function with new types.
        super().__init__(name="LorentzXYZTType()")


@numba.extending.typeof_impl.register(LorentzXYZTFree)
def typeof_LorentzXYZTFree(obj, c):
    return LorentzXYZTType()


@numba.extending.type_callable(operator.getitem)
def typer_LorentzXYZT_getitem(context):
    def generic(self, ind):
        # Only accept compile-time constants. It's a fair restriction.
        if isinstance(self, LorentzXYZTType) and ind in {
            numba.types.StringLiteral(c) for c in {"x", "y", "z", "t"}
        }:
            return numba.float64

    return generic


@numba.extending.type_callable(operator.add)
def typer_LorentzXYZT_add(context):
    def generic(self, other):
        # Only accept compile-time constants. It's a fair restriction.
        if isinstance(self, LorentzXYZTType) and isinstance(other, LorentzXYZTType):
            return LorentzXYZTType()

    return generic


@numba.extending.type_callable(operator.mul)
def typer_LorentzXYZT_mul(context):
    def generic(self, other):
        # Only accept compile-time constants. It's a fair restriction.
        if isinstance(self, LorentzXYZTType) and isinstance(other, LorentzXYZTType):
            return numba.float64
        elif isinstance(self, LorentzXYZTType) and isinstance(other, numba.types.Float):
            return LorentzXYZTType()

    return generic


# Type model


@numba.extending.register_model(LorentzXYZTType)
class LorentzXYZTModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        # This is the C-style struct that will be used wherever LorentzXYZT are needed.
        members = [
            ("x", numba.float64),
            ("y", numba.float64),
            ("z", numba.float64),
            ("t", numba.float64),
        ]
        super().__init__(dmm, fe_type, members)


# Unboxing & Boxing


@numba.extending.unbox(LorentzXYZTType)
def unbox_LorentzXYZT(lxyztype, lxyzobj, c):
    # How to turn LorentzXYZTFree Python objects into LorentzXYZTModel structs.
    x_obj = c.pyapi.object_getattr_string(lxyzobj, "x")
    y_obj = c.pyapi.object_getattr_string(lxyzobj, "y")
    z_obj = c.pyapi.object_getattr_string(lxyzobj, "z")
    t_obj = c.pyapi.object_getattr_string(lxyzobj, "t")

    # "values" are raw LLVM code; "proxies" have getattr/setattr logic to access fields.
    outproxy = c.context.make_helper(c.builder, lxyztype)

    # https://github.com/numba/numba/blob/master/numba/core/pythonapi.py
    outproxy.x = c.pyapi.float_as_double(x_obj)
    outproxy.y = c.pyapi.float_as_double(y_obj)
    outproxy.z = c.pyapi.float_as_double(z_obj)
    outproxy.t = c.pyapi.float_as_double(t_obj)

    # Yes, we're in that world...
    c.pyapi.decref(x_obj)
    c.pyapi.decref(y_obj)
    c.pyapi.decref(z_obj)
    c.pyapi.decref(t_obj)

    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(outproxy._getvalue(), is_error)


@numba.extending.box(LorentzXYZTType)
def box_LorentzXYZT(lxyztype, lxyzval, c):
    # This proxy is initialized with a value, used for getattr, rather than setattr.
    inproxy = c.context.make_helper(c.builder, lxyztype, lxyzval)
    x_obj = c.pyapi.float_from_double(inproxy.x)
    y_obj = c.pyapi.float_from_double(inproxy.y)
    z_obj = c.pyapi.float_from_double(inproxy.z)
    t_obj = c.pyapi.float_from_double(inproxy.t)

    # The way we get Python objects into this lowered world is by pickling them.
    LorentzXYZTFree_obj = c.pyapi.unserialize(c.pyapi.serialize_object(LorentzXYZTFree))

    out = c.pyapi.call_function_objargs(
        LorentzXYZTFree_obj, (x_obj, y_obj, z_obj, t_obj)
    )

    c.pyapi.decref(LorentzXYZTFree_obj)
    c.pyapi.decref(x_obj)
    c.pyapi.decref(y_obj)
    c.pyapi.decref(z_obj)
    c.pyapi.decref(t_obj)

    return out


# Constructor

# Defining an in-Numba constructor is a separate thing.
@numba.extending.type_callable(LorentzXYZTFree)
def typer_LorentzXYZTFree_constructor(context):
    def typer(x, y, z, t):
        if (
            x == numba.types.float64
            and y == numba.types.float64
            and z == numba.types.float64
            and t == numba.types.float64
        ):
            return LorentzXYZTType()

    return typer


# Lowering


@numba.extending.lower_builtin(
    LorentzXYZTFree,
    numba.types.Float,
    numba.types.Float,
    numba.types.Float,
    numba.types.Float,
)
def lower_LorentzXYZTFree_constructor(context, builder, sig, args):
    xval, yval, zval, tval = args

    outproxy = context.make_helper(builder, sig.return_type)
    outproxy.x = xval
    outproxy.y = yval
    outproxy.z = zval
    outproxy.t = tval

    return outproxy._getvalue()


# Now it's time to define the methods and properties.

# To simply map model attributes to user-accessible properties, use a macro.
numba.extending.make_attribute_wrapper(LorentzXYZTType, "x", "x")
numba.extending.make_attribute_wrapper(LorentzXYZTType, "y", "y")
numba.extending.make_attribute_wrapper(LorentzXYZTType, "z", "z")
numba.extending.make_attribute_wrapper(LorentzXYZTType, "t", "t")


# For more general cases, there's an AttributeTemplate.
@numba.extending.infer_getattr
class typer_LorentzXYZT_methods(numba.core.typing.templates.AttributeTemplate):
    key = LorentzXYZTType

    def generic_resolve(self, lxyztype, attr):
        if attr == "pt":
            return numba.float64
        elif attr == "eta":
            return numba.float64
        elif attr == "phi":
            return numba.float64
        elif attr == "mag":
            return numba.float64
        else:
            # typers that return None defer to other typers.
            return None

    # If we had any methods with arguments, this is how we'd do it.
    #
    # @numba.typing.templates.bound_function("pt")
    # def pt_resolve(self, lxyztype, args, kwargs):
    #     ...


# To lower these functions, we can duck-type the Python functions above.
# Since they're defined in terms of NumPy functions, they apply to
#
#    * Python scalars
#    * NumPy arrays
#    * Awkward arrays
#    * lowered Numba values


@numba.extending.lower_getattr(LorentzXYZTType, "pt")
def lower_LorentzXYZT_pt(context, builder, lxyztype, lxyzval):
    return context.compile_internal(
        builder, core.lorentz.xyzt.pt, numba.float64(lxyztype), (lxyzval,)
    )


@numba.extending.lower_getattr(LorentzXYZTType, "eta")
def lower_LorentzXYZT_eta(context, builder, lxyztype, lxyzval):
    return context.compile_internal(
        builder, core.lorentz.xyzt.eta, numba.float64(lxyztype), (lxyzval,)
    )


@numba.extending.lower_getattr(LorentzXYZTType, "phi")
def lower_LorentzXYZT_phi(context, builder, lxyztype, lxyzval):
    return context.compile_internal(
        builder, core.lorentz.xyzt.phi, numba.float64(lxyztype), (lxyzval,)
    )


@numba.extending.lower_getattr(LorentzXYZTType, "mag")
def lower_LorentzXYZT_mag(context, builder, lxyztype, lxyzval):
    return context.compile_internal(
        builder, core.lorentz.xyzt.mag, numba.float64(lxyztype), (lxyzval,)
    )


@numba.extending.lower_builtin(operator.add, LorentzXYZTType, LorentzXYZTType)
def lower_add_LorentzXYZT(context, builder, sig, args):
    a_type, b_type = sig.args
    a_val, b_val = args

    def f(a, b):
        return LorentzXYZTFree(a.x + b.x, a.y + b.y, a.z + b.z, a.t + b.t)

    res = context.compile_internal(
        builder, f, LorentzXYZTType()(a_type, b_type), (a_val, b_val)
    )
    return res


@numba.extending.lower_builtin(operator.mul, LorentzXYZTType, numba.float64)
def lower_mul_scalar_LorentzXYZT(context, builder, sig, args):
    a_type, b_type = sig.args
    a_val, b_val = args

    def f(a, b):
        return LorentzXYZTFree(a.x * b, a.y * b, a.z * b, a.t * b)

    res = context.compile_internal(
        builder, f, LorentzXYZTType()(a_type, b_type), (a_val, b_val)
    )
    return res


@numba.extending.lower_builtin(operator.mul, LorentzXYZTType, LorentzXYZTType)
def lower_mul_LorentzXYZT(context, builder, sig, args):
    a_type, b_type = sig.args
    a_val, b_val = args

    return context.compile_internal(
        builder, core.lorentz.xyzt.dot, numba.float64(a_type, b_type), (a_val, b_val)
    )


@numba.extending.lower_builtin(
    operator.getitem, LorentzXYZTType, numba.types.StringLiteral
)
def lower_getitem_LorentzXYZT(context, builder, sig, args):
    lxyztype, wheretype = sig.args
    lxyzval, whereval = args

    inproxy = context.make_helper(builder, lxyztype, lxyzval)

    # The value of a StringLiteral is in its compile-time type.
    if wheretype.literal_value == "x":
        return inproxy.x
    elif wheretype.literal_value == "y":
        return inproxy.y
    elif wheretype.literal_value == "z":
        return inproxy.z
    elif wheretype.literal_value == "t":
        return inproxy.t
