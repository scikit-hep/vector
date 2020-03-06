# Copyright (c) 2019-2020, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import division, absolute_import, print_function

import numba
import operator

from ..common.lorentz import LorentzXYZCommon
from ..single.lorentz import LorentzXYZFree
from .. import core


@numba.extending.typeof_impl.register(LorentzXYZFree)
def typeof_LorentzXYZFree(obj, c):
    return LorentzXYZType()


class LorentzXYZType(numba.types.Type):
    def __init__(self):
        # Type names have to be unique identifiers; they determine whether Numba
        # will recompile a function with new types.
        super(LorentzXYZType, self).__init__(name="LorentzXYZType()")


@numba.extending.register_model(LorentzXYZType)
class LorentzXYZModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        # This is the C-style struct that will be used wherever LorentzXYZ are needed.
        members = [
            ("x", numba.float64),
            ("y", numba.float64),
            ("z", numba.float64),
            ("t", numba.float64),
        ]
        super(LorentzXYZModel, self).__init__(dmm, fe_type, members)


@numba.extending.unbox(LorentzXYZType)
def unbox_LorentzXYZ(lxyztype, lxyzobj, c):
    # How to turn LorentzXYZFree Python objects into LorentzXYZModel structs.
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

    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(outproxy._getvalue(), is_error)


@numba.extending.box(LorentzXYZType)
def box_LorentzXYZ(lxyztype, lxyzval, c):
    # This proxy is initialized with a value, used for getattr, rather than setattr.
    inproxy = c.context.make_helper(c.builder, lxyztype, lxyzval)
    x_obj = c.pyapi.float_from_double(inproxy.x)
    y_obj = c.pyapi.float_from_double(inproxy.y)
    z_obj = c.pyapi.float_from_double(inproxy.z)
    t_obj = c.pyapi.float_from_double(inproxy.t)

    # The way we get Python objects into this lowered world is by pickling them.
    LorentzXYZFree_obj = c.pyapi.unserialize(c.pyapi.serialize_object(LorentzXYZFree))

    out = c.pyapi.call_function_objargs(
        LorentzXYZFree_obj, (x_obj, y_obj, z_obj, t_obj)
    )

    c.pyapi.decref(LorentzXYZFree_obj)
    c.pyapi.decref(x_obj)
    c.pyapi.decref(y_obj)
    c.pyapi.decref(z_obj)
    c.pyapi.decref(t_obj)

    return out


# Defining an in-Numba constructor is a separate thing.
@numba.extending.type_callable(LorentzXYZFree)
def typer_LorentzXYZFree_constructor(context):
    def typer(x, y, z, t):
        if (
            x == numba.types.float64
            and y == numba.types.float64
            and z == numba.types.float64
            and t == numba.types.float64
        ):
            return LorentzXYZType()

    return typer


@numba.extending.lower_builtin(
    LorentzXYZFree,
    numba.types.float64,
    numba.types.float64,
    numba.types.float64,
    numba.types.float64,
)
def lower_LorentzXYZFree_constructor(context, builder, sig, args):
    rettype, (xtype, ytype, ztype, ttype) = sig.return_type, sig.args
    xval, yval, zval, tval = args

    outproxy = context.make_helper(builder, rettype)
    outproxy.x = xval
    outproxy.y = yval
    outproxy.z = zval
    outproxy.t = tval

    return outproxy._getvalue()


# Now it's time to define the methods and properties.

# To simply map model attributes to user-accessible properties, use a macro.
numba.extending.make_attribute_wrapper(LorentzXYZType, "x", "x")
numba.extending.make_attribute_wrapper(LorentzXYZType, "y", "y")
numba.extending.make_attribute_wrapper(LorentzXYZType, "z", "z")
numba.extending.make_attribute_wrapper(LorentzXYZType, "t", "t")

# For more general cases, there's an AttributeTemplate.
@numba.typing.templates.infer_getattr
class typer_LorentzXYZ_methods(numba.typing.templates.AttributeTemplate):
    key = LorentzXYZType

    def generic_resolve(self, lxyztype, attr):
        if attr == "pt":
            return numba.float64
        elif attr == "eta":
            return numba.float64
        elif attr == "phi":
            return numba.float64
        elif attr == "mass":
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


@numba.extending.lower_getattr(LorentzXYZType, "pt")
def lower_LorentzXYZ_pt(context, builder, lxyztype, lxyzval):
    return context.compile_internal(
        builder, core.lorentz.xyzt.pt, numba.float64(lxyztype), (lxyzval,)
    )


@numba.extending.lower_getattr(LorentzXYZType, "eta")
def lower_LorentzXYZ_eta(context, builder, lxyztype, lxyzval):
    return context.compile_internal(
        builder, core.lorentz.xyzt.eta, numba.float64(lxyztype), (lxyzval,)
    )


@numba.extending.lower_getattr(LorentzXYZType, "phi")
def lower_LorentzXYZ_phi(context, builder, lxyztype, lxyzval):
    return context.compile_internal(
        builder, core.lorentz.xyzt.phi, numba.float64(lxyztype), (lxyzval,)
    )


@numba.extending.lower_getattr(LorentzXYZType, "mass")
def lower_LorentzXYZ_mass(context, builder, lxyztype, lxyzval):
    return context.compile_internal(
        builder, core.lorentz.xyzt.mass, numba.float64(lxyztype), (lxyzval,)
    )


# And the __getitem__ access...
@numba.typing.templates.infer_global(operator.getitem)
class typer_LorentzXYZ_getitem(numba.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 2 and len(kwargs) == 0 and isinstance(args[0], LorentzXYZType):
            # Only accept compile-time constants. It's a fair restriction.
            if isinstance(args[1], numba.types.StringLiteral):
                if args[1].literal_value in ("x", "y", "z", "t"):
                    return numba.float64(*args)


@numba.extending.lower_builtin(
    operator.getitem, LorentzXYZType, numba.types.StringLiteral
)
def lower_getitem_LorentzXYZ(context, builder, sig, args):
    rettype, (lxyztype, wheretype) = sig.return_type, sig.args
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
