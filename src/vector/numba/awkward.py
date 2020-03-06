# Copyright (c) 2019-2020, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import division, absolute_import, print_function

import awkward1 as ak
from .lorentz import LorentzXYZType
from ..awkward import lorentzbehavior
from ..single.lorentz import LorentzXYZFree
import operator


def lower_ArrayBuilder_append_LorentzXYZ(context, builder, sig, args):
    def doit(output, lxyz):
        output.beginrecord("LorentzXYZ")
        output.field("x")
        output.real(lxyz.x)
        output.field("y")
        output.real(lxyz.y)
        output.field("z")
        output.real(lxyz.z)
        output.field("t")
        output.real(lxyz.t)
        output.endrecord()

    return context.compile_internal(builder, doit, sig, args)


lorentzbehavior[
    "__numba_lower__", ak.ArrayBuilder.append, LorentzXYZType
] = lower_ArrayBuilder_append_LorentzXYZ


def typer_lorentz_xyz_add(binop, left, right):
    return LorentzXYZType()(left, right)


def lower_lorentz_xyz_add(context, builder, sig, args):
    def compute(left, right):
        return LorentzXYZFree(
            left.x + right.x, left.y + right.y, left.z + right.z, left.t + right.t
        )

    return context.compile_internal(builder, compute, sig, args)


lorentzbehavior[
    "__numba_typer__", "LorentzXYZ", operator.add, "LorentzXYZ"
] = typer_lorentz_xyz_add
lorentzbehavior[
    "__numba_lower__", "LorentzXYZ", operator.add, "LorentzXYZ"
] = lower_lorentz_xyz_add
