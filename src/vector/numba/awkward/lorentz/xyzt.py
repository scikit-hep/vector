# -*- coding: utf-8 -*-
# Copyright (c) 2019-2020, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import absolute_import, division, print_function

import operator

import awkward as ak

from vector.awkward.lorentz.xyzt import behavior
from vector.numba.lorentz.xyzt import LorentzXYZTType
from vector.numba.lorentz.xyzt import lower_add_LorentzXYZT


def lower_ArrayBuilder_append_LorentzXYZT(context, builder, sig, args):
    def doit(output, lxyz):
        output.begin_record("LorentzXYZT")
        output.field("x")
        output.real(lxyz.x)
        output.field("y")
        output.real(lxyz.y)
        output.field("z")
        output.real(lxyz.z)
        output.field("t")
        output.real(lxyz.t)
        output.end_record()

    return context.compile_internal(builder, doit, sig, args)


behavior[
    "__numba_lower__", ak.ArrayBuilder.append, LorentzXYZTType
] = lower_ArrayBuilder_append_LorentzXYZT


def typer_lorentz_xyz_add(binop, left, right):
    return LorentzXYZTType()(left, right)


behavior[
    "__numba_typer__", "LorentzXYZT", operator.add, "LorentzXYZT"
] = typer_lorentz_xyz_add
behavior[
    "__numba_lower__", "LorentzXYZT", operator.add, "LorentzXYZT"
] = lower_add_LorentzXYZT
