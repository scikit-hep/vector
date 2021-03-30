# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import types
import typing

import numba

import vector.compute.lorentz
import vector.compute.planar
import vector.compute.spatial

names_and_modules = [
    ("planar", vector.compute.planar),
    ("spatial", vector.compute.spatial),
    ("lorentz", vector.compute.lorentz),
]

numba_modules: typing.Any = {}

registered = set()

for groupname, module in names_and_modules:
    numba_modules[groupname] = {}
    for modname, submodule in module.__dict__.items():
        if isinstance(submodule, types.ModuleType) and submodule.__name__.startswith(
            "vector.compute."
        ):
            new_name = submodule.__name__.replace(
                "vector.compute.", "vector.compute.numba."
            )
            numba_modules[groupname][modname] = {}

            for name, obj in submodule.__dict__.items():
                if (
                    isinstance(obj, types.FunctionType)
                    and name != "dispatch"
                    and obj.__module__ == submodule.__name__
                ):
                    numba.extending.register_jitable(obj)
                    registered.add(obj)

            for key, value in getattr(submodule, "dispatch_map").items():
                function, *returns = value
                if function not in registered:
                    numba.extending.register_jitable(function)
                    registered.add(function)

                numba_modules[groupname][modname][key] = tuple([function] + returns)
