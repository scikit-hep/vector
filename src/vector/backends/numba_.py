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


def make_dispatcher(function, new_module):
    new_function = types.FunctionType(
        function.__code__,
        new_module.__dict__,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    return numba.jit(nopython=True)(new_function)


scope = [
    ("planar", vector.compute.planar),
    ("spatial", vector.compute.spatial),
    ("lorentz", vector.compute.lorentz),
]

numba_modules: typing.Any = {}
for groupname, module in scope:
    numba_modules[groupname] = {}
    for modname, submodule in module.__dict__.items():
        if isinstance(submodule, types.ModuleType) and submodule.__name__.startswith(
            "vector.compute."
        ):
            new_module = types.ModuleType("<dynamic>")
            numba_modules[groupname][modname] = {None: new_module}
            copied = {}
            for name, obj in submodule.__dict__.items():
                if (
                    isinstance(obj, types.FunctionType)
                    and name != "dispatch"
                    and obj.__module__ == submodule.__name__
                ):
                    copied_function = make_dispatcher(obj, new_module)
                    copied[obj] = copied_function
                    setattr(new_module, name, copied_function)

            for key, value in getattr(submodule, "dispatch_map").items():
                function, *returns = value
                if function not in copied:
                    copied_function = make_dispatcher(function, new_module)
                    copied[function] = copied_function
                numba_modules[groupname][modname][key] = tuple(
                    [copied[function]] + returns
                )

for groupname, module in scope:
    for modname, submodule in module.__dict__.items():
        if isinstance(submodule, types.ModuleType) and submodule.__name__.startswith(
            "vector.compute."
        ):
            for name, refmodule in submodule.__dict__.items():
                if isinstance(
                    refmodule, types.ModuleType
                ) and refmodule.__name__.startswith("vector.compute."):
                    splitname = refmodule.__name__.split(".")
                    setattr(
                        numba_modules[groupname][modname][None],
                        name,
                        numba_modules[splitname[2]][splitname[3]][None],
                    )
