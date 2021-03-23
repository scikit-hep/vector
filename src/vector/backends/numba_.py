# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import sys
import types
import typing

import numba

import vector.compute.lorentz
import vector.compute.planar
import vector.compute.spatial


def make_cell(cell_contents):
    if hasattr(types, "CellType"):
        return types.CellType(cell_contents)

    else:

        def f():
            z = 123

            def g():
                return z

            return g

        h = f()
        h.__closure__[0].cell_contents = cell_contents
        return h.__closure__[0]


def make_dispatcher(function, new_module):
    closure = None
    if function.__closure__ is not None:
        closure = tuple(make_cell(x.cell_contents) for x in function.__closure__)
    new_function = types.FunctionType(
        function.__code__,
        new_module.__dict__,  # make the function's surrounding scope the new module
        function.__name__,
        function.__defaults__,
        closure,
    )
    return numba.jit(nopython=True)(new_function)


names_and_modules = [
    ("planar", vector.compute.planar),
    ("spatial", vector.compute.spatial),
    ("lorentz", vector.compute.lorentz),
]

numba_modules: typing.Any = {}

copied: typing.Any = {}

# Make a copy of all the vector.compute.* modules to be wrapped as CPUDispatchers,
# leaving the originals untouched so they still work in TensorFlow/JAX/Torch/whatever.
for groupname, module in names_and_modules:
    numba_modules[groupname] = {}
    for modname, submodule in module.__dict__.items():
        if isinstance(submodule, types.ModuleType) and submodule.__name__.startswith(
            "vector.compute."
        ):
            # Dynamically created modules need to be in sys.modules for Numba.
            new_name = submodule.__name__.replace(
                "vector.compute.", "vector.compute.numba."
            )
            new_module = types.ModuleType(new_name)
            sys.modules[new_name] = new_module
            numba_modules[groupname][modname] = {None: new_module}

            # Copy (and Numbafy) all the functions defined in this module except "dispatch".
            for name, obj in submodule.__dict__.items():
                if (
                    isinstance(obj, types.FunctionType)
                    and name != "dispatch"
                    and obj.__module__ == submodule.__name__
                ):
                    copied_function = make_dispatcher(obj, new_module)
                    copied[obj] = copied_function
                    setattr(new_module, name, copied_function)

            # Copy (and Numbafy) all the functions in the dispatch_map that aren't
            # defined at module-level.
            for key, value in getattr(submodule, "dispatch_map").items():
                function, *returns = value
                if function not in copied:
                    copied_function = make_dispatcher(function, new_module)
                    copied[function] = copied_function
                numba_modules[groupname][modname][key] = tuple(
                    [copied[function]] + returns
                )

# Now do a second pass, in which references to other modules in the old set are
# replaced with the corresponding modules in the new set.
for groupname, module in names_and_modules:
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

# Now do a third pass, in which any closures of a function over other functions
# get mapped to the corresponding Numba CPUDispatcher instead.
for copied_function in copied.values():
    if copied_function.py_func.__closure__ is not None:
        for cell in copied_function.py_func.__closure__:
            if cell.cell_contents in copied:
                cell.cell_contents = copied[cell.cell_contents]
