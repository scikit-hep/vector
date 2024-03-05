# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.
"""
Vector comes loaded with 3 + 2 backends; a pure Python object backend, a NumPy backend, an Awkward
Array backend, an Object-Numba, and an Awkward-Numba backend to leverage JIT (Just In Time) compiled
calculations on vectors. Other potential future vanilla backends include Tensorflow and JAX, and
other possible future Numba-backends include Numba-NumPy.
"""
