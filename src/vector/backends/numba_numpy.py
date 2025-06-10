# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
Implements VectorNumpys in Numba. Only __getitem__ should be overloaded to return
lowered VectorObjects, and maybe the ``vector.array`` constructor should also
be handled.

As mentioned in scikit-hep/vector#43, this capability is blocked by numba/numba#6148.
"""
