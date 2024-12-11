# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
Defines behaviors for PyTorch Tensor. New tensors created with the

.. code-block:: python

    vector.tensor(...)

function will have these behaviors built in (and will pass them to any derived
tensors).
"""

from __future__ import annotations

from packaging.version import parse as parse_version

import torch

if parse_version(torch.__version__) < parse_version("1.7.0"):
    # https://pytorch.org/docs/stable/notes/extending.html#subclassing-torch-tensor
    raise ImportError("Vector's PyTorch backend requires PyTorch >= 1.7.0")


class Index:
    pass


class AzimuthalIndex(Index):
    pass


class AzimuthalXYIndex(AzimuthalIndex):
    def __init__(self, x, y):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def _constructors(self):
        return f"x_index={self._x}, y_index={self._y}"


class VectorTensor(torch.Tensor):
    def __repr__(self):
        step1 = torch.Tensor.__repr__(self)

        pos_paren1 = step1.index("(")
        prefix = "vector.tensor"
        ws_before = "\n" + (" " * pos_paren1)
        ws_after = "\n" + (" " * len(prefix))
        step2 = prefix + step1[pos_paren1:].replace(ws_before, ws_after)

        pos_paren2 = step2.rindex(")")
        eoln = "\n    " if "\n" in step2 else " "
        step3 = step2[:pos_paren2] + f",{eoln}{self._constructors()})"

        return step3

    @property
    def raw(self):
        return torch.Tensor.__new__(torch.Tensor, self)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # RIGHT HERE, we need to deal with all the types of functions:
        #   * 1 vector  -> non-vector      e.g. coordinate, abs, deltaphi
        #   * 1 vector  -> vector          e.g. rotation, scalar multiplication,
        #                                       coordinate transformation
        #   * 2 vectors -> non-vector      e.g. is_parallel, equal
        #   * 2 vectors -> vector          e.g. add, cross-product, boost
        #   * vectors are untouched        e.g. copy, dtype or device change
        #
        # Which functions with more than 1 vector arguments...
        #   * need to shuffle indexes to put them in the same index positions?
        #   * ~~need to convert coordinate systems~~ NO: compute funcs do that
        #   * need to project N-d to n-d?
        #   * need to complain if vector dimensions are mismatched?

        raw_args = [x.raw if isinstance(x, VectorTensor) else x for x in args]

        out = func(*raw_args, **kwargs)

        if isinstance(out, torch.Tensor):
            args[0]._apply_one(out)

        return out


class Vector2DTensor(VectorTensor):
    @staticmethod
    def __new__(cls, data, azimuthal_index, **kwargs):
        out = torch.Tensor.__new__(torch.Tensor, data, **kwargs)
        out.__class__ = cls
        out._azimuthal_index = azimuthal_index
        return out

    @property
    def azimuthal_index(self):
        return self._azimuthal_index

    def _constructors(self):
        return self._azimuthal_index._constructors()

    def _apply_one(self, out):
        out.__class__ = type(self)
        out._azimuthal_index = self._azimuthal_index
        return out
