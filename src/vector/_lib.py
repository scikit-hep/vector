from __future__ import annotations

import types
import typing

from vector._typeutils import FloatArray

if typing.TYPE_CHECKING:
    import sympy


class Lib:
    def __init__(self, lib: types.ModuleType):
        if lib.__name__ == "sympy":
            import sympy

            self.lib: types.ModuleType = sympy
        elif lib.__name__ == "numpy":
            import numpy

            self.lib = numpy
        assert self.lib is not None

    def sqrt(self, val: FloatArray | sympy.Symbol) -> FloatArray | sympy.Symbol:
        return self.lib.sqrt(val)

    def nan_to_num(
        self, val: FloatArray | sympy.Symbol, **kwargs: int | float
    ) -> FloatArray | sympy.Symbol:
        if self.lib.__name__ == "sympy":
            return val
        else:
            return self.lib.nan_to_num(val, **kwargs)

    def sin(self, val: FloatArray | sympy.Symbol) -> FloatArray | sympy.Symbol:
        return self.lib.sin(val)

    def cos(self, val: FloatArray | sympy.Symbol) -> FloatArray | sympy.Symbol:
        return self.lib.cos(val)

    def tan(self, val: FloatArray | sympy.Symbol) -> FloatArray | sympy.Symbol:
        return self.lib.tan(val)

    def sinh(self, val: FloatArray | sympy.Symbol) -> FloatArray | sympy.Symbol:
        return self.lib.sinh(val)

    def cosh(self, val: FloatArray | sympy.Symbol) -> FloatArray | sympy.Symbol:
        return self.lib.cosh(val)

    def maximum(
        self, val1: FloatArray | sympy.Symbol, val2: FloatArray | sympy.Symbol
    ) -> FloatArray | sympy.Symbol:
        return self.lib.maximum(val1, val2)

    def arctan2(
        self, val1: FloatArray | sympy.Symbol, val2: FloatArray | sympy.Symbol
    ) -> FloatArray | sympy.Symbol:
        if self.lib.__name__ == "sympy":
            return self.lib.atan2(val1, val2)
        else:
            return self.lib.arctan2(val1, val2)

    def arcsinh(self, val: FloatArray | sympy.Symbol) -> FloatArray | sympy.Symbol:
        if self.lib.__name__ == "sympy":
            return self.lib.asinh(val)
        else:
            return self.lib.arcsinh(val)

    def arccos(self, val: FloatArray | sympy.Symbol) -> FloatArray | sympy.Symbol:
        if self.lib.__name__ == "sympy":
            return self.lib.acos(val)
        else:
            return self.lib.arccos(val)

    def arctan(self, val: FloatArray | sympy.Symbol) -> FloatArray | sympy.Symbol:
        if self.lib.__name__ == "sympy":
            return self.lib.atan(val)
        else:
            return self.lib.arctan(val)

    def absolute(self, val: FloatArray | sympy.Symbol) -> FloatArray | sympy.Symbol:
        if self.lib.__name__ == "sympy":
            return self.lib.Abs(val)
        else:
            return self.lib.absolute(val)

    def log(self, val: FloatArray | sympy.Symbol) -> FloatArray | sympy.Symbol:
        return self.lib.log(val)

    @property
    def pi(self) -> float | sympy.Symbol:
        return self.lib.pi

    def exp(self, val: FloatArray | sympy.Symbol) -> FloatArray | sympy.Symbol:
        return self.lib.exp(val)

    @property
    def inf(self) -> float | sympy.Symbol:
        if self.lib.__name__ == "sympy":
            return self.lib.oo
        else:
            return self.lib.inf

    def isclose(
        self,
        val1: FloatArray | sympy.Symbol,
        val2: FloatArray | sympy.Symbol,
        *args: float,
    ) -> FloatArray | sympy.Equality:
        if self.lib.__name__ == "sympy":
            return self.lib.Eq(val1, val2)
        else:
            return self.lib.isclose(val1, val2, *args)

    def copysign(
        self, val1: FloatArray | sympy.Symbol, val2: FloatArray | sympy.Symbol
    ) -> FloatArray | sympy.Symbol:
        if self.lib.__name__ == "sympy":
            return val1
        else:
            return self.lib.copysign(val1, val2)

    def sign(self, val: FloatArray | sympy.Symbol) -> FloatArray | sympy.Symbol:
        if self.lib.__name__ == "sympy":
            return val
        else:
            return self.lib.sign(val)
