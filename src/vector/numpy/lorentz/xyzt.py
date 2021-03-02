# Copyright (c) 2019-2020, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.


from typing import TYPE_CHECKING, Any, cast

import vector.mixins.lorentz.xyzt
from vector.core import numpy as np

if TYPE_CHECKING:
    ArrayLike = Any


class LorentzXYZT(
    vector.mixins.lorentz.xyzt.LorentzXYZTMethodMixin,
    vector.mixins.lorentz.xyzt.LorentzXYZTDunderMixin,
):
    def __init__(self, x, y, z, t):
        # type: (ArrayLike, ArrayLike, ArrayLike, ArrayLike) -> None
        """
        Notes
        =====

        For now, all arrays are broadcast here - in the future, arrays may remain unbroadcast until an action is taken.
        """

        self.x, self.y, self.z, self.t = np.broadcast_arrays(x, y, z, t)

    def __repr__(self):
        # type: () -> str
        return f"Lxyz({self.x}, {self.y}, {self.z}, {self.t})"

    def __getitem__(self, attr):
        # type: (str) -> ArrayLike
        # It has to behave the same way as the bound objects or users will get confused.
        if attr in ("x", "y", "z", "t"):
            return getattr(self, attr)
        else:
            raise ValueError(f"key {attr} does not exist in x,y,z,t")


if TYPE_CHECKING:
    from vector.protocols.lorentz import LorentzVector

    _ = cast(LorentzXYZT, None)  # type: LorentzVector
