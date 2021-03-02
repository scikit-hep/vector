# Copyright (c) 2019-2020, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.


import json
from typing import TYPE_CHECKING, cast

import vector.mixins.lorentz.xyzt


class LorentzXYZTFree(
    vector.mixins.lorentz.xyzt.LorentzXYZTMethodMixin,
    vector.mixins.lorentz.xyzt.LorentzXYZTDunderMixin,
):
    def __init__(self, x, y, z, t):
        # type: (float, float, float, float) -> None
        self.x = x
        self.y = y
        self.z = z
        self.t = t

    def __repr__(self):
        # type: () -> str
        return "Lxyz({:.3g} {:.3g} {:.3g} {:.3g})".format(
            self.x, self.y, self.z, self.t
        )

    def __getitem__(self, attr):
        # type: (str) -> float
        # It has to behave the same way as the bound objects or users will get confused.
        if attr in ("x", "y", "z", "t"):
            return getattr(self, attr)
        else:
            raise ValueError(
                "key {} does not exist (not in record)".format(json.dumps(attr))
            )


# This is a "standard" trick to validate a Protocol.
# Standard is in quotes because the PEP literally says don't do it this way, but
# since a nicer way was never added, this is the way everyone recommends doing it.
# Think of it like instantiating an ABC, but at type-check time.
if TYPE_CHECKING:
    from vector.protocols.lorentz import LorentzVector

    _ = cast(LorentzXYZTFree, None)  # type: LorentzVector
