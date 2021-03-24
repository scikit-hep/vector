# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.


from .backends.numpy_ import (  # noqa: 401
    MomentumNumpy2D,
    MomentumNumpy3D,
    MomentumNumpy4D,
    VectorNumpy2D,
    VectorNumpy3D,
    VectorNumpy4D,
    array,
)
from .backends.object_ import (  # noqa: 401
    MomentumObject2D,
    MomentumObject3D,
    MomentumObject4D,
    VectorObject2D,
    VectorObject3D,
    VectorObject4D,
    obj,
)
from .methods import (  # noqa: 401
    Azimuthal,
    AzimuthalRhoPhi,
    AzimuthalXY,
    Coordinates,
    Longitudinal,
    LongitudinalEta,
    LongitudinalTheta,
    LongitudinalZ,
    Momentum,
    Temporal,
    TemporalT,
    TemporalTau,
    Vector,
    Vector2D,
    Vector3D,
    Vector4D,
    dim,
)

# from .version import version as __version__

# __all__ = ("__version__",)
