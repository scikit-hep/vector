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


def register_numba():
    import vector.backends.numba_numpy  # noqa: 401
    import vector.backends.numba_object  # noqa: 401


_awkward_registered = False


def register_awkward():
    import awkward

    import vector.backends.awkward_  # noqa: 401

    global _awkward_registered
    awkward.behavior.update(vector.backends.awkward_.behavior)
    _awkward_registered = True


def Array(*args, **kwargs):
    "vector.Array docs"
    import awkward

    import vector.backends.awkward_  # noqa: 401

    akarray = awkward.Array(*args, **kwargs)
    fields = awkward.fields(akarray)

    complaint1 = "duplicate coordinates (through momentum-aliases): " + ", ".join(
        repr(x) for x in fields
    )
    complaint2 = (
        "unrecognized combination of coordinates, allowed combinations are:\n\n"
        "    (2D) x= y=\n"
        "    (2D) rho= phi=\n"
        "    (3D) x= y= z=\n"
        "    (3D) x= y= theta=\n"
        "    (3D) x= y= eta=\n"
        "    (3D) rho= phi= z=\n"
        "    (3D) rho= phi= theta=\n"
        "    (3D) rho= phi= eta=\n"
        "    (4D) x= y= z= t=\n"
        "    (4D) x= y= z= tau=\n"
        "    (4D) x= y= theta= t=\n"
        "    (4D) x= y= theta= tau=\n"
        "    (4D) x= y= eta= t=\n"
        "    (4D) x= y= eta= tau=\n"
        "    (4D) rho= phi= z= t=\n"
        "    (4D) rho= phi= z= tau=\n"
        "    (4D) rho= phi= theta= t=\n"
        "    (4D) rho= phi= theta= tau=\n"
        "    (4D) rho= phi= eta= t=\n"
        "    (4D) rho= phi= eta= tau="
    )

    is_momentum = False
    dimension = 0
    names = []
    arrays = []

    if "x" in fields and "y" in fields:
        if dimension != 0:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 2
        names.extend(["x", "y"])
        arrays.extend([akarray["x"], akarray["y"]])
        fields.remove("x")
        fields.remove("y")
    if "rho" in fields and "phi" in fields:
        if dimension != 0:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 2
        names.extend(["rho", "phi"])
        arrays.extend([akarray["rho"], akarray["phi"]])
        fields.remove("rho")
        fields.remove("phi")
    if "x" in fields and "py" in fields:
        is_momentum = True
        if dimension != 0:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 2
        names.extend(["x", "y"])
        arrays.extend([akarray["x"], akarray["py"]])
        fields.remove("x")
        fields.remove("py")
    if "px" in fields and "y" in fields:
        is_momentum = True
        if dimension != 0:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 2
        names.extend(["x", "y"])
        arrays.extend([akarray["px"], akarray["y"]])
        fields.remove("px")
        fields.remove("y")
    if "px" in fields and "py" in fields:
        is_momentum = True
        if dimension != 0:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 2
        names.extend(["x", "y"])
        arrays.extend([akarray["px"], akarray["py"]])
        fields.remove("px")
        fields.remove("py")
    if "pt" in fields and "phi" in fields:
        is_momentum = True
        if dimension != 0:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 2
        names.extend(["rho", "phi"])
        arrays.extend([akarray["pt"], akarray["phi"]])
        fields.remove("pt")
        fields.remove("phi")

    if "z" in fields:
        if dimension != 2:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 3
        names.append("z")
        arrays.append(akarray["z"])
        fields.remove("z")
    if "theta" in fields:
        if dimension != 2:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 3
        names.append("theta")
        arrays.append(akarray["theta"])
        fields.remove("theta")
    if "eta" in fields:
        if dimension != 2:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 3
        names.append("eta")
        arrays.append(akarray["eta"])
        fields.remove("eta")
    if "pz" in fields:
        is_momentum = True
        if dimension != 2:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 3
        names.append("z")
        arrays.append(akarray["pz"])
        fields.remove("pz")

    if "t" in fields:
        if dimension != 3:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 4
        names.append("t")
        arrays.append(akarray["t"])
        fields.remove("t")
    if "tau" in fields:
        if dimension != 3:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 4
        names.append("tau")
        arrays.append(akarray["tau"])
        fields.remove("tau")
    if "E" in fields:
        is_momentum = True
        if dimension != 3:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 4
        names.append("t")
        arrays.append(akarray["E"])
        fields.remove("E")
    if "energy" in fields:
        is_momentum = True
        if dimension != 3:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 4
        names.append("t")
        arrays.append(akarray["energy"])
        fields.remove("energy")
    if "M" in fields:
        is_momentum = True
        if dimension != 3:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 4
        names.append("tau")
        arrays.append(akarray["M"])
        fields.remove("M")
    if "mass" in fields:
        is_momentum = True
        if dimension != 3:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 4
        names.append("tau")
        arrays.append(akarray["mass"])
        fields.remove("mass")

    if dimension == 0:
        raise TypeError(complaint1 if is_momentum else complaint2)

    for name in fields:
        names.append(name)
        arrays.append(akarray[name])

    needs_behavior = not _awkward_registered
    for x in arrays:
        if needs_behavior:
            if x.behavior is None:
                x.behavior = vector.backends.awkward_.behavior
            else:
                x.behavior = dict(x.behavior)
                x.behavior.update(vector.backends.awkward_.behavior)
        else:
            x.behavior = None
        needs_behavior = False

    depth = akarray.layout.purelist_depth

    if dimension == 2 and not is_momentum:
        recname = "Vector2D"
    elif dimension == 2 and is_momentum:
        recname = "Momentum2D"
    elif dimension == 3 and not is_momentum:
        recname = "Vector3D"
    elif dimension == 3 and is_momentum:
        recname = "Momentum3D"
    elif dimension == 4 and not is_momentum:
        recname = "Vector4D"
    elif dimension == 4 and is_momentum:
        recname = "Momentum4D"

    return awkward.zip(dict(zip(names, arrays)), depth_limit=depth, with_name=recname)


# __all__ = ("__version__",)
