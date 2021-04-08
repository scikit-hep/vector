<img alt="Vector logo" width="433" src="https://raw.githubusercontent.com/scikit-hep/vector/main/docs/_images/vector-logo.png"/>

# Vector

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]
[![Join the chat at https://gitter.im/Scikit-HEP/vector][gitter-badge]][gitter-link]
[![Code style: black][black-badge]](https://github.com/psf/black)
[![Scikit-HEP][sk-badge]](https://scikit-hep.org/)

Vector is a Python library for 2D, 3D, and [Lorentz vectors](https://en.wikipedia.org/wiki/Special_relativity#Physics_in_spacetime), especially _arrays of vectors_, to solve common physics problems in a NumPy-like way.

Main features of Vector:

   * Pure Python with NumPy as its only dependency. This makes it easier to install.
   * Vectors may be represented in a variety of coordinate systems: Cartesian, cylindrical, spherical, and any combination of these with time or proper time for Lorentz vectors. In all, there are 12 coordinate systems: {_x_-_y_ vs _ρ_-_φ_ in the azimuthal plane} × {_z_ vs _θ_ vs _η_ longitudinally} × {_t_ vs _τ_ temporally}.
   * Uses names and conventions set by [ROOT](https://root.cern/)'s [TLorentzVector](https://root.cern.ch/doc/master/classTLorentzVector.html) and [Math::LorentzVector](https://root.cern.ch/doc/master/classROOT_1_1Math_1_1LorentzVector.html), as well as [scikit-hep/math](https://github.com/scikit-hep/scikit-hep/tree/master/skhep/math), [uproot-methods TLorentzVector](https://github.com/scikit-hep/uproot3-methods/blob/master/uproot3_methods/classes/TLorentzVector.py), [henryiii/hepvector](https://github.com/henryiii/hepvector), and [coffea.nanoevents.methods.vector](https://coffeateam.github.io/coffea/modules/coffea.nanoevents.methods.vector.html).
   * Implemented on a variety of backends:
      - pure Python objects
      - NumPy arrays of vectors (as a [structured array](https://numpy.org/doc/stable/user/basics.rec.html) subclass)
      - [Awkward Arrays](https://awkward-array.org/) of vectors
      - potential for more: CuPy, TensorFlow, Torch, JAX...
   * Implemented in [Numba](https://numba.pydata.org/) for JIT-compiled calculations on vectors.
   * Distinction between geometrical vectors, which have a minimum of attribute and method names, and vectors representing momentum, which have synonyms like `pt` = `rho`, `energy` = `t`, `mass` = `tau`.

This overview is based on the [documentation here](https://vector.readthedocs.io/en/develop/usage/intro.html).


```python
import vector
import numpy as np
import awkward as ak  # at least version 1.2.0
import numba as nb
```

## Constructing a vector or an array of vectors

The easiest way to create one or many vectors is with a helper function:

   * `vector.obj` to make a pure Python vector object,
   * `vector.arr` to make a NumPy array of vectors (lowercase, like `np.array`),
   * `vector.awk` to make an Awkward Array of vectors (uppercase, like `ak.Array`).

### Pure Python vectors


```python
vector.obj(x=3, y=4)   # Cartesian 2D vector
vector.obj(rho=5, phi=0.9273)   # same in polar coordinates
vector.obj(x=3, y=4).isclose(vector.obj(rho=5, phi=0.9273))   # use "isclose" unless they are exactly equal
vector.obj(x=3, y=4, z=-2)   # Cartesian 3D vector
vector.obj(x=3, y=4, z=-2, t=10)   # Cartesian 4D vector
vector.obj(rho=5, phi=0.9273, eta=-0.39, t=10)   # in rho-phi-eta-t cylindrical coordinates
vector.obj(pt=5, phi=0.9273, eta=-0.39, E=10)   # use momentum-synonyms to get a momentum vector
vector.obj(rho=5, phi=0.9273, eta=-0.39, t=10) == vector.obj(pt=5, phi=0.9273, eta=-0.390035, E=10)
vector.obj(rho=5, phi=0.9273, eta=-0.39, t=10).tau   # geometrical vectors have to use geometrical names ("tau", not "mass")
vector.obj(pt=5, phi=0.9273, eta=-0.39, E=10).mass   # momentum vectors can use momentum names (as well as geometrical ones)
vector.obj(pt=5, phi=0.9273, theta=1.9513, mass=8.4262)   # any combination of azimuthal, longitudinal, and temporal coordinates is allowed
vector.obj(x=3, y=4, z=-2, t=10).isclose(vector.obj(pt=5, phi=0.9273, theta=1.9513, mass=8.4262))

# Test instance type for any level of granularity.
(
    isinstance(vector.obj(x=1.1, y=2.2), vector.Vector),                                    # is a vector or array of vectors
    isinstance(vector.obj(x=1.1, y=2.2), vector.Vector2D),                                  # is 2D (not 3D or 4D)
    isinstance(vector.obj(x=1.1, y=2.2), vector.VectorObject),                              # is a vector object (not an array)
    isinstance(vector.obj(px=1.1, py=2.2), vector.Momentum),                                # has momentum synonyms
    isinstance(vector.obj(x=1.1, y=2.2, z=3.3, t=4.4), vector.Planar),                      # has transverse plane (2D, 3D, or 4D)
    isinstance(vector.obj(x=1.1, y=2.2, z=3.3, t=4.4), vector.Spatial),                     # has all spatial coordinates (3D or 4D)
    isinstance(vector.obj(x=1.1, y=2.2, z=3.3, t=4.4), vector.Lorentz),                     # has temporal coordinates (4D)
    isinstance(vector.obj(x=1.1, y=2.2, z=3.3, t=4.4).azimuthal, vector.AzimuthalXY),       # azimuthal coordinate type
    isinstance(vector.obj(x=1.1, y=2.2, z=3.3, t=4.4).longitudinal, vector.LongitudinalZ),  # longitudinal coordinate type
    isinstance(vector.obj(x=1.1, y=2.2, z=3.3, t=4.4).temporal, vector.TemporalT),          # temporal coordinate type
)
```


The allowed keyword arguments for 2D vectors are:

   * `x` and `y` for Caresian azimuthal coordinates,
   * `px` and `py` for momentum,
   * `rho` and `phi` for polar azimuthal coordinates,
   * `pt` and `phi` for momentum.

For 3D vectors, you need the above and:

   * `z` for the Cartesian longitudinal coordinate,
   * `pz` for momentum,
   * `theta` for the spherical polar angle (from $0$ to $\pi$, inclusive),
   * `eta` for pseudorapidity, which is a kind of spherical polar angle.

For 4D vectors, you need the above and:

   * `t` for the Cartesian temporal coordinate,
   * `E` or `energy` to get four-momentum,
   * `tau` for the "proper time" (temporal coordinate in the vector's rest coordinate system),
   * `M` or `mass` to get four-moemtum.

Since momentum vectors have momentum-synonyms _in addition_ to the geometrical names, any momentum-synonym will make the whole vector a momentum vector.

If you want to bypass the dimension and coordinate system inference through keyword arguments (e.g. for static typing), you can use specialized constructors:


```python
vector.VectorObject2D.from_xy(1.1, 2.2)
vector.MomentumObject3D.from_rhophiz(1.1, 2.2, 3.3)
vector.VectorObject4D.from_xyetatau(1.1, 2.2, 3.3, 4.4)
```

and so on, for all combinations of azimuthal, longitudinal, and temporal coordinates, geometric and momentum-flavored.

### NumPy arrays of vectors


```python
# NumPy-like arguments (literally passed through to NumPy)
vector.arr([
    (1.1, 2.1), (1.2, 2.2), (1.3, 2.3), (1.4, 2.4), (1.5, 2.5)
], dtype=[("x", float), ("y", float)])

# Pandas-like arguments (dict from names to column arrays)
vector.arr({"x": [1.1, 1.2, 1.3, 1.4, 1.5], "y": [2.1, 2.2, 2.3, 2.4, 2.5]})

# As with objects, the coordinate system and dimension is taken from the names of the fields.
vector.arr({
    "x": [1.1, 1.2, 1.3, 1.4, 1.5],
    "y": [2.1, 2.2, 2.3, 2.4, 2.5],
    "z": [3.1, 3.2, 3.3, 3.4, 3.5],
    "t": [4.1, 4.2, 4.3, 4.4, 4.5],
})

vector.arr({
    "pt": [1.1, 1.2, 1.3, 1.4, 1.5],
    "phi": [2.1, 2.2, 2.3, 2.4, 2.5],
    "eta": [3.1, 3.2, 3.3, 3.4, 3.5],
    "M": [4.1, 4.2, 4.3, 4.4, 4.5],
})
```

Existing NumPy arrays can be viewed as arrays of vectors, but it needs to be a [structured array](https://numpy.org/doc/stable/user/basics.rec.html) with recognized field names.


```python
# NumPy array         # interpret groups of four values as named fields              # give it vector properties and methods
np.arange(0, 24, 0.1).view([("x", float), ("y", float), ("z", float), ("t", float)]).view(vector.VectorNumpy4D)
```


Since `VectorNumpy2D`, `VectorNumpy3D`, `VectorNumpy4D`, and their momentum equivalents are NumPy array subclasses, all of the normal NumPy methods and functions work on them.


```python
np.arange(0, 24, 0.1).view([("x", float), ("y", float), ("z", float), ("t", float)]).view(vector.VectorNumpy4D).reshape(6, 5, 2)
```


All of the keyword arguments and rules that apply to `vector.obj` construction apply to `vector.arr` dtypes.

Geometrical names are used in the dtype, even if momentum-synonyms are used in construction.


```python
vector.arr({"px": [1, 2, 3, 4], "py": [1.1, 2.2, 3.3, 4.4], "pz": [0.1, 0.2, 0.3, 0.4]})
```


### Awkward Arrays of vectors

[Awkward Arrays](https://awkward-array.org/) are arrays with more complex data structures than NumPy allows, such as variable-length lists, nested records, missing and even heterogeneous data (multiple data types: use sparingly).

The `vector.awk` function behaves exactly like the [ak.Array](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html) constructor, except that it makes arrays of vectors.


```python
vector.awk([
    [{"x": 1, "y": 1.1, "z": 0.1}, {"x": 2, "y": 2.2, "z": 0.2}],
    [],
    [{"x": 3, "y": 3.3, "z": 0.3}],
    [{"x": 4, "y": 4.4, "z": 0.4}, {"x": 5, "y": 5.5, "z": 0.5}, {"x": 6, "y": 6.6, "z": 0.6}],
])
```


If you want _any_ records named "`Vector2D`", "`Vector3D`", "`Vector4D`", "`Momentum2D`", "`Momentum3D`", or "`Momentum4D`" to be interpreted as vectors, register the behaviors globally.


```python
vector.register_awkward()

ak.Array([
    [{"x": 1, "y": 1.1, "z": 0.1}, {"x": 2, "y": 2.2, "z": 0.2}],
    [],
    [{"x": 3, "y": 3.3, "z": 0.3}],
    [{"x": 4, "y": 4.4, "z": 0.4}, {"x": 5, "y": 5.5, "z": 0.5}, {"x": 6, "y": 6.6, "z": 0.6}],
],
    with_name="Vector3D"
)
```






All of the keyword arguments and rules that apply to `vector.obj` construction apply to `vector.awk` field names.

## Vector properties

Any geometrical coordinate can be computed from vectors in any coordinate system; they'll be provided or computed as needed.


```python
vector.obj(x=3, y=4).rho
vector.obj(rho=5, phi=0.9273).x
vector.obj(rho=5, phi=0.9273).y
vector.obj(x=1, y=2, z=3).theta
vector.obj(x=1, y=2, z=3).eta
```


Some properties are not coordinates, but derived from them.


```python
vector.obj(x=1, y=2, z=3).costheta
vector.obj(x=1, y=2, z=3).mag   # spatial magnitude
vector.obj(x=1, y=2, z=3).mag2   # spatial magnitude squared
```


These properties are provided because they can be computed faster or with more numerical stability in different coordinate systems. For instance, the magnitude ignores `phi` in polar coordinates.


```python
vector.obj(rho=3, phi=0.123456789, z=4).mag2
```

Momentum vectors have geometrical properties as well as their momentum-synonyms.


```python
vector.obj(px=3, py=4).rho
vector.obj(px=3, py=4).pt
vector.obj(x=1, y=2, z=3, E=4).tau
vector.obj(x=1, y=2, z=3, E=4).mass
```




Here's the key thing: _arrays of vectors return arrays of coordinates_.


```python
vector.arr({
    "x": [1.0, 2.0, 3.0, 4.0, 5.0],
    "y": [1.1, 2.2, 3.3, 4.4, 5.5],
    "z": [0.1, 0.2, 0.3, 0.4, 0.5],
}).theta

vector.awk([
    [{"x": 1, "y": 1.1, "z": 0.1}, {"x": 2, "y": 2.2, "z": 0.2}],
    [],
    [{"x": 3, "y": 3.3, "z": 0.3}],
    [{"x": 4, "y": 4.4, "z": 0.4}, {"x": 5, "y": 5.5, "z": 0.5}],
]).theta

# Make a large, random NumPy array of 3D momentum vectors.
array = np.random.normal(0, 1, 150).view([(x, float) for x in ("x", "y", "z")]).view(vector.MomentumNumpy3D).reshape(5, 5, 2)

# Get the transverse momentum of each one.
array.pt

# The array and its components have the same shape.
array.shape
array.pt.shape

# Make a large, random Awkward Array of 3D momentum vectors.
array = vector.awk([[{x: np.random.normal(0, 1) for x in ("px", "py", "pz")} for inner in range(np.random.poisson(1.5))] for outer in range(50)])

# Get the transverse momentum of each one, in the same nested structure.
array.pt

# The array and its components have the same list lengths (and can therefore be used together in subsequent calculations).
ak.num(array)
ak.num(array.pt)
```


## Vector methods

Vector methods require arguments (in parentheses), which may be scalars or other vectors, depending on the calculation.


```python
vector.obj(x=3, y=4).rotateZ(0.1)
vector.obj(rho=5, phi=0.4).rotateZ(0.1)

# Broadcasts a scalar rotation angle of 0.5 to all elements of the NumPy array.
print(vector.arr({"rho": [1, 2, 3, 4, 5], "phi": [0.1, 0.2, 0.3, 0.4, 0.5]}).rotateZ(0.5))

# Matches each rotation angle to an element of the NumPy array.
print(vector.arr({"rho": [1, 2, 3, 4, 5], "phi": [0.1, 0.2, 0.3, 0.4, 0.5]}).rotateZ(np.array([0.1, 0.2, 0.3, 0.4, 0.5])))

# Broadcasts a scalar rotation angle of 0.5 to all elements of the Awkward Array.
print(vector.awk([[{"rho": 1, "phi": 0.1}, {"rho": 2, "phi": 0.2}], [], [{"rho": 3, "phi": 0.3}]]).rotateZ(0.5))

# Broadcasts a rotation angle of 0.1 to both elements of the first list, 0.2 to the empty list, and 0.3 to the only element of the last list.
print(vector.awk([[{"rho": 1, "phi": 0.1}, {"rho": 2, "phi": 0.2}], [], [{"rho": 3, "phi": 0.3}]]).rotateZ([0.1, 0.2, 0.3]))

# Matches each rotation angle to an element of the Awkward Array.
print(vector.awk([[{"rho": 1, "phi": 0.1}, {"rho": 2, "phi": 0.2}], [], [{"rho": 3, "phi": 0.3}]]).rotateZ([[0.1, 0.2], [], [0.3]]))
```


Some methods are equivalent to binary operators.


```python
vector.obj(x=3, y=4).scale(10)
vector.obj(x=3, y=4) * 10
10 * vector.obj(x=3, y=4)
vector.obj(rho=5, phi=0.5) * 10
```


Some methods involve more than one vector.


```python
vector.obj(x=1, y=2).add(vector.obj(x=5, y=5))
vector.obj(x=1, y=2) + vector.obj(x=5, y=5)
vector.obj(x=1, y=2).dot(vector.obj(x=5, y=5))
vector.obj(x=1, y=2) @ vector.obj(x=5, y=5)
```


The vectors can use different coordinate systems. Conversions are necessary, but minimized for speed and numeric stability.


```python
vector.obj(x=3, y=4) @ vector.obj(x=6, y=8)   # both are Cartesian, dot product is exact
vector.obj(rho=5, phi=0.9273) @ vector.obj(x=6, y=8)   # one is polar, dot product is approximate
vector.obj(x=3, y=4) @ vector.obj(rho=10, phi=0.9273)   # one is polar, dot product is approximate
vector.obj(rho=5, phi=0.9273) @ vector.obj(rho=10, phi=0.9273)   # both are polar, a formula that depends on phi differences is used
```




In Python, some "operators" are actually built-in functions, such as `abs`.


```python
abs(vector.obj(x=3, y=4))
```




Note that `abs` returns

   * `rho` for 2D vectors
   * `mag` for 3D vectors
   * `tau` (`mass`) for 4D vectors

Use the named properties when you want magnitude in a specific number of dimensions; use `abs` when you want the magnitude for any number of dimensions.

The vectors can be from different backends. Normal rules for broadcasting Python numbers, NumPy arrays, and Awkward Arrays apply.


```python
vector.arr({"x": [1, 2, 3, 4, 5], "y": [0.1, 0.2, 0.3, 0.4, 0.5]}) + vector.obj(x=10, y=5)

(
    vector.awk([                                                   # an Awkward Array of vectors
        [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}],
        [],
        [{"x": 3, "y": 3.3}],
        [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}],
    ])
    + vector.obj(x=10, y=5)                                          # and a single vector object
)

(
    vector.awk([                                                   # an Awkward Array of vectors
        [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}],
        [],
        [{"x": 3, "y": 3.3}],
        [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}],
    ])
    + vector.arr({"x": [4, 3, 2, 1], "y": [0.1, 0.1, 0.1, 0.1]})   # and a NumPy array of vectors
)
```


Some operations are defined for 2D or 3D vectors, but are usable on higher-dimensional vectors because the additional components can be ignored or are passed through unaffected.


```python
vector.obj(rho=1, phi=0.5).deltaphi(vector.obj(rho=2, phi=0.3))   # deltaphi is a planar operation (defined on the transverse plane)
vector.obj(rho=1, phi=0.5, z=10).deltaphi(vector.obj(rho=2, phi=0.3, theta=1.4))   # but we can use it on 3D vectors
vector.obj(rho=1, phi=0.5, z=10, t=100).deltaphi(vector.obj(rho=2, phi=0.3, theta=1.4, tau=1000))   # and 4D vectors
vector.obj(rho=1, phi=0.5).deltaphi(vector.obj(rho=2, phi=0.3, theta=1.4, tau=1000))   # and mixed dimensionality
```

This is especially useful for giving 4D vectors all the capabilities of 3D vectors.


```python
vector.obj(x=1, y=2, z=3).rotateX(np.pi/4)
vector.obj(x=1, y=2, z=3, tau=10).rotateX(np.pi/4)
vector.obj(pt=1, phi=1.3, eta=2).deltaR(vector.obj(pt=2, phi=0.3, eta=1))
vector.obj(pt=1, phi=1.3, eta=2, mass=5).deltaR(vector.obj(pt=2, phi=0.3, eta=1, mass=10))
```


The opposite—using low-dimensional vectors in operations defined for higher numbers of dimensions—is sometimes defined. In these cases, a zero longitudinal or temporal component has to be imputed.


```python
vector.obj(x=1, y=2, z=3) - vector.obj(x=1, y=2)
vector.obj(x=1, y=2, z=0).is_parallel(vector.obj(x=1, y=2))
```


And finally, in some cases, the function excludes a higher-dimensional component, even if the input vectors had them.

It would be confusing if the 3D cross-product returned a fourth component.


```python
vector.obj(x=0.1, y=0.2, z=0.3, t=10).cross(vector.obj(x=0.4, y=0.5, z=0.6, t=20))
```


The (current) list of properties and methods is:

**Planar (2D, 3D, 4D):**

   * `x` (`px`)
   * `y` (`py`)
   * `rho` (`pt`): two-dimensional magnitude
   * `rho2` (`pt2`): two-dimensional magnitude squared
   * `phi`
   * `deltaphi(vector)`: difference in `phi` (signed and rectified to $-\pi$ through $\pi$)
   * `rotateZ(angle)`
   * `transform2D(obj)`: the `obj` must supply components through `obj["xx"]`, `obj["xy"]`, `obj["yx"]`, `obj["yy"]`
   * `is_parallel(vector, tolerance=1e-5)`: only true _if they're pointing in the same direction_
   * `is_antiparallel(vector, tolerance=1e-5)`: only true _if they're pointing in opposite directions_
   * `is_perpendicular(vector, tolerance=1e-5)`

**Spatial (3D, 4D):**

   * `z` (`pz`)
   * `theta`
   * `eta`
   * `costheta`
   * `cottheta`
   * `mag` (`p`): three-dimensional magnitude, does not include temporal component
   * `mag2` (`p2`): three-dimensional magnitude squared
   * `cross`: cross-product (strictly 3D)
   * `deltaangle(vector)`: difference in angle (always non-negative)
   * `deltaeta(vector)`: difference in `eta` (signed)
   * `deltaR(vector)`: $\Delta R = \sqrt{\Delta\phi^2 + \Delta\eta^2}$
   * `deltaR2(vector)`: the above, squared
   * `rotateX(angle)`
   * `rotateY(angle)`
   * `rotate_axis(axis, angle)`: the magnitude of `axis` is ignored, but it must be at least 3D
   * `rotate_euler(phi, theta, psi, order="zxz")`: the arguments are in the same order as [ROOT::Math::EulerAngles](https://root.cern.ch/doc/master/classROOT_1_1Math_1_1EulerAngles.html), and `order="zxz"` agrees with ROOT's choice of conventions
   * `rotate_nautical(yaw, pitch, roll)`
   * `rotate_quaternion(u, i, j, k)`: again, the conventions match [ROOT::Math::Quaternion](https://root.cern.ch/doc/master/classROOT_1_1Math_1_1Quaternion.html).
   * `transform3D(obj)`: the `obj` must supply components through `obj["xx"]`, `obj["xy"]`, etc.
   * `is_parallel(vector, tolerance=1e-5)`: only true _if they're pointing in the same direction_
   * `is_antiparallel(vector, tolerance=1e-5)`: only true _if they're pointing in opposite directions_
   * `is_perpendicular(vector, tolerance=1e-5)`

**Lorentz (4D only):**

   * `t` (`E`, `energy`): follows the [ROOT::Math::LorentzVector](https://root.cern/doc/master/LorentzVectorPage.html) behavior of treating spacelike vectors as negative `t` and negative `tau` and truncating wrong-direction timelike vectors
   * `t2` (`E2`, `energy2`)
   * `tau` (`M`, `mass`): see note above
   * `tau2` (`M2`, `mass2`)
   * `beta`: scalar(s) between $0$ (inclusive) and $1$ (exclusive, unless the vector components are infinite)
   * `gamma`: scalar(s) between $1$ (inclusive) and $\infty$
   * `rapidity`: scalar(s) between $0$ (inclusive) and $\infty$
   * `boost_p4(four_vector)`: change coordinate system using another 4D vector as the difference
   * `boost_beta(three_vector)`: change coordinate system using a 3D beta vector (all components between $-1$ and $+1$)
   * `boost(vector)`: uses the dimension of the given `vector` to determine behavior
   * `boostX(beta=None, gamma=None)`: supply `beta` xor `gamma`, but not both
   * `boostY(beta=None, gamma=None)`: supply `beta` xor `gamma`, but not both
   * `boostZ(beta=None, gamma=None)`: supply `beta` xor `gamma`, but not both
   * `transform4D(obj)`: the `obj` must supply components through `obj["xx"]`, `obj["xy"]`, etc.
   * `to_beta3()`: turns a `four_vector` (for `boost_p4`) into a `three_vector` (for `boost_beta3`)
   * `is_timelike(tolerance=0)`
   * `is_spacelike(tolerance=0)`
   * `is_lightlike(tolerance=1e-5)`: note the different tolerance

**All numbers of dimensions:**

   * `unit()`: note the parentheses
   * `dot(vector)`: can also use the `@` operator
   * `add(vector)`: can also use the `+` operator
   * `subtract(vector)`: can also use the `-` operator
   * `scale(factor)`: can also use the `*` operator
   * `equal(vector)`: can also use the `==` operator, but consider `isclose` instead
   * `not_equal(vector)`: can also use the `!=` operator, but consider `isclose` instead
   * `isclose(vector, rtol=1e-5, atol=1e-8, equal_nan=False)`: works like [np.isclose](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html); arrays also have an [allclose](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html) method

## Compiling your Python with Numba

[Numba](https://numba.pydata.org/) is a just-in-time (JIT) compiler for a mathematically relevant subset of NumPy and Python. It allows you to write fast code without leaving the Python environment. The drawback of Numba is that it can only compile code blocks involving objects and functions that it recognizes.

The Vector library includes extensions to inform Numba about vector objects, vector NumPy arrays, and vector Awkward Arrays. At the time of writing, the implementation of vector NumPy arrays is incomplete due to [numba/numba#6148](https://github.com/numba/numba/pull/6148).

For instance, consider the following function:


```python
@nb.njit
def compute_mass(v1, v2):
    return (v1 + v2).mass

compute_mass(vector.obj(px=1, py=2, pz=3, E=4), vector.obj(px=-1, py=-2, pz=-3, E=4))
```




When the two `MomentumObject4D` objects are passed as arguments, Numba recognizes them and replaces the Python objects with low-level structs. When it compiles the function, it recognizes `+` as the 4D `add` function and recognizes `.mass` as the `tau` component of the result.

Although this demonstrates that Numba can manipulate vector objects, there is no performance advantage (and a likely disadvantage) to compiling a calculation on just a few vectors. The advantage comes when many vectors are involved, in arrays.


```python
# This is still not a large number. You want millions.
array = vector.awk([[dict({x: np.random.normal(0, 1) for x in ("px", "py", "pz")}, E=np.random.normal(10, 1)) for inner in range(np.random.poisson(1.5))] for outer in range(50)])

@nb.njit
def compute_masses(array):
    out = np.empty(len(array), np.float64)
    for i, event in enumerate(array):
        total = vector.obj(px=0.0, py=0.0, pz=0.0, E=0.0)
        for vec in event:
            total = total + vec
        out[i] = total.mass
    return out


compute_masses(array)
```


### Status as of March 30, 2020

Undoubtedly, there are rough edges, but most of the functionality is there and Vector is ready for user-testing. It can only be improved by your feedback!



See [CONTRIBUTING.md](./.github/CONTRIBUTING.md)
for information on setting up a development environment.



[gitter-badge]:  https://badges.gitter.im/Scikit-HEP/vector.svg
[gitter-link]:   https://gitter.im/Scikit-HEP/vector?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
[actions-badge]: https://github.com/scikit-hep/vector/workflows/CI/badge.svg
[actions-link]:  https://github.com/scikit-hep/vector/actions
[rtd-badge]:     https://readthedocs.org/projects/vector/badge/?version=latest
[rtd-link]:      https://vector.readthedocs.io/en/latest/?badge=latest
[sk-badge]:      https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg
[black-badge]:   https://img.shields.io/badge/code%20style-black-000000.svg
