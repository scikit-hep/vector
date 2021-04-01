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

Vector is a Python library for 2D, 3D, and [Lorentz vectors](https://en.wikipedia.org/wiki/Special_relativity#Physics_in_spacetime), especially _arrays of vectors_, to solve common physics problems in a NumPy-like way.


```python
import vector
import numpy as np
import awkward as ak   # at least version 1.2.0rc5
import numba as nb
```

## Constructing a vector or an array of vectors

The easiest way to create one or many vectors is with a helper function:

   * `vector.obj` to make a pure Python vector object,
   * `vector.array` to make a NumPy array of vectors (lowercase, like `np.array`),
   * `vector.Array` to make an Awkward Array of vectors (uppercase, like `ak.Array`).

### Pure Python vectors


```python
vector.obj(x=3, y=4)   # Cartesian 2D vector
```




    vector.obj(x=3, y=4)




```python
vector.obj(rho=5, phi=0.9273)   # same in polar coordinates
```




    vector.obj(rho=5, phi=0.9273)




```python
vector.obj(x=3, y=4).isclose(vector.obj(rho=5, phi=0.9273))   # use "isclose" unless they are exactly equal
```




    True




```python
vector.obj(x=3, y=4, z=-2)   # Cartesian 3D vector
```




    vector.obj(x=3, y=4, z=-2)




```python
vector.obj(x=3, y=4, z=-2, t=10)   # Cartesian 4D vector
```




    vector.obj(x=3, y=4, z=-2, t=10)




```python
vector.obj(rho=5, phi=0.9273, eta=-0.39, t=10)   # in rho-phi-eta-t cylindrical coordinates
```




    vector.obj(rho=5, phi=0.9273, eta=-0.39, t=10)




```python
vector.obj(pt=5, phi=0.9273, eta=-0.39, E=10)   # use momentum-synonyms to get a momentum vector
```




    vector.obj(pt=5, phi=0.9273, eta=-0.39, E=10)




```python
vector.obj(rho=5, phi=0.9273, eta=-0.39, t=10) == vector.obj(pt=5, phi=0.9273, eta=-0.390035, E=10)
```




    False




```python
vector.obj(rho=5, phi=0.9273, eta=-0.39, t=10).tau   # geometrical vectors have to use geometrical names ("tau", not "mass")
```




    8.426194916448265




```python
vector.obj(pt=5, phi=0.9273, eta=-0.39, E=10).mass   # momentum vectors can use momentum names (as well as geometrical ones)
```




    8.426194916448265




```python
vector.obj(pt=5, phi=0.9273, theta=1.9513, mass=8.4262)   # any combination of azimuthal, longitudinal, and temporal coordinates is allowed
```




    vector.obj(pt=5, phi=0.9273, theta=1.9513, mass=8.4262)




```python
vector.obj(x=3, y=4, z=-2, t=10).isclose(vector.obj(pt=5, phi=0.9273, theta=1.9513, mass=8.4262))
```




    True




```python
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




    (True, True, True, True, True, True, True, True, True, True)



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
```




    vector.obj(x=1.1, y=2.2)




```python
vector.MomentumObject3D.from_rhophiz(1.1, 2.2, 3.3)
```




    vector.obj(pt=1.1, phi=2.2, pz=3.3)




```python
vector.VectorObject4D.from_xyetatau(1.1, 2.2, 3.3, 4.4)
```




    vector.obj(x=1.1, y=2.2, eta=3.3, tau=4.4)



and so on, for all combinations of azimuthal, longitudinal, and temporal coordinates, geometric and momentum-flavored.

### NumPy arrays of vectors


```python
# NumPy-like arguments (literally passed through to NumPy)
vector.array([
    (1.1, 2.1), (1.2, 2.2), (1.3, 2.3), (1.4, 2.4), (1.5, 2.5)
], dtype=[("x", float), ("y", float)])
```




    VectorNumpy2D([(1.1, 2.1), (1.2, 2.2), (1.3, 2.3), (1.4, 2.4), (1.5, 2.5)],
                  dtype=[('x', '<f8'), ('y', '<f8')])




```python
# Pandas-like arguments (dict from names to column arrays)
vector.array({"x": [1.1, 1.2, 1.3, 1.4, 1.5], "y": [2.1, 2.2, 2.3, 2.4, 2.5]})
```




    VectorNumpy2D([(1.1, 2.1), (1.2, 2.2), (1.3, 2.3), (1.4, 2.4), (1.5, 2.5)],
                  dtype=[('x', '<f8'), ('y', '<f8')])




```python
# As with objects, the coordinate system and dimension is taken from the names of the fields.
vector.array({
    "x": [1.1, 1.2, 1.3, 1.4, 1.5],
    "y": [2.1, 2.2, 2.3, 2.4, 2.5],
    "z": [3.1, 3.2, 3.3, 3.4, 3.5],
    "t": [4.1, 4.2, 4.3, 4.4, 4.5],
})
```




    VectorNumpy4D([(1.1, 2.1, 3.1, 4.1), (1.2, 2.2, 3.2, 4.2), (1.3, 2.3, 3.3, 4.3),
                   (1.4, 2.4, 3.4, 4.4), (1.5, 2.5, 3.5, 4.5)],
                  dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('t', '<f8')])




```python
vector.array({
    "pt": [1.1, 1.2, 1.3, 1.4, 1.5],
    "phi": [2.1, 2.2, 2.3, 2.4, 2.5],
    "eta": [3.1, 3.2, 3.3, 3.4, 3.5],
    "M": [4.1, 4.2, 4.3, 4.4, 4.5],
})
```




    MomentumNumpy4D([(1.1, 2.1, 3.1, 4.1), (1.2, 2.2, 3.2, 4.2), (1.3, 2.3, 3.3, 4.3),
                     (1.4, 2.4, 3.4, 4.4), (1.5, 2.5, 3.5, 4.5)],
                    dtype=[('rho', '<f8'), ('phi', '<f8'), ('eta', '<f8'), ('tau', '<f8')])



Existing NumPy arrays can be viewed as arrays of vectors, but it needs to be a [structured array](https://numpy.org/doc/stable/user/basics.rec.html) with recognized field names.


```python
# NumPy array         # interpret groups of four values as named fields              # give it vector properties and methods
np.arange(0, 24, 0.1).view([("x", float), ("y", float), ("z", float), ("t", float)]).view(vector.VectorNumpy4D)
```




    VectorNumpy4D([( 0. ,  0.1,  0.2,  0.3), ( 0.4,  0.5,  0.6,  0.7),
                   ( 0.8,  0.9,  1. ,  1.1), ( 1.2,  1.3,  1.4,  1.5),
                   ( 1.6,  1.7,  1.8,  1.9), ( 2. ,  2.1,  2.2,  2.3),
                   ( 2.4,  2.5,  2.6,  2.7), ( 2.8,  2.9,  3. ,  3.1),
                   ( 3.2,  3.3,  3.4,  3.5), ( 3.6,  3.7,  3.8,  3.9),
                   ( 4. ,  4.1,  4.2,  4.3), ( 4.4,  4.5,  4.6,  4.7),
                   ( 4.8,  4.9,  5. ,  5.1), ( 5.2,  5.3,  5.4,  5.5),
                   ( 5.6,  5.7,  5.8,  5.9), ( 6. ,  6.1,  6.2,  6.3),
                   ( 6.4,  6.5,  6.6,  6.7), ( 6.8,  6.9,  7. ,  7.1),
                   ( 7.2,  7.3,  7.4,  7.5), ( 7.6,  7.7,  7.8,  7.9),
                   ( 8. ,  8.1,  8.2,  8.3), ( 8.4,  8.5,  8.6,  8.7),
                   ( 8.8,  8.9,  9. ,  9.1), ( 9.2,  9.3,  9.4,  9.5),
                   ( 9.6,  9.7,  9.8,  9.9), (10. , 10.1, 10.2, 10.3),
                   (10.4, 10.5, 10.6, 10.7), (10.8, 10.9, 11. , 11.1),
                   (11.2, 11.3, 11.4, 11.5), (11.6, 11.7, 11.8, 11.9),
                   (12. , 12.1, 12.2, 12.3), (12.4, 12.5, 12.6, 12.7),
                   (12.8, 12.9, 13. , 13.1), (13.2, 13.3, 13.4, 13.5),
                   (13.6, 13.7, 13.8, 13.9), (14. , 14.1, 14.2, 14.3),
                   (14.4, 14.5, 14.6, 14.7), (14.8, 14.9, 15. , 15.1),
                   (15.2, 15.3, 15.4, 15.5), (15.6, 15.7, 15.8, 15.9),
                   (16. , 16.1, 16.2, 16.3), (16.4, 16.5, 16.6, 16.7),
                   (16.8, 16.9, 17. , 17.1), (17.2, 17.3, 17.4, 17.5),
                   (17.6, 17.7, 17.8, 17.9), (18. , 18.1, 18.2, 18.3),
                   (18.4, 18.5, 18.6, 18.7), (18.8, 18.9, 19. , 19.1),
                   (19.2, 19.3, 19.4, 19.5), (19.6, 19.7, 19.8, 19.9),
                   (20. , 20.1, 20.2, 20.3), (20.4, 20.5, 20.6, 20.7),
                   (20.8, 20.9, 21. , 21.1), (21.2, 21.3, 21.4, 21.5),
                   (21.6, 21.7, 21.8, 21.9), (22. , 22.1, 22.2, 22.3),
                   (22.4, 22.5, 22.6, 22.7), (22.8, 22.9, 23. , 23.1),
                   (23.2, 23.3, 23.4, 23.5), (23.6, 23.7, 23.8, 23.9)],
                  dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('t', '<f8')])



Since `VectorNumpy2D`, `VectorNumpy3D`, `VectorNumpy4D`, and their momentum equivalents are NumPy array subclasses, all of the normal NumPy methods and functions work on them.


```python
np.arange(0, 24, 0.1).view([("x", float), ("y", float), ("z", float), ("t", float)]).view(vector.VectorNumpy4D).reshape(6, 5, 2)
```




    VectorNumpy4D([[[( 0. ,  0.1,  0.2,  0.3), ( 0.4,  0.5,  0.6,  0.7)],
                    [( 0.8,  0.9,  1. ,  1.1), ( 1.2,  1.3,  1.4,  1.5)],
                    [( 1.6,  1.7,  1.8,  1.9), ( 2. ,  2.1,  2.2,  2.3)],
                    [( 2.4,  2.5,  2.6,  2.7), ( 2.8,  2.9,  3. ,  3.1)],
                    [( 3.2,  3.3,  3.4,  3.5), ( 3.6,  3.7,  3.8,  3.9)]],

                   [[( 4. ,  4.1,  4.2,  4.3), ( 4.4,  4.5,  4.6,  4.7)],
                    [( 4.8,  4.9,  5. ,  5.1), ( 5.2,  5.3,  5.4,  5.5)],
                    [( 5.6,  5.7,  5.8,  5.9), ( 6. ,  6.1,  6.2,  6.3)],
                    [( 6.4,  6.5,  6.6,  6.7), ( 6.8,  6.9,  7. ,  7.1)],
                    [( 7.2,  7.3,  7.4,  7.5), ( 7.6,  7.7,  7.8,  7.9)]],

                   [[( 8. ,  8.1,  8.2,  8.3), ( 8.4,  8.5,  8.6,  8.7)],
                    [( 8.8,  8.9,  9. ,  9.1), ( 9.2,  9.3,  9.4,  9.5)],
                    [( 9.6,  9.7,  9.8,  9.9), (10. , 10.1, 10.2, 10.3)],
                    [(10.4, 10.5, 10.6, 10.7), (10.8, 10.9, 11. , 11.1)],
                    [(11.2, 11.3, 11.4, 11.5), (11.6, 11.7, 11.8, 11.9)]],

                   [[(12. , 12.1, 12.2, 12.3), (12.4, 12.5, 12.6, 12.7)],
                    [(12.8, 12.9, 13. , 13.1), (13.2, 13.3, 13.4, 13.5)],
                    [(13.6, 13.7, 13.8, 13.9), (14. , 14.1, 14.2, 14.3)],
                    [(14.4, 14.5, 14.6, 14.7), (14.8, 14.9, 15. , 15.1)],
                    [(15.2, 15.3, 15.4, 15.5), (15.6, 15.7, 15.8, 15.9)]],

                   [[(16. , 16.1, 16.2, 16.3), (16.4, 16.5, 16.6, 16.7)],
                    [(16.8, 16.9, 17. , 17.1), (17.2, 17.3, 17.4, 17.5)],
                    [(17.6, 17.7, 17.8, 17.9), (18. , 18.1, 18.2, 18.3)],
                    [(18.4, 18.5, 18.6, 18.7), (18.8, 18.9, 19. , 19.1)],
                    [(19.2, 19.3, 19.4, 19.5), (19.6, 19.7, 19.8, 19.9)]],

                   [[(20. , 20.1, 20.2, 20.3), (20.4, 20.5, 20.6, 20.7)],
                    [(20.8, 20.9, 21. , 21.1), (21.2, 21.3, 21.4, 21.5)],
                    [(21.6, 21.7, 21.8, 21.9), (22. , 22.1, 22.2, 22.3)],
                    [(22.4, 22.5, 22.6, 22.7), (22.8, 22.9, 23. , 23.1)],
                    [(23.2, 23.3, 23.4, 23.5), (23.6, 23.7, 23.8, 23.9)]]],
                  dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('t', '<f8')])



All of the keyword arguments and rules that apply to `vector.obj` construction apply to `vector.array` dtypes.

Geometrical names are used in the dtype, even if momentum-synonyms are used in construction.


```python
vector.array({"px": [1, 2, 3, 4], "py": [1.1, 2.2, 3.3, 4.4], "pz": [0.1, 0.2, 0.3, 0.4]})
```




    MomentumNumpy3D([(1., 1.1, 0.1), (2., 2.2, 0.2), (3., 3.3, 0.3), (4., 4.4, 0.4)],
                    dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8')])



### Awkward Arrays of vectors

[Awkward Arrays](https://awkward-array.org/) are arrays with more complex data structures than NumPy allows, such as variable-length lists, nested records, missing and even heterogeneous data (multiple data types: use sparingly).

The `vector.Array` function behaves exactly like the [ak.Array](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html) constructor, except that it makes arrays of vectors.


```python
vector.Array([
    [{"x": 1, "y": 1.1, "z": 0.1}, {"x": 2, "y": 2.2, "z": 0.2}],
    [],
    [{"x": 3, "y": 3.3, "z": 0.3}],
    [{"x": 4, "y": 4.4, "z": 0.4}, {"x": 5, "y": 5.5, "z": 0.5}, {"x": 6, "y": 6.6, "z": 0.6}],
])
```




    <VectorArray3D [[{x: 1, y: 1.1, z: 0.1, ... z: 0.6}]] type='4 * var * Vector3D["...'>



If you want _any_ records named "`Vector2D`", "`Vector3D`", "`Vector4D`", "`Momentum2D`", "`Momentum3D`", or "`Momentum4D`" to be interpreted as vectors, register the behaviors globally.


```python
vector.register_awkward()
```


```python
ak.Array([
    [{"x": 1, "y": 1.1, "z": 0.1}, {"x": 2, "y": 2.2, "z": 0.2}],
    [],
    [{"x": 3, "y": 3.3, "z": 0.3}],
    [{"x": 4, "y": 4.4, "z": 0.4}, {"x": 5, "y": 5.5, "z": 0.5}, {"x": 6, "y": 6.6, "z": 0.6}],
],
    with_name="Vector3D"
)
```




    <VectorArray3D [[{x: 1, y: 1.1, z: 0.1, ... z: 0.6}]] type='4 * var * Vector3D["...'>



All of the keyword arguments and rules that apply to `vector.obj` construction apply to `vector.Array` field names.

## Vector properties

Any geometrical coordinate can be computed from vectors in any coordinate system; they'll be provided or computed as needed.


```python
vector.obj(x=3, y=4).rho
```




    5.0




```python
vector.obj(rho=5, phi=0.9273).x
```




    2.9999808719721477




```python
vector.obj(rho=5, phi=0.9273).y
```




    4.000014345949428




```python
vector.obj(x=1, y=2, z=3).theta
```




    0.6405223126794245




```python
vector.obj(x=1, y=2, z=3).eta
```




    1.1035868415601453



Some properties are not coordinates, but derived from them.


```python
vector.obj(x=1, y=2, z=3).costheta
```




    0.8017837257372732




```python
vector.obj(x=1, y=2, z=3).mag   # spatial magnitude
```




    3.7416573867739413




```python
vector.obj(x=1, y=2, z=3).mag2   # spatial magnitude squared
```




    14



These properties are provided because they can be computed faster or with more numerical stability in different coordinate systems. For instance, the magnitude ignores `phi` in polar coordinates.


```python
vector.obj(rho=3, phi=0.123456789, z=4).mag2
```




    25



Momentum vectors have geometrical properties as well as their momentum-synonyms.


```python
vector.obj(px=3, py=4).rho
```




    5.0




```python
vector.obj(px=3, py=4).pt
```




    5.0




```python
vector.obj(x=1, y=2, z=3, E=4).tau
```




    1.4142135623730951




```python
vector.obj(x=1, y=2, z=3, E=4).mass
```




    1.4142135623730951



Here's the key thing: _arrays of vectors return arrays of coordinates_.


```python
vector.array({
    "x": [1.0, 2.0, 3.0, 4.0, 5.0],
    "y": [1.1, 2.2, 3.3, 4.4, 5.5],
    "z": [0.1, 0.2, 0.3, 0.4, 0.5],
}).theta
```




    array([1.50363023, 1.50363023, 1.50363023, 1.50363023, 1.50363023])




```python
vector.Array([
    [{"x": 1, "y": 1.1, "z": 0.1}, {"x": 2, "y": 2.2, "z": 0.2}],
    [],
    [{"x": 3, "y": 3.3, "z": 0.3}],
    [{"x": 4, "y": 4.4, "z": 0.4}, {"x": 5, "y": 5.5, "z": 0.5}],
]).theta
```




    <Array [[1.5, 1.5], [], [1.5], [1.5, 1.5]] type='4 * var * float64'>




```python
# Make a large, random NumPy array of 3D momentum vectors.
array = np.random.normal(0, 1, 150).view([(x, float) for x in ("x", "y", "z")]).view(vector.MomentumNumpy3D).reshape(5, 5, 2)
array
```




    MomentumNumpy3D([[[(-0.5816785 , -0.05413944,  0.76144538),
                       (-1.28722717, -1.08334531, -0.6673024 )],
                      [(-0.62358686,  0.55198952,  1.7445032 ),
                       ( 0.01700467,  1.15120078, -0.25539554)],
                      [(-1.34635672, -2.12940091, -1.54467987),
                       (-0.81493735,  0.20080621, -1.55001381)],
                      [( 1.24975086, -0.51898145,  0.60519884),
                       ( 0.96658282,  1.09592942, -0.06253526)],
                      [( 2.20226434, -0.31735941, -1.65264014),
                       (-0.24360299, -1.35841058,  0.58728995)]],

                     [[( 1.34917966,  2.09671483, -0.11296369),
                       (-0.95766198,  0.61822435,  0.00631232)],
                      [(-0.63470122, -1.57605684,  1.93922382),
                       (-1.49464097, -0.25947235, -1.42989385)],
                      [( 0.13181703, -0.23925908, -1.29684743),
                       (-1.30332525,  0.22076072, -1.85090259)],
                      [( 0.23572319, -1.37731231,  0.47495316),
                       (-1.03116698,  1.31698481,  0.05923118)],
                      [( 0.26381498,  0.81470087, -0.4088332 ),
                       ( 1.07145768, -0.02538051,  0.56726254)]],

                     [[( 0.78343072, -0.02741195,  0.7812837 ),
                       (-0.9815062 ,  0.45809367,  1.70387148)],
                      [( 0.22003486,  0.27980794,  0.47145122),
                       ( 0.93975121,  0.06854498, -0.78840851)],
                      [( 1.91018848,  0.65588855, -0.26309178),
                       (-0.24689939, -0.63389731, -0.18429427)],
                      [(-0.45547946, -0.60659614,  0.3710947 ),
                       ( 0.20913049, -0.0568765 , -0.03245573)],
                      [(-0.15149451, -0.53761606,  0.61024707),
                       (-1.72565708,  1.34034237, -0.05086564)]],

                     [[( 0.32961237,  0.13714739, -0.61025587),
                       (-0.36806236,  1.64625621,  0.44977818)],
                      [(-0.16190701,  0.44515714, -1.00738623),
                       ( 1.53029679, -0.15509494, -0.82669165)],
                      [(-0.71576504, -1.21193201, -0.61774313),
                       (-0.05330236,  0.33569103,  1.22197535)],
                      [(-1.14971585,  2.25842249,  0.07937847),
                       ( 0.93505701, -1.01045082, -0.40314804)],
                      [( 0.05238702, -0.27996488,  1.15669341),
                       (-0.23345516,  0.50328019,  0.13368106)]],

                     [[(-1.34294052, -1.40223737,  0.58557272),
                       ( 0.14018108,  1.24216928, -0.64543334)],
                      [(-0.03884685,  2.61315639, -0.88051183),
                       ( 0.24816409, -0.41001159,  0.94234544)],
                      [( 0.1734323 , -0.67060395,  1.01330813),
                       (-1.57644514,  0.51398789, -0.5226023 )],
                      [( 0.61337692, -0.45606094,  1.79962181),
                       (-0.19398749,  0.12963915,  0.46498903)],
                      [( 0.64323387,  0.27050203, -0.54877836),
                       ( 0.97756399, -0.43162287, -0.29129695)]]],
                    dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8')])




```python
# Get the transverse momentum of each one.
array.pt
```




    array([[[0.58419257, 1.68243599],
            [0.83279829, 1.15132637],
            [2.5193302 , 0.83931282],
            [1.35322538, 1.4612815 ],
            [2.22501354, 1.38008034]],

           [[2.49329076, 1.13987623],
            [1.69905879, 1.51699622],
            [0.27316778, 1.32188956],
            [1.39733841, 1.67264889],
            [0.85635031, 1.07175824]],

           [[0.78391014, 1.08314552],
            [0.35596043, 0.94224771],
            [2.01965587, 0.68028311],
            [0.75856471, 0.21672679],
            [0.55855314, 2.18504234]],

           [[0.35700661, 1.68689935],
            [0.47368635, 1.53813611],
            [1.40751511, 0.33989647],
            [2.53422941, 1.37671437],
            [0.28482404, 0.55479028]],

           [[1.94158669, 1.2500541 ],
            [2.61344512, 0.47926498],
            [0.69266761, 1.65812027],
            [0.76434471, 0.23331836],
            [0.69779736, 1.06861109]]])




```python
# The array and its components have the same shape.
array.shape
```




    (5, 5, 2)




```python
array.pt.shape
```




    (5, 5, 2)




```python
# Make a large, random Awkward Array of 3D momentum vectors.
array = vector.Array([[{x: np.random.normal(0, 1) for x in ("px", "py", "pz")} for inner in range(np.random.poisson(1.5))] for outer in range(50)])
array
```




    <MomentumArray3D [[{x: -1.24, y: 2.08, ... z: 1.56}]] type='50 * var * Momentum3...'>




```python
# Get the transverse momentum of each one, in the same nested structure.
array.pt
```




    <Array [[2.42, 1.29, 0.945, ... [0.789, 0.942]] type='50 * var * float64'>




```python
# The array and its components have the same list lengths (and can therefore be used together in subsequent calculations).
ak.num(array)
```




    <Array [3, 0, 1, 2, 0, 1, ... 0, 1, 2, 0, 1, 2] type='50 * int64'>




```python
ak.num(array.pt)
```




    <Array [3, 0, 1, 2, 0, 1, ... 0, 1, 2, 0, 1, 2] type='50 * int64'>



## Vector methods

Vector methods require arguments (in parentheses), which may be scalars or other vectors, depending on the calculation.


```python
vector.obj(x=3, y=4).rotateZ(0.1)
```




    vector.obj(x=2.585678829246765, y=4.279516911052588)




```python
vector.obj(rho=5, phi=0.4).rotateZ(0.1)
```




    vector.obj(rho=5, phi=0.5)




```python
# Broadcasts a scalar rotation angle of 0.5 to all elements of the NumPy array.
print(vector.array({"rho": [1, 2, 3, 4, 5], "phi": [0.1, 0.2, 0.3, 0.4, 0.5]}).rotateZ(0.5))
```

    [(1., 0.6) (2., 0.7) (3., 0.8) (4., 0.9) (5., 1. )]



```python
# Matches each rotation angle to an element of the NumPy array.
print(vector.array({"rho": [1, 2, 3, 4, 5], "phi": [0.1, 0.2, 0.3, 0.4, 0.5]}).rotateZ(np.array([0.1, 0.2, 0.3, 0.4, 0.5])))
```

    [(1., 0.2) (2., 0.4) (3., 0.6) (4., 0.8) (5., 1. )]



```python
# Broadcasts a scalar rotation angle of 0.5 to all elements of the Awkward Array.
print(vector.Array([[{"rho": 1, "phi": 0.1}, {"rho": 2, "phi": 0.2}], [], [{"rho": 3, "phi": 0.3}]]).rotateZ(0.5))
```

    [[{rho: 1, phi: 0.6}, {rho: 2, phi: 0.7}], [], [{rho: 3, phi: 0.8}]]



```python
# Broadcasts a rotation angle of 0.1 to both elements of the first list, 0.2 to the empty list, and 0.3 to the only element of the last list.
print(vector.Array([[{"rho": 1, "phi": 0.1}, {"rho": 2, "phi": 0.2}], [], [{"rho": 3, "phi": 0.3}]]).rotateZ([0.1, 0.2, 0.3]))
```

    [[{rho: 1, phi: 0.2}, {rho: 2, phi: 0.3}], [], [{rho: 3, phi: 0.6}]]



```python
# Matches each rotation angle to an element of the Awkward Array.
print(vector.Array([[{"rho": 1, "phi": 0.1}, {"rho": 2, "phi": 0.2}], [], [{"rho": 3, "phi": 0.3}]]).rotateZ([[0.1, 0.2], [], [0.3]]))
```

    [[{rho: 1, phi: 0.2}, {rho: 2, phi: 0.4}], [], [{rho: 3, phi: 0.6}]]


Some methods are equivalent to binary operators.


```python
vector.obj(x=3, y=4).scale(10)
```




    vector.obj(x=30, y=40)




```python
vector.obj(x=3, y=4) * 10
```




    vector.obj(x=30, y=40)




```python
10 * vector.obj(x=3, y=4)
```




    vector.obj(x=30, y=40)




```python
vector.obj(rho=5, phi=0.5) * 10
```




    vector.obj(rho=50, phi=0.5)



Some methods involve more than one vector.


```python
vector.obj(x=1, y=2).add(vector.obj(x=5, y=5))
```




    vector.obj(x=6, y=7)




```python
vector.obj(x=1, y=2) + vector.obj(x=5, y=5)
```




    vector.obj(x=6, y=7)




```python
vector.obj(x=1, y=2).dot(vector.obj(x=5, y=5))
```




    15




```python
vector.obj(x=1, y=2) @ vector.obj(x=5, y=5)
```




    15



The vectors can use different coordinate systems. Conversions are necessary, but minimized for speed and numeric stability.


```python
vector.obj(x=3, y=4) @ vector.obj(x=6, y=8)   # both are Cartesian, dot product is exact
```




    50




```python
vector.obj(rho=5, phi=0.9273) @ vector.obj(x=6, y=8)   # one is polar, dot product is approximate
```




    49.99999999942831




```python
vector.obj(x=3, y=4) @ vector.obj(rho=10, phi=0.9273)   # one is polar, dot product is approximate
```




    49.99999999942831




```python
vector.obj(rho=5, phi=0.9273) @ vector.obj(rho=10, phi=0.9273)   # both are polar, a formula that depends on phi differences is used
```




    50.0



In Python, some "operators" are actually built-in functions, such as `abs`.


```python
abs(vector.obj(x=3, y=4))
```




    5.0



Note that `abs` returns

   * `rho` for 2D vectors
   * `mag` for 3D vectors
   * `tau` (`mass`) for 4D vectors

Use the named properties when you want magnitude in a specific number of dimensions; use `abs` when you want the magnitude for any number of dimensions.

The vectors can be from different backends. Normal rules for broadcasting Python numbers, NumPy arrays, and Awkward Arrays apply.


```python
vector.array({"x": [1, 2, 3, 4, 5], "y": [0.1, 0.2, 0.3, 0.4, 0.5]}) + vector.obj(x=10, y=5)
```




    VectorNumpy2D([(11., 5.1), (12., 5.2), (13., 5.3), (14., 5.4), (15., 5.5)],
                  dtype=[('x', '<f8'), ('y', '<f8')])




```python
(
    vector.Array([                                                   # an Awkward Array of vectors
        [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}],
        [],
        [{"x": 3, "y": 3.3}],
        [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}],
    ])
    + vector.obj(x=10, y=5)                                          # and a single vector object
)
```




    <VectorArray2D [[{x: 11, y: 6.1}, ... x: 15, y: 10.5}]] type='4 * var * Vector2D...'>




```python
(
    vector.Array([                                                   # an Awkward Array of vectors
        [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}],
        [],
        [{"x": 3, "y": 3.3}],
        [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}],
    ])
    + vector.array({"x": [4, 3, 2, 1], "y": [0.1, 0.1, 0.1, 0.1]})   # and a NumPy array of vectors
)
```




    <VectorArray2D [[{x: 5, y: 1.2}, ... x: 6, y: 5.6}]] type='4 * var * Vector2D["x...'>



Some operations are defined for 2D or 3D vectors, but are usable on higher-dimensional vectors because the additional components can be ignored or are passed through unaffected.


```python
vector.obj(rho=1, phi=0.5).deltaphi(vector.obj(rho=2, phi=0.3))   # deltaphi is a planar operation (defined on the transverse plane)
```




    0.20000000000000018




```python
vector.obj(rho=1, phi=0.5, z=10).deltaphi(vector.obj(rho=2, phi=0.3, theta=1.4))   # but we can use it on 3D vectors
```




    0.20000000000000018




```python
vector.obj(rho=1, phi=0.5, z=10, t=100).deltaphi(vector.obj(rho=2, phi=0.3, theta=1.4, tau=1000))   # and 4D vectors
```




    0.20000000000000018




```python
vector.obj(rho=1, phi=0.5).deltaphi(vector.obj(rho=2, phi=0.3, theta=1.4, tau=1000))   # and mixed dimensionality
```




    0.20000000000000018



This is especially useful for giving 4D vectors all the capabilities of 3D vectors.


```python
vector.obj(x=1, y=2, z=3).rotateX(np.pi/4)
```




    vector.obj(x=1, y=-0.7071067811865472, z=3.5355339059327378)




```python
vector.obj(x=1, y=2, z=3, tau=10).rotateX(np.pi/4)
```




    vector.obj(x=1, y=-0.7071067811865472, z=3.5355339059327378, tau=10)




```python
vector.obj(pt=1, phi=1.3, eta=2).deltaR(vector.obj(pt=2, phi=0.3, eta=1))
```




    1.4142135623730951




```python
vector.obj(pt=1, phi=1.3, eta=2, mass=5).deltaR(vector.obj(pt=2, phi=0.3, eta=1, mass=10))
```




    1.4142135623730951



The opposite—using low-dimensional vectors in operations defined for higher numbers of dimensions—is sometimes defined. In these cases, a zero longitudinal or temporal component has to be imputed.


```python
vector.obj(x=1, y=2, z=3) - vector.obj(x=1, y=2)
```




    vector.obj(x=0, y=0, z=3)




```python
vector.obj(x=1, y=2, z=0).is_parallel(vector.obj(x=1, y=2))
```




    True



And finally, in some cases, the function excludes a higher-dimensional component, even if the input vectors had them.

It would be confusing if the 3D cross-product returned a fourth component.


```python
vector.obj(x=0.1, y=0.2, z=0.3, t=10).cross(vector.obj(x=0.4, y=0.5, z=0.6, t=20))
```




    vector.obj(x=-0.03, y=0.06, z=-0.030000000000000013)



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
```


```python
compute_mass(vector.obj(px=1, py=2, pz=3, E=4), vector.obj(px=-1, py=-2, pz=-3, E=4))
```




    8.0



When the two `MomentumObject4D` objects are passed as arguments, Numba recognizes them and replaces the Python objects with low-level structs. When it compiles the function, it recognizes `+` as the 4D `add` function and recognizes `.mass` as the `tau` component of the result.

Although this demonstrates that Numba can manipulate vector objects, there is no performance advantage (and a likely disadvantage) to compiling a calculation on just a few vectors. The advantage comes when many vectors are involved, in arrays.


```python
# This is still not a large number. You want millions.
array = vector.Array([[dict({x: np.random.normal(0, 1) for x in ("px", "py", "pz")}, E=np.random.normal(10, 1)) for inner in range(np.random.poisson(1.5))] for outer in range(50)])
array
```




    <MomentumArray4D [[{x: 1.09, y: 0.537, ... t: 10.1}]] type='50 * var * Momentum4...'>




```python
@nb.njit
def compute_masses(array):
    out = np.empty(len(array), np.float64)
    for i, event in enumerate(array):
        total = vector.obj(px=0.0, py=0.0, pz=0.0, E=0.0)
        for vec in event:
            total = total + vec
        out[i] = total.mass
    return out
```


```python
compute_masses(array)
```




    array([ 8.95477603,  0.        ,  9.8871152 ,  0.        , 39.07925142,
           48.55053642,  9.55009174, 31.75711709, 10.04008594,  0.        ,
           51.7984392 ,  8.66502552,  9.69483698, 29.51663446, 12.41687358,
           31.415441  ,  0.        , 19.80136543, 29.56169674,  9.87277159,
            9.73184162, 11.38881315, 22.24734135, 21.37722606,  8.03300384,
            0.        , 29.03811593,  9.88647946, 21.95344143, 10.68257124,
            9.96117033, 23.35887444,  0.        ,  9.3885009 , 18.96812068,
           29.25050679,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  8.48410078, 18.50092416,  0.        ,  9.43988115,
            0.        ,  9.0743725 ,  8.96849697, 17.25900391, 29.60567615])



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
