# Making Awkward Arrays of vectors

An Awkward Array of vectors is an Awkward Array containing appropriately named records, appropriately named fields, and the Vector [behaviors](https://awkward-array.org/doc/main/reference/ak.behavior.html) registered in the array. Here's a complete example for illustration:

```python
>>> import awkward as ak
>>> import vector
>>> vector.register_awkward()
>>>
>>> vec = ak.Array([
...     [{"x": 1.1, "y": 2.2}, {"x": 3.3, "y": 4.4}],
...     [],
...     [{"x": 5.5, "y": 6.6}],
... ], with_name="Vector2D")
>>>
>>> abs(vec)
<Array [[2.46, 5.5], [], [8.59]] type='3 * var * float64'>
```

In the above,

1. `vector.register_awkward()` loads Vector's `vector.backends.awkward.behavior` dict of functionality into the global `ak.behavior`
2. the Awkward Array contains records (inside variable-length lists) with field names `"x"` and `"y"`
3. those records are labeled with type name `"Vector2D"`

and thus the `abs` function computes the magnitude of each record as `sqrt(x**2 + y**2)`, through the variable-length lists.

It is not necessary to install Vector's behaviors globally. They could be installed in the `vec` array only by passing `behavior=vector.backends.awkward.behavior` to the [ak.Array](https://awkward-array.org/doc/main/reference/generated/ak.Array.html) constructor.

The records can contain more fields than those that specify coordinates, which can be useful for specifying properties of a particle other than its momentum. Only the coordinate names are considered when performing vector calculations. Coordinates must be numbers (not, for instance, lists of numbers). Be careful about field names that coincide with coordinates, such as `rho` (azimuthal magnitude) and `tau` (proper time).

The `vector.Array` function (`vector.awk` is a synonym) is an alternative to the [ak.Array](https://awkward-array.org/doc/main/reference/generated/ak.Array.html) constructor, which installs Vector's behavior in the new array (not globally in `ak.behavior`).

The `vector.zip` function is an alternative to the [ak.zip](https://awkward-array.org/doc/main/reference/generated/ak.zip.html) function, which installs Vector's behavior in the new array (not globally in `ak.behavior`).

Awkward Arrays can be used in [Numba-compiled functions](https://numba.pydata.org/), including those that contain vectors.

```{eval-rst}
.. autofunction:: vector.register_awkward
```

```{eval-rst}
.. autofunction:: vector.Array
```

```{eval-rst}
.. autofunction:: vector.zip
```
