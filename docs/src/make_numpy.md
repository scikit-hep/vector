# Making NumPy arrays of vectors

A NumPy array of vectors is a subclass of [np.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) with vector properties and methods. The [dtype](https://numpy.org/doc/stable/reference/arrays.dtypes.html) of this array is [structured](https://numpy.org/doc/stable/user/basics.rec.html) to specify the coordinate names; an array with fields `x` and `y` (Cartesian) performs computations differently from an array with fields `rho` and `phi` (polar).

To create a NumPy array of vectors,

1. use the `vector.array` function (`vector.arr` is a synonym)
2. use the `vector.VectorNumpy` class constructor
3. or cast a structured NumPy array as the appropriate class, which can avoid copying data.

## General constructor

```{eval-rst}
.. autofunction:: vector.array
```

```{eval-rst}
.. autoclass:: vector.VectorNumpy
```

## Casting structured arrays

[NumPy structured arrays](https://numpy.org/doc/stable/user/basics.rec.html) with appropriately named fields (see above) can be _cast_ as arrays of vectors using [np.ndarray.view](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.view.html). Use the NumPy array subclass with the appropriate dimension below.

```{eval-rst}
.. autoclass:: vector.VectorNumpy2D
```

```{eval-rst}
.. autoclass:: vector.MomentumNumpy2D
```

```{eval-rst}
.. autoclass:: vector.VectorNumpy3D
```

```{eval-rst}
.. autoclass:: vector.MomentumNumpy3D
```

```{eval-rst}
.. autoclass:: vector.VectorNumpy4D
```

```{eval-rst}
.. autoclass:: vector.MomentumNumpy4D
```
