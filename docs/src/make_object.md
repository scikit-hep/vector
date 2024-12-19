# Making vector objects

A vector object represents a single vector, rather than an array of vectors. Lists of vector objects are slower to compute and have more memory overhead than arrays of vectors, _unless_ those computations are performed in [Numba-compiled functions](https://numba.pydata.org/).

To create a vector object, use the `vector.obj` function with appropriate arguments for 2D/3D/4D and geometric versus momentum.

## General constructor

```{eval-rst}
.. autofunction:: vector.obj
```

## 2D constructors

```{eval-rst}
.. autoclass:: vector.VectorObject2D
    :members: from_rhophi,from_xy
```

```{eval-rst}
.. autoclass:: vector.MomentumObject2D
    :members: from_rhophi,from_xy
```

## 3D constructors

```{eval-rst}
.. autoclass:: vector.VectorObject3D
    :members: from_rhophieta,from_rhophitheta,from_rhophiz,from_xyeta,from_xytheta,from_xyz
```

```{eval-rst}
.. autoclass:: vector.MomentumObject3D
    :members: from_rhophieta,from_rhophitheta,from_rhophiz,from_xyeta,from_xytheta,from_xyz
```

## 4D constructors

```{eval-rst}
.. autoclass:: vector.VectorObject4D
    :members: from_rhophietat,from_rhophietatau,from_rhophithetat,from_rhophithetatau,from_rhophizt,from_rhophiztau,from_xyetat,from_xyetatau,from_xythetat,from_xythetatau,from_xyzt,from_xyztau
```

```{eval-rst}
.. autoclass:: vector.MomentumObject4D
    :members: from_rhophietat,from_rhophietatau,from_rhophithetat,from_rhophithetatau,from_rhophizt,from_rhophiztau,from_xyetat,from_xyetatau,from_xythetat,from_xythetatau,from_xyzt,from_xyztau
```
