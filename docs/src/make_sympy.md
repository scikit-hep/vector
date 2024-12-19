# Making SymPy vector expressions

SymPy expressions are not numerical, they're purely algebraic. However, the same Vector computations can be performed on them.

To construct a symbolic vector, first create symbols for its components and ensure that they are real-valued (not complex),

```python
>>> import sympy
>>> x, y, z, t, px, py, pz, eta, tau = sympy.symbols(
...     "x y z t px py pz eta tau", real=True
... )
```

then use one of Vector's SymPy constructors (geometric or momentum),

```python
>>> vector.VectorSympy2D(x=x, y=y)
VectorSympy2D(x=x, y=y)
>>>
>>> vector.MomentumSympy3D(px=px, py=py, pz=pz)
MomentumSympy3D(px=px, py=py, pz=pz)
>>>
>>> vector.VectorSympy4D(x=x, y=y, eta=eta, tau=tau)
vector.VectorSympy4D(x=x, y=y, eta=eta, tau=tau)
```

which are documented below.

## 2D constructors

```{eval-rst}
.. autoclass:: vector.VectorSympy2D
```

```{eval-rst}
.. autoclass:: vector.MomentumSympy2D
```

## 3D constructors

```{eval-rst}
.. autoclass:: vector.VectorSympy3D
```

```{eval-rst}
.. autoclass:: vector.MomentumSympy3D
```

## 4D constructors

```{eval-rst}
.. autoclass:: vector.VectorSympy4D
```

```{eval-rst}
.. autoclass:: vector.MomentumSympy4D
```
