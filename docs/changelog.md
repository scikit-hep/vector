# Changelog

## Version 0.8

### Version 0.8.1

* Fix issue importing without Awkward installed [#76][]

[#76]: https://github.com/scikit-hep/vector/pull/76

### Version 0.8.0

First release to PyPI. Initial implementation. Initial features:

* 2D, 3D, and Lorentz vectors
* Single, Array, and Awkward forms
* Supports Numba / Awkward + Numba
* Multiple coordinate systems
* Geometric / momentum versions
* Statically typed

You can currently construct vectors using `obj`/`arr`/`awk` (or
`obj`/`array`/`Array`) for single, NumPy, and Awkward vectors, respectively.
The next version is likely to improve the vector construction process.

