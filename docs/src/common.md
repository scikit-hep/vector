# Interface for all vectors

Vectors of all backends, 2D/3D/4D, geometric and momentum, have the following attributes, properties, and methods.

For interfaces specialized to 2D/3D/4D vectors or momentum vectors, see

- [Interface for 2D vectors](vector2d.md)
- [Interface for 3D vectors](vector3d.md)
- [Interface for 4D vectors](vector4d.md)
- [Interface for 2D momentum](momentum2d.md)
- [Interface for 3D momentum](momentum3d.md)
- [Interface for 4D momentum](momentum4d.md)

```{eval-rst}
.. autoclass:: vector._methods.VectorProtocol
    :members:
    :inherited-members:
    :member-order: bysource
    :exclude-members: lib,ProjectionClass2D,ProjectionClass3D,ProjectionClass4D,GenericClass,MomentumClass
```
