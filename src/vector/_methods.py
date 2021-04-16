# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

from vector._typeutils import (
    BoolCollection,
    ScalarCollection,
    TransformProtocol2D,
    TransformProtocol3D,
    TransformProtocol4D,
)

Module = typing.Any  # returns a module, but we can't be specific about which one


class Coordinates:
    pass


class Azimuthal(Coordinates):
    @property
    def elements(self) -> typing.Tuple[ScalarCollection, ScalarCollection]:
        """
        Azimuthal coordinates as a tuple.

        Each coordinate may be a scalar, a NumPy array, an Awkward Array, etc.,
        but they are not vectors.
        """
        raise AssertionError


class Longitudinal(Coordinates):
    @property
    def elements(self) -> typing.Tuple[ScalarCollection]:
        """
        Longitudinal coordinates as a tuple.

        Each coordinate may be a scalar, a NumPy array, an Awkward Array, etc.,
        but they are not vectors.
        """
        raise AssertionError


class Temporal(Coordinates):
    @property
    def elements(self) -> typing.Tuple[ScalarCollection]:
        """
        Temporal coordinates as a tuple.

        Each coordinate may be a scalar, a NumPy array, an Awkward Array, etc.,
        but they are not vectors.
        """
        raise AssertionError


class AzimuthalXY(Azimuthal):
    """
    Attributes:
        x (scalar, ``np.ndarray``, ``ak.Array``, etc.): The $x$ coordinate(s).
        y (scalar, ``np.ndarray``, ``ak.Array``, etc.): The $y$ coordinate(s).
    """

    x: ScalarCollection
    y: ScalarCollection


class AzimuthalRhoPhi(Azimuthal):
    r"""
    Attributes:
        rho (scalar, ``np.ndarray``, ``ak.Array``, etc.): The $\rho$ coordinate(s).
        phi (scalar, ``np.ndarray``, ``ak.Array``, etc.): The $\phi$ coordinate(s).
    """

    rho: ScalarCollection
    phi: ScalarCollection


class LongitudinalZ(Longitudinal):
    """
    Attributes:
        z (scalar, ``np.ndarray``, ``ak.Array``, etc.): The $z$ coordinate(s).
    """

    z: ScalarCollection


class LongitudinalTheta(Longitudinal):
    r"""
    Attributes:
        theta (scalar, ``np.ndarray``, ``ak.Array``, etc.): The $\theta$ coordinate(s).
    """

    theta: ScalarCollection


class LongitudinalEta(Longitudinal):
    r"""
    Attributes:
        eta (scalar, ``np.ndarray``, ``ak.Array``, etc.): The $\eta$ coordinate(s).
    """

    eta: ScalarCollection


class TemporalT(Temporal):
    """
    Attributes:
        t (scalar, ``np.ndarray``, ``ak.Array``, etc.): The $t$ coordinate(s).
    """

    t: ScalarCollection


class TemporalTau(Temporal):
    r"""
    Attributes:
        tau (scalar, ``np.ndarray``, ``ak.Array``, etc.): The $\tau$ coordinate(s).
    """

    tau: ScalarCollection


SameVectorType = typing.TypeVar("SameVectorType", bound="VectorProtocol")


class VectorProtocol:
    """
    Attributes:
        lib (module): The module used for functions used in compute functions
            (such as ``sqrt``, ``sin``, ``cos``). Usually ``numpy``.
        ProjectionClass2D (type): The class that would result from projecting this
            vector onto azimuthal coordinates only.
        ProjectionClass3D (type): The class that would result from projecting this
            vector onto azimuthal and longitudinal coordinates only.
        ProjectionClass4D (type): The class that would result from projecting this
            vector onto azimuthal, longitudinal, and temporal coordinates.
        GenericClass (type): The most generic concrete class for this type, for
            vectors without momentum-synonyms.
    """

    lib: Module

    def _wrap_result(
        self,
        cls: typing.Any,
        result: typing.Any,
        returns: typing.Any,
        num_vecargs: typing.Any,
    ) -> typing.Any:
        """
        Args:
            result: Value or tuple of values from a compute function.
            returns: Signature from a ``dispatch_map``.
            num_vecargs (int): Number of vector arguments in the function
                that would be treated on an equal footing (i.e. ``add``
                has two, but ``rotate_axis`` has only one: the ``axis``
                is secondary).

        Wraps the raw result of a compute function as a scalar, an array of scalars,
        a vector, or an array of vectors.
        """
        raise AssertionError

    ProjectionClass2D: typing.Type["VectorProtocolPlanar"]
    ProjectionClass3D: typing.Type["VectorProtocolSpatial"]
    ProjectionClass4D: typing.Type["VectorProtocolLorentz"]
    GenericClass: typing.Type["VectorProtocol"]

    def to_Vector2D(self) -> "VectorProtocolPlanar":
        """
        Projects this vector/these vectors onto azimuthal coordinates only.
        """
        raise AssertionError

    def to_Vector3D(self) -> "VectorProtocolSpatial":
        """
        Projects this vector/these vectors onto azimuthal and longitudinal
        coordinates only.

        If 2D, a $z$ component of $0$ is imputed.
        """
        raise AssertionError

    def to_Vector4D(self) -> "VectorProtocolLorentz":
        """
        Projects this vector/these vectors onto azimuthal, longitudinal,
        and temporal coordinates.

        If 2D or 3D, a $t$ component of $0$ is imputed.

        If 2D, a $z$ component of $0$ is imputed.
        """
        raise AssertionError

    def to_xy(self) -> "VectorProtocolPlanar":
        """
        Converts to $x$-$y$ coordinates, possibly eliminating dimensions with a
        projection.
        """
        raise AssertionError

    def to_rhophi(self) -> "VectorProtocolPlanar":
        r"""
        Converts to $\rho$-$\phi$ coordinates, possibly eliminating dimensions with a
        projection.
        """
        raise AssertionError

    def to_xyz(self) -> "VectorProtocolSpatial":
        """
        Converts to $x$-$y$-$z$ coordinates, possibly eliminating or imputing
        dimensions with a projection.
        """
        raise AssertionError

    def to_xytheta(self) -> "VectorProtocolSpatial":
        r"""
        Converts to $x$-$y$-$\theta$ coordinates, possibly eliminating or imputing
        dimensions with a projection.
        """
        raise AssertionError

    def to_xyeta(self) -> "VectorProtocolSpatial":
        r"""
        Converts to $x$-$y$-$\eta$ coordinates, possibly eliminating or imputing
        dimensions with a projection.
        """
        raise AssertionError

    def to_rhophiz(self) -> "VectorProtocolSpatial":
        r"""
        Converts to $\rho$-$\phi$-$z$ coordinates, possibly eliminating or imputing
        dimensions with a projection.
        """
        raise AssertionError

    def to_rhophitheta(self) -> "VectorProtocolSpatial":
        r"""
        Converts to $\rho$-$\phi$-$\theta$ coordinates, possibly eliminating or
        imputing dimensions with a projection.
        """
        raise AssertionError

    def to_rhophieta(self) -> "VectorProtocolSpatial":
        r"""
        Converts to $\rho$-$\phi$-$\eta$ coordinates, possibly eliminating or
        imputing dimensions with a projection.
        """
        raise AssertionError

    def to_xyzt(self) -> "VectorProtocolLorentz":
        """
        Converts to $x$-$y$-$z$-$t$ coordinates, possibly imputing dimensions with
        a projection.
        """
        raise AssertionError

    def to_xyztau(self) -> "VectorProtocolLorentz":
        r"""
        Converts to $x$-$y$-$z$-$\tau$ coordinates, possibly imputing dimensions
        with a projection.
        """
        raise AssertionError

    def to_xythetat(self) -> "VectorProtocolLorentz":
        r"""
        Converts to $x$-$y$-$\theta$-$t$ coordinates, possibly imputing dimensions
        with a projection.
        """
        raise AssertionError

    def to_xythetatau(self) -> "VectorProtocolLorentz":
        r"""
        Converts to $x$-$y$-$\theta$-$\tau$ coordinates, possibly imputing
        dimensions with a projection.
        """
        raise AssertionError

    def to_xyetat(self) -> "VectorProtocolLorentz":
        r"""
        Converts to $x$-$y$-$\eta$-$t$ coordinates, possibly imputing dimensions
        with a projection.
        """
        raise AssertionError

    def to_xyetatau(self) -> "VectorProtocolLorentz":
        r"""
        Converts to $x$-$y$-$\eta$-$\tau$ coordinates, possibly imputing dimensions
        with a projection.
        """
        raise AssertionError

    def to_rhophizt(self) -> "VectorProtocolLorentz":
        r"""
        Converts to $\rho$-$\phi$-$z$-$t$ coordinates, possibly imputing dimensions
        with a projection.
        """
        raise AssertionError

    def to_rhophiztau(self) -> "VectorProtocolLorentz":
        r"""
        Converts to $\rho$-$\phi$-$z$-$\tau$ coordinates, possibly imputing
        dimensions with a projection.
        """
        raise AssertionError

    def to_rhophithetat(self) -> "VectorProtocolLorentz":
        r"""
        Converts to $\rho$-$\phi$-$\theta$-$t$ coordinates, possibly imputing
        dimensions with a projection.
        """
        raise AssertionError

    def to_rhophithetatau(self) -> "VectorProtocolLorentz":
        r"""
        Converts to $\rho$-$\phi$-$\theta$-$\tau$ coordinates, possibly imputing
        dimensions with a projection.
        """
        raise AssertionError

    def to_rhophietat(self) -> "VectorProtocolLorentz":
        r"""
        Converts to $\rho$-$\phi$-$\eta$-$t$ coordinates, possibly imputing
        dimensions with a projection.
        """
        raise AssertionError

    def to_rhophietatau(self) -> "VectorProtocolLorentz":
        r"""
        Converts to $\rho$-$\phi$-$\eta$-$\tau$ coordinates, possibly imputing
        dimensions with a projection.
        """
        raise AssertionError

    def unit(self: SameVectorType) -> SameVectorType:
        """
        Returns vector(s) normalized to unit length, which is `rho == 1` for 2D
        vectors, `mag == 1` for 3D vectors, and `tau == 1` for 4D vectors.
        """
        raise AssertionError

    def dot(self, other: "VectorProtocol") -> ScalarCollection:
        """
        Vector dot product of ``self`` with ``other``.

        This method is equivalent to the ``@`` operator.
        """
        raise AssertionError

    def add(self, other: "VectorProtocol") -> "VectorProtocol":
        """
        Sum of ``self`` and ``other``.

        This method is equivalent to the ``+`` operator.
        """
        raise AssertionError

    def subtract(self, other: "VectorProtocol") -> "VectorProtocol":
        """
        Difference of ``self`` minus ``other``.

        This method is equivalent to the ``-`` operator.
        """
        raise AssertionError

    def scale(self: SameVectorType, factor: ScalarCollection) -> SameVectorType:
        """
        Returns vector(s) scaled by a ``factor``, changing the length(s) but not
        the direction(s).

        This method is equivalent to the ``*`` operator.
        """
        raise AssertionError

    def equal(self, other: "VectorProtocol") -> BoolCollection:
        """
        Returns True if ``self`` is exactly equal to ``other`` (possibly for arrays
        of vectors), False otherwise.

        This method is equivalent to the ``==`` operator.

        Typically, you'll want to check :doc:`vector._methods.VectorProtocol.isclose`
        to allow for numerical errors.
        """
        raise AssertionError

    def not_equal(self, other: "VectorProtocol") -> BoolCollection:
        """
        Returns False if ``self`` is exactly equal to ``other`` (possibly for arrays
        of vectors), True otherwise.

        This method is equivalent to the ``!=`` operator.

        Typically, you'll want to check :doc:`vector._methods.VectorProtocol.isclose`
        to allow for numerical errors.
        """
        raise AssertionError

    def isclose(
        self,
        other: "VectorProtocol",
        rtol: ScalarCollection = 1e-05,
        atol: ScalarCollection = 1e-08,
        equal_nan: BoolCollection = False,
    ) -> BoolCollection:
        """
        Returns True if ``self`` is approximately equal to ``other`` (possibly for
        arrays of vectors), False otherwise.

        The relative tolerance (``rtol``) and absolute tolerance (``atol``) are
        interpreted as in ``np.isclose``:

        .. code-block:: python

            close_enough = abs(self - other) <= atol + rtol * abs(other)
        """
        raise AssertionError


class VectorProtocolPlanar(VectorProtocol):
    @property
    def azimuthal(self) -> Azimuthal:
        """
        Container of azimuthal coordinates, for use in dispatching to compute
        functions or to identify coordinate system with ``isinstance``.
        """
        raise AssertionError

    @property
    def x(self) -> ScalarCollection:
        """
        The Cartesian $x$ coordinate of the vector or every vector in the array.
        """
        raise AssertionError

    @property
    def y(self) -> ScalarCollection:
        """
        The Cartesian $y$ coordinate of the vector or every vector in the array.
        """
        raise AssertionError

    @property
    def rho(self) -> ScalarCollection:
        r"""
        The polar $\rho$ coordinate of the vector or every vector in the array.

        This is also the magnitude of the 2D azimuthal part of the vector (not
        including any longitudinal or temporal parts).
        """
        raise AssertionError

    @property
    def rho2(self) -> ScalarCollection:
        r"""
        The polar $\rho$ coordinate squared of the vector or every vector in the array.
        """
        raise AssertionError

    @property
    def phi(self) -> ScalarCollection:
        r"""
        The polar $\phi$ coordinate of the vector or every vector in the array
        (in radians, always between $-\pi$ and $\pi$).
        """
        raise AssertionError

    def deltaphi(self, other: VectorProtocol) -> ScalarCollection:
        r"""
        Signed difference in $\phi$ of ``self`` minus ``other`` (in radians).
        """
        raise AssertionError

    def rotateZ(self: SameVectorType, angle: ScalarCollection) -> SameVectorType:
        """
        Rotates the vector(s) by a given ``angle`` (in radians) around the
        longitudinal axis.

        Note that the ``angle`` can be an array with the same length as the vectors,
        if the vectors are in an array.
        """
        raise AssertionError

    def transform2D(self: SameVectorType, obj: TransformProtocol2D) -> SameVectorType:
        """
        Arbitrarily transforms the vector(s) by

        .. code-block:: python

            obj["xx"] obj["xy"]
            obj["yx"] obj["yy"]

        leaving any longitudinal or temporal coordinates unchanged. There is no
        restriction on the type of ``obj``; it just has to provide those components
        (which can be arrays if the vectors are in an array).
        """
        raise AssertionError

    def is_parallel(
        self, other: VectorProtocol, tolerance: ScalarCollection = 1e-5
    ) -> BoolCollection:
        r"""
        Returns True if ``self`` and ``other`` are pointing in the same direction
        (i.e. not "antiparallel"; dot product is nearly ``abs(self) * abs(other)``).

        The ``tolerance`` is measured in units of $\cos(\Delta\alpha)$ where $\Delta\alpha$
        is ``self.deltaangle(other)`.
        """
        raise AssertionError

    def is_antiparallel(
        self, other: VectorProtocol, tolerance: ScalarCollection = 1e-5
    ) -> BoolCollection:
        r"""
        Returns True if ``self`` and ``other`` are pointing in opposite directions
        (i.e. dot product is nearly ``-abs(self) * abs(other)``).

        The ``tolerance`` is measured in units of $\cos(\Delta\alpha)$ where $\Delta\alpha$
        is ``self.deltaangle(other)`.
        """
        raise AssertionError

    def is_perpendicular(
        self, other: VectorProtocol, tolerance: ScalarCollection = 1e-5
    ) -> BoolCollection:
        r"""
        Returns True if ``self`` and ``other`` are pointing in perpendicular directions
        (i.e. dot product is nearly ``0``).

        The ``tolerance`` is measured in units of $\cos(\Delta\alpha)$ where $\Delta\alpha$
        is ``self.deltaangle(other)`.
        """
        raise AssertionError


class VectorProtocolSpatial(VectorProtocolPlanar):
    @property
    def longitudinal(self) -> Longitudinal:
        """
        Container of longitudinal coordinates, for use in dispatching to compute
        functions or to identify coordinate system with ``isinstance``.
        """
        raise AssertionError

    @property
    def z(self) -> ScalarCollection:
        """
        The Cartesian $z$ coordinate of the vector or every vector in the array.
        """
        raise AssertionError

    @property
    def theta(self) -> ScalarCollection:
        r"""
        The spherical $\theta$ coordinate (polar angle) of the vector or every vector
        in the array (in radians, always between $0$ ($+z$) and $\pi$ ($-z$)).
        """
        raise AssertionError

    @property
    def eta(self) -> ScalarCollection:
        r"""
        The pseudorapidity $\eta$ coordinate of the vector or every vector
        in the array (in radians, always between $0$ ($+z$) and $\pi$ ($-z$)).
        """
        raise AssertionError

    @property
    def costheta(self) -> ScalarCollection:
        r"""
        The $\cos\theta$ coordinate of the vector or every vector in the array
        (has the same sign as $z$).
        """
        raise AssertionError

    @property
    def cottheta(self) -> ScalarCollection:
        r"""
        The $\cot\theta$ coordinate of the vector or every vector in the array
        (has the same sign as $z$).
        """
        raise AssertionError

    @property
    def mag(self) -> ScalarCollection:
        """
        The magnitude of the vector(s) in 3D (not including any temporal parts).
        """
        raise AssertionError

    @property
    def mag2(self) -> ScalarCollection:
        """
        The magnitude-squared of the vector(s) in 3D (not including any temporal parts).
        """
        raise AssertionError

    def cross(self, other: "VectorProtocol") -> "VectorProtocolSpatial":
        """
        The 3D cross-product of ``self`` with ``other``.

        Even if ``self`` or ``other`` is 4D, the resulting vector(s) is/are 3D.
        """
        raise AssertionError

    def deltaangle(self, other: "VectorProtocol") -> ScalarCollection:
        r"""
        Angle in 3D space between ``self`` and ``other``, which is always
        positive, between $0$ and $\pi$.
        """
        raise AssertionError

    def deltaeta(self, other: "VectorProtocol") -> ScalarCollection:
        r"""
        Signed difference in $\eta$ of ``self`` minus ``other``.
        """
        raise AssertionError

    def deltaR(self, other: "VectorProtocol") -> ScalarCollection:
        r"""
        Sum in quadrature of :doc:`vector._methods.VectorProtocolPlanar.deltaphi`
        and :doc:`vector._methods.VectorProtocolPlanar.deltaeta`:

        $$\Delta R = \sqrt{\Delta\phi^2 + \Delta\eta^2}$$
        """
        raise AssertionError

    def deltaR2(self, other: "VectorProtocol") -> ScalarCollection:
        r"""
        Square of the sum in quadrature of
        :doc:`vector._methods.VectorProtocolPlanar.deltaphi` and
        :doc:`vector._methods.VectorProtocolSpatial.deltaeta`:

        $$\Delta R^2 = \Delta\phi^2 + \Delta\eta^2$$
        """
        raise AssertionError

    def rotateX(self: SameVectorType, angle: ScalarCollection) -> SameVectorType:
        """
        Rotates the vector(s) by a given ``angle`` (in radians) around the
        $x$ axis.

        Note that the ``angle`` can be an array with the same length as the vectors,
        if the vectors are in an array.
        """
        raise AssertionError

    def rotateY(self: SameVectorType, angle: ScalarCollection) -> SameVectorType:
        """
        Rotates the vector(s) by a given ``angle`` (in radians) around the
        $y$ axis.

        Note that the ``angle`` can be an array with the same length as the vectors,
        if the vectors are in an array.
        """
        raise AssertionError

    def rotate_axis(
        self: SameVectorType, axis: "VectorProtocol", angle: ScalarCollection
    ) -> SameVectorType:
        """
        Rotates the vector(s) by a given ``angle`` (in radians) around the
        axis indicated by another vector, ``axis``. The magnitude of ``axis`` is
        ignored.

        Note that the ``axis`` and ``angle`` can be arrays with the same length
        as the vectors, if the vectors are in an array.
        """
        raise AssertionError

    def rotate_euler(
        self: SameVectorType,
        phi: ScalarCollection,
        theta: ScalarCollection,
        psi: ScalarCollection,
        order: str = "zxz",
    ) -> SameVectorType:
        """
        Rotates the vector(s) by three given angles: ``phi``, ``theta``, and ``psi``
        (in radians). The ``order`` string determines which axis each rotation is
        applied around:

        - ``"zxz"``, ``"xyx"``, ``"yzy"``, ``"zyz"``, ``"xzx"``, and ``"yxy"``
          are proper Euler angles
        - ``"zxz"``, ``"xyx"``, ``"yzy"``, ``"zyz"``, ``"xzx"``, and ``"yxy"``
          are Tait-Bryan angles (see
          :doc:`vector._methods.VectorProtocolSpatial.rotate_nautical`)

        The names ``phi``, ``theta``, and ``psi`` agree with
        `Wikipedia's terminology <https://en.wikipedia.org/wiki/Euler_angles>`_,
        and both the names and order agree with
        `ROOT's Math::EulerAngles <https://root.cern/doc/v612/classROOT_1_1Math_1_1EulerAngles.html>`_.
        The default ``order = "zxz"`` is also ROOT's convention.

        Note that the angles can be arrays with the same lengths as the vectors,
        if the vectors are in an array.
        """
        raise AssertionError

    def rotate_nautical(
        self: SameVectorType,
        yaw: ScalarCollection,
        pitch: ScalarCollection,
        roll: ScalarCollection,
    ) -> SameVectorType:
        """
        Rotates the vector(s) by three given angles: ``yaw``, ``pitch``, and ``roll``
        (in radians). These are Tait-Bryan angles often used for boats and planes
        (see `this lesson <http://planning.cs.uiuc.edu/node102.html>`_ and
        `this lesson <http://www.chrobotics.com/library/understanding-euler-angles>`_).

        This function is entirely equivalent to

        .. code-block:: python

            rotate_euler(roll, pitch, yaw, order="zyx")

        Note that the angles can be arrays with the same lengths as the vectors,
        if the vectors are in an array.
        """
        raise AssertionError

    def rotate_quaternion(
        self: SameVectorType,
        u: ScalarCollection,
        i: ScalarCollection,
        j: ScalarCollection,
        k: ScalarCollection,
    ) -> SameVectorType:
        """
        Rotates the vector(s) by four angles as quaternion coefficients (in radians).
        Four angles are sometimes preferred over three because the latter has a
        pathology known as "gimbal lock."

        This function follows the same conventions as
        `ROOT's Math::Quaternion <https://root.cern/doc/v612/classROOT_1_1Math_1_1Quaternion.html>`_.

        Note that the angles can be arrays with the same lengths as the vectors,
        if the vectors are in an array.
        """
        raise AssertionError

    def transform3D(self: SameVectorType, obj: TransformProtocol3D) -> SameVectorType:
        """
        Arbitrarily transforms the vector(s) by

        .. code-block:: python

            obj["xx"] obj["xy"] obj["xz"]
            obj["yx"] obj["yy"] obj["yz"]
            obj["zx"] obj["zy"] obj["zz"]

        leaving any temporal coordinate unchanged. There is no restriction on the
        type of ``obj``; it just has to provide those components (which can be
        arrays if the vectors are in an array).
        """
        raise AssertionError

    def is_parallel(
        self, other: "VectorProtocol", tolerance: ScalarCollection = 1e-5
    ) -> BoolCollection:
        raise AssertionError

    def is_antiparallel(
        self, other: "VectorProtocol", tolerance: ScalarCollection = 1e-5
    ) -> BoolCollection:
        raise AssertionError

    def is_perpendicular(
        self, other: "VectorProtocol", tolerance: ScalarCollection = 1e-5
    ) -> BoolCollection:
        raise AssertionError


class VectorProtocolLorentz(VectorProtocolSpatial):
    @property
    def temporal(self) -> Temporal:
        """
        Container of temporal coordinates, for use in dispatching to compute
        functions or to identify coordinate system with ``isinstance``.
        """
        raise AssertionError

    @property
    def t(self) -> ScalarCollection:
        r"""
        The Cartesian $t$ (time) coordinate of the vector or every vector in the array.

        If $t$ is derived from $\tau$, it is not allowed to be ``NaN``.

        .. code-block:: python

            t = sqrt(max(copysign(tau**2, tau) + mag**2, 0))
        """
        raise AssertionError

    @property
    def t2(self) -> ScalarCollection:
        r"""
        The Cartesian $t$ (time) coordinate squared of the vector or every vector
        in the array.

        If $t^2$ is derived from $\tau$, it is not allowed to be negative.

        .. code-block:: python

            t2 = max(copysign(tau**2, tau) + mag**2, 0)
        """
        raise AssertionError

    @property
    def tau(self) -> ScalarCollection:
        r"""
        The Lorentz magnitude $\tau$ (proper time) of the vector or every vector
        in the array.

        If $\tau$ is derived from $t$, spacelike vectors are represented by negative
        proper times.

        .. code-block:: python

            tau = copysign(sqrt(abs(t**2 - mag**2)), t**2 - mag**2)
        """
        raise AssertionError

    @property
    def tau2(self) -> ScalarCollection:
        r"""
        The Lorentz magnitude $\tau$ (proper time) squared of the vector or every
        vector in the array.

        .. code-block:: python

            tau2 = t**2 - mag**2
        """
        raise AssertionError

    @property
    def beta(self) -> ScalarCollection:
        """
        The speed(s) of the Lorentz vector or array of vectors, in which lightlike
        vectors have ``beta == 1``.
        """
        raise AssertionError

    @property
    def gamma(self) -> ScalarCollection:
        r"""
        The time dilation/length contraction factor(s) of the Lorentz vector or
        array of vectors: $t/\tau$.
        """
        raise AssertionError

    @property
    def rapidity(self) -> ScalarCollection:
        """
        The rapidity relative to the longitudinal axis of the Lorentz vector or
        array of vectors.

        .. code-block:: python

            0.5 * log((t + z) / (t - z))
        """
        raise AssertionError

    def boost_p4(self: SameVectorType, p4: "VectorProtocolLorentz") -> SameVectorType:
        """
        Boosts the vector or array of vectors in a direction and magnitude given
        by the 4D vector or array of vectors ``p4``.

        This function is equivalent to but more numerically stable than

        .. code-block:: python

            boost_beta3(p4.to_beta3)

        where :doc:`vector._methods.VectorProtocolLorentz.to_beta3` converts a
        4D Lorentz vector into a 3D velocity (in which lightlike velocities have
        ``mag == 1``).
        """
        raise AssertionError

    def boost_beta3(
        self: SameVectorType, beta3: "VectorProtocolSpatial"
    ) -> SameVectorType:
        """
        Boosts the vector or array of vectors in a direction and magnitude given
        by the 3D velocity or array of velocity vectors ``beta3``.
        """
        raise AssertionError

    def boost(self: SameVectorType, booster: "VectorProtocol") -> SameVectorType:
        """
        Boosts the vector or array of vectors using the 3D or 4D ``booster``.

        If ``booster`` is 3D, it is interpreted as a velocity (in which lightlike
        velocities have ``mag == 1``) and :doc:`vector._methods.VectorProtocolLorentz.boost_beta3`
        is called.

        If ``booster`` is 4D, it is interpreted as a Lorentz vector and
        :doc:`vector._methods.VectorProtocolLorentz.boost_p4` is called.
        """
        raise AssertionError

    def boostX(
        self: SameVectorType,
        beta: typing.Optional[ScalarCollection] = None,
        gamma: typing.Optional[ScalarCollection] = None,
    ) -> SameVectorType:
        """
        Boosts the vector or array of vectors in the $x$ direction by a speed
        ``beta`` (in which lightlike boosts have ``beta == 1``) or time dilation/length
        contraction factor ``gamma``.

        Either ``beta`` xor ``gamma`` must be specified, not both or neither.

        If ``beta`` or ``gamma`` is negative, it is taken as a boost in the $-x$
        direction.
        """
        raise AssertionError

    def boostY(
        self: SameVectorType,
        beta: typing.Optional[ScalarCollection] = None,
        gamma: typing.Optional[ScalarCollection] = None,
    ) -> SameVectorType:
        """
        Boosts the vector or array of vectors in the $y$ direction by a speed
        ``beta`` (in which lightlike boosts have ``beta == 1``) or time dilation/length
        contraction factor ``gamma``.

        Either ``beta`` xor ``gamma`` must be specified, not both or neither.

        If ``beta`` or ``gamma`` is negative, it is taken as a boost in the $-y$
        direction.
        """
        raise AssertionError

    def boostZ(
        self: SameVectorType,
        beta: typing.Optional[ScalarCollection] = None,
        gamma: typing.Optional[ScalarCollection] = None,
    ) -> SameVectorType:
        """
        Boosts the vector or array of vectors in the $z$ direction by a speed
        ``beta`` (in which lightlike boosts have ``beta == 1``) or time dilation/length
        contraction factor ``gamma``.

        Either ``beta`` xor ``gamma`` must be specified, not both or neither.

        If ``beta`` or ``gamma`` is negative, it is taken as a boost in the $-z$
        direction.
        """
        raise AssertionError

    def transform4D(self: SameVectorType, obj: TransformProtocol4D) -> SameVectorType:
        """
        Arbitrarily transforms the vector(s) by

        .. code-block:: python

            obj["xx"] obj["xy"] obj["xz"] obj["xt"]
            obj["yx"] obj["yy"] obj["yz"] obj["yt"]
            obj["zx"] obj["zy"] obj["zz"] obj["zt"]
            obj["tx"] obj["ty"] obj["tz"] obj["tt"]

        There is no restriction on the type of ``obj``; it just has to provide
        those components (which can be arrays if the vectors are in an array).
        """
        raise AssertionError

    def to_beta3(self) -> "VectorProtocolSpatial":
        """
        Converts the 4D Lorentz vector or array of vectors into a 3D velocity
        vector or array of vectors, in which lightlike velocities have
        ``mag == 1``.
        """
        raise AssertionError

    def is_timelike(self, tolerance: ScalarCollection = 0) -> BoolCollection:
        """
        Returns True if the vector or a vector in the array is pointing in a
        timelike direction, ``t**2 > mag**2``, False otherwise.

        The ``tolerance`` is in units of ``t`` and ``mag``. Note that

        - the default ``tolerance`` for :doc:`vector._methods.VectorProtocolLorentz.is_timelike`
          is ``0``
        - the default ``tolerance`` for :doc:`vector._methods.VectorProtocolLorentz.is_spacelike`
          is ``0``
        - the default ``tolerance`` for :doc:`vector._methods.VectorProtocolLorentz.is_lightlike`
          is ``1e-5``

        If you want to use these methods to divide space-time into non-overlapping
        regions (the light-cone), use the same ``tolerance`` for each.
        """
        raise AssertionError

    def is_spacelike(self, tolerance: ScalarCollection = 0) -> BoolCollection:
        """
        Returns True if the vector or a vector in the array is pointing in a
        spacelike direction, ``t**2 < mag**2``, False otherwise.

        The ``tolerance`` is in units of ``t`` and ``mag``. Note that

        - the default ``tolerance`` for :doc:`vector._methods.VectorProtocolLorentz.is_timelike`
          is ``0``
        - the default ``tolerance`` for :doc:`vector._methods.VectorProtocolLorentz.is_spacelike`
          is ``0``
        - the default ``tolerance`` for :doc:`vector._methods.VectorProtocolLorentz.is_lightlike`
          is ``1e-5``

        If you want to use these methods to divide space-time into non-overlapping
        regions (the light-cone), use the same ``tolerance`` for each.
        """
        raise AssertionError

    def is_lightlike(self, tolerance: ScalarCollection = 1e-5) -> BoolCollection:
        """
        Returns True if the vector or a vector in the array is pointing in a
        lightlike direction, ``t**2 == mag**2``, False otherwise.

        The ``tolerance`` is in units of ``t`` and ``mag``. Note that

        - the default ``tolerance`` for :doc:`vector._methods.VectorProtocolLorentz.is_timelike`
          is ``0``
        - the default ``tolerance`` for :doc:`vector._methods.VectorProtocolLorentz.is_spacelike`
          is ``0``
        - the default ``tolerance`` for :doc:`vector._methods.VectorProtocolLorentz.is_lightlike`
          is ``1e-5``

        If you want to use these methods to divide space-time into non-overlapping
        regions (the light-cone), use the same ``tolerance`` for each.
        """
        raise AssertionError


class MomentumProtocolPlanar(VectorProtocolPlanar):
    @property
    def px(self) -> ScalarCollection:
        """
        Momentum-synonym for :doc:`vector._methods.VectorProtocolPlanar.x`.
        """
        raise AssertionError

    @property
    def py(self) -> ScalarCollection:
        """
        Momentum-synonym for :doc:`vector._methods.VectorProtocolPlanar.y`.
        """
        raise AssertionError

    @property
    def pt(self) -> ScalarCollection:
        """
        Momentum-synonym for :doc:`vector._methods.VectorProtocolPlanar.rho`.
        """
        raise AssertionError

    @property
    def pt2(self) -> ScalarCollection:
        """
        Momentum-synonym for :doc:`vector._methods.VectorProtocolPlanar.rho2`.
        """
        raise AssertionError


class MomentumProtocolSpatial(VectorProtocolSpatial, MomentumProtocolPlanar):
    @property
    def pz(self) -> ScalarCollection:
        """
        Momentum-synonym for :doc:`vector._methods.VectorProtocolSpatial.z`.
        """
        raise AssertionError

    @property
    def pseudorapidity(self) -> ScalarCollection:
        """
        Momentum-synonym for :doc:`vector._methods.VectorProtocolSpatial.eta`.
        """
        raise AssertionError

    @property
    def p(self) -> ScalarCollection:
        """
        Momentum-synonym for :doc:`vector._methods.VectorProtocolSpatial.mag`.
        """
        raise AssertionError

    @property
    def p2(self) -> ScalarCollection:
        """
        Momentum-synonym for :doc:`vector._methods.VectorProtocolSpatial.mag2`.
        """
        raise AssertionError


class MomentumProtocolLorentz(VectorProtocolLorentz, MomentumProtocolSpatial):
    @property
    def E(self) -> ScalarCollection:
        """
        Momentum-synonym for :doc:`vector._methods.VectorProtocolLorentz.t`.
        """
        raise AssertionError

    @property
    def e(self) -> ScalarCollection:
        """
        Momentum-synonym for :doc:`vector._methods.VectorProtocolLorentz.t`.
        """
        raise AssertionError

    @property
    def energy(self) -> ScalarCollection:
        """
        Momentum-synonym for :doc:`vector._methods.VectorProtocolLorentz.t`.
        """
        raise AssertionError

    @property
    def E2(self) -> ScalarCollection:
        """
        Momentum-synonym for :doc:`vector._methods.VectorProtocolLorentz.t2`.
        """
        raise AssertionError

    @property
    def e2(self) -> ScalarCollection:
        """
        Momentum-synonym for :doc:`vector._methods.VectorProtocolLorentz.t2`.
        """
        raise AssertionError

    @property
    def energy2(self) -> ScalarCollection:
        """
        Momentum-synonym for :doc:`vector._methods.VectorProtocolLorentz.t2`.
        """
        raise AssertionError

    @property
    def M(self) -> ScalarCollection:
        """
        Momentum-synonym for :doc:`vector._methods.VectorProtocolLorentz.tau`.
        """
        raise AssertionError

    @property
    def m(self) -> ScalarCollection:
        """
        Momentum-synonym for :doc:`vector._methods.VectorProtocolLorentz.tau`.
        """
        raise AssertionError

    @property
    def mass(self) -> ScalarCollection:
        """
        Momentum-synonym for :doc:`vector._methods.VectorProtocolLorentz.tau`.
        """
        raise AssertionError

    @property
    def M2(self) -> ScalarCollection:
        """
        Momentum-synonym for :doc:`vector._methods.VectorProtocolLorentz.tau2`.
        """
        raise AssertionError

    @property
    def m2(self) -> ScalarCollection:
        """
        Momentum-synonym for :doc:`vector._methods.VectorProtocolLorentz.tau2`.
        """
        raise AssertionError

    @property
    def mass2(self) -> ScalarCollection:
        """
        Momentum-synonym for :doc:`vector._methods.VectorProtocolLorentz.tau2`.
        """
        raise AssertionError

    @property
    def Et(self) -> ScalarCollection:
        r"""
        Transverse energy of the four-momentum vector or array of vectors:
        $E_T = E \sin\theta$.
        """
        raise AssertionError

    @property
    def et(self) -> ScalarCollection:
        r"""
        Transverse energy of the four-momentum vector or array of vectors:
        $E_T = E \sin\theta$.
        """
        raise AssertionError

    @property
    def transverse_energy(self) -> ScalarCollection:
        """
        Synonym for :doc:`vector._methods.MomentumProtocolLorentz.Et`.
        """
        raise AssertionError

    @property
    def Et2(self) -> ScalarCollection:
        r"""
        Transverse energy squared of the four-momentum vector or array of
        vectors: $E_T^2 = E^2 \sin^2\theta$.
        """
        raise AssertionError

    @property
    def et2(self) -> ScalarCollection:
        r"""
        Transverse energy squared of the four-momentum vector or array of
        vectors: $E_T^2 = E^2 \sin^2\theta$.
        """
        raise AssertionError

    @property
    def transverse_energy2(self) -> ScalarCollection:
        """
        Synonym for :doc:`vector._methods.MomentumProtocolLorentz.Et2`.
        """
        raise AssertionError

    @property
    def Mt(self) -> ScalarCollection:
        r"""
        Transverse mass of the four-momentum vector or array of vectors:
        $M_T = \sqrt{t^2 - z^2}$.
        """
        raise AssertionError

    @property
    def mt(self) -> ScalarCollection:
        r"""
        Transverse mass of the four-momentum vector or array of vectors:
        $M_T = \sqrt{t^2 - z^2}$.
        """
        raise AssertionError

    @property
    def transverse_mass(self) -> ScalarCollection:
        """
        Synonym for :doc:`vector._methods.MomentumProtocolLorentz.Mt`.
        """
        raise AssertionError

    @property
    def Mt2(self) -> ScalarCollection:
        r"""
        Transverse mass squared of the four-momentum vector or array of vectors:
        $M_T^2 = t^2 - z^2$.
        """
        raise AssertionError

    @property
    def mt2(self) -> ScalarCollection:
        r"""
        Transverse mass squared of the four-momentum vector or array of vectors:
        $M_T^2 = t^2 - z^2$.
        """
        raise AssertionError

    @property
    def transverse_mass2(self) -> ScalarCollection:
        """
        Synonym for :doc:`vector._methods.MomentumProtocolLorentz.Mt2`.
        """
        raise AssertionError


class Vector(VectorProtocol):
    def to_xy(self) -> VectorProtocolPlanar:
        from vector._compute import planar

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self)),
            [AzimuthalXY, None],
            1,
        )

    def to_rhophi(self) -> VectorProtocolPlanar:
        from vector._compute import planar

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self)),
            [AzimuthalRhoPhi, None],
            1,
        )

    def to_xyz(self) -> VectorProtocolSpatial:
        from vector._compute import planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.z.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord),
            [AzimuthalXY, LongitudinalZ, None],
            1,
        )

    def to_xytheta(self) -> VectorProtocolSpatial:
        from vector._compute import planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.theta.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord),
            [AzimuthalXY, LongitudinalTheta, None],
            1,
        )

    def to_xyeta(self) -> VectorProtocolSpatial:
        from vector._compute import planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.eta.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord),
            [AzimuthalXY, LongitudinalEta, None],
            1,
        )

    def to_rhophiz(self) -> VectorProtocolSpatial:
        from vector._compute import planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.z.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord),
            [AzimuthalRhoPhi, LongitudinalZ, None],
            1,
        )

    def to_rhophitheta(self) -> VectorProtocolSpatial:
        from vector._compute import planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.theta.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord),
            [AzimuthalRhoPhi, LongitudinalTheta, None],
            1,
        )

    def to_rhophieta(self) -> VectorProtocolSpatial:
        from vector._compute import planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.eta.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord),
            [AzimuthalRhoPhi, LongitudinalEta, None],
            1,
        )

    def to_xyzt(self) -> VectorProtocolLorentz:
        from vector._compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.z.dispatch(self)
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.t.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord, tcoord),
            [AzimuthalXY, LongitudinalZ, TemporalT],
            1,
        )

    def to_xyztau(self) -> VectorProtocolLorentz:
        from vector._compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.z.dispatch(self)
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.tau.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord, tcoord),
            [AzimuthalXY, LongitudinalZ, TemporalTau],
            1,
        )

    def to_xythetat(self) -> VectorProtocolLorentz:
        from vector._compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.theta.dispatch(self)
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.t.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord, tcoord),
            [AzimuthalXY, LongitudinalTheta, TemporalT],
            1,
        )

    def to_xythetatau(self) -> VectorProtocolLorentz:
        from vector._compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.theta.dispatch(self)
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.tau.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord, tcoord),
            [AzimuthalXY, LongitudinalTheta, TemporalTau],
            1,
        )

    def to_xyetat(self) -> VectorProtocolLorentz:
        from vector._compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.eta.dispatch(self)
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.t.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord, tcoord),
            [AzimuthalXY, LongitudinalEta, TemporalT],
            1,
        )

    def to_xyetatau(self) -> VectorProtocolLorentz:
        from vector._compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.eta.dispatch(self)
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.tau.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.x.dispatch(self), planar.y.dispatch(self), lcoord, tcoord),
            [AzimuthalXY, LongitudinalEta, TemporalTau],
            1,
        )

    def to_rhophizt(self) -> VectorProtocolLorentz:
        from vector._compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.z.dispatch(self)
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.t.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord, tcoord),
            [AzimuthalRhoPhi, LongitudinalZ, TemporalT],
            1,
        )

    def to_rhophiztau(self) -> VectorProtocolLorentz:
        from vector._compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.z.dispatch(self)
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.tau.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord, tcoord),
            [AzimuthalRhoPhi, LongitudinalZ, TemporalTau],
            1,
        )

    def to_rhophithetat(self) -> VectorProtocolLorentz:
        from vector._compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.theta.dispatch(self)
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.t.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord, tcoord),
            [AzimuthalRhoPhi, LongitudinalTheta, TemporalT],
            1,
        )

    def to_rhophithetatau(self) -> VectorProtocolLorentz:
        from vector._compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.theta.dispatch(self)
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.tau.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord, tcoord),
            [AzimuthalRhoPhi, LongitudinalTheta, TemporalTau],
            1,
        )

    def to_rhophietat(self) -> VectorProtocolLorentz:
        from vector._compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.eta.dispatch(self)
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.t.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord, tcoord),
            [AzimuthalRhoPhi, LongitudinalEta, TemporalT],
            1,
        )

    def to_rhophietatau(self) -> VectorProtocolLorentz:
        from vector._compute import lorentz, planar, spatial

        lcoord = 0
        if isinstance(self, (Vector3D, Vector4D)):
            lcoord = spatial.eta.dispatch(self)
        tcoord = 0
        if isinstance(self, Vector4D):
            tcoord = lorentz.tau.dispatch(self)

        return self._wrap_result(
            type(self),
            (planar.rho.dispatch(self), planar.phi.dispatch(self), lcoord, tcoord),
            [AzimuthalRhoPhi, LongitudinalEta, TemporalTau],
            1,
        )


class Vector2D(Vector, VectorProtocolPlanar):
    def to_Vector2D(self) -> VectorProtocolPlanar:
        return self

    def to_Vector3D(self) -> VectorProtocolSpatial:
        return self._wrap_result(
            type(self),
            self.azimuthal.elements + (0,),
            [_aztype(self), LongitudinalZ, None],
            1,
        )

    def to_Vector4D(self) -> VectorProtocolLorentz:
        return self._wrap_result(
            type(self),
            self.azimuthal.elements + (0, 0),
            [_aztype(self), LongitudinalZ, TemporalT],
            1,
        )


class Vector3D(Vector, VectorProtocolSpatial):
    def to_Vector2D(self) -> VectorProtocolPlanar:
        return self._wrap_result(
            type(self),
            self.azimuthal.elements,
            [_aztype(self), None],
            1,
        )

    def to_Vector3D(self) -> VectorProtocolSpatial:
        return self

    def to_Vector4D(self) -> VectorProtocolLorentz:
        return self._wrap_result(
            type(self),
            self.azimuthal.elements + self.longitudinal.elements + (0,),
            [_aztype(self), _ltype(self), TemporalT],
            1,
        )


class Vector4D(Vector, VectorProtocolLorentz):
    def to_Vector2D(self) -> VectorProtocolPlanar:
        return self._wrap_result(
            type(self),
            self.azimuthal.elements,
            [_aztype(self), None],
            1,
        )

    def to_Vector3D(self) -> VectorProtocolSpatial:
        return self._wrap_result(
            type(self),
            self.azimuthal.elements + self.longitudinal.elements,
            [_aztype(self), _ltype(self), None],
            1,
        )

    def to_Vector4D(self) -> VectorProtocolLorentz:
        return self


class Planar(VectorProtocolPlanar):
    @property
    def x(self) -> ScalarCollection:
        from vector._compute.planar import x

        return x.dispatch(self)

    @property
    def y(self) -> ScalarCollection:
        from vector._compute.planar import y

        return y.dispatch(self)

    @property
    def rho(self) -> ScalarCollection:
        from vector._compute.planar import rho

        return rho.dispatch(self)

    @property
    def rho2(self) -> ScalarCollection:
        from vector._compute.planar import rho2

        return rho2.dispatch(self)

    @property
    def phi(self) -> ScalarCollection:
        from vector._compute.planar import phi

        return phi.dispatch(self)

    def deltaphi(self, other: VectorProtocol) -> ScalarCollection:
        from vector._compute.planar import deltaphi

        return deltaphi.dispatch(self, other)

    def rotateZ(self: SameVectorType, angle: ScalarCollection) -> SameVectorType:
        from vector._compute.planar import rotateZ

        return rotateZ.dispatch(angle, self)

    def transform2D(self: SameVectorType, obj: TransformProtocol2D) -> SameVectorType:
        from vector._compute.planar import transform2D

        return transform2D.dispatch(obj, self)

    def is_parallel(
        self, other: VectorProtocol, tolerance: ScalarCollection = 1e-5
    ) -> BoolCollection:
        from vector._compute.planar import is_parallel

        if not isinstance(other, Vector2D):
            return self.to_Vector3D().is_parallel(other, tolerance=tolerance)
        else:
            return is_parallel.dispatch(tolerance, self, other)

    def is_antiparallel(
        self, other: VectorProtocol, tolerance: ScalarCollection = 1e-5
    ) -> BoolCollection:
        from vector._compute.planar import is_antiparallel

        if not isinstance(other, Vector2D):
            return self.to_Vector3D().is_antiparallel(other, tolerance=tolerance)
        else:
            return is_antiparallel.dispatch(tolerance, self, other)

    def is_perpendicular(
        self, other: VectorProtocol, tolerance: ScalarCollection = 1e-5
    ) -> BoolCollection:
        from vector._compute.planar import is_perpendicular

        if not isinstance(other, Vector2D):
            return self.to_Vector3D().is_perpendicular(other, tolerance=tolerance)
        else:
            return is_perpendicular.dispatch(tolerance, self, other)

    def unit(self: SameVectorType) -> SameVectorType:
        from vector._compute.planar import unit

        return unit.dispatch(self)

    def dot(self, other: VectorProtocol) -> ScalarCollection:
        module = _compute_module_of(self, other)
        return module.dot.dispatch(self, other)

    def add(self, other: VectorProtocol) -> VectorProtocol:
        module = _compute_module_of(self, other)
        return module.add.dispatch(self, other)

    def subtract(self, other: VectorProtocol) -> VectorProtocol:
        module = _compute_module_of(self, other)
        return module.subtract.dispatch(self, other)

    def scale(self: SameVectorType, factor: ScalarCollection) -> SameVectorType:
        from vector._compute.planar import scale

        return scale.dispatch(factor, self)

    def equal(self, other: VectorProtocol) -> BoolCollection:
        from vector._compute.planar import equal

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return equal.dispatch(self, other)

    def not_equal(self, other: VectorProtocol) -> BoolCollection:
        from vector._compute.planar import not_equal

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return not_equal.dispatch(self, other)

    def isclose(
        self,
        other: VectorProtocol,
        rtol: ScalarCollection = 1e-05,
        atol: ScalarCollection = 1e-08,
        equal_nan: BoolCollection = False,
    ) -> BoolCollection:
        from vector._compute.planar import isclose

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return isclose.dispatch(rtol, atol, equal_nan, self, other)


class Spatial(Planar, VectorProtocolSpatial):
    @property
    def z(self) -> ScalarCollection:
        from vector._compute.spatial import z

        return z.dispatch(self)

    @property
    def theta(self) -> ScalarCollection:
        from vector._compute.spatial import theta

        return theta.dispatch(self)

    @property
    def eta(self) -> ScalarCollection:
        from vector._compute.spatial import eta

        return eta.dispatch(self)

    @property
    def costheta(self) -> ScalarCollection:
        from vector._compute.spatial import costheta

        return costheta.dispatch(self)

    @property
    def cottheta(self) -> ScalarCollection:
        from vector._compute.spatial import cottheta

        return cottheta.dispatch(self)

    @property
    def mag(self) -> ScalarCollection:
        from vector._compute.spatial import mag

        return mag.dispatch(self)

    @property
    def mag2(self) -> ScalarCollection:
        from vector._compute.spatial import mag2

        return mag2.dispatch(self)

    def cross(self, other: VectorProtocol) -> VectorProtocolSpatial:
        from vector._compute.spatial import cross

        return cross.dispatch(self, other)

    def deltaangle(self, other: VectorProtocol) -> ScalarCollection:
        from vector._compute.spatial import deltaangle

        return deltaangle.dispatch(self, other)

    def deltaeta(self, other: VectorProtocol) -> ScalarCollection:
        from vector._compute.spatial import deltaeta

        return deltaeta.dispatch(self, other)

    def deltaR(self, other: VectorProtocol) -> ScalarCollection:
        from vector._compute.spatial import deltaR

        return deltaR.dispatch(self, other)

    def deltaR2(self, other: VectorProtocol) -> ScalarCollection:
        from vector._compute.spatial import deltaR2

        return deltaR2.dispatch(self, other)

    def rotateX(self: SameVectorType, angle: ScalarCollection) -> SameVectorType:
        from vector._compute.spatial import rotateX

        return rotateX.dispatch(angle, self)

    def rotateY(self: SameVectorType, angle: ScalarCollection) -> SameVectorType:
        from vector._compute.spatial import rotateY

        return rotateY.dispatch(angle, self)

    def rotate_axis(
        self: SameVectorType, axis: VectorProtocol, angle: ScalarCollection
    ) -> SameVectorType:
        from vector._compute.spatial import rotate_axis

        return rotate_axis.dispatch(angle, axis, self)

    def rotate_euler(
        self: SameVectorType,
        phi: ScalarCollection,
        theta: ScalarCollection,
        psi: ScalarCollection,
        order: str = "zxz",
    ) -> SameVectorType:
        from vector._compute.spatial import rotate_euler

        return rotate_euler.dispatch(phi, theta, psi, order.lower(), self)

    def rotate_nautical(
        self: SameVectorType,
        yaw: ScalarCollection,
        pitch: ScalarCollection,
        roll: ScalarCollection,
    ) -> SameVectorType:
        # The order of arguments is reversed because rotate_euler
        # follows ROOT's argument order: phi, theta, psi.
        from vector._compute.spatial import rotate_euler

        return rotate_euler.dispatch(roll, pitch, yaw, "zyx", self)

    def rotate_quaternion(
        self: SameVectorType,
        u: ScalarCollection,
        i: ScalarCollection,
        j: ScalarCollection,
        k: ScalarCollection,
    ) -> SameVectorType:
        from vector._compute.spatial import rotate_quaternion

        return rotate_quaternion.dispatch(u, i, j, k, self)

    def transform3D(self: SameVectorType, obj: TransformProtocol3D) -> SameVectorType:
        from vector._compute.spatial import transform3D

        return transform3D.dispatch(obj, self)

    def is_parallel(
        self, other: VectorProtocol, tolerance: ScalarCollection = 1e-5
    ) -> BoolCollection:
        from vector._compute.spatial import is_parallel

        if isinstance(other, Vector2D):
            return is_parallel.dispatch(tolerance, self, other.to_Vector3D())
        else:
            return is_parallel.dispatch(tolerance, self, other)

    def is_antiparallel(
        self, other: VectorProtocol, tolerance: ScalarCollection = 1e-5
    ) -> BoolCollection:
        from vector._compute.spatial import is_antiparallel

        if isinstance(other, Vector2D):
            return is_antiparallel.dispatch(tolerance, self, other.to_Vector3D())
        else:
            return is_antiparallel.dispatch(tolerance, self, other)

    def is_perpendicular(
        self, other: VectorProtocol, tolerance: ScalarCollection = 1e-5
    ) -> BoolCollection:
        from vector._compute.spatial import is_perpendicular

        if isinstance(other, Vector2D):
            return is_perpendicular.dispatch(tolerance, self, other.to_Vector3D())
        else:
            return is_perpendicular.dispatch(tolerance, self, other)

    def unit(self: SameVectorType) -> SameVectorType:
        from vector._compute.spatial import unit

        return unit.dispatch(self)

    def dot(self, other: VectorProtocol) -> ScalarCollection:
        module = _compute_module_of(self, other)
        return module.dot.dispatch(self, other)

    def add(self, other: VectorProtocol) -> VectorProtocol:
        module = _compute_module_of(self, other)
        return module.add.dispatch(self, other)

    def subtract(self, other: VectorProtocol) -> VectorProtocol:
        module = _compute_module_of(self, other)
        return module.subtract.dispatch(self, other)

    def scale(self: SameVectorType, factor: ScalarCollection) -> SameVectorType:
        from vector._compute.spatial import scale

        return scale.dispatch(factor, self)

    def equal(self, other: VectorProtocol) -> BoolCollection:
        from vector._compute.spatial import equal

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return equal.dispatch(self, other)

    def not_equal(self, other: VectorProtocol) -> BoolCollection:
        from vector._compute.spatial import not_equal

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return not_equal.dispatch(self, other)

    def isclose(
        self,
        other: VectorProtocol,
        rtol: ScalarCollection = 1e-05,
        atol: ScalarCollection = 1e-08,
        equal_nan: BoolCollection = False,
    ) -> BoolCollection:
        from vector._compute.spatial import isclose

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return isclose.dispatch(rtol, atol, equal_nan, self, other)


class Lorentz(Spatial, VectorProtocolLorentz):
    @property
    def t(self) -> ScalarCollection:
        from vector._compute.lorentz import t

        return t.dispatch(self)

    @property
    def t2(self) -> ScalarCollection:
        from vector._compute.lorentz import t2

        return t2.dispatch(self)

    @property
    def tau(self) -> ScalarCollection:
        from vector._compute.lorentz import tau

        return tau.dispatch(self)

    @property
    def tau2(self) -> ScalarCollection:
        from vector._compute.lorentz import tau2

        return tau2.dispatch(self)

    @property
    def beta(self) -> ScalarCollection:
        from vector._compute.lorentz import beta

        return beta.dispatch(self)

    @property
    def gamma(self) -> ScalarCollection:
        from vector._compute.lorentz import gamma

        return gamma.dispatch(self)

    @property
    def rapidity(self) -> ScalarCollection:
        from vector._compute.lorentz import rapidity

        return rapidity.dispatch(self)

    def boost_p4(self: SameVectorType, p4: VectorProtocolLorentz) -> SameVectorType:
        from vector._compute.lorentz import boost_p4

        return boost_p4.dispatch(self, p4)

    def boost_beta3(
        self: SameVectorType, beta3: VectorProtocolSpatial
    ) -> SameVectorType:
        from vector._compute.lorentz import boost_beta3

        return boost_beta3.dispatch(self, beta3)

    def boost(self: SameVectorType, booster: VectorProtocol) -> SameVectorType:
        from vector._compute.lorentz import boost_beta3, boost_p4

        if isinstance(booster, Vector3D):
            return boost_beta3.dispatch(self, booster)
        elif isinstance(booster, Vector4D):
            return boost_p4.dispatch(self, booster)
        else:
            raise TypeError(
                "specify a Vector3D to boost by beta (velocity with c=1) or "
                "a Vector4D to boost by a momentum 4-vector"
            )

    def boostX(
        self: SameVectorType,
        beta: typing.Optional[ScalarCollection] = None,
        gamma: typing.Optional[ScalarCollection] = None,
    ) -> SameVectorType:
        from vector._compute.lorentz import boostX_beta, boostX_gamma

        if beta is not None and gamma is None:
            return boostX_beta.dispatch(beta, self)
        elif beta is None and gamma is not None:
            return boostX_gamma.dispatch(gamma, self)
        else:
            raise TypeError("specify 'beta' xor 'gamma', not both or neither")

    def boostY(
        self: SameVectorType,
        beta: typing.Optional[ScalarCollection] = None,
        gamma: typing.Optional[ScalarCollection] = None,
    ) -> SameVectorType:
        from vector._compute.lorentz import boostY_beta, boostY_gamma

        if beta is not None and gamma is None:
            return boostY_beta.dispatch(beta, self)
        elif beta is None and gamma is not None:
            return boostY_gamma.dispatch(gamma, self)
        else:
            raise TypeError("specify 'beta' xor 'gamma', not both or neither")

    def boostZ(
        self: SameVectorType,
        beta: typing.Optional[ScalarCollection] = None,
        gamma: typing.Optional[ScalarCollection] = None,
    ) -> SameVectorType:
        from vector._compute.lorentz import boostZ_beta, boostZ_gamma

        if beta is not None and gamma is None:
            return boostZ_beta.dispatch(beta, self)
        elif beta is None and gamma is not None:
            return boostZ_gamma.dispatch(gamma, self)
        else:
            raise TypeError("specify 'beta' xor 'gamma', not both or neither")

    def transform4D(self: SameVectorType, obj: TransformProtocol4D) -> SameVectorType:
        from vector._compute.lorentz import transform4D

        return transform4D.dispatch(obj, self)

    def to_beta3(self) -> VectorProtocolSpatial:
        from vector._compute.lorentz import to_beta3

        return to_beta3.dispatch(self)

    def is_timelike(self, tolerance: ScalarCollection = 0) -> BoolCollection:
        from vector._compute.lorentz import is_timelike

        return is_timelike.dispatch(tolerance, self)

    def is_spacelike(self, tolerance: ScalarCollection = 0) -> BoolCollection:
        from vector._compute.lorentz import is_spacelike

        return is_spacelike.dispatch(tolerance, self)

    def is_lightlike(self, tolerance: ScalarCollection = 1e-5) -> BoolCollection:
        from vector._compute.lorentz import is_lightlike

        return is_lightlike.dispatch(tolerance, self)

    def unit(self: SameVectorType) -> SameVectorType:
        from vector._compute.lorentz import unit

        return unit.dispatch(self)

    def dot(self, other: VectorProtocol) -> ScalarCollection:
        module = _compute_module_of(self, other)
        return module.dot.dispatch(self, other)

    def add(self, other: VectorProtocol) -> VectorProtocol:
        module = _compute_module_of(self, other)
        return module.add.dispatch(self, other)

    def subtract(self, other: VectorProtocol) -> VectorProtocol:
        module = _compute_module_of(self, other)
        return module.subtract.dispatch(self, other)

    def scale(self: SameVectorType, factor: ScalarCollection) -> SameVectorType:
        from vector._compute.lorentz import scale

        return scale.dispatch(factor, self)

    def equal(self, other: VectorProtocol) -> BoolCollection:
        from vector._compute.lorentz import equal

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return equal.dispatch(self, other)

    def not_equal(self, other: VectorProtocol) -> BoolCollection:
        from vector._compute.lorentz import not_equal

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return not_equal.dispatch(self, other)

    def isclose(
        self,
        other: VectorProtocol,
        rtol: ScalarCollection = 1e-05,
        atol: ScalarCollection = 1e-08,
        equal_nan: BoolCollection = False,
    ) -> BoolCollection:
        from vector._compute.lorentz import isclose

        if dim(self) != dim(other):
            raise TypeError(
                f"{repr(self)} and {repr(other)} do not have the same dimension"
            )
        return isclose.dispatch(rtol, atol, equal_nan, self, other)


class Momentum:
    pass


class PlanarMomentum(Momentum, MomentumProtocolPlanar):
    @property
    def px(self) -> ScalarCollection:
        return self.x

    @property
    def py(self) -> ScalarCollection:
        return self.y

    @property
    def pt(self) -> ScalarCollection:
        return self.rho

    @property
    def pt2(self) -> ScalarCollection:
        return self.rho2


class SpatialMomentum(PlanarMomentum, MomentumProtocolSpatial):
    @property
    def pz(self) -> ScalarCollection:
        return self.z

    @property
    def pseudorapidity(self) -> ScalarCollection:
        return self.eta

    @property
    def p(self) -> ScalarCollection:
        return self.mag

    @property
    def p2(self) -> ScalarCollection:
        return self.mag2


class LorentzMomentum(SpatialMomentum, MomentumProtocolLorentz):
    @property
    def E(self) -> ScalarCollection:
        return self.t

    @property
    def e(self) -> ScalarCollection:
        return self.t

    @property
    def energy(self) -> ScalarCollection:
        return self.t

    @property
    def E2(self) -> ScalarCollection:
        return self.t2

    @property
    def e2(self) -> ScalarCollection:
        return self.t2

    @property
    def energy2(self) -> ScalarCollection:
        return self.t2

    @property
    def M(self) -> ScalarCollection:
        return self.tau

    @property
    def m(self) -> ScalarCollection:
        return self.tau

    @property
    def mass(self) -> ScalarCollection:
        return self.tau

    @property
    def M2(self) -> ScalarCollection:
        return self.tau2

    @property
    def m2(self) -> ScalarCollection:
        return self.tau2

    @property
    def mass2(self) -> ScalarCollection:
        return self.tau2

    @property
    def Et(self) -> ScalarCollection:
        from vector._compute.lorentz import Et

        return Et.dispatch(self)

    @property
    def et(self) -> ScalarCollection:
        from vector._compute.lorentz import Et

        return Et.dispatch(self)

    @property
    def transverse_energy(self) -> ScalarCollection:
        return self.Et

    @property
    def Et2(self) -> ScalarCollection:
        from vector._compute.lorentz import Et2

        return Et2.dispatch(self)

    @property
    def et2(self) -> ScalarCollection:
        from vector._compute.lorentz import Et2

        return Et2.dispatch(self)

    @property
    def transverse_energy2(self) -> ScalarCollection:
        return self.Et2

    @property
    def Mt(self) -> ScalarCollection:
        from vector._compute.lorentz import Mt

        return Mt.dispatch(self)

    @property
    def mt(self) -> ScalarCollection:
        from vector._compute.lorentz import Mt

        return Mt.dispatch(self)

    @property
    def transverse_mass(self) -> ScalarCollection:
        return self.Mt

    @property
    def Mt2(self) -> ScalarCollection:
        from vector._compute.lorentz import Mt2

        return Mt2.dispatch(self)

    @property
    def mt2(self) -> ScalarCollection:
        from vector._compute.lorentz import Mt2

        return Mt2.dispatch(self)

    @property
    def transverse_mass2(self) -> ScalarCollection:
        return self.Mt2


def dim(v: VectorProtocol) -> int:
    """
    Returns the number of dimensions in a vector: 2, 3, or 4.
    """
    if isinstance(v, Vector2D):
        return 2
    elif isinstance(v, Vector3D):
        return 3
    elif isinstance(v, Vector4D):
        return 4
    else:
        raise TypeError(f"{repr(v)} is not a vector.Vector")


def _compute_module_of(
    one: VectorProtocol, two: VectorProtocol, nontemporal: bool = False
) -> Module:
    """
    Determines which compute module to use for functions of two vectors
    (the one with minimum dimension).

    If ``nontemporal``, use a spatial module even if both vectors are 4D.
    """
    if not isinstance(one, Vector):
        raise TypeError(f"{repr(one)} is not a Vector")
    if not isinstance(two, Vector):
        raise TypeError(f"{repr(two)} is not a Vector")

    if isinstance(one, Vector2D):
        import vector._compute.planar

        return vector._compute.planar

    elif isinstance(one, Vector3D):
        if isinstance(two, Vector2D):
            import vector._compute.planar

            return vector._compute.planar
        else:
            import vector._compute.spatial

            return vector._compute.spatial

    elif isinstance(one, Vector4D):
        if isinstance(two, Vector2D):
            import vector._compute.planar

            return vector._compute.planar
        elif isinstance(two, Vector3D) or nontemporal:
            import vector._compute.spatial

            return vector._compute.spatial
        else:
            import vector._compute.lorentz

            return vector._compute.lorentz

    raise AssertionError(repr(one))


_coordinate_class_to_names = {
    AzimuthalXY: ("x", "y"),
    AzimuthalRhoPhi: ("rho", "phi"),
    LongitudinalZ: ("z",),
    LongitudinalTheta: ("theta",),
    LongitudinalEta: ("eta",),
    TemporalT: ("t",),
    TemporalTau: ("tau",),
}


_repr_generic_to_momentum = {
    "x": "px",
    "y": "py",
    "rho": "pt",
    "z": "pz",
    "t": "E",
    "tau": "mass",
}


_repr_momentum_to_generic = {
    "px": "x",
    "py": "y",
    "pt": "rho",
    "pz": "z",
    "E": "t",
    "e": "t",
    "energy": "t",
    "M": "tau",
    "m": "tau",
    "mass": "tau",
}


_coordinate_order = [
    "x",
    "px",
    "y",
    "py",
    "rho",
    "pt",
    "phi",
    "z",
    "pz",
    "theta",
    "eta",
    "t",
    "E",
    "e",
    "energy",
    "tau",
    "M",
    "m",
    "mass",
]


def _aztype(obj: VectorProtocolPlanar) -> typing.Type[Coordinates]:
    """
    Determines the Azimuthal type of a vector for use in looking up a
    dispatched function.
    """
    if hasattr(obj, "azimuthal"):
        for t in type(obj.azimuthal).__mro__:
            if t in (AzimuthalXY, AzimuthalRhoPhi):
                return t
    raise AssertionError(repr(obj))


def _ltype(obj: VectorProtocolSpatial) -> typing.Type[Coordinates]:
    """
    Determines the Longitudinal type of a vector for use in looking up a
    dispatched function.
    """
    if hasattr(obj, "longitudinal"):
        for t in type(obj.longitudinal).__mro__:
            if t in (LongitudinalZ, LongitudinalTheta, LongitudinalEta):
                return t
    raise AssertionError(repr(obj))


def _ttype(obj: VectorProtocolLorentz) -> typing.Type[Coordinates]:
    """
    Determines the Temporal type of a vector for use in looking up a
    dispatched function.
    """
    if hasattr(obj, "temporal"):
        for t in type(obj.temporal).__mro__:
            if t in (TemporalT, TemporalTau):
                return t
    raise AssertionError(repr(obj))


def _lib_of(*objects: VectorProtocol) -> Module:  # NumPy-like module
    """
    Determines the ``lib`` of a vector or set of vectors, complaining
    if they're incompatible.
    """
    lib = None
    for obj in objects:
        if isinstance(obj, Vector):
            if lib is None:
                lib = obj.lib
            elif lib is not obj.lib:
                raise TypeError(
                    f"cannot use {lib} and {obj.lib} in the same calculation"
                )

    assert lib is not None
    return lib


def _from_signature(
    name: str,
    dispatch_map: typing.Dict[typing.Any, typing.Any],
    signature: typing.Tuple[typing.Any, ...],
) -> typing.Tuple[typing.Any, ...]:
    """
    Gets a function and its return type from a ``dispatch_map`` and the
    ``signature`` to search for (complaining if none is found).
    """
    result = dispatch_map.get(signature)
    if result is None:
        raise TypeError(
            f"function {repr('.'.join(name.split('.')[-2:]))} has no signature {signature}"
        )
    return result


_handler_priority = [
    "vector._backends.object_",
    "vector._backends.numpy_",
    "vector._backends.awkward_",
]


def _handler_of(*objects: VectorProtocol) -> VectorProtocol:
    """
    Determines which vector should wrap the output of a dispatched function.

    Awkward Arrays have higher priority than NumPy arrays, which have higher
    priority than Python objects, which has the effect of "promoting" Python
    objects to NumPy arrays to Awkward Arrays whenever two are used in the
    same formula.
    """
    handler = None
    for obj in objects:
        if isinstance(obj, Vector):
            if handler is None:
                handler = obj
            elif _handler_priority.index(
                type(obj).__module__
            ) > _handler_priority.index(type(handler).__module__):
                handler = obj

    assert handler is not None
    return handler


def _flavor_of(*objects: VectorProtocol) -> typing.Type[VectorProtocol]:
    """
    Determines the flavor of the output of a dispatched function, where
    "flavor" is generic vs momentum.
    """
    from vector._backends.numpy_ import VectorNumpy
    from vector._backends.object_ import VectorObject

    handler = None
    is_momentum = True
    for obj in objects:
        if isinstance(obj, Vector):
            if not isinstance(obj, Momentum):
                is_momentum = False
            if handler is None:
                handler = obj
            elif isinstance(obj, VectorObject):
                pass
            elif isinstance(obj, VectorNumpy):
                handler = obj

    assert handler is not None
    if is_momentum:
        return type(handler)
    else:
        return handler.GenericClass
