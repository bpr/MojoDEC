# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Time stepping (`mojodec.routines.time`).
========================================
The explicit leapfrog (Störmer–Verlet; Yee's scheme for Maxwell's
equations) for linear wave systems; not in PyDEC.

The system is two fields `q` and `p` with diagonal masses,

    Mq q' =  K p
    Mp p' = -K^T q,

skew by construction, so it conserves `1/2 q^T Mq q + 1/2 p^T Mp p`. With
Hodge stars as the masses every step is explicit: two sparse products and
two row scalings, no solve. Maxwell's equations on a 2D mesh, with the
electric field `e` a primal 1-cochain and the magnetic flux `b` a primal
2-cochain, are `star1 e' = d1^T star2 b`, `star2 b' = -star2 d1 e`; the
acoustic wave equation, pressure and velocity, has the same shape one
degree lower.

One step is a half-kick of `p`, a drift of `q`, and a half-kick of `p`.
It is symplectic and second order, and conserves the staggered energy

    E = 1/2 q^T Mq q + 1/2 p^T Mp p - dt^2/8 (K^T q)^T Mp^-1 (K^T q)

to rounding, for any number of steps. `E` is positive, and the scheme
stable, exactly when `dt < 2 / sqrt(lambda_max)`, `lambda_max` the largest
eigenvalue of `Mq^-1 K Mp^-1 K^T` (the squared highest frequency); the
constructor raises for a larger `dt`. A mode of frequency `omega` advances
at `omega_dt = (2/dt) asin(omega dt / 2)`, slightly faster.

```mojo
var e = tc.cochain[1, PRIMAL]()            # electric field on the edges
var b = tc.cochain[2, PRIMAL]()            # magnetic flux through triangles
var K = tc.d[1, PRIMAL]().T() @ tc.star[2, PRIMAL]()
var lf = TypedLeapfrog(tc.star[1, PRIMAL](), tc.star[2, PRIMAL](), K, dt)
lf.step(e, b, steps=100)
print(lf.energy(e, b))                     # the same after every step
```
"""

from std.math import isfinite, sqrt

from mojodec.core.diagonal import DiagonalOperator
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.sparse import CSRMatrix
from mojodec.core.typed import DiagonalOp, MatrixOp, TypedCochain
from mojodec.core.utils import RealArray, axpy, real_vector


struct Leapfrog(Movable):
    """The explicit leapfrog for `Mq q' = K p`, `Mp p' = -K^T q`.

    See `mojodec.routines.time`. `TypedLeapfrog` is the same on typed
    cochains and operators.
    """

    var dt: Float64
    """The time step."""
    var max_stable_dt: Float64
    """The stability limit `2 / sqrt(lambda_max)`, `lambda_max` estimated
    by power iteration (from below, so the limit from above); infinite when
    `K` is zero."""
    var _mq: DiagonalOperator
    var _mp: DiagonalOperator
    var _A: CSRMatrix
    """`Mq^-1 K`: `q' = A p`."""
    var _B: CSRMatrix
    """`-Mp^-1 K^T`: `p' = B q`."""
    var _tq: RealArray
    var _tp: RealArray

    def __init__(
        out self,
        Mq: DiagonalOperator,
        Mp: DiagonalOperator,
        K: CSRMatrix,
        dt: Float64,
    ) raises:
        """The scheme for `Mq q' = K p`, `Mp p' = -K^T q` with step `dt`.

        Args:
            Mq: The mass of `q`, positive.
            Mp: The mass of `p`, positive.
            K: The coupling, `len(Mq) x len(Mp)`.
            dt: The time step, positive and below the stability limit.

        Raises:
            If the shapes do not match, a mass entry is not positive and
            finite, or `dt` is not positive or not below the stability
            limit (the message gives the limit).
        """
        var nq = len(Mq)
        var np = len(Mp)
        if K.nrows != nq or K.ncols != np:
            raise Error(
                String(
                    "Leapfrog: K is ",
                    K.nrows,
                    " x ",
                    K.ncols,
                    ", expected ",
                    nq,
                    " x ",
                    np,
                )
            )
        _require_positive(Mq, "Mq")
        _require_positive(Mp, "Mp")
        if not (dt > 0) or not isfinite(dt):
            raise Error(String("Leapfrog: dt must be positive, got ", dt))
        self.dt = dt
        self._A = Mq.inverse() @ K
        self._B = Mp.inverse() @ K.transpose() * -1.0
        self._mq = Mq.copy()
        self._mp = Mp.copy()
        self._tq = real_vector(nq)
        self._tp = real_vector(np)
        self.max_stable_dt = Float64.MAX
        var lam = self._max_eigenvalue()
        if lam > 0:
            self.max_stable_dt = 2 / sqrt(lam)
        if not dt < self.max_stable_dt:
            raise Error(
                String(
                    "Leapfrog: dt = ",
                    dt,
                    " is not below the stability limit ",
                    self.max_stable_dt,
                )
            )

    def step(
        mut self, mut q: RealArray, mut p: RealArray, steps: Int = 1
    ) raises:
        """Advance `q` and `p` in place.

        Args:
            q: The first field, length `len(Mq)`.
            p: The second field, length `len(Mp)`.
            steps: The number of steps of `dt`.

        Raises:
            If a field has the wrong length or `steps` is negative.
        """
        if q.size != self._tq.size or p.size != self._tp.size:
            raise Error("Leapfrog.step: field lengths do not match the masses")
        if steps < 0:
            raise Error("Leapfrog.step: steps must not be negative")
        var h = 0.5 * self.dt
        for _ in range(steps):
            self._B.matvec_into(q, self._tp)
            axpy(h, self._tp, p)
            self._A.matvec_into(p, self._tq)
            axpy(self.dt, self._tq, q)
            self._B.matvec_into(q, self._tp)
            axpy(h, self._tp, p)

    def energy(self, q: RealArray, p: RealArray) raises -> Float64:
        """The staggered energy, which `step` conserves to rounding.

        Args:
            q: The first field.
            p: The second field.

        Returns:
            `1/2 q^T Mq q + 1/2 p^T Mp p - dt^2/8 (K^T q)^T Mp^-1 (K^T q)`.

        Raises:
            If a field has the wrong length.
        """
        var bq = self._B @ q  # -Mp^-1 K^T q
        return (
            0.5 * self._mq.inner_product(q, q)
            + 0.5 * self._mp.inner_product(p, p)
            - self.dt * self.dt / 8 * self._mp.inner_product(bq, bq)
        )

    def _max_eigenvalue(mut self) raises -> Float64:
        """Largest eigenvalue of `-A B = Mq^-1 K Mp^-1 K^T`, self-adjoint in
        the `Mq` inner product, by power iteration with Rayleigh quotients.
        """
        var n = self._tq.size
        var x = real_vector(n)
        # a fixed, irregular start, not orthogonal to any mode in practice
        var s: UInt64 = 0x9E3779B97F4A7C15
        for i in range(n):
            s = s * 6364136223846793005 + 1442695040888963407
            x.unsafe_set(i, Float64((s >> 11) & 0xFFFFF) / 1048576.0 + 0.5)
        var y = real_vector(n)
        var lam: Float64 = 0
        for _ in range(2000):
            var nx = self._mq.inner_product(x, x)
            if nx == 0:
                return 0
            self._B.matvec_into(x, self._tp)
            self._A.matvec_into(self._tp, y)
            # y = -A B x
            var new = -self._mq.inner_product(x, y) / nx
            for i in range(n):
                x.unsafe_set(i, -y.unsafe_get(i) / sqrt(nx))
            if abs(new - lam) <= 1e-13 * abs(new):
                return new
            lam = new
        return lam


struct TypedLeapfrog[n: Int, kq: Int, pq: Bool, kp: Int, pp: Bool](Movable):
    """The leapfrog of `mojodec.routines.time` on typed cochains: `q` a
    `kq`-cochain, `p` a `kp`-cochain.

    The masses take each field to the dual kind, and `K` takes `p` to the
    kind `Mq` gives, so the kinds are checked when the scheme is built.

    Parameters:
        n: The complex dimension.
        kq: The degree of `q`.
        pq: Whether `q` is primal.
        kp: The degree of `p`.
        pp: Whether `p` is primal.
    """

    var core: Leapfrog
    """The untyped scheme."""
    var complex: Optional[SimplicialComplex]
    """The complex of the operators, if they carry one."""

    def __init__[
        oq: Int, opq: Bool, op: Int, opp: Bool
    ](
        out self,
        Mq: DiagonalOp[Self.n, Self.kq, Self.pq, oq, opq],
        Mp: DiagonalOp[Self.n, Self.kp, Self.pp, op, opp],
        K: MatrixOp[Self.n, Self.kp, Self.pp, oq, opq],
        dt: Float64,
    ) raises:
        """The scheme for `Mq q' = K p`, `Mp p' = -K^T q`.

        Parameters:
            oq: The degree `Mq` and `K` give, `n - kq`.
            opq: Whether `Mq` and `K` give primal cochains, `not pq`.
            op: The degree `Mp` gives, `n - kp`.
            opp: Whether `Mp` gives primal cochains, `not pp`.

        Args:
            Mq: The mass of `q`, such as `tc.star[kq, PRIMAL]()`.
            Mp: The mass of `p`.
            K: The coupling from `p` to the kind `Mq` gives.
            dt: The time step.

        Raises:
            If the operators are on different complexes, or as `Leapfrog`.
        """
        comptime assert oq == Self.n - Self.kq and opq == (
            not Self.pq
        ), "TypedLeapfrog: Mq must map q to the dual kind"
        comptime assert op == Self.n - Self.kp and opp == (
            not Self.pp
        ), "TypedLeapfrog: Mp must map p to the dual kind"
        var complex = Mq.complex
        for c in [Mp.complex, K.complex]:
            if not complex:
                complex = c
            elif c and not c.value().is_same(complex.value()):
                raise Error("TypedLeapfrog: operators on different complexes")
        self.core = Leapfrog(Mq.diag, Mp.diag, K.mat, dt)
        self.complex = complex^

    def step(
        mut self,
        mut q: TypedCochain[Self.n, Self.kq, Self.pq],
        mut p: TypedCochain[Self.n, Self.kp, Self.pp],
        steps: Int = 1,
    ) raises:
        """Advance `q` and `p` in place.

        Args:
            q: The first field.
            p: The second field.
            steps: The number of steps of `dt`.

        Raises:
            If a field is on another complex than the operators, or as
            `Leapfrog.step`.
        """
        self._check(q.complex)
        self._check(p.complex)
        self.core.step(q.v, p.v, steps)

    def energy(
        self,
        q: TypedCochain[Self.n, Self.kq, Self.pq],
        p: TypedCochain[Self.n, Self.kp, Self.pp],
    ) raises -> Float64:
        """The staggered energy, which `step` conserves to rounding.

        Args:
            q: The first field.
            p: The second field.

        Returns:
            See `Leapfrog.energy`.

        Raises:
            If a field is on another complex than the operators.
        """
        self._check(q.complex)
        self._check(p.complex)
        return self.core.energy(q.v, p.v)

    def dt(self) -> Float64:
        """The time step.

        Returns:
            `dt`.
        """
        return self.core.dt

    def max_stable_dt(self) -> Float64:
        """The stability limit; see `Leapfrog.max_stable_dt`.

        Returns:
            The largest stable time step.
        """
        return self.core.max_stable_dt

    def _check(self, c: SimplicialComplex) raises:
        if self.complex and not self.complex.value().is_same(c):
            raise Error("TypedLeapfrog: field on another complex")


def _require_positive(M: DiagonalOperator, name: String) raises:
    for i in range(len(M)):
        var m = M.values[i]
        if not (m > 0) or not isfinite(m):
            raise Error(
                String(
                    "Leapfrog: ",
                    name,
                    "[",
                    i,
                    "] = ",
                    m,
                    (
                        "; masses must be positive (a Hodge star on a mesh that"
                        " is not well-centered can be negative)"
                    ),
                )
            )
