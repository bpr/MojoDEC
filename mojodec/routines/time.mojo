# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Time stepping (`mojodec.routines.time`).
========================================
The explicit leapfrog (Störmer–Verlet; Yee's scheme for Maxwell's
equations) for linear wave systems, and the implicit midpoint rule for the
same systems (`ImplicitMidpoint`, any symmetric positive definite mass for
`q`, any `dt`) and for diffusion (`CrankNicolson`); not in PyDEC.

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
from mojodec.core.utils import RealArray, axpy, dot, real_vector
from mojodec.routines.linalg import cg, jacobi


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


def _require_positive(
    M: DiagonalOperator, name: String, who: String = "Leapfrog"
) raises:
    for i in range(len(M)):
        var m = M.values[i]
        if not (m > 0) or not isfinite(m):
            raise Error(
                String(
                    who,
                    ": ",
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


# ===----------------------------------------------------------------------=== #
# Implicit midpoint: the wave system with a general mass, and diffusion
# ===----------------------------------------------------------------------=== #


struct ImplicitMidpoint(Movable):
    """The implicit midpoint rule for `Mq q' = K p`, `Mp p' = -K^T q`.

    Each step evaluates the right-hand side at the midpoint of the step:
    `Mq (q1 - q0) = dt K pm`, `Mp (p1 - p0) = -dt K^T qm`, with
    `qm = (q0 + q1) / 2`, `pm = (p0 + p1) / 2`. It is second order,
    unconditionally stable, and conserves the energy
    `1/2 q^T Mq q + 1/2 p^T Mp p` exactly (up to the solve's tolerance),
    for any `dt`. A mode of frequency `omega` turns by
    `theta = 2 atan(omega dt / 2)` per step: slower than `omega dt`, where
    the leapfrog is faster.

    Eliminating `pm` leaves one symmetric positive definite solve per step,

        (Mq + dt^2/4 K Mp^-1 K^T) qm = Mq q0 + dt/2 K p0,

    so `Mp` must be diagonal and positive, while `Mq` may be any
    symmetric positive definite matrix, such as the Whitney mass matrix.
    For Maxwell's equations that puts the electric field on the edges with
    `whitney_innerproduct(sc, 1)` and the flux on the triangles with the
    star `1 / area`, which is positive on every mesh: unlike the leapfrog,
    no well-centered mesh is needed. The solve is Jacobi-preconditioned CG
    for the change of `qm` from `q0`.
    """

    var dt: Float64
    """The time step."""
    var rtol: Float64
    """The relative residual at which each step's CG stops."""
    var last_iterations: Int
    """CG iterations of the last step."""
    var _mq: CSRMatrix
    var _mp: DiagonalOperator
    var _K: CSRMatrix
    var _B: CSRMatrix
    """`Mp^-1 K^T`."""
    var _S: CSRMatrix
    """`Mq + dt^2/4 K Mp^-1 K^T`."""
    var _P: DiagonalOperator

    def __init__(
        out self,
        Mq: CSRMatrix,
        Mp: DiagonalOperator,
        K: CSRMatrix,
        dt: Float64,
        rtol: Float64 = 1e-12,
    ) raises:
        """The scheme with step `dt`.

        Args:
            Mq: The mass of `q`, symmetric positive definite.
            Mp: The mass of `p`, diagonal and positive.
            K: The coupling, `len(Mq) x len(Mp)`.
            dt: The time step, positive; any size is stable.
            rtol: The relative residual at which each step's CG stops.

        Raises:
            If the shapes do not match, an entry of `Mp` is not positive
            and finite, `Mq` is not symmetric, or `dt` or `rtol` is not
            positive.
        """
        var nq = Mq.nrows
        var np = len(Mp)
        if Mq.ncols != nq or K.nrows != nq or K.ncols != np:
            raise Error(
                String(
                    "ImplicitMidpoint: Mq is ",
                    Mq.nrows,
                    " x ",
                    Mq.ncols,
                    " and K ",
                    K.nrows,
                    " x ",
                    K.ncols,
                    "; expected Mq square and K ",
                    nq,
                    " x ",
                    np,
                )
            )
        _require_positive(Mp, "Mp", "ImplicitMidpoint")
        if not (dt > 0) or not isfinite(dt):
            raise Error(
                String("ImplicitMidpoint: dt must be positive, got ", dt)
            )
        if not (rtol > 0):
            raise Error("ImplicitMidpoint: rtol must be positive")
        self.dt = dt
        self.rtol = rtol
        self.last_iterations = 0
        self._B = Mp.inverse() @ K.transpose()
        self._S = Mq + (K @ self._B) * (dt * dt / 4)
        self._P = jacobi(self._S)
        self._mq = Mq.copy()
        self._mp = Mp.copy()
        self._K = K.copy()
        # cg checks symmetry on its first solve; check it here, once
        _ = cg(self._S, real_vector(nq), M=self._P, maxiter=0)

    def __init__(
        out self,
        Mq: DiagonalOperator,
        Mp: DiagonalOperator,
        K: CSRMatrix,
        dt: Float64,
        rtol: Float64 = 1e-12,
    ) raises:
        """The scheme with a diagonal `Mq`, such as a Hodge star.

        Args:
            Mq: The mass of `q`, diagonal and positive.
            Mp: The mass of `p`, diagonal and positive.
            K: The coupling, `len(Mq) x len(Mp)`.
            dt: The time step, positive; any size is stable.
            rtol: The relative residual at which each step's CG stops.

        Raises:
            As the overload with a general `Mq`, or if an entry of `Mq`
            is not positive and finite.
        """
        _require_positive(Mq, "Mq", "ImplicitMidpoint")
        self = Self(Mq.to_csr(), Mp, K, dt, rtol)

    def step(
        mut self, mut q: RealArray, mut p: RealArray, steps: Int = 1
    ) raises:
        """Advance `q` and `p` in place.

        Args:
            q: The first field, length `len(Mq)`.
            p: The second field, length `len(Mp)`.
            steps: The number of steps of `dt`.

        Raises:
            If a field has the wrong length, `steps` is negative, or CG
            does not converge.
        """
        if q.size != self._mq.nrows or p.size != len(self._mp):
            raise Error(
                "ImplicitMidpoint.step: field lengths do not match the masses"
            )
        if steps < 0:
            raise Error("ImplicitMidpoint.step: steps must not be negative")
        var h = 0.5 * self.dt
        for _ in range(steps):
            # S (qm - q) = h K (p - h B q)
            var w = p.copy()
            axpy(-h, self._B @ q, w)
            var rhs = self._K @ w
            for i in range(rhs.size):
                rhs.unsafe_set(i, h * rhs.unsafe_get(i))
            var res = cg(self._S, rhs, M=self._P, rtol=self.rtol)
            if not res.converged():
                raise Error(
                    String(
                        "ImplicitMidpoint.step: CG stopped (",
                        res.stop,
                        ") at residual ",
                        res.residual_norm,
                    )
                )
            self.last_iterations = res.iterations
            var dq = res^.take_x()
            # qm = q + dq, pm = p - h B qm; q1 = 2 qm - q, p1 = 2 pm - p
            axpy(1.0, dq, q)  # q = qm
            var bq = self._B @ q
            axpy(1.0, dq, q)  # q1 = qm + dq
            axpy(-self.dt, bq, p)

    def energy(self, q: RealArray, p: RealArray) raises -> Float64:
        """The energy, which `step` conserves.

        Args:
            q: The first field.
            p: The second field.

        Returns:
            `1/2 q^T Mq q + 1/2 p^T Mp p`.

        Raises:
            If a field has the wrong length.
        """
        return 0.5 * dot(q, self._mq @ q) + 0.5 * self._mp.inner_product(p, p)


struct CrankNicolson(Movable):
    """The implicit midpoint rule (Crank–Nicolson) for `M u' = -K u + f`.

    One step solves `(M + dt/2 K) u1 = (M - dt/2 K) u0 + dt f`, with `M`
    symmetric positive definite (a Hodge star or a Whitney mass) and `K`
    symmetric positive semidefinite (a stiffness `d^T star d`). It is
    second order and unconditionally stable: without a source, the energy
    `u^T M u` and the Dirichlet energy `u^T K u` never increase. A mode
    with `K x = lambda M x` is multiplied by
    `(1 - dt lambda / 2) / (1 + dt lambda / 2)` each step, which tends to
    -1 for `dt lambda >> 1`: stiff modes are damped slowly and flip sign
    every step instead of decaying (the scheme is A-stable, not L-stable).
    The solve is Jacobi-preconditioned CG for `u1 - u0`.
    """

    var dt: Float64
    """The time step."""
    var rtol: Float64
    """The relative residual at which each step's CG stops."""
    var last_iterations: Int
    """CG iterations of the last step."""
    var _M: CSRMatrix
    var _K: CSRMatrix
    var _A: CSRMatrix
    """`M + dt/2 K`."""
    var _P: DiagonalOperator

    def __init__(
        out self,
        M: CSRMatrix,
        K: CSRMatrix,
        dt: Float64,
        rtol: Float64 = 1e-12,
    ) raises:
        """The scheme with step `dt`.

        Args:
            M: The mass, symmetric positive definite.
            K: The stiffness, symmetric positive semidefinite, same size.
            dt: The time step, positive; any size is stable.
            rtol: The relative residual at which each step's CG stops.

        Raises:
            If the shapes do not match, `M + dt/2 K` is not symmetric or
            has a zero diagonal entry, or `dt` or `rtol` is not positive.
        """
        var n = M.nrows
        if M.ncols != n or K.nrows != n or K.ncols != n:
            raise Error(
                String(
                    "CrankNicolson: M is ",
                    M.nrows,
                    " x ",
                    M.ncols,
                    " and K ",
                    K.nrows,
                    " x ",
                    K.ncols,
                    "; both must be the same square size",
                )
            )
        if not (dt > 0) or not isfinite(dt):
            raise Error(String("CrankNicolson: dt must be positive, got ", dt))
        if not (rtol > 0):
            raise Error("CrankNicolson: rtol must be positive")
        self.dt = dt
        self.rtol = rtol
        self.last_iterations = 0
        self._A = M + K * (0.5 * dt)
        self._P = jacobi(self._A)
        self._M = M.copy()
        self._K = K.copy()
        _ = cg(self._A, real_vector(n), M=self._P, maxiter=0)

    def __init__(
        out self,
        M: DiagonalOperator,
        K: CSRMatrix,
        dt: Float64,
        rtol: Float64 = 1e-12,
    ) raises:
        """The scheme with a diagonal mass, such as a Hodge star.

        Args:
            M: The mass, diagonal and positive.
            K: The stiffness, symmetric positive semidefinite, same size.
            dt: The time step, positive; any size is stable.
            rtol: The relative residual at which each step's CG stops.

        Raises:
            As the overload with a general `M`, or if an entry of `M` is
            not positive and finite.
        """
        _require_positive(M, "M", "CrankNicolson")
        self = Self(M.to_csr(), K, dt, rtol)

    def step(mut self, mut u: RealArray, steps: Int = 1) raises:
        """Advance `u` in place, without a source.

        Args:
            u: The field, length `len(M)`.
            steps: The number of steps of `dt`.

        Raises:
            If `u` has the wrong length, `steps` is negative, or CG does
            not converge.
        """
        self._steps(u, None, steps)

    def step(mut self, mut u: RealArray, f: RealArray, steps: Int = 1) raises:
        """Advance `u` in place with a constant source `f`.

        Args:
            u: The field, length `len(M)`.
            f: The source, a load vector (`M` times a density, or
                `whitney_load`), of the same length.
            steps: The number of steps of `dt`.

        Raises:
            If `u` or `f` has the wrong length, `steps` is negative, or CG
            does not converge.
        """
        if f.size != self._M.nrows:
            raise Error("CrankNicolson.step: f has the wrong length")
        self._steps(u, f.copy(), steps)

    def _steps(
        mut self, mut u: RealArray, f: Optional[RealArray], steps: Int
    ) raises:
        if u.size != self._M.nrows:
            raise Error("CrankNicolson.step: u has the wrong length")
        if steps < 0:
            raise Error("CrankNicolson.step: steps must not be negative")
        for _ in range(steps):
            # (M + dt/2 K) (u1 - u0) = dt (f - K u0)
            var rhs = self._K @ u
            for i in range(rhs.size):
                var s = -rhs.unsafe_get(i)
                if f:
                    s += f.value().unsafe_get(i)
                rhs.unsafe_set(i, self.dt * s)
            var res = cg(self._A, rhs, M=self._P, rtol=self.rtol)
            if not res.converged():
                raise Error(
                    String(
                        "CrankNicolson.step: CG stopped (",
                        res.stop,
                        ") at residual ",
                        res.residual_norm,
                    )
                )
            self.last_iterations = res.iterations
            axpy(1.0, res^.take_x(), u)

    def energy(self, u: RealArray) raises -> Float64:
        """`1/2 u^T M u`, which never increases without a source.

        Args:
            u: The field.

        Returns:
            Half the squared `M` norm.

        Raises:
            If `u` has the wrong length.
        """
        return 0.5 * dot(u, self._M @ u)


def _join_complexes(
    ops: List[Optional[SimplicialComplex]], who: String
) raises -> Optional[SimplicialComplex]:
    """The complex the operators share, if any carries one."""
    var complex = Optional[SimplicialComplex](None)
    for ref c in ops:
        if not complex:
            complex = c.copy()
        elif c and not c.value().is_same(complex.value()):
            raise Error(who + ": operators on different complexes")
    return complex^


def _check_field(
    complex: Optional[SimplicialComplex], c: SimplicialComplex, who: String
) raises:
    if complex and not complex.value().is_same(c):
        raise Error(who + ": field on another complex")


struct TypedImplicitMidpoint[n: Int, kq: Int, pq: Bool, kp: Int, pp: Bool](
    Movable
):
    """`ImplicitMidpoint` on typed cochains: `q` a `kq`-cochain, `p` a
    `kp`-cochain, the kinds checked as `TypedLeapfrog` checks them.

    Parameters:
        n: The complex dimension.
        kq: The degree of `q`.
        pq: Whether `q` is primal.
        kp: The degree of `p`.
        pp: Whether `p` is primal.
    """

    var core: ImplicitMidpoint
    """The untyped scheme."""
    var complex: Optional[SimplicialComplex]
    """The complex of the operators, if they carry one."""

    def __init__[
        oq: Int, opq: Bool, op: Int, opp: Bool
    ](
        out self,
        Mq: MatrixOp[Self.n, Self.kq, Self.pq, oq, opq],
        Mp: DiagonalOp[Self.n, Self.kp, Self.pp, op, opp],
        K: MatrixOp[Self.n, Self.kp, Self.pp, oq, opq],
        dt: Float64,
        rtol: Float64 = 1e-12,
    ) raises:
        """The scheme with a general mass for `q`, such as
        `whitney_innerproduct[kq](tc)`.

        Parameters:
            oq: The degree `Mq` and `K` give, `n - kq`.
            opq: Whether `Mq` and `K` give primal cochains, `not pq`.
            op: The degree `Mp` gives, `n - kp`.
            opp: Whether `Mp` gives primal cochains, `not pp`.

        Args:
            Mq: The mass of `q`, symmetric positive definite.
            Mp: The mass of `p`, diagonal and positive.
            K: The coupling from `p` to the kind `Mq` gives.
            dt: The time step.
            rtol: The relative residual at which each step's CG stops.

        Raises:
            If the operators are on different complexes, or as
            `ImplicitMidpoint`.
        """
        comptime assert oq == Self.n - Self.kq and opq == (
            not Self.pq
        ), "TypedImplicitMidpoint: Mq must map q to the dual kind"
        comptime assert op == Self.n - Self.kp and opp == (
            not Self.pp
        ), "TypedImplicitMidpoint: Mp must map p to the dual kind"
        self.complex = _join_complexes(
            [Mq.complex, Mp.complex, K.complex], "TypedImplicitMidpoint"
        )
        self.core = ImplicitMidpoint(Mq.mat, Mp.diag, K.mat, dt, rtol)

    def __init__[
        oq: Int, opq: Bool, op: Int, opp: Bool
    ](
        out self,
        Mq: DiagonalOp[Self.n, Self.kq, Self.pq, oq, opq],
        Mp: DiagonalOp[Self.n, Self.kp, Self.pp, op, opp],
        K: MatrixOp[Self.n, Self.kp, Self.pp, oq, opq],
        dt: Float64,
        rtol: Float64 = 1e-12,
    ) raises:
        """The scheme with a diagonal mass for `q`, such as a Hodge star.

        Parameters:
            oq: The degree `Mq` and `K` give, `n - kq`.
            opq: Whether `Mq` and `K` give primal cochains, `not pq`.
            op: The degree `Mp` gives, `n - kp`.
            opp: Whether `Mp` gives primal cochains, `not pp`.

        Args:
            Mq: The mass of `q`, diagonal and positive.
            Mp: The mass of `p`, diagonal and positive.
            K: The coupling from `p` to the kind `Mq` gives.
            dt: The time step.
            rtol: The relative residual at which each step's CG stops.

        Raises:
            If the operators are on different complexes, or as
            `ImplicitMidpoint`.
        """
        comptime assert oq == Self.n - Self.kq and opq == (
            not Self.pq
        ), "TypedImplicitMidpoint: Mq must map q to the dual kind"
        comptime assert op == Self.n - Self.kp and opp == (
            not Self.pp
        ), "TypedImplicitMidpoint: Mp must map p to the dual kind"
        self.complex = _join_complexes(
            [Mq.complex, Mp.complex, K.complex], "TypedImplicitMidpoint"
        )
        self.core = ImplicitMidpoint(Mq.diag, Mp.diag, K.mat, dt, rtol)

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
            `ImplicitMidpoint.step`.
        """
        _check_field(self.complex, q.complex, "TypedImplicitMidpoint")
        _check_field(self.complex, p.complex, "TypedImplicitMidpoint")
        self.core.step(q.v, p.v, steps)

    def energy(
        self,
        q: TypedCochain[Self.n, Self.kq, Self.pq],
        p: TypedCochain[Self.n, Self.kp, Self.pp],
    ) raises -> Float64:
        """The energy, which `step` conserves.

        Args:
            q: The first field.
            p: The second field.

        Returns:
            See `ImplicitMidpoint.energy`.

        Raises:
            If a field is on another complex than the operators.
        """
        _check_field(self.complex, q.complex, "TypedImplicitMidpoint")
        _check_field(self.complex, p.complex, "TypedImplicitMidpoint")
        return self.core.energy(q.v, p.v)


struct TypedCrankNicolson[n: Int, k: Int, primal: Bool](Movable):
    """`CrankNicolson` on typed cochains: `u` a k-cochain, the mass and
    stiffness from its kind to the dual kind, and the source a load of
    the dual kind.

    Parameters:
        n: The complex dimension.
        k: The degree of `u`.
        primal: Whether `u` is primal.
    """

    var core: CrankNicolson
    """The untyped scheme."""
    var complex: Optional[SimplicialComplex]
    """The complex of the operators, if they carry one."""

    def __init__(
        out self,
        M: MatrixOp[
            Self.n, Self.k, Self.primal, Self.n - Self.k, not Self.primal
        ],
        K: MatrixOp[
            Self.n, Self.k, Self.primal, Self.n - Self.k, not Self.primal
        ],
        dt: Float64,
        rtol: Float64 = 1e-12,
    ) raises:
        """The scheme with a general mass, such as
        `whitney_innerproduct[k](tc)`.

        Args:
            M: The mass, symmetric positive definite.
            K: The stiffness, such as `d.T() @ star @ d`.
            dt: The time step.
            rtol: The relative residual at which each step's CG stops.

        Raises:
            If the operators are on different complexes, or as
            `CrankNicolson`.
        """
        self.complex = _join_complexes(
            [M.complex, K.complex], "TypedCrankNicolson"
        )
        self.core = CrankNicolson(M.mat, K.mat, dt, rtol)

    def __init__(
        out self,
        M: DiagonalOp[
            Self.n, Self.k, Self.primal, Self.n - Self.k, not Self.primal
        ],
        K: MatrixOp[
            Self.n, Self.k, Self.primal, Self.n - Self.k, not Self.primal
        ],
        dt: Float64,
        rtol: Float64 = 1e-12,
    ) raises:
        """The scheme with a diagonal mass, such as a Hodge star.

        Args:
            M: The mass, diagonal and positive.
            K: The stiffness.
            dt: The time step.
            rtol: The relative residual at which each step's CG stops.

        Raises:
            If the operators are on different complexes, or as
            `CrankNicolson`.
        """
        self.complex = _join_complexes(
            [M.complex, K.complex], "TypedCrankNicolson"
        )
        self.core = CrankNicolson(M.diag, K.mat, dt, rtol)

    def step(
        mut self,
        mut u: TypedCochain[Self.n, Self.k, Self.primal],
        steps: Int = 1,
    ) raises:
        """Advance `u` in place, without a source.

        Args:
            u: The field.
            steps: The number of steps of `dt`.

        Raises:
            If `u` is on another complex than the operators, or as
            `CrankNicolson.step`.
        """
        _check_field(self.complex, u.complex, "TypedCrankNicolson")
        self.core.step(u.v, steps)

    def step(
        mut self,
        mut u: TypedCochain[Self.n, Self.k, Self.primal],
        f: TypedCochain[Self.n, Self.n - Self.k, not Self.primal],
        steps: Int = 1,
    ) raises:
        """Advance `u` in place with a constant source.

        Args:
            u: The field.
            f: The source as a load, of the kind `M` gives.
            steps: The number of steps of `dt`.

        Raises:
            If a field is on another complex than the operators, or as
            `CrankNicolson.step`.
        """
        _check_field(self.complex, u.complex, "TypedCrankNicolson")
        _check_field(self.complex, f.complex, "TypedCrankNicolson")
        self.core.step(u.v, f.v, steps)

    def energy(
        self, u: TypedCochain[Self.n, Self.k, Self.primal]
    ) raises -> Float64:
        """`1/2 u^T M u`; see `CrankNicolson.energy`.

        Args:
            u: The field.

        Returns:
            Half the squared `M` norm.

        Raises:
            If `u` is on another complex than the operators.
        """
        _check_field(self.complex, u.complex, "TypedCrankNicolson")
        return self.core.energy(u.v)
