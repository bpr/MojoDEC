# Time stepping

`mojodec.routines.time` has three schemes:
- **`Leapfrog`** steps linear wave systems explicitly. It is the
  Störmer–Verlet method, which for Maxwell's equations is Yee's scheme.
  With DEC's diagonal Hodge stars as mass matrices every step is explicit,
  and a discrete energy is conserved exactly, for any number of steps.
- **`ImplicitMidpoint`** steps the same systems with one solve per step. In
  exchange it takes any symmetric positive definite mass for one field
  (the Whitney mass), any time step, and conserves the plain energy.
- **`CrankNicolson`** is the same rule for diffusion, $M u' = -K u + f$.

Each has a typed version (`TypedLeapfrog`, `TypedImplicitMidpoint`,
`TypedCrankNicolson`). Most of this page is about the leapfrog; the
implicit schemes are described [at the end](#implicit-schemes).

## The systems it solves

Two fields $q$ and $p$, with diagonal positive mass matrices $M_q$ and
$M_p$ and one coupling matrix $K$:

$$
M_q \, \dot q = K p, \qquad M_p \, \dot p = -K^T q.
$$

The energy $H = \tfrac12 q^T M_q q + \tfrac12 p^T M_p p$ is conserved,
because the system is skew:

$$
\dot H = q^T M_q \dot q + p^T M_p \dot p = q^T K p - p^T K^T q = 0.
$$

`Leapfrog(Mq, Mp, K, dt)` takes the one matrix $K$ and builds both
couplings from it, so the system you give it is skew by construction.
There is no way to pass two couplings that are not each other's negative
transpose.

### Maxwell's equations in 2D

With the speed of light, permittivity, and permeability set to 1, the
transverse electric field in the plane is a 1-form $E$ and the magnetic
field a 2-form $B$ (its one component $B_z$ times the area form).
Faraday's law is $\partial_t B = -dE$, and Ampère's law (no current) is
$\partial_t E = \delta B$. Discretize $E$ as a primal 1-cochain $e$ on the
edges (its line integrals) and $B$ as a primal 2-cochain $b$ on the
triangles (its fluxes):

$$
\star_1 \dot e = d_1^T \star_2 b, \qquad
\star_2 \dot b = -\star_2 d_1 e.
$$

That is the system above with $q = e$, $p = b$, $M_q = \star_1$,
$M_p = \star_2$, and $K = d_1^T \star_2$. Multiplying out,
$\dot b = -d_1 e$ is Faraday's law exactly (Stokes' theorem on each
triangle), and $\dot e = \star_1^{-1} d_1^T \star_2 b$ is Ampère's law, the
codifferential of $b$.

### The acoustic wave equation

Pressure $\pi$ as a primal 0-cochain on the vertices and velocity $v$ as a
primal 1-cochain on the edges:

$$
\star_1 \dot v = \star_1 d_0 \pi, \qquad
\star_0 \dot \pi = -d_0^T \star_1 v,
$$

so $q = v$, $p = \pi$, $M_q = \star_1$, $M_p = \star_0$, $K = \star_1 d_0$.
Eliminating $v$ gives $\ddot \pi = -\star_0^{-1} d_0^T \star_1 d_0 \, \pi$,
the DEC wave equation $\ddot \pi = \Delta \pi$, with natural (Neumann)
conditions on the boundary.

## The scheme

One step of length $\Delta t$, with $h = \Delta t / 2$, is a half-kick, a
drift, and a half-kick:

$$
\begin{aligned}
p_{n+1/2} &= p_n - h \, M_p^{-1} K^T q_n, \\
q_{n+1} &= q_n + \Delta t \, M_q^{-1} K p_{n+1/2}, \\
p_{n+1} &= p_{n+1/2} - h \, M_p^{-1} K^T q_{n+1}.
\end{aligned}
$$

$M_q^{-1} K$ and $M_p^{-1} K^T$ are formed once, as row scalings, so a step
costs three sparse matrix–vector products and three vector updates, into
preallocated buffers. Nothing is solved. The scheme is second-order
accurate, time-reversible, and symplectic.

`lf.step(q, p, steps=k)` advances both fields in place by k steps.

## The conserved energy

The leapfrog does not conserve $H$ itself, which oscillates by
$O(\Delta t^2)$, but it conserves a nearby **staggered energy** exactly:

$$
E(q, p) = \tfrac12 q^T M_q q + \tfrac12 p^T M_p p
  - \frac{\Delta t^2}{8} \, (K^T q)^T M_p^{-1} (K^T q).
$$

`lf.energy(q, p)` returns it. Why it is conserved: substitute
$p_n = p_{n+1/2} + h M_p^{-1} K^T q_n$ and the $\Delta t^2/8$ terms cancel,
leaving

$$
E(q_n, p_n) = \tfrac12 q_n^T M_q q_n + \tfrac12 p_{n+1/2}^T M_p p_{n+1/2}
  + h \, q_n^T K p_{n+1/2}.
$$

The same substitution with $p_{n+1} = p_{n+1/2} - h M_p^{-1} K^T q_{n+1}$
gives the same expression with $q_{n+1}$ and a minus sign on the last term.
Their difference is

$$
\tfrac12 (q_{n+1} - q_n)^T M_q (q_{n+1} + q_n)
  - h \, (q_{n+1} + q_n)^T K p_{n+1/2} = 0,
$$

since $M_q (q_{n+1} - q_n) = \Delta t \, K p_{n+1/2}$ and $h = \Delta t / 2$.
So $E$ is constant up to rounding: the CavityWaves example keeps it to
$2 \times 10^{-15}$ over five periods. Checking it is a strong test that a
time-stepping loop is wired correctly.

## Stability: the time-step limit

The middle expression above is a quadratic form in $(q_n, p_{n+1/2})$ with
matrix $\begin{pmatrix} M_q & hK \\ hK^T & M_p \end{pmatrix}$, positive
definite exactly when $M_q - h^2 K M_p^{-1} K^T$ is, that is when

$$
\Delta t < \frac{2}{\sqrt{\lambda_{\max}}}, \qquad
\lambda_{\max} = \text{largest eigenvalue of } M_q^{-1} K M_p^{-1} K^T.
$$

$\lambda_{\max}$ is the square of the highest frequency the mesh can carry.
Below the limit $E$ is a norm and conserved, so the fields stay bounded:
the scheme is stable. Above it, high-frequency modes grow exponentially.
This is the CFL condition; $\lambda_{\max}$ grows like $1/h^2$ with the edge
length $h$, so the limit shrinks in proportion to the smallest edges.

The constructor estimates $\lambda_{\max}$ by power iteration and raises
if `dt` is not below the limit, giving the limit in the message;
`lf.max_stable_dt` holds it. Power iteration approaches $\lambda_{\max}$
from below, so the reported limit can sit above the true one by the
iteration's tolerance (relative $10^{-13}$ when it converges); stay a
little below it. Half the limit is a sound default:

```mojo
var probe = TypedLeapfrog(Mq, Mp, K, 1e-12)        # any tiny dt
var lf = TypedLeapfrog(Mq, Mp, K, 0.5 * probe.max_stable_dt())
```

## Dispersion

A mode of frequency $\omega$ (an eigenvector of
$M_q^{-1} K M_p^{-1} K^T$ with eigenvalue $\omega^2$) started with
$p = 0$ evolves exactly as $q_n = \cos(n \theta) \, q_0$ with
$\cos \theta = 1 - (\omega \Delta t)^2 / 2$, that is, at the frequency

$$
\omega_{\Delta t} = \frac{2}{\Delta t} \arcsin \frac{\omega \Delta t}{2}
  \approx \omega \left(1 + \frac{(\omega \Delta t)^2}{24}\right),
$$

slightly faster than $\omega$. The tests and the example check this to
$10^{-9}$ and $3 \times 10^{-12}$. The space discretization adds its own
$O(h^2)$ error to $\omega$ itself.

## Masses must be positive: mesh quality

The constructor raises if any mass entry is not positive and finite. With
Hodge stars as masses, that is a condition on the mesh:

- $\star_2 = 1 / \text{area}$ on triangles is always positive.
- $\star_1$ on an edge is the ratio of its dual edge's signed length to
  its own length. For an interior edge whose opposite angles in the two
  triangles are $\alpha$ and $\beta$, $\star_1 = (\cot \alpha + \cot \beta)/2$,
  positive exactly when $\alpha + \beta < \pi$: the **Delaunay** condition.
  For a boundary edge, $\star_1 = \cot \alpha / 2$, positive when its
  opposite angle is acute.
- $\star_0$ (the dual area around a vertex) can be negative on poor meshes
  too.

So the leapfrog needs a Delaunay mesh with acute angles opposite boundary
edges, and a well-centered mesh (every circumcenter inside its triangle)
is always enough. Check with `well_centered_fraction(sc)` from
`mojodec.routines.quality`. ResonantCavity's mesh, for example, is not
well-centered and has negative $\star_1$ entries, so the leapfrog refuses
it.
(Its eigenvalue computation uses the Whitney mass matrix, which is
positive definite on any mesh but not diagonal.)

A Whitney (Galerkin) mass would need a linear solve per step; this
leapfrog takes diagonal masses only.

## Boundary conditions

A perfectly conducting wall makes the tangential electric field zero:
$e = 0$ on boundary edges. Zero those rows of $K$ with a projection that
drops the boundary edges, and start with $e = 0$ there:

```mojo
var boundary = tc.boundary_cells[1]()
var P = tc.identity[1, PRIMAL]().exclude_subset(boundary)
var K = P.T() @ tc.d[1, PRIMAL]().T() @ tc.star[2, PRIMAL]().to_matrix_op()
```

The boundary rows of $M_q^{-1} K$ are then zero, so $e$ never changes
there, and since the same $K$ gives $-K^T$, the system stays skew and the
energy is still conserved. The same pattern fixes any field on any
`TypedSubset`. Natural conditions (Neumann for the acoustic pressure,
"magnetic wall" for Maxwell) need nothing.

## The typed API

`TypedLeapfrog(Mq, Mp, K, dt)` takes typed operators and infers the kinds
of the two fields from them:

- `Mq` must map $q$'s kind to its dual kind (a $k$-cochain to an
  $(n-k)$-cochain of the other primality), and so must `Mp` for $p$;
- `K` must map $p$'s kind to the kind `Mq` gives.

Anything else fails to compile. For Maxwell, `TypedLeapfrog(tc.star[1,
PRIMAL](), tc.star[2, PRIMAL](), K, dt)` is a
`TypedLeapfrog[2, 1, PRIMAL, 2, PRIMAL]`, and `lf.step(e, b)` accepts only a
primal 1-cochain and a primal 2-cochain. It also raises if the operators or
the fields come from different complexes. `lf.dt()` and
`lf.max_stable_dt()` give the step and the limit, and `lf.core` the
untyped `Leapfrog`.

## Example: CavityWaves

`pixi run examples CavityWaves` runs Maxwell's equations in an
equilateral-triangle cavity of side 1, cut into 256 equilateral triangles
(every star positive). It:

1. computes the resonances, eigenvalues of $K d_1 P e = \omega^2 \star_1 e$:
   $\omega^2 / (4\pi/3)^2 = 0.9986, 0.9986, 2.987, 3.977, 3.977, 6.93,
   6.93, 8.88$, against the exact $m^2 + mn + n^2 = 1, 1, 3, 4, 4, 7, 7, 9$
   (the Neumann eigenvalues of the equilateral triangle, which are the TE
   resonances);
2. starts from the lowest mode with $b = 0$ at half the stability limit;
3. steps five periods (586 steps) and finds the amplitude
   $\langle e(t), e_0 \rangle / \langle e_0, e_0 \rangle$ equal to
   $\cos(\omega_{\Delta t} t)$ within $3 \times 10^{-12}$, and the staggered
   energy constant within $2 \times 10^{-15}$, while $H$ oscillates by
   $7 \times 10^{-4}$ (the `energy.png` figure).

It writes `waves.txt`, `amplitude.png`, `energy.png`, and `waves.vtu`
(the initial $E$ field and $B_z$ a quarter period later).

## Implicit schemes

The **implicit midpoint rule** evaluates the right-hand side at the
midpoint of each step:

$$
M_q \frac{q_1 - q_0}{\Delta t} = K \, \frac{p_0 + p_1}{2}, \qquad
M_p \frac{p_1 - p_0}{\Delta t} = -K^T \frac{q_0 + q_1}{2} .
$$

It is a Gauss–Legendre Runge–Kutta method with one stage: second order,
symplectic, and A-stable. It conserves every quadratic invariant of the
system exactly. Here that is the plain energy
$\tfrac12 q^T M_q q + \tfrac12 p^T M_p p$, with no staggered correction and
for **any** $\Delta t$.

A mode of frequency $\omega$ turns by $\theta = 2 \arctan(\omega \Delta
t / 2)$ per step, so

$$
\omega_{\Delta t} = \frac{2}{\Delta t} \arctan\frac{\omega \Delta t}{2}
  \approx \omega \left(1 - \frac{(\omega \Delta t)^2}{12}\right),
$$

slightly slower than $\omega$; the leapfrog errs by half as much, in the
other direction.
Large steps are stable but resolve fast modes badly: at
$\omega \Delta t = 2$ the phase per step is $\pi/2$ instead of 2.

**One solve per step.** Write $q_m$, $p_m$ for the midpoint values and
eliminate $p_m$:

$$
\Big(M_q + \frac{\Delta t^2}{4} K M_p^{-1} K^T\Big)\, q_m = M_q q_0 +
\frac{\Delta t}{2} K p_0, \qquad p_m = p_0 - \frac{\Delta t}{2} M_p^{-1}
K^T q_m ,
$$

then $q_1 = 2 q_m - q_0$ and $p_1 = 2 p_m - p_0$. The matrix is symmetric
positive definite and is assembled once. Each step solves it with
Jacobi-preconditioned CG for the change $q_m - q_0$, so the tolerance
`rtol` (default $10^{-12}$) applies to that change and the energy stays
constant to about that relative accuracy.

This needs $M_p^{-1}$, so **$M_p$ must be diagonal**, while **$M_q$ may be
any symmetric positive definite matrix**. For Maxwell's equations in 2D:
- put $e$ on the edges with the Whitney mass `whitney_innerproduct(sc, 1)`;
- put $b$ on the triangles with $\star_2 = 1/\text{area}$, which is
  positive on every mesh.

Then no well-centered or Delaunay mesh is needed. A Kuhn grid, whose
diagonal edges have zero dual length, works; the leapfrog refuses it.

```mojo
var W1 = whitney_innerproduct[1](tc)
var star2 = tc.star[2, PRIMAL]()
var K = tc.d[1, PRIMAL]().T() @ star2.to_matrix_op()
var im = TypedImplicitMidpoint(W1, star2, K, dt)   # any dt > 0
im.step(e, b, steps=100)
```

**Crank–Nicolson** is the same rule for $M u' = -K u + f$, with $M$
symmetric positive definite (a star or a Whitney mass) and $K$ symmetric
positive semidefinite (a stiffness $d^T \star d$):

$$
\Big(M + \frac{\Delta t}{2} K\Big) u_1 = \Big(M - \frac{\Delta t}{2} K\Big)
u_0 + \Delta t\, f .
$$

Without a source, $u^T M u$ and $u^T K u$ never increase. A mode with
$K x = \lambda M x$ is multiplied by
$g = (1 - \Delta t \lambda/2) / (1 + \Delta t \lambda / 2)$ per step.

The scheme is A-stable but not L-stable: for $\Delta t \lambda \gg 1$,
$g \to -1$. Stiff modes then flip sign every step and decay slowly, which
shows as checkerboard oscillations after a rough start or a jump in the
boundary data. Take a few small steps first, or smaller steps throughout,
if that matters.

With a constant source, `step(u, f, steps)` approaches the steady state
$K u = f$. Here `f` is a load vector, such as `whitney_load` or a mass
times nodal values; the typed version types it as the dual kind.

**Boundary conditions.** Fix values the way the leapfrog does: drop the
fixed cells from $q$ (or $u$) and from the rows of $K$ and the mass, as
with `select(free, free)`. Nonzero Dirichlet data enters as a source.

## Limitations

- The leapfrog takes diagonal masses only. With a Whitney mass, use
  `ImplicitMidpoint`.
- No currents or damping terms in the wave schemes; the system is the skew
  one above.
- The implicit schemes are one-stage, so second order. Higher-order Gauss
  or Radau IIA stages would need Kronecker block systems.

## Tests to read

`tests/test_time.mojo`:
- **Leapfrog:**
  - an oscillator against $\cos(n\theta)$ for 1,000 steps;
  - the constructor's errors;
  - the Maxwell cavity mode against the dense eigensolver, including the
    stability limit.
- **Implicit midpoint:**
  - the oscillator against $\cos(n\theta)$ with
    $\theta = 2\arctan(\omega \Delta t/2)$, at twice the leapfrog's
    limit, energy constant to $10^{-14}$;
  - Maxwell on a Kuhn grid with the Whitney mass, which the leapfrog
    refuses: the lowest mode turns as predicted to $10^{-10}$ over 200
    steps, with energy constant to $10^{-12}$.
- **Crank–Nicolson:**
  - the slowest and stiffest modes against $g^n$, including $g < -0.9$ at
    a large step;
  - monotone decay of both energies;
  - the steady state with the consistent and a lumped mass;
  - the typed versions and the errors.

## Reference

E. Hairer, C. Lubich, G. Wanner. "Geometric numerical integration
illustrated by the Störmer–Verlet method." *Acta Numerica* 12 (2003),
399–450. [DOI 10.1017/S0962492902000144](https://doi.org/10.1017/S0962492902000144)
