# The Hodge-Laplacian spectrum and the curvature of a triangulated sphere.
#
# On the unit sphere the Laplacian of 0-forms (and of 2-forms) has the
# eigenvalues n(n+1), n = 0, 1, 2, ..., with multiplicity 2n+1: the
# spherical harmonics. On 1-forms the eigenvalues are the same without 0,
# each with twice the multiplicity, since the sphere has no harmonic
# 1-forms. This program computes all three spectra on an icosphere and
# compares them with these values, after checking the Euler characteristic,
# Gauss-Bonnet, and the Gaussian and mean curvatures (both 1 on the unit
# sphere). The 1-form spectrum is also exactly the 0-form and 2-form
# spectra merged, without their zeros: the discrete Hodge decomposition.
#
# Run from the repository root:  pixi run examples Sphere [--show] [--subdivisions N]
# or from this directory:  pixi run mojo run -I ../.. -I ../../build driver.mojo [--show]
# --show also opens the figures in windows (see examples/README.md).
# --subdivisions N (default 2) refines the mesh: 10*4^N + 2 vertices.
# Writes spectrum.txt, spectrum.png, and sphere.vtu (curvatures and the
# first nonconstant eigenfunction, for ParaView).

from std.math import pi
from std.python import Python
from std.sys import argv

from mojodec import (
    CSRMatrix,
    SimplicialComplex,
    VTKField,
    angle_defect,
    eigh_generalized,
    gaussian_curvature,
    icosphere,
    laplace_derham,
    mean_curvature,
    mean_curvature_normal,
    shape_regularity,
    well_centered_fraction,
    write_vtu,
)
from mojodec.core.utils import (
    RealArray,
    fget,
    real_vector,
    real_vector_from_list,
)
from mojodec.vis import pyplot, show_requested


def main() raises:
    var subdivisions = 2
    var args = argv()
    for i in range(len(args) - 1):
        if args[i] == "--subdivisions":
            subdivisions = atol(args[i + 1])

    var sc = SimplicialComplex(icosphere(subdivisions))
    print(sc)

    # topology, mesh quality, and curvature
    var chi = sc.euler_characteristic()
    var defect: Float64 = 0
    var D = angle_defect(sc)
    for v in range(D.size):
        defect += D.unsafe_get(v)
    var reg = shape_regularity(sc)
    var worst: Float64 = 1
    for t in range(reg.size):
        worst = min(worst, reg.unsafe_get(t))
    print("Euler characteristic:", chi)
    print("total angle defect / (2 pi chi):", defect / (2 * pi * Float64(chi)))
    print("well-centered triangles:", well_centered_fraction(sc) * 100, "%")
    print("worst shape regularity:", worst)
    var K = gaussian_curvature(sc)
    var H = mean_curvature(sc)
    var Kr = _range(K)
    var Hr = _range(H)
    print("Gaussian curvature in [", Kr[0], ",", Kr[1], "] (exact: 1)")
    print("mean curvature in     [", Hr[0], ",", Hr[1], "] (exact: 1)")

    # spectra: (star_k Laplacian_k) x = lambda star_k x, a symmetric
    # generalized eigenproblem with the Hodge star as the mass matrix
    var NUM = 16
    var computed = List[List[Float64]]()
    var expected = List[List[Float64]]()
    var first_mode = real_vector(0)
    var full = List[RealArray]()
    with open("spectrum.txt", "w") as f:
        f.write("# k index computed expected\n")
        print("\nfirst", NUM, "eigenvalues of the Hodge Laplacian:")
        print("   k=0 (expected)        k=1 (expected)        k=2 (expected)")
        for k in range(3):
            ref star = sc[k].star
            var L = laplace_derham(sc.get_cochain_basis(k)).v.copy()
            var eig = eigh_generalized((star @ L).todense(), star)
            var values = List[Float64]()
            for i in range(NUM):
                values.append(eig.values.unsafe_get(i))
            var exact = _harmonic_spectrum(NUM, skip_zero=(k == 1))
            for i in range(NUM):
                f.write(k, " ", i, " ", values[i], " ", exact[i], "\n")
            if k == 0:
                # an l = 1 spherical harmonic: a linear function of position
                first_mode = real_vector(sc[0].num_simplices)
                for v in range(first_mode.size):
                    first_mode.unsafe_set(v, fget(eig.vectors, v, 1))
            computed.append(values^)
            expected.append(exact^)
            full.append(eig.values.copy())
        for i in range(NUM):
            var line = String("   ")
            for k in range(3):
                line += _fixed(computed[k][i]) + " (" + _fixed(expected[k][i])
                line += ")   "
            print(line)
    print("wrote spectrum.txt")

    # d d = 0 splits 1-forms into exact and coexact parts, and the sphere has
    # no harmonic 1-forms, so the 1-form spectrum is the nonzero 0-form and
    # 2-form spectra merged
    var merged = List[Float64]()
    for k in [0, 2]:
        for i in range(1, full[k].size):
            merged.append(full[k].unsafe_get(i))
    sort(merged)
    var gap: Float64 = 0
    for i in range(full[1].size):
        gap = max(gap, abs(full[1].unsafe_get(i) - merged[i]))
    print(
        (
            "1-form spectrum vs. merged nonzero 0- and 2-form spectra: max"
            " difference"
        ),
        gap,
    )

    var show = show_requested()
    var plt = pyplot(interactive=show)
    var fig_axes = plt.subplots(
        1, 3, figsize=Python.tuple(12.0, 4.0), sharey=True
    )
    var names: List[String] = ["0-forms", "1-forms", "2-forms"]
    for k in range(3):
        var ax = fig_axes[1][k]
        ax.plot(
            real_vector_from_list(expected[k]).to_numpy(),
            "k-",
            drawstyle="steps-mid",
            label="n(n+1)",
        )
        ax.plot(
            real_vector_from_list(computed[k]).to_numpy(), "ro", label="DEC"
        )
        ax.set_title(names[k])
        ax.set_xlabel("index")
    fig_axes[1][0].set_ylabel("eigenvalue")
    fig_axes[1][0].legend()
    plt.suptitle(
        "Hodge-Laplacian spectrum on an icosphere ("
        + String(sc[0].num_simplices)
        + " vertices)"
    )
    plt.savefig("spectrum.png")
    print("wrote spectrum.png")

    write_vtu(
        "sphere.vtu",
        sc.vertices(),
        sc.simplices(),
        point_data=[
            VTKField("gaussian_curvature", K),
            VTKField("mean_curvature", H),
            VTKField("mean_curvature_normal", mean_curvature_normal(sc)),
            VTKField("eigenfunction_l1", first_mode),
        ],
        format=.APPENDED,
    )
    print("wrote sphere.vtu")
    if show:
        plt.show()  # one window per figure; returns when all are closed


def _harmonic_spectrum(count: Int, skip_zero: Bool) -> List[Float64]:
    """n(n+1) with multiplicity 2n+1, or 2(2n+1) without n = 0 for 1-forms."""
    var out = List[Float64]()
    var n = 1 if skip_zero else 0
    while len(out) < count:
        var multiplicity = 2 * (2 * n + 1) if skip_zero else 2 * n + 1
        for _ in range(multiplicity):
            if len(out) < count:
                out.append(Float64(n * (n + 1)))
        n += 1
    return out^


def _range(x: RealArray) -> Tuple[Float64, Float64]:
    var lo = x.unsafe_get(0)
    var hi = lo
    for i in range(x.size):
        lo = min(lo, x.unsafe_get(i))
        hi = max(hi, x.unsafe_get(i))
    return (lo, hi)


def _fixed(x: Float64) -> String:
    """`x` rounded to 6 decimals, right-aligned in 9 characters."""
    var s = String(Float64(Int(x * 1e6 + (0.5 if x >= 0 else -0.5))) / 1e6)
    while s.byte_length() < 9:
        s = " " + s
    return s
