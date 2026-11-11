//! dexterior's and formoniq's side of the cross-library comparison.
//!
//! ```sh
//! dec-compare dexterior|formoniq MESH_DIR REPEATS [MESH...]
//! ```
//!
//! Prints one JSON object per line, as `mojodec_bench.mojo` does.
//! dexterior builds its whole complex, geometry included, in
//! `SimplicialMesh::new`, so it reports one `build` phase. formoniq takes
//! its geometry from squared edge lengths and has no circumcentric duals,
//! so its `geometry` phase is the edge lengths and has no star checks.

use std::{fs, time::Instant};

use dexterior::{Primal, SimplicialMesh, na};
use formoniq::whitney_complex::{HilbertComplex, WhitneyComplex};
use regge::coord::mesh::MeshCoords;
use simplicial::topology::{complex::Complex, simplex::Simplex, skeleton::Skeleton};

/// A rank-2 array from PyDEC's binary array format: a line holding the
/// number of `key=value` lines, those lines, then raw little-endian values.
struct Array<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

fn read_array<T: Copy, const W: usize>(
    path: &str,
    dtype: &str,
    from_le: fn([u8; W]) -> T,
) -> Array<T> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut pos = 0;
    let mut line = || {
        let end = pos + bytes[pos..].iter().position(|&b| b == b'\n').unwrap();
        let s = std::str::from_utf8(&bytes[pos..end]).unwrap().to_owned();
        pos = end + 1;
        s
    };
    let nkeys: usize = line().trim().parse().unwrap();
    let (mut rows, mut cols) = (0, 0);
    for _ in 0..nkeys {
        let l = line();
        let (key, value) = l.split_once('=').unwrap();
        match key {
            "dims" => {
                let d: Vec<usize> = value.split(',').map(|x| x.parse().unwrap()).collect();
                (rows, cols) = (d[0], d[1]);
            }
            "dtype" => assert_eq!(value, dtype, "{path}"),
            "format" => assert_eq!(value, "binary", "{path}"),
            _ => {}
        }
    }
    let data: Vec<T> = bytes[pos..]
        .chunks_exact(W)
        .map(|c| from_le(c.try_into().unwrap()))
        .collect();
    assert_eq!(data.len(), rows * cols, "{path}");
    Array { rows, cols, data }
}

fn emit(lib: &str, mesh: &str, phase: &str, ms: f64, checks: &[f64]) {
    let checks: Vec<String> = checks.iter().map(|c| format!("{c:?}")).collect();
    println!(
        r#"{{"lib": "{lib}", "mesh": "{mesh}", "phase": "{phase}", "ms": {ms:?}, "check": [{}]}}"#,
        checks.join(", ")
    );
}

/// Best time in ms of `run()` over `repeats` runs, and the last result.
fn best_of<T>(repeats: usize, mut run: impl FnMut() -> T) -> (f64, T) {
    let mut best = f64::INFINITY;
    let mut result = None;
    for _ in 0..repeats {
        let t = Instant::now();
        let r = run();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        result = Some(r);
    }
    (best, result.unwrap())
}

macro_rules! dexterior_bench {
    ($name:ident, $dim:literal, $($k:literal),+) => {
        fn $name(mesh: &str, v: &Array<f64>, e: &Array<i64>, repeats: usize) {
            let vertices: Vec<na::SVector<f64, $dim>> = v
                .data
                .chunks_exact(v.cols)
                .map(na::SVector::<f64, $dim>::from_column_slice)
                .collect();
            let indices: Vec<usize> = e.data.iter().map(|&i| i as usize).collect();
            let (ms, m) = best_of(repeats, || {
                SimplicialMesh::<$dim>::new(vertices.clone(), indices.clone())
            });
            emit("dexterior", mesh, "build", ms, &[]);
            emit("dexterior", mesh, "topology_check", 0.0,
                 &[$(m.simplex_count::<$k>() as f64),+]);
            emit("dexterior", mesh, "geometry_check", 0.0,
                 &[$(m.star::<$k, Primal>().diagonal.sum()),+]);
        }
    };
}

dexterior_bench!(dexterior_2d, 2, 0, 1, 2);
dexterior_bench!(dexterior_3d, 3, 0, 1, 2, 3);

fn frobenius(values: &[f64]) -> f64 {
    values.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn formoniq_bench(mesh: &str, v: &Array<f64>, e: &Array<i64>, repeats: usize) {
    let n = e.cols - 1;
    let (ms, topology) = best_of(repeats, || {
        let cells: Vec<Simplex> = e
            .data
            .chunks_exact(e.cols)
            .map(|c| {
                let mut s: Vec<usize> = c.iter().map(|&i| i as usize).collect();
                s.sort_unstable();
                Simplex::new(s)
            })
            .collect();
        Complex::from_cells(Skeleton::new(cells))
    });
    let counts: Vec<f64> = (0..=n).map(|k| topology.skeleton(k).len() as f64).collect();
    emit("formoniq", mesh, "topology", ms, &counts);

    // one column per vertex: the row-major vertex array read column-major
    let coords = simplicial::linalg::Matrix::from_column_slice(v.cols, v.rows, &v.data);
    let (ms, lengths) = best_of(repeats, || {
        MeshCoords::new(coords.clone()).to_edge_lengths_sq(&topology)
    });
    emit("formoniq", mesh, "geometry", ms, &[]);

    let whitney = WhitneyComplex::new(&topology, &lengths);
    for k in 0..=n {
        let (ms, mass) = best_of(repeats, || whitney.mass(k));
        emit("formoniq", mesh, &format!("whitney_{k}"), ms, &[frobenius(mass.values())]);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: dec-compare dexterior|formoniq MESH_DIR REPEATS [MESH...]");
        std::process::exit(2);
    }
    let (lib, dir) = (args[1].as_str(), args[2].as_str());
    let repeats: usize = args[3].parse().unwrap();
    let mut names: Vec<String> = args[4..].to_vec();
    if names.is_empty() {
        names = fs::read_to_string(format!("{dir}/meshes.txt"))
            .unwrap()
            .lines()
            .filter_map(|l| l.split_whitespace().next().map(str::to_owned))
            .collect();
    }
    for name in &names {
        let v = read_array(&format!("{dir}/{name}.vertices.arr"), "float64", f64::from_le_bytes);
        let e = read_array(&format!("{dir}/{name}.elements.arr"), "int64", i64::from_le_bytes);
        match (lib, v.cols) {
            ("dexterior", 2) => dexterior_2d(name, &v, &e, repeats),
            ("dexterior", 3) => dexterior_3d(name, &v, &e, repeats),
            ("formoniq", _) => formoniq_bench(name, &v, &e, repeats),
            _ => panic!("unknown library {lib} or dimension {}", v.cols),
        }
    }
}
