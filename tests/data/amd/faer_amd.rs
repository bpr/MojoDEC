use faer::sparse::linalg::amd;
use faer::sparse::SymbolicSparseColMatRef;
use dyn_stack::{MemBuffer, MemStack};
use std::io::Read;

fn main() {
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s).unwrap();
    let mut it = s.split_whitespace().map(|x| x.parse::<usize>().unwrap());
    let n = it.next().unwrap();
    let nnz = it.next().unwrap();
    let mut col_ptr = vec![0usize; n + 1];
    for k in 0..=n { col_ptr[k] = it.next().unwrap(); }
    let mut row_idx = vec![0usize; nnz];
    for k in 0..nnz { row_idx[k] = it.next().unwrap(); }
    let a = SymbolicSparseColMatRef::new_checked(n, n, &col_ptr, None, &row_idx);
    let mut perm = vec![0usize; n];
    let mut perm_inv = vec![0usize; n];
    let mut mem = MemBuffer::new(amd::order_scratch::<usize>(n, nnz));
    amd::order(&mut perm, &mut perm_inv, a, amd::Control::default(), MemStack::new(&mut mem)).unwrap();
    let out: Vec<String> = perm.iter().map(|x| x.to_string()).collect();
    println!("{}", out.join(" "));
}
