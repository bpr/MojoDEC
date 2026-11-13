# MojoDEC

Discrete exterior calculus on simplicial and cubical complexes in Mojo,
largely inspired by [PyDEC](https://github.com/hirani/pydec), with NuMojo arrays in
place of NumPy and its own sparse matrices in place of `scipy.sparse`.

- **[API reference](api/index.md)**: every public module, struct, and
  function, generated from the doc strings.
- **[User guide](guide/index.md)**: the mathematics and use of
  orientation checks, homology, and time stepping.
- **[Architecture](architecture.md)**: the layers, what a complex
  stores and what it computes on demand, where build time and memory go,
  and how parallel results stay equal to serial ones.
- **[Plan](plan.md)**: ideas not yet done, with the measurements behind
  them, and ideas rejected, with the reasons.
- **[Writing doc strings](writing-docs.md)**: the format, and the checks
  that keep the reference complete.

The repository's `README.md` is the user guide for now: installation, a
tour of complexes, cochains, and operators, the differences from PyDEC,
and the examples.
