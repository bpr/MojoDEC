# Writing doc strings

The [API reference](api/index.md) is generated from the doc strings, so
they are the documentation. Every public module, struct, trait, field,
alias, and function needs one.

## Format

Google style, as Mojo's standard library uses. The first paragraph is the
summary; then any longer description, in Markdown; then the sections
that apply:

```mojo
def subset(self, sc: SimplicialComplex, name: String, dim: Int) raises -> Subset:
    """The `dim`-simplices of the physical groups named `name`.

    Groups of a higher dimension contribute the faces of their cells.

    Args:
        sc: The complex built from `self.mesh()`.
        name: A physical group's name.
        dim: The dimension of the simplices to return.

    Returns:
        A primal `Subset` of `sc`'s `dim`-simplices.

    Raises:
        If no group has that name, or none has cells of dimension >= `dim`.
    """
```

- **`Parameters:`** documents compile-time parameters, on the struct or
  function that declares them (`TypedCochain[n, k, primal]`).
- **`Args:`** lists every argument except `self`.
- **`Returns:`** is needed whenever the function returns a value.
- **`Raises:`** is needed whenever the function is `raises`.
- Fields and `comptime` aliases take a one-line doc string below them.
- A module's doc string starts with its title, underlined with `=`; the
  generator uses the title in the index.

Code in doc strings is not compiled or run (Mojo has no doctest runner),
so keep examples short and check them by hand; the tests and the examples
are the checked usage.

## Checking

```sh
pixi run doc-check                       # fail if any file got worse
pixi run doc-check --list core/cochain   # the warnings in matching files
pixi run doc-check --update              # record the counts after fixing some
```

`mojo doc --diagnose-missing-doc-strings` warns about missing doc strings
and missing sections. `doc-check` counts its warnings per file and fails
when a file has more than `docs/docstring-baseline.txt` records. Every
file is at zero, so the baseline is empty and any warning fails the
check: document new code as you write it. (`--update` exists for a
future tool version that adds new kinds of warning; it should not be
used to excuse undocumented code.)

## Generating

```sh
pixi run docs         # Markdown pages in docs/api/ (not committed)
pixi run docs-site    # the site, with search, in build/site
pixi run docs-serve   # the site at http://127.0.0.1:8000, rebuilt on edits
```

`docs/gen_api.py` renders `mojo doc`'s JSON: a page per module, with
signatures, the sections above, and links between MojoDEC's types. The
site tasks run in pixi's `docs` environment, which adds MkDocs Material.
`docs-serve` rebuilds when the Markdown changes, not when the Mojo code
does; rerun `pixi run docs` for that.
