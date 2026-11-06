"""Generate MojoDEC's API reference as Markdown, from `mojo doc`'s JSON.

    pixi run docs                     # docs/api/, one page per module
    python docs/gen_api.py api.json   # from JSON made earlier

`mojo doc` collects the doc strings of the package into JSON but does not
render them. This writes `docs/api/index.md` and a page per module
(`docs/api/mojodec/core/subset.md`, ...) with the module's aliases, traits,
structs, and functions: signatures, summaries and descriptions, and the
Parameters, Args, Returns, and Raises sections. Types defined in MojoDEC
link to their pages. The pages read on GitHub as they are, and `pixi run
docs-site` builds them into a searchable site with MkDocs.

Run from the repository root, after `pixi run numojo`.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "api")


def load(json_path=None):
    if json_path:
        with open(json_path) as f:
            return json.load(f)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "api.json")
        proc = subprocess.run(
            ["mojo", "doc", "-I", "build", "-o", out, "mojodec"],
            cwd=ROOT, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            sys.exit(proc.stdout + proc.stderr + "\nmojo doc failed (did `pixi run numojo` run?)")
        with open(out) as f:
            return json.load(f)


def modules(pkg, prefix=()):
    """(path parts, module) for every module, depth first; a package's
    `__init__` stands for the package itself."""
    parts = prefix + (pkg["name"],)
    for m in pkg.get("modules", []):
        yield (parts if m["name"] == "__init__" else parts + (m["name"],)), m, m["name"] == "__init__"
    for sub in pkg.get("packages", []):
        yield from modules(sub, parts)


class Page:
    """One module's page; `link` resolves `mojo doc` paths relative to it."""

    def __init__(self, parts, is_package):
        self.parts = parts
        # a package's page is its index, so its modules sit beside it
        self.file = os.path.join(*parts, "index.md") if is_package else os.path.join(*parts) + ".md"
        self.lines = []

    def w(self, text=""):
        self.lines.append(text)

    def link(self, type_name, path):
        """`type_name`, linked to its page when MojoDEC defines it."""
        code = f"`{type_name}`"
        if not path or not path.startswith("/mojodec/"):
            return code
        parts = path.strip("/").split("/")
        if parts[-1].startswith("#"):
            target, anchor = parts[:-1], parts[-1][1:]
        else:
            target, anchor = parts[:-1], parts[-1].lower()
        target_file = target_page_file(target)
        if target_file is None:
            return code
        rel = os.path.relpath(target_file, os.path.dirname(self.file) or ".")
        return f"[{code}]({rel}#{anchor})"


PACKAGES = set()


def target_page_file(parts):
    parts = tuple(parts)
    if parts in PACKAGES:
        return os.path.join(*parts, "index.md")
    return os.path.join(*parts) + ".md" if parts else None


TITLE = re.compile(r"^(?P<title>.*?)\s+={3,}\s*(?P<rest>.*)$", re.S)


def split_title(summary):
    """A module doc string opens with an underlined title, which `mojo doc`
    folds into the summary: ("Subsets of cells", the rest of the summary)."""
    m = TITLE.match(summary.strip())
    if not m:
        return "", summary.strip()
    title = re.sub(r"\s*\(`[^`]*`\)\.?$", "", m["title"].strip())
    return title, m["rest"].strip()


def text(item, module=False):
    """Summary and description, as Markdown paragraphs."""
    summary = item.get("summary", "").strip()
    if module:
        summary = split_title(summary)[1]
    out = [summary, item.get("description", "").strip()]
    return "\n\n".join(t for t in out if t)


def signature(fn):
    sig = fn["signature"]
    # mojo doc leaves out `raises`
    if fn.get("raises") and " raises" not in sig:
        i = sig.rfind(" -> ")
        sig = sig[:i] + " raises" + sig[i:] if i >= 0 else sig + " raises"
    return sig


def items(page, title, entries, show_default=True):
    """A Parameters / Args / Fields list."""
    if not entries:
        return
    page.w(f"**{title}:**")
    page.w()
    for e in entries:
        typ = page.link(e.get("type", ""), e.get("path")) if e.get("type") else ""
        bits = [typ] if typ else []
        if show_default and e.get("default"):
            bits.append(f"default `{e['default']}`")
        conv = e.get("convention")
        if conv in ("mut", "var", "out", "ref"):
            bits.append(conv)
        head = f"- **`{e['name']}`**" + (f" ({', '.join(bits)})" if bits else "")
        desc = (e.get("description") or e.get("summary") or "").strip()
        page.w(head + (": " + desc.replace("\n", "\n  ") if desc else ""))
    page.w()


def function(page, fn, level, anchor):
    """Every overload of a function or method."""
    page.w(f'<a id="{anchor}"></a>')
    page.w()
    page.w(f"{'#' * level} `{fn['name']}`")
    page.w()
    for ov in fn["overloads"]:
        if ov.get("isStatic") and fn["name"] != "__init__":
            page.w("*static*")
            page.w()
        page.w("```mojo")
        page.w(signature(ov))
        page.w("```")
        page.w()
        if ov.get("deprecated"):
            page.w(f"**Deprecated:** {ov['deprecated']}")
            page.w()
        body = text(ov)
        if body:
            page.w(body)
            page.w()
        if ov.get("constraints"):
            page.w(f"**Constraints:** {ov['constraints'].strip()}")
            page.w()
        items(page, "Parameters", ov.get("parameters", []))
        items(page, "Args", [a for a in ov.get("args", []) if a["name"] != "self"])
        ret = ov.get("returns") or {}
        if ret.get("doc"):
            typ = page.link(ret.get("type", ""), ret.get("path"))
            page.w(f"**Returns:** {typ}: {ret['doc'].strip()}")
            page.w()
        if ov.get("raisesDoc"):
            page.w(f"**Raises:** {ov['raisesDoc'].strip()}")
            page.w()


def aliases(page, entries, level, prefix=""):
    if not entries:
        return
    page.w(f"{'#' * level} Aliases")
    page.w()
    for a in entries:
        page.w(f'<a id="{prefix}{a["name"].lower()}"></a>')
        page.w()
        value = f" = {a['value']}" if a.get("value") else ""
        page.w(f"- **`{a['name']}`**`{value}`" if value else f"- **`{a['name']}`**")
        body = text(a)
        if body:
            page.w("  " + body.replace("\n", "\n  "))
    page.w()


IMPLICIT_TRAITS = {"AnyType", "Deinitable", "UnknownDestructibility"}


def method_order(fn):
    """`__init__` first, then named methods, then the other dunders; stable
    within each group."""
    name = fn["name"]
    return 0 if name == "__init__" else 2 if name.startswith("__") else 1


def struct(page, s, kind):
    anchor = s["name"].lower()
    page.w(f'<a id="{anchor}"></a>')
    page.w()
    page.w(f"## {kind} `{s['name']}`")
    page.w()
    if s.get("signature") and s["signature"] != f"{kind} {s['name']}":
        page.w("```mojo")
        page.w(s["signature"])
        page.w("```")
        page.w()
    body = text(s)
    if body:
        page.w(body)
        page.w()
    traits = [t.get("name", "") if isinstance(t, dict) else str(t) for t in s.get("parentTraits", [])]
    traits = [t for t in traits if t not in IMPLICIT_TRAITS]
    if traits:
        page.w("**Implements:** " + ", ".join(f"`{t}`" for t in traits if t))
        page.w()
    items(page, "Parameters", s.get("parameters", []))
    items(page, "Fields", s.get("fields", []))
    aliases(page, s.get("aliases", []), 3, anchor + ".")
    fns = sorted(s.get("functions", []), key=method_order)
    if fns:
        page.w("### Methods")
        page.w()
        page.w(", ".join(f"[`{f['name']}`](#{anchor}.{f['name'].lower()})" for f in fns))
        page.w()
        for f in fns:
            function(page, f, 4, f"{anchor}.{f['name'].lower()}")


def module_page(parts, m, is_package):
    page = Page(parts, is_package)
    name = ".".join(parts)
    title = split_title(m.get("summary", ""))[0]
    page.w(f"# `{name}`" + (f": {title}" if title else ""))
    page.w()
    body = text(m, module=True)
    if body:
        page.w(body)
        page.w()
    contents = (
        [f"[`{a['name']}`](#{a['name'].lower()})" for a in m.get("aliases", [])]
        + [f"[`{t['name']}`](#{t['name'].lower()})" for t in m.get("traits", [])]
        + [f"[`{s['name']}`](#{s['name'].lower()})" for s in m.get("structs", [])]
        + [f"[`{f['name']}`](#{f['name'].lower()})" for f in m.get("functions", [])]
    )
    if contents:
        page.w("**Contents:** " + ", ".join(contents))
        page.w()
    aliases(page, m.get("aliases", []), 2)
    for t in m.get("traits", []):
        struct(page, t, "trait")
    for s in m.get("structs", []):
        struct(page, s, "struct")
    if m.get("functions"):
        page.w("## Functions")
        page.w()
        for f in m["functions"]:
            function(page, f, 3, f["name"].lower())
    return page


def main():
    doc = load(sys.argv[1] if len(sys.argv) > 1 else None)
    decl = doc["decl"]
    mods = list(modules(decl))
    for parts, _, is_package in mods:
        if is_package:
            PACKAGES.add(parts)
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    pages = []
    for parts, m, is_package in mods:
        page = module_page(parts, m, is_package)
        path = os.path.join(OUT, page.file)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(page.lines).rstrip() + "\n")
        pages.append((parts, page.file, m))

    with open(os.path.join(OUT, "index.md"), "w") as f:
        f.write("# API reference\n\n")
        f.write(
            "Generated from the doc strings by `pixi run docs` "
            f"(Mojo {doc.get('version', '?')}); do not edit.\n\n"
        )
        for parts, file, m in pages:
            depth = "  " * (len(parts) - 1)
            title, rest = split_title(m.get("summary", ""))
            summary = title or rest.split("\n")[0]
            f.write(f"{depth}- [`{'.'.join(parts)}`]({file})" + (f": {summary}" if summary else "") + "\n")
    print(f"wrote {len(pages)} pages to {os.path.relpath(OUT, ROOT)}/")


if __name__ == "__main__":
    main()
