# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Mesh files (`mojodec.routines.io.meshio`).
==========================================
Port of `pydec.io.meshio`: a mesh is an XML file naming one array file per
mesh field, e.g.

```xml
<mesh type="simplicial_mesh">
    <elements><file name="square_8.elements"/></elements>
    <vertices><file name="square_8.vertices"/></vertices>
</mesh>
```

The array files sit next to the XML file and use `mojodec.routines.io.arrayio`.
"""

from mojodec.core.mesh import SimplicialMesh
from mojodec.routines.io.arrayio import ArrayFormat, read_array, write_array


def read_mesh(path: String) raises -> SimplicialMesh:
    """Read a simplicial mesh from a PyDEC XML mesh file.

    ```mojo
    var mesh = read_mesh("torus.xml")
    ```

    Args:
        path: The XML file; the array files it names are read from the same
            directory.

    Returns:
        The mesh, with `float64` vertices and integer elements.

    Raises:
        If a file cannot be opened; the XML has no `<mesh>` node, a `type`
        other than `simplicial_mesh`, or no `<vertices>` or `<elements>`
        node with a named `<file>`; an array file is malformed (see
        `read_array`); or `SimplicialMesh` rejects the arrays.
    """
    var xml: String
    with open(path, "r") as f:
        xml = f.read()
    var mesh_tag = _find_tag(xml, "mesh", 0)
    if mesh_tag < 0:
        raise Error("read_mesh: invalid XML root node")
    var mesh_type = _attribute(xml, mesh_tag, "type")
    if mesh_type != "" and mesh_type != "simplicial_mesh":
        raise Error("read_mesh: unsupported mesh type [" + mesh_type + "]")
    var dir = _dirname(path)
    var vertices_file = _field_file(xml, "vertices")
    var elements_file = _field_file(xml, "elements")
    var vertices = read_array[DType.float64](_join(dir, vertices_file))
    var elements = read_array[DType.int](_join(dir, elements_file))
    return SimplicialMesh(vertices, elements)


def write_mesh(
    path: String, mesh: SimplicialMesh, format: ArrayFormat = .BINARY
) raises:
    """Write a simplicial mesh as an XML file plus one array file per field.

    The array files are named `<basename>.vertices` and `<basename>.elements`
    and written in `format`, next to the XML file. `<basename>` is the file name up to its first `.`.

    Args:
        path: The XML file to write; existing files are replaced.
        mesh: The mesh to write.
        format: The array file format: `BASIC`, `ASCII`, or `BINARY`.

    Raises:
        If a file cannot be written.
    """
    var dir = _dirname(path)
    var filename = _basename(path)
    var base = String(filename.split(".")[0])
    var xml = String('<?xml version="1.0" ?>\n<mesh type="simplicial_mesh">\n')
    var keys: List[String] = ["vertices", "elements"]
    for key in keys:
        var data_filename = base + "." + key
        xml += (
            "\t<"
            + key
            + '>\n\t\t<file name="'
            + data_filename
            + '"/>\n\t</'
            + key
            + ">\n"
        )
        if key == "vertices":
            write_array(_join(dir, data_filename), mesh.vertices, format)
        else:
            write_array(_join(dir, data_filename), mesh.elements, format)
    xml += "</mesh>\n"
    with open(path, "w") as f:
        f.write(xml)


def _find_tag(xml: String, name: String, start: Int) -> Int:
    """Offset of the first `<name` tag (followed by space, `>` or `/`)."""
    var pos = start
    while True:
        var i = xml.find("<" + name, pos)
        if i < 0:
            return -1
        var after = i + 1 + name.byte_length()
        if after < xml.byte_length():
            var c = xml[byte=after]
            if c == " " or c == ">" or c == "/" or c == "\t" or c == "\n":
                return i
        pos = after


def _attribute(xml: String, tag_start: Int, name: String) -> String:
    # attribute values may be single- or double-quoted
    var end = xml.find(">", tag_start)
    var quotes: List[String] = ['"', "'"]
    for q in quotes:
        var key = name + "=" + q
        var i = xml.find(key, tag_start)
        # skip longer names ending in `name`, such as `filename=`
        while (
            i > 0
            and xml[byte=i - 1] != " "
            and xml[byte=i - 1] != "\t"
            and xml[byte=i - 1] != "\n"
        ):
            i = xml.find(key, i + 1)
        if i < 0 or (end >= 0 and i > end):
            continue
        var vstart = i + key.byte_length()
        var vend = xml.find(q, vstart)
        if vend < 0 or (end >= 0 and vend > end):
            continue
        return String(xml[byte=vstart:vend])
    return ""


def _field_file(xml: String, field: String) raises -> String:
    var tag = _find_tag(xml, field, 0)
    if tag < 0:
        raise Error("read_mesh: missing <" + field + "> node")
    # the <file> must be inside this node, not a later one's
    var open_end = xml.find(">", tag)
    if open_end < 0:
        raise Error("read_mesh: unterminated <" + field + "> tag")
    var close = -1
    if xml[byte=open_end - 1] != "/":
        close = xml.find("</" + field, open_end)
        if close < 0:
            raise Error("read_mesh: <" + field + "> node is not closed")
    var file_tag = _find_tag(xml, "file", open_end)
    if file_tag < 0 or close < 0 or file_tag > close:
        raise Error("read_mesh: missing <file> node for " + field)
    var name = _attribute(xml, file_tag, "name")
    if name == "":
        raise Error("read_mesh: <file> node without a name")
    return name


def _dirname(path: String) -> String:
    var i = path.rfind("/")
    return String(path[byte=:i]) if i >= 0 else String("")


def _basename(path: String) -> String:
    var i = path.rfind("/")
    return String(path[byte = i + 1 :]) if i >= 0 else path


def _join(dir: String, name: String) -> String:
    return name if dir == "" else dir + "/" + name
