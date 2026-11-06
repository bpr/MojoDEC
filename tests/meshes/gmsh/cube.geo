// Unit cube of tetrahedra with tagged faces, for tests/test_gmsh.mojo.
SetFactory("OpenCASCADE");
Box(1) = {0, 0, 0, 1, 1, 1};
Mesh.MeshSizeMax = 0.5;
Physical Surface("bottom") = {5};
Physical Surface("top") = {6};
Physical Volume("solid") = {1};
