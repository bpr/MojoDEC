These meshes are used to test the correctness of simplicial_complex

sphere3.xml
  simplicial_mesh< 2D manifold, 3D embedding, 66 vertices, 128 elements >

torus_tet.xml
  simplicial_mesh< 3D manifold, 3D embedding, 204 vertices, 592 elements >

cube.xml
  simplicial_mesh< 2D manifold, 3D embedding, 8 vertices, 12 elements >


ply/sphere_{ascii,binary_le,binary_be}.ply
  a 34-vertex, 64-triangle sphere of radius 1.5 written by VTK's vtkPLYWriter
  (vtkSphereSource, theta 8, phi 6) in the three PLY encodings, with float
  coordinates, normals, and colors; ply/sphere_points.txt and
  ply/sphere_triangles.txt hold the same points and triangles as text

gmsh/square.geo, gmsh/cube.geo
  Gmsh geometries for tests/test_gmsh.mojo: the unit square split into
  two regions with named boundary curves, interface, corner point, and an
  unnamed group (tag 42); the unit cube of tetrahedra with "bottom", "top",
  and "solid". Meshed by Gmsh 4.15.2 into square_{v41,v41_bin,v22,v22_bin}.msh
  and cube_*.msh, e.g.
    pixi exec --spec gmsh gmsh -2 square.geo -format msh41 -bin -o square_v41_bin.msh
  square_edges_only.geo/.msh tags only a boundary curve, so Gmsh saves no
  triangles.
