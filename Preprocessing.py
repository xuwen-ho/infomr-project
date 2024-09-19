import vedo
import re
import os
import numpy as np

#Step 2: Preprocessing and cleaning

#Step 2.1: Analyzing a single shape
# Output for each shape:
#
# the class of the shape
# the number of faces and vertices of the shape
# the type of faces (e.g. only triangles, only quads, mixes of triangles and quads)
# the axis-aligned 3D bounding box of the shapes

file_path = "ShapeDatabase_INFOMR/AquaticAnimal/m54.obj" #copy the root path

mesh = vedo.load(file_path)
vertices = mesh.vertices
num_vertices = vertices.shape[0]
num_faces = len(mesh.cells)
num_vertices_of_faces = mesh.count_vertices()
type_of_faces = np.unique(num_vertices_of_faces)
bounding_box = mesh.bounds()

#Step 2.2: Statistics over the whole database
