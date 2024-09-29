from vedo import*
import re
#Step1: Visualization

# test

# Get the file name
file_path = "ShapeDatabase_INFOMR/Apartment/D00045.obj" #copy the root path
file_name = re.search(r'([^/]+)(?=\.obj$)', file_path)

# Create a scene window
plotter = Plotter(title="Visualization of " + file_name.group(0), axes=0)  # visualization without axes
print(file_name.group(0))

# Load the model.obj
# Test different shape
mesh = load(file_path)

# Set display parameters
model_color = 'white'
edges_color = 'tomato'
lighting = 'plastic'

mesh.c(model_color).bc(edges_color).lw(0.5)  # set model_color and edges_color
mesh.flat().lighting(lighting)  # set light effect

# Show model with interactive controls
plotter.show(mesh, interactive=True)

# Shut down when the window is closed
plotter.close()