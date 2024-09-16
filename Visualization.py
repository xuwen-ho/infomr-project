from vedo import*

#Step1: Visualization

# test
# Create a scene window
plotter = Plotter(title="OBJ Shape Viewer", axes=1)  # visualization with axes

# Load the model.obj
mesh = load("ShapeDatabase_INFOMR/AircraftBuoyant/m1337.obj")

# Set display parameters
model_color = 'white'
edges_color = 'tomato'
lighting = 'plastic'

mesh.c(model_color).bc(edges_color).lw(0.5)  # set model_color and edges_color
mesh.flat().lighting(lighting)  # set light effect

# Show model with interactive controls
plotter.show(mesh, interactive=True)

# Enable interaction
plotter.interactor.Start()