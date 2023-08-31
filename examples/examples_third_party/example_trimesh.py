import numpy as np
import trimesh

# Define the radii of the ellipsoid
radii = np.array([1.0, 2.0, 3.0])

# Create a unit sphere
sphere = trimesh.creation.icosphere(subdivisions=1, radius=1.0)

# Scale it to form the ellipsoid
ellipsoid_mesh = sphere.apply_transform(np.diag(np.append(radii, 1)))

# Print mesh vertices and faces
print("Vertices:")
print(ellipsoid_mesh.vertices)

print("\nFaces:")
print(ellipsoid_mesh.faces)

# Print additional mesh properties
print("\nVolume:")
print(ellipsoid_mesh.volume)

print("\nSurface area:")
print(ellipsoid_mesh.area)

print("\nCenter of mass:")
print(ellipsoid_mesh.center_mass)

print("\nBounding box:")
print(ellipsoid_mesh.bounds)

# Simplify the mesh
simplified_mesh = ellipsoid_mesh.simplify_quadratic_decimation(5)
print("\nSimplified mesh:")
print(simplified_mesh)

# Export the mesh to a file
ellipsoid_mesh.export('ellipsoid_mesh.stl')
ellipsoid_mesh.export('ellipsoid_mesh.ply')

# Visualize the mesh using the built-in viewer
ellipsoid_mesh.show()
