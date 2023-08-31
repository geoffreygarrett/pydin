import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pydin

# Classical orbital elements
p = 11000.0  # km
e = 0.1
a = p / (1.0 - e ** 2)  # km
i = 45.0 * np.pi / 180.0  # radians
Omega = 30.0 * np.pi / 180.0  # radians
omega = 60.0 * np.pi / 180.0  # radians
mu = 398600.4418  # km^3/s^2

# Sample the orbit
n_points = 100
true_anomalies = pydin.sample_true_from_eccentric_anomaly(e, n_points)

# Convert anomalies to position and velocity vectors
r_vectors = []
for f in true_anomalies:
    r, v = pydin.coe2rv(p, a, e, i, Omega, omega, f, mu)
    r_vectors.append(r)

# Convert position vectors to numpy array
r_vectors = np.array(r_vectors)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the orbit
ax.plot(r_vectors[:, 0], r_vectors[:, 1], r_vectors[:, 2])

# Label axes
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')

# Show the plot
plt.savefig('orbit_plot.png')
