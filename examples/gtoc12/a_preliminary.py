import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydin

GM = 398600.4418
AU = 149597870.7  # km
RAD2DEG = 180 / np.pi

FILENAME = 'pydin/examples/data/GTOC12_Asteroids_Data.txt'

# Define column names for the dataset
col_names = ['ID', 'epoch(MJD)', 'a(AU)', 'e', 'i(deg)', 'LAN(deg)', 'argperi(deg)', 'M(deg)']


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, delim_whitespace=True)
    return df


# Import the data
df = load_and_preprocess_data(FILENAME)

# rename columns and convert to SI (we leave a as km)
df['id'] = df['ID']
df['epoch_mjd'] = df['epoch(MJD)']
df['a_km'] = df['a(AU)'] * AU
df['p_km'] = df['a_km'] * (1 - df['e'] ** 2)
df['e'] = df['e']
df['i'] = df['i(deg)'] * RAD2DEG
df['RAAN'] = df['LAN(deg)'] * RAD2DEG
df['argp'] = df['argperi(deg)'] * RAD2DEG
df['M'] = df['M(deg)'] * RAD2DEG
df['true_anomaly'] = pydin.anomaly_mean_to_true(df['M'], df['e'])
df = df.drop(columns=['ID', 'epoch(MJD)', 'a(AU)', 'i(deg)', 'LAN(deg)', 'argperi(deg)', 'M(deg)'])

# compute position and velocity vectors
r, v = pydin.coe2rv(398600.4418, df['p_km'], df['e'], df['i'], df['RAAN'], df['argp'],
                    df['true_anomaly'])

# eigen stores in column major order, so we transpose for numpy's row major order
r, v = r.T, v.T
df['r_x'] = r[:, 0]
df['r_y'] = r[:, 1]
df['r_z'] = r[:, 2]
df['v_x'] = v[:, 0]
df['v_y'] = v[:, 1]
df['v_z'] = v[:, 2]

# compute angular momentum vector
h = np.cross(r, v)
df['h_x'] = h[:, 0]
df['h_y'] = h[:, 1]
df['h_z'] = h[:, 2]

# compute eccentricity vector
norm_v_squared = np.linalg.norm(v, axis=1) ** 2
norm_r = np.linalg.norm(r, axis=1)
r_dot_v = np.einsum('ij,ij->i', r, v)

# Reshape norm_v_squared, norm_r, and r_dot_v for broadcasting
norm_v_squared = norm_v_squared[:, np.newaxis]
norm_r = norm_r[:, np.newaxis]
r_dot_v = r_dot_v[:, np.newaxis]

e_vec = (r * (norm_v_squared - GM / norm_r) - v * r_dot_v) / GM
df['e_x'] = e_vec[:, 0]
df['e_y'] = e_vec[:, 1]
df['e_z'] = e_vec[:, 2]

# cluster according to h_x, h_y, h_z using scikit-learn
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=300, random_state=0, n_init=10).fit(df[['h_x', 'h_y', 'h_z']])

df['cluster'] = kmeans.labels_

print(df['cluster'].value_counts())

# plot
from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure(figsize=(10, 10), dpi=300)
# ax = fig.add_subplot(111, projection='3d')
#
# for cluster in df['cluster'].unique():
#     # plot the cluster mean orbit
#     cluster_mean = df[df['cluster'] == cluster][['p_km', 'e', 'i', 'RAAN', 'argp']].mean()
#     nu_samples = pydin.sample_true_from_mean_anomaly(cluster_mean['e'], 30)
#     r, v = pydin.coe2rv(GM,
#                         cluster_mean['p_km'],
#                         cluster_mean['e'],
#                         cluster_mean['i'],
#                         cluster_mean['RAAN'],
#                         cluster_mean['argp'], nu_samples)
#     r, v = r.T, v.T
#     ax.plot(r[:, 0], r[:, 1], r[:, 2], label=f'Cluster {cluster}')
#
# ax.legend()
# plt.savefig('cluster_mean_orbits.png')
#
# # plot 50, or all, orbits from each cluster, and plot it onto a 3x5 a4 page
# for cluster in df['cluster'].unique():
#     fig = plt.figure(figsize=(10, 10), dpi=300)
#     ax = fig.add_subplot(111, projection='3d')
#
#     # plot the cluster mean orbit
#     cluster_mean = df[df['cluster'] == cluster][['p_km', 'e', 'i', 'RAAN', 'argp']].mean()
#     nu_samples = pydin.sample_true_from_mean_anomaly(cluster_mean['e'], 30)
#     r, _ = pydin.coe2rv(GM,
#                         cluster_mean['p_km'],
#                         cluster_mean['e'],
#                         cluster_mean['i'],
#                         cluster_mean['RAAN'],
#                         cluster_mean['argp'], nu_samples)
#     r = r.T
#     ax.plot(r[:, 0], r[:, 1], r[:, 2], label=f'Cluster {cluster}', color='red')
#
#     # plot 50 orbits from each cluster
#     for i in range(50):
#         orbit = df[df['cluster'] == cluster].sample().iloc[0]
#         r, _ = pydin.coe2rv(GM,
#                             orbit['p_km'],
#                             orbit['e'],
#                             orbit['i'],
#                             orbit['RAAN'],
#                             orbit['argp'], nu_samples)
#         r = r.T
#         ax.plot(r[:, 0], r[:, 1], r[:, 2], alpha=0.2, lw=0.3, color='blue')
#
#     ax.legend()
#     plt.savefig(f'cluster_{cluster}_orbits.png')
#     plt.close()

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

features = df[['h_x', 'h_y', 'h_z', 'e_x', 'e_y', 'e_z', 'argp', 'true_anomaly']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

dbscan = DBSCAN(eps=0.3, min_samples=5)
clusters = dbscan.fit_predict(features_scaled)

df['cluster_dbscan'] = clusters
print(df['cluster_dbscan'].value_counts())

fig = plt.figure(figsize=(10, 10), dpi=300)
ax = fig.add_subplot(111, projection='3d')

import random

# Get unique clusters, excluding -1 and 0
clusters = [c for c in df['cluster_dbscan'].unique() if c not in [-1, 0]]

# Randomly sample 4 clusters
clusters = random.sample(clusters, 10)

# Define colors
colors = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'pink', 'yellow', 'brown', 'gray',
          'cyan', 'magenta']

# create a cycle generator for the colors
color_cycle = itertools.cycle(colors)

# # Plot each asteroid and its velocity
for i, cluster in enumerate(clusters):
    if cluster == -1 or cluster == 0:
        continue

    subset = df[df['cluster_dbscan'] == cluster]
    color = next(color_cycle)
    # Scatter plot for asteroid positions, using the same color for the same cluster
    ax.scatter(subset['r_x'], subset['r_y'], subset['r_z'], color=color, alpha=0.3)

    # Quiver plot for asteroid velocities, using the same color for the same cluster
    ax.quiver(subset['r_x'], subset['r_y'], subset['r_z'], subset['v_x'], subset['v_y'],
              subset['v_z'], length=1e6, color=color, alpha=0.3)

    # Calculate the average orbital elements for the cluster
    cluster_mean = subset[['p_km', 'e', 'i', 'RAAN', 'argp', 'true_anomaly']].mean()
    cluster_max = subset[['p_km', 'e', 'i', 'RAAN', 'argp', 'true_anomaly']].max()
    cluster_min = subset[['p_km', 'e', 'i', 'RAAN', 'argp', 'true_anomaly']].min()

    # Sample 100 points along the orbit of the average elements
    nu_samples = np.linspace(0, 2 * np.pi, 50)

    # Compute position and velocity vectors for the average orbit
    r, v = pydin.coe2rv(GM, cluster_mean['p_km'], cluster_mean['e'], cluster_mean['i'],
                        cluster_mean['RAAN'], cluster_mean['argp'], nu_samples)
    r, v = r.T, v.T

    # Plot the orbit of the cluster mean, using the same color for the same cluster
    ax.plot(r[:, 0], r[:, 1], r[:, 2], label=f'Cluster {cluster}', color=color, alpha=0.3)

    # Compute position and velocity vectors for the average orbit
    r, v = pydin.coe2rv(GM, cluster_max['p_km'], cluster_max['e'], cluster_max['i'],
                        cluster_max['RAAN'], cluster_max['argp'], nu_samples)
    r, v = r.T, v.T

    # Plot the orbit of the cluster mean, using the same color for the same cluster
    ax.plot(r[:, 0], r[:, 1], r[:, 2], label=f'Cluster {cluster}', color=color, alpha=0.3)

    # Compute position and velocity vectors for the average orbit
    r, v = pydin.coe2rv(GM, cluster_min['p_km'], cluster_min['e'], cluster_min['i'],
                        cluster_min['RAAN'], cluster_min['argp'], nu_samples)
    r, v = r.T, v.T

    # Plot the orbit of the cluster mean, using the same color for the same cluster
    ax.plot(r[:, 0], r[:, 1], r[:, 2], label=f'Cluster {cluster}', color=color, alpha=0.3)

ax.legend()
plt.title('Clusters of Asteroids with velocities according to DBSCAN')
plt.savefig('dbscan_clusters_velocities.png')

# Get a list of unique clusters
clusters_unique = df['cluster_dbscan'].unique()

for cluster in clusters_unique:
    # Filter data for each cluster
    subset = df[df['cluster_dbscan'] == cluster]

    # Create a 3D scatter plot for each cluster
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(subset['r_x'], subset['r_y'], subset['r_z'], label=f'Cluster {cluster}')
    ax.quiver(subset['r_x'], subset['r_y'], subset['r_z'], subset['v_x'], subset['v_y'],
              subset['v_z'], length=1e6)

    ax.legend()
    plt.title(f'Cluster {cluster} of Asteroids with velocities according to DBSCAN')
    plt.savefig(f'dbscan_cluster_{cluster}.png')
    plt.close()  # close the figure to free up memory

# Plot distribution for each selected column
# for col in df.columns.drop(['id', 'epoch_mjd']):
#     sns.displot(df[col])
#     plt.title(f'Distribution for {col}')
#     plt.show()
