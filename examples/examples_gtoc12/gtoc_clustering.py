import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np
import matplotlib.pyplot as plt


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, delim_whitespace=True)
    return df


def scale_data(df, columns_to_cluster):
    scaler = RobustScaler()
    df_scaled = df.copy()
    df_scaled[columns_to_cluster] = scaler.fit_transform(df[columns_to_cluster])
    return df_scaled, scaler


def perform_kmeans_clustering(df, columns_to_cluster, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    df_copy = df.copy()
    # scaler = RobustScaler()
    df_scaled = df_copy.copy()
    # df_scaled[columns_to_cluster] = scaler.fit_transform(df_copy[columns_to_cluster])
    kmeans.fit(df_scaled[columns_to_cluster])
    labels = kmeans.labels_
    # detransform the centroids
    # centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    centroids = kmeans.cluster_centers_
    return labels, centroids


import pydin

mu = 398600.4418  # km^3/s^2


def plot_clusters(df, labels, centroids, columns_to_cluster, specific_clusters=None):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    for label in set(labels):

        df_label = df[labels == label].drop(columns=["ID", "epoch(MJD)"])
        # Classical orbital elements, mean of the set
        print(df_label.mean())
        a, e, i, raan, argp, m_anomaly, t_anomaly, h_x, h_y, h_z, _, _, _, _, _, _ = df_label.mean().values
        p = a * (1 - e ** 2)

        # Sample the orbit
        true_anomalies = pydin.sample_true_from_eccentric_anomaly(e, 30)

        # Convert anomalies to position and velocity vectors
        r_vectors = []
        for f in true_anomalies:
            r, v = pydin.coe2rv(p, a, e, i, raan, argp, f, mu)
            r_vectors.append(r)

        # Convert position vectors to numpy array
        r_vectors = np.array(r_vectors)

        # Plot the orbit
        ax.plot(r_vectors[:, 0], r_vectors[:, 1], r_vectors[:, 2], label=f"Centroid {i}")

    # Plot centroids
    # ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', s=100)

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    # plt.figure(figsize=(10, 10), dpi=300)
    plt.savefig('clustered_3d.png')
    plt.close()


def plot_cluster_samples2(df, labels, centroids, columns_to_cluster, specific_clusters=None):
    for label in set(labels):
        fig = plt.figure(figsize=(7, 7), dpi=300)
        ax = fig.add_subplot(111, projection='3d')

        df_label = df[labels == label].drop(columns=["ID", "epoch(MJD)"])
        df_samples = df_label.sample(50, replace=True, random_state=42)
        # Classical orbital elements, mean of the set
        # print(df_label.mean())

        # plot the mean
        a, e, i, raan, argp, m_anomaly, t_anomaly, h_x, h_y, h_z, _, _, _, _, _, _ = df_label.mean().values
        p = a * (1 - e ** 2)

        # Sample the orbit
        true_anomalies = pydin.sample_true_from_eccentric_anomaly(e, 30)

        # Convert anomalies to position and velocity vectors
        r_vectors = []
        for f in true_anomalies:
            r, v = pydin.coe2rv(p, a, e, i, raan, argp, f, mu)
            r_vectors.append(r)

        # Convert position vectors to numpy array
        r_vectors = np.array(r_vectors)

        # Plot the orbit
        ax.plot(r_vectors[:, 0], r_vectors[:, 1], r_vectors[:, 2], label=f"Centroid {i}",
                color='red')

        for index, row in df_samples.iterrows():
            a, e, i, raan, argp, m_anomaly, t_anomaly, h_x, h_y, h_z, _, _, _, _, _, _ = row.values
            p = a * (1 - e ** 2)

            # Sample the orbit
            true_anomalies = pydin.sample_true_from_eccentric_anomaly(e, 30)

            # Convert anomalies to position and velocity vectors
            r_vectors = []
            for f in true_anomalies:
                r, v = pydin.coe2rv(p, a, e, i, raan, argp, f, mu)
                r_vectors.append(r)

            # Convert position vectors to numpy array
            r_vectors = np.array(r_vectors)

            # Plot the orbit
            ax.plot(r_vectors[:, 0], r_vectors[:, 1], r_vectors[:, 2], lw=0.1, color='blue')
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_zlabel("Z (km)")
        # plt.figure(figsize=(10, 10), dpi=300)
        plt.savefig('clustered_3d_{}.png'.format(label))
        plt.close()


import math


def plot_cluster_samples(df, labels, centroids, columns_to_cluster, specific_clusters=None):
    # Convert labels to list and find unique labels
    labels_list = list(set(labels))

    # Calculate total number of pages
    total_pages = math.ceil(len(labels_list) / 15)

    for page in range(total_pages):
        # Create a new figure for each page
        fig = plt.figure(figsize=(20, 30), dpi=300)

        for i in range(15):
            label_index = page * 15 + i

            # If label index is out of range, break the loop
            if label_index >= len(labels_list):
                break

            label = labels_list[label_index]

            # Create subplot
            ax = fig.add_subplot(5, 3, i + 1, projection='3d')

            df_label = df[labels == label].drop(columns=["ID", "epoch(MJD)"])
            df_samples = df_label.sample(50, replace=True)

            # plot the mean
            a, e, i, raan, argp, m_anomaly, t_anomaly, h_x, h_y, h_z, _, _, _, _, _, _ = df_label.mean().values
            p = a * (1 - e ** 2)

            # Sample the orbit
            true_anomalies = pydin.sample_true_from_eccentric_anomaly(e, 30)

            # Convert anomalies to position and velocity vectors
            r_vectors = []
            for f in true_anomalies:
                r, v = pydin.coe2rv(p, a, e, i, raan, argp, f, mu)
                r_vectors.append(r)

            # Convert position vectors to numpy array
            r_vectors = np.array(r_vectors)

            # Plot the orbit
            ax.plot(r_vectors[:, 0], r_vectors[:, 1], r_vectors[:, 2], label=f"Centroid {i}",
                    color='red')

            for index, row in df_samples.iterrows():
                a, e, i, raan, argp, m_anomaly, t_anomaly, h_x, h_y, h_z, _, _, _, _, _, _ = row.values
                p = a * (1 - e ** 2)

                # Sample the orbit
                true_anomalies = pydin.sample_true_from_eccentric_anomaly(e, 30)

                # Convert anomalies to position and velocity vectors
                r_vectors = []
                for f in true_anomalies:
                    r, v = pydin.coe2rv(p, a, e, i, raan, argp, f, mu)
                    r_vectors.append(r)

                # Convert position vectors to numpy array
                r_vectors = np.array(r_vectors)

                # Plot the orbit
                ax.plot(r_vectors[:, 0], r_vectors[:, 1], r_vectors[:, 2], lw=0.1, color='blue')

            ax.set_xlabel("X (km)")
            ax.set_ylabel("Y (km)")
            ax.set_zlabel("Z (km)")

        plt.tight_layout()
        plt.savefig('clustered_3d_page_{}.png'.format(page + 1))
        plt.close()


# Plot centroids
# ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', s=100)


# ID    epoch(MJD)       a(AU)            e             i(deg)         LAN(deg)      argperi(deg)
# M(deg)
if __name__ == "__main__":
    file_path = 'pydin/examples/data/GTOC12_Asteroids_Data.txt'
    n_clusters = 300
    # labels, centroids = perform_kmeans_clustering(df_scaled, columns_to_cluster, n_clusters)
    # plot_clusters(df_scaled, labels, centroids, scaler, columns_to_cluster)

    col_id = "ID"
    col_epoch = "epoch(MJD)"
    col_a = "a(AU)"
    col_e = "e"
    col_i = "i(deg)"
    col_lan = "LAN(deg)"
    col_argp = "argperi(deg)"
    col_m_anomaly = "M(deg)"

    df = load_and_preprocess_data(file_path)
    df['a(AU)'] = df['a(AU)'].apply(lambda x: x * 149597870.7)
    df['M(deg)'] = df['M(deg)'].apply(np.deg2rad)
    df['i(deg)'] = df['i(deg)'].apply(np.deg2rad)
    df['argperi(deg)'] = df['argperi(deg)'].apply(np.deg2rad)
    df['LAN(deg)'] = df['LAN(deg)'].apply(np.deg2rad)

    # calculate the current r and v vectors
    df['true_anomaly'] = df[col_m_anomaly].apply(
        lambda x: pydin.anomaly_mean_to_true(x, df[col_e].values[0])
    )


    def compute_h(r, v):
        h_x = r[1] * v[2] - r[2] * v[1]
        h_y = r[2] * v[0] - r[0] * v[2]
        h_z = r[0] * v[1] - r[1] * v[0]
        return h_x, h_y, h_z


    def compute_vectors(row):
        r, v = pydin.coe2rv(row[col_a] * (1 - row[col_e] ** 2),
                            row[col_a],
                            row[col_e],
                            row[col_i],
                            row[col_lan],
                            row[col_argp],
                            row['true_anomaly'],
                            mu)
        h = compute_h(r, v)
        return r, v, h


    results = df.apply(compute_vectors, axis=1)

    df['r'], df['v'], df['h'] = zip(*results)
    df['h_x'], df['h_y'], df['h_z'] = zip(*df['h'])
    df['r_x'], df['r_y'], df['r_z'] = zip(*df['r'])
    df['v_x'], df['v_y'], df['v_z'] = zip(*df['v'])

    # compute and add eccentricity vector components using the vectors above
    # Vector3<Scalar> e     = ((v_mag * v_mag - mu / r_mag) * r - (r.dot(v)) * v) / mu;
    df['e_x'] = ((df['v_x'] ** 2 - mu / df['r_x']) * df['r_x'] - (
            df['r_x'] * df['v_x'] + df['r_y'] * df['v_y'] + df['r_z'] * df['v_z']) * df[
                     'v_x']) / mu
    df['e_y'] = ((df['v_y'] ** 2 - mu / df['r_y']) * df['r_y'] - (
            df['r_x'] * df['v_x'] + df['r_y'] * df['v_y'] + df['r_z'] * df['v_z']) * df[
                     'v_y']) / mu
    df['e_z'] = ((df['v_z'] ** 2 - mu / df['r_z']) * df['r_z'] - (
            df['r_x'] * df['v_x'] + df['r_y'] * df['v_y'] + df['r_z'] * df['v_z']) * df[
                     'v_z']) / mu

    # scale the eccentricity to some meaningless range that is comparable to the angular momentum
    # according the the golden ratio (1/1.618) * norm(h)
    df['norm_h'] = df['h'].apply(np.linalg.norm)
    df['e_x'] = df['e_x'].values / df['e'] * (1 / (1.618 * df['norm_h'].mean()))
    df['e_y'] = df['e_y'].values / df['e'] * (1 / (1.618 * df['norm_h'].mean()))
    df['e_z'] = df['e_z'].values / df['e'] * (1 / (1.618 * df['norm_h'].mean()))

    df.drop(columns=['norm_h'], inplace=True)
    df.drop(columns=['h'], inplace=True)
    df.drop(columns=['r'], inplace=True)
    df.drop(columns=['v'], inplace=True)

    # cluster according to the angular momentum vectors
    columns_to_cluster = ['h_x', 'h_y', 'h_z', 'e_x', 'e_y', 'e_z']
    labels, centroids = perform_kmeans_clustering(df, columns_to_cluster, n_clusters)
    df.drop(columns=['e_x', 'e_y', 'e_z'], inplace=True)
    print(labels)
    plot_clusters(df, labels, centroids, columns_to_cluster)
    plot_cluster_samples(df, labels, centroids, columns_to_cluster)
    plot_cluster_samples2(df, labels, centroids, columns_to_cluster)
