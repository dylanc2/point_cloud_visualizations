import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import numpy as np
import random

def main():
    ### Deformed cloud ###

    deformed_cloud_rod_array = np.load('deformed_cloud_rod.npy')
    num_points_def = deformed_cloud_rod_array.shape[0]

    # Plot a subset of the points
    sample_size = round(num_points_def / 2)
    kept_indices = []
    for _ in range(sample_size):
        kept_indices.append(random.randint(0, num_points_def))
    sampled = []
    for i, p in enumerate(deformed_cloud_rod_array):
        if i in kept_indices:
            sampled.append(p)
    sampled_deformed_cloud_rod_array = np.asarray(sampled)

    X_def = sampled_deformed_cloud_rod_array[0:sample_size, 0:1]
    Y_def = sampled_deformed_cloud_rod_array[0:sample_size, 1:2]
    Z_def = sampled_deformed_cloud_rod_array[0:sample_size, 2:3]

    coords = deformed_cloud_rod_array
    pca = PCA(n_components=1)
    pca.fit(coords)
    direction_vector = pca.components_[0]
    print("Direction vector: {}".format(direction_vector))

    ### Full cloud ###

    filtered_cloud_rod_array = np.load('filtered_cloud_rod.npy')
    num_points = filtered_cloud_rod_array.shape[0]

    # Plot a subset of the points, otherwise they will blot out the deformed points
    sample_size = round(num_points / 8)
    kept_indices = []
    for _ in range(sample_size):
        kept_indices.append(random.randint(0, num_points))
    sampled = []

    dist_threshold = 2e-3
    for i, p in enumerate(filtered_cloud_rod_array):
        distances = np.linalg.norm(deformed_cloud_rod_array - p, axis=1)
        min_distance = np.min(distances)
        if i in kept_indices and p not in deformed_cloud_rod_array and min_distance > dist_threshold:
            sampled.append(p)
    sampled_filtered_cloud_rod_array = np.asarray(sampled)

    X = sampled_filtered_cloud_rod_array[0:sample_size, 0:1]
    Y = sampled_filtered_cloud_rod_array[0:sample_size, 1:2]
    Z = sampled_filtered_cloud_rod_array[0:sample_size, 2:3]

    origin = np.mean(sampled_filtered_cloud_rod_array, axis=0)
    euclidian_distance = np.linalg.norm(sampled_filtered_cloud_rod_array - origin, axis=1)
    extent = np.max(euclidian_distance)

    line = np.vstack((origin - direction_vector * extent,
                    origin + direction_vector * extent))

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    scale_factor = 100

    # Set equal aspect ratios
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0 * scale_factor

    mid_x = (X.max()+X.min()) * 0.5 * scale_factor
    mid_y = (Y.max()+Y.min()) * 0.5 * scale_factor
    mid_z = (Z.max()+Z.min()) * 0.5 * scale_factor
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.scatter(X_def * scale_factor, Y_def * scale_factor, Z_def * scale_factor, color="red", marker='o', alpha=0.5) # Deformed cloud
    ax.scatter(X * scale_factor, Y * scale_factor, Z * scale_factor, color="blue", marker='o', alpha=0.5) # Full cloud
    ax.plot(line[:, 0] * scale_factor, line[:, 1] * scale_factor, line[:, 2] * scale_factor, color="red", linewidth=5) # PCA direction

    # Plot PCA direction as arrow
    line_start = line[1] * scale_factor
    quiver_scale_factor = 1
    ax.quiver(line_start[0], line_start[1], line_start[2], 
    direction_vector[0] * quiver_scale_factor, direction_vector[1] * quiver_scale_factor, direction_vector[2] * quiver_scale_factor, 
    color='red', linewidth=5, arrow_length_ratio=1, pivot='tail', alpha=0.8)

    plt.show()

if __name__ == "__main__":
    main()