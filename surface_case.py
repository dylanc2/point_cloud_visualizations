import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import numpy as np
import random

def main():
    ### Deformed cloud ###

    deformed_cloud_plane_array = np.load('deformed_cloud_plane.npy')
    num_points_def = deformed_cloud_plane_array.shape[0]
 
    # Plot a subset of the points
    sample_size = round(num_points_def / 3)
    kept_indices = []
    for _ in range(sample_size):
        kept_indices.append(random.randint(0, num_points_def))
    sampled = []
    for i, p in enumerate(deformed_cloud_plane_array):
        if i in kept_indices:
            sampled.append(p)
    sampled_deformed_cloud_plane_array = np.asarray(sampled)

    X_def = sampled_deformed_cloud_plane_array[0:sample_size, 0:1]
    Y_def = sampled_deformed_cloud_plane_array[0:sample_size, 1:2]
    Z_def = sampled_deformed_cloud_plane_array[0:sample_size, 2:3]

    ### Full cloud ###

    filtered_cloud_plane_array = np.load('filtered_cloud_plane.npy')
    num_points = filtered_cloud_plane_array.shape[0]

    # Plot a subset of the points, otherwise they will blot out the deformed points
    sample_size = round(num_points / 7)
    kept_indices = []
    for _ in range(sample_size):
        kept_indices.append(random.randint(0, num_points))
    sampled = []

    dist_threshold = 2e-3
    for i, p in enumerate(filtered_cloud_plane_array):
        distances = np.linalg.norm(deformed_cloud_plane_array - p, axis=1)
        min_distance = np.min(distances)
        if i in kept_indices and p not in deformed_cloud_plane_array and min_distance > dist_threshold:
            sampled.append(p)
    sampled_filtered_cloud_plane_array = np.asarray(sampled)

    X = sampled_filtered_cloud_plane_array[0:sample_size, 0:1]
    Y = sampled_filtered_cloud_plane_array[0:sample_size, 1:2]
    Z = sampled_filtered_cloud_plane_array[0:sample_size, 2:3]

    # Get points that are intersected by a plane
    # Self-defined quantities
    pc_centroid = np.mean(sampled_filtered_cloud_plane_array, axis=0)
    plane_normal = np.array([1, 0, 0]) 
    threshold = 1e-3

    points_to_centroid = sampled_filtered_cloud_plane_array - pc_centroid
    projection_dist = np.abs(np.dot(points_to_centroid, plane_normal))
    close_idx = np.where(projection_dist < threshold)

    # Run PCA on the points that are intersected by the plane (up to a threshold)
    close_coords = np.array((X[close_idx], Y[close_idx], Z[close_idx])).T
    close_coords = np.reshape(close_coords, (close_coords.shape[1], close_coords.shape[2]))
    close_coords_num = close_coords.shape[0]
    X_close = close_coords[0:close_coords_num, 0:1]
    Y_close = close_coords[0:close_coords_num, 1:2]
    Z_close = close_coords[0:close_coords_num, 2:3]

    pca = PCA(n_components=1)
    pca.fit(close_coords)
    direction_vector = pca.components_[0]
    print("Direction vector: {}".format(direction_vector))

    origin = np.mean(sampled_filtered_cloud_plane_array, axis=0)
    euclidian_distance = np.linalg.norm(sampled_filtered_cloud_plane_array - origin, axis=1)
    extent = np.max(euclidian_distance)

    line = np.vstack((origin - direction_vector * extent,
                    origin + direction_vector * extent))
    
    print(line)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    scale_factor = 100 # Scale up point cloud so that it matches the scale of the plane

    # Set equal aspect ratios
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0 * scale_factor
    mid_x = (X.max()+X.min()) * 0.5 * scale_factor
    mid_y = (Y.max()+Y.min()) * 0.5 * scale_factor
    mid_z = (Z.max()+Z.min()) * 0.5 * scale_factor
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Plot plane
    yy, zz = np.meshgrid(range(-5, 5, 1), range(10, 17, 1))
    xx = 0
    ax.plot_surface(xx, yy, zz, alpha=0.25)

    ax.scatter(X_def * scale_factor, Y_def * scale_factor, Z_def * scale_factor, color="red", marker='o', alpha=0.5) # Deformed cloud
    ax.scatter(X * scale_factor, Y * scale_factor, Z * scale_factor, color="blue", marker='o', alpha=0.5) # Full cloud
    ax.scatter(X_close * scale_factor, Y_close * scale_factor, Z_close * scale_factor, color="black", marker='o', alpha=0.5) # Intersected points
    # ax.plot(line[:, 0] * scale_factor, line[:, 1] * scale_factor, line[:, 2] * scale_factor, color="red", linewidth=5) # PCA direction as line
    
    # Plot PCA direction as arrow
    line_start = line[0]
    quiver_scale_factor = 12
    ax.quiver(line_start[0] * scale_factor, line_start[1] * scale_factor, line_start[2] * scale_factor, 
    direction_vector[0] * quiver_scale_factor, direction_vector[1] * quiver_scale_factor, direction_vector[2] * quiver_scale_factor, 
    color='red', linewidth=5, arrow_length_ratio=0.1, alpha=0.8)

    plt.show()

if __name__ == "__main__":
    main()