import os
import numpy as np
import open3d as o3d
from utils import process_dicom_to_point_cloud


def center_point_cloud(pcd):
    """
    Center the point cloud around the origin.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.

    Returns:
        o3d.geometry.PointCloud: Centered point cloud.
    """
    center = pcd.get_center()
    pcd.translate(-center)
    return pcd


def plot_nose_tip(input_path):
    """
    Visualize the nose tip (highest Z point) from a DICOM folder or point cloud file.

    Steps:
        1. Convert DICOM to point cloud if directory is given.
        2. Center the point cloud.
        3. Find the point with maximum Z value.
        4. Mark it with a red sphere.
        5. Display in Open3D viewer.

    Args:
        input_path (str): Path to DICOM folder or .ply/.pcd file.
    """
    if os.path.isdir(input_path):
        output_file = 'cloud_from_dicom.ply'
        print(f"Processing DICOM directory: {input_path}")
        process_dicom_to_point_cloud(input_path, output_file)
        input_path = output_file

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"No such file: {input_path}")

    pcd = o3d.io.read_point_cloud(input_path)
    if len(pcd.points) == 0:
        raise ValueError("Loaded point cloud is empty.")

    pcd = center_point_cloud(pcd)
    points = np.asarray(pcd.points)

    max_z_index = np.argmax(points[:, 2])
    nose_tip = points[max_z_index]

    print("Nose tip (farthest point along Z-axis):", nose_tip)

    # Create a small red sphere at the nose tip
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    sphere.translate(nose_tip)
    sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red
    
    # Visualize
    o3d.visualization.draw_geometries([pcd, sphere])

