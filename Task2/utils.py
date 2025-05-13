import os
import glob
import numpy as np
import pydicom
import open3d as o3d
from skimage import measure


def get_rotation_angles(dicom_path):
    """
    Compute the rotation angles from the DICOM ImageOrientationPatient tag.

    Args:
        dicom_path (str): Path to a DICOM file.

    Returns:
        np.ndarray: Rotation angles in radians.
    """
    dicom = pydicom.dcmread(dicom_path)
    orientation = [float(val) for val in dicom.ImageOrientationPatient]
    rotation_vector = -np.pi / 2 * np.array([i + j for i, j in zip(orientation[:3], orientation[3:])])
    return rotation_vector


def load_scan_series(directory):
    """
    Load and sort a series of DICOM slices from a directory.

    Args:
        directory (str): Path to directory containing DICOM files.

    Returns:
        List[pydicom.Dataset]: List of sorted DICOM slices.
    """
    slices = [pydicom.dcmread(os.path.join(directory, f)) for f in os.listdir(directory)]
    slices.sort(key=lambda s: int(s.InstanceNumber))

    try:
        slice_thickness = abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except Exception:
        slice_thickness = abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_in_hu(slices):
    """
    Convert DICOM pixel data to Hounsfield Units (HU).

    Args:
        slices (List[pydicom.Dataset]): List of DICOM slices.

    Returns:
        np.ndarray: 3D array of pixel data in HU.
    """
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16)
    image[image == -2000] = 0

    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope

    if slope != 1:
        image = (slope * image.astype(np.float64)).astype(np.int16)

    return image + np.int16(intercept)


def load_dicom_volume(directory):
    """
    Load a 3D volume from a series of DICOM files.

    Args:
        directory (str): Path to folder with DICOM files.

    Returns:
        tuple: (3D numpy array of volume, List of DICOM slices)
    """
    slices = [pydicom.dcmread(os.path.join(directory, f)) for f in os.listdir(directory)]
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    volume = np.stack([s.pixel_array for s in slices])
    return volume, slices


def convert_volume_to_hu(volume, slices):
    """
    Apply DICOM rescale to convert volume to HU.

    Args:
        volume (np.ndarray): Raw pixel data.
        slices (List[pydicom.Dataset]): DICOM slices.

    Returns:
        np.ndarray: Volume in HU.
    """
    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope
    return volume * slope + intercept


def extract_surface_mesh(volume, threshold=-300):
    """
    Extract surface mesh using the Marching Cubes algorithm.

    Args:
        volume (np.ndarray): HU volume.
        threshold (float): Intensity value for surface extraction.

    Returns:
        np.ndarray: Vertices of the surface mesh.
    """
    verts, _, _, _ = measure.marching_cubes(volume, level=threshold)
    return verts


def create_point_cloud(vertices):
    """
    Convert mesh vertices to Open3D point cloud.

    Args:
        vertices (np.ndarray): Mesh vertices.

    Returns:
        o3d.geometry.PointCloud: Point cloud object.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    return pcd


def process_dicom_to_point_cloud(input_dir, output_file):
    """
    Convert a DICOM directory into an oriented and filtered point cloud.

    Args:
        input_dir (str): Path to the folder containing DICOM slices.
        output_file (str): Output path for the point cloud file (.ply or .pcd).
    """
    slices = load_scan_series(input_dir)
    rotation = get_rotation_angles(glob.glob(os.path.join(input_dir, "*"))[0])
    volume_raw, slice_data = load_dicom_volume(input_dir)
    volume_hu = convert_volume_to_hu(volume_raw, slice_data)
    vertices = extract_surface_mesh(volume_hu, threshold=-100)
    point_cloud = create_point_cloud(vertices)

    # Rotate based on extracted DICOM orientation
    rotation_matrix = point_cloud.get_rotation_matrix_from_xyz(rotation)
    point_cloud.rotate(rotation_matrix, center=(0, 0, 0))

    # Filter using DBSCAN to keep the largest cluster
    labels = np.array(point_cloud.cluster_dbscan(eps=2.0, min_points=10, print_progress=True))
    if len(labels[labels >= 0]) == 0:
        raise ValueError("No clusters found in point cloud.")

    largest_cluster = np.bincount(labels[labels >= 0]).argmax()
    inliers = labels == largest_cluster
    filtered_pcd = point_cloud.select_by_index(np.where(inliers)[0])

    o3d.io.write_point_cloud(output_file, filtered_pcd)
