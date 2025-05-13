# Task 2: Landmark Detection on a 3D Facial Point Cloud

## Objective
Detect the **tip of the nose** from a 3D facial point cloud reconstructed from DICOM medical imaging data.

---

## Approach

1. **3D Volume Reconstruction**
   - Load and sort DICOM slices.
   - Convert pixel data to Hounsfield Units (HU).
   - Extract a 3D surface using the Marching Cubes algorithm.

2. **Point Cloud Generation**
   - Convert surface mesh vertices to a point cloud using Open3D.
   - Apply orientation correction from DICOM metadata.
   - Denoise using DBSCAN clustering to isolate the facial region.

3. **Landmark Detection**
   - Center the point cloud.
   - Identify the nose tip as the point with **maximum Z-coordinate** (most protruding).

4. **Visualization**
   - Overlay a red sphere at the detected nose tip for verification.

---

## Usage

### Requirements
- Python 3.8+
- Dependencies: `open3d`, `pydicom`, `numpy`, `scikit-image`

Install dependencies:
```bash
pip install open3d pydicom numpy scikit-image
