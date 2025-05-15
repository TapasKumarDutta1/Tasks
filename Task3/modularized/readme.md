# Surgery Instrument Object Detection Using YOLOv12

## Objective
The goal of this project is to implement an efficient and accurate object detection system for identifying and classifying surgical instruments in medical environments. By leveraging YOLOv12, a state-of-the-art deep learning model, this system aims to assist healthcare professionals by automatically detecting instruments in real-time, aiding in surgical procedures, and improving workflow efficiency.

## Approach
The model uses YOLOv12 (You Only Look Once version 12), a cutting-edge, real-time object detection algorithm known for its speed and accuracy. The approach involves:

1. **Dataset Preparation**: Collecting and annotating a dataset of surgical instruments, ensuring high-quality, labeled images for training.
2. **Model Training**: Fine-tuning YOLOv12 on the annotated dataset for accurate classification and localization of surgical instruments.
3. **Inference and Evaluation**: Running the trained model on unseen data to test its performance and adjust hyperparameters as needed.

## Visualization
The trained YOLOv12 model outputs bounding boxes, class labels, and confidence scores for each detected surgical instrument. Visualizing these predictions on input images enables real-time monitoring and quality assessment.

### Example:
- **Input Image**: A photo of a surgical table with multiple instruments.
- **Output Image**: The same image with bounding boxes drawn around the instruments, each labeled with its predicted class (e.g., scalpel, forceps, etc.) and a confidence score.

## Demo
To run the demo and visualize object detection in action, follow these steps:

### Requirements
- Python 3.x
- YOLOv12 model and weights
- OpenCV, PyTorch, and other dependencies (refer to `requirements.txt`)

### Instructions:
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/surgery-instrument-detection.git
    cd surgery-instrument-detection
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the detection script:
    ```bash
    python detect.py --input path_to_image_or_video --output path_to_save_results
    ```

4. View the output:
   - The results will be saved as images or videos with visualized detections, including bounding boxes and class labels.

## Code Example
Here is a sample Python script to perform object detection:

```python
import cv2
from yolov12 import YOLOv12

# Load YOLOv12 model
model = YOLOv12(weights="yolov12.weights")

# Load an image
image = cv2.imread("surgical_scene.jpg")

# Perform inference
results = model.detect(image)

# Visualize results
for box, label, score in zip(results['boxes'], results['labels'], results['scores']):
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Save and show the output image
cv2.imwrite("output_image.jpg", image)
cv2.imshow("Detection Results", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
