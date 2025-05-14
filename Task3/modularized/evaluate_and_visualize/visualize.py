
import os
import cv2
import matplotlib.pyplot as plt
import yaml

def visualize_predictions(model, dataset_yaml, class_dict, indices=range(20)):
    """
    Visualize predictions on a few test images from the test set defined in datasets.yaml.
    Displays the top 20 images in a 5x4 grid.

    Args:
        model: YOLO model object.
        dataset_yaml (str): Path to the dataset yaml file.
        class_dict (dict): Mapping from class IDs to class names.
        indices (list): Indices of images to visualize (default is the first 20 images).
    """
    # Read dataset.yaml to locate test set
    with open(dataset_yaml, 'r') as f:
        data = yaml.safe_load(f)

    test_img_dir = os.path.join('datasets',data.get('path'),data.get('test'))
    if not test_img_dir or not os.path.isdir(test_img_dir):
        print(test_img_dir)
        raise ValueError(f"Test image directory not found or invalid in {dataset_yaml}")

    # Get all image files in the test set directory
    image_files = sorted(os.listdir(test_img_dir))
    selected_indices = [i for i in indices if i < len(image_files)]

    # Create a 5x4 grid of subplots
    fig, axes = plt.subplots(5, 4, figsize=(15, 15))
    axes = axes.ravel()  # Flatten axes array to easily index

    for idx, ax in zip(selected_indices, axes):
        image_path = os.path.join(test_img_dir, image_files[idx])
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Perform predictions with the YOLO model
        results = model.predict(image_path, device='cuda', verbose=False)
        annotated = image_rgb.copy()

        # Annotate the image with bounding boxes and class labels
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = f"{class_dict.get(class_id, 'Unknown')} {confidence:.2f}"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display the annotated image on the appropriate subplot
        ax.imshow(annotated)
        ax.set_title(f"Prediction: {os.path.basename(image_path)}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("prediction_grid.png")  # Or any filename you prefer
    plt.show()

