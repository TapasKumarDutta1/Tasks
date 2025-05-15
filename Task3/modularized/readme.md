# Surgery Instrument Object Detection Using YOLOv12

## Objective
Detect surgerical tools in images and videos.


## Approach


1. **Dataset Preparation**:
  1.   The Dataset consist of train, test and validation ids
  2.   Convert the annotations from PASCAL VOC to YOLO format
2. **Model Training**:
   1. Fine-tuning YOLOv12 on the annotated dataset for accurate classification and localization of surgical instruments.
   2. Save the model's best weights for validation set
3. **Inference and Evaluation**:
   1. Load the best weights for validation set
   2. Running the trained model on test data and video.

## Results

| Class         | Images | Instances | Box(P) | Box(R) | mAP50 | mAP95 |
|---------------|--------|-----------|--------|--------|-------|-------|
| all           | 563    | 780       | 0.709  | 0.65   | 0.729 | 0.383 |
| Bipolar       | 95     | 95        | 0.815  | 0.789  | 0.844 | 0.421 |
| SpecimenBag   | 96     | 96        | 0.597  | 0.656  | 0.669 | 0.356 |
| Grasper       | 234    | 293       | 0.586  | 0.807  | 0.74  | 0.359 |
| Irrigator     | 84     | 84        | 0.592  | 0.675  | 0.696 | 0.325 |
| Scissors      | 84     | 84        | 0.901  | 0.434  | 0.692 | 0.336 |
| Hook          | 64     | 64        | 0.862  | 0.562  | 0.794 | 0.475 |
| Clipper       | 64     | 64        | 0.61   | 0.625  | 0.671 | 0.411 |


## Visualization
### Images
### Videos


## Demo
A working demo of the entire pipeline is available in the Jupyter notebook here:
https://github.com/TapasKumarDutta1/Tasks/blob/main/Task2/Task_2_Demo.ipynb](https://www.kaggle.com/code/tapaskd123/task-1-final-run-100?scriptVersionId=239962497
