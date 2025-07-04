# Surgery Instrument Object Detection Using YOLOv12

## Objective
Detect surgerical tools in images and videos.


## Approach


1. **Dataset Preparation**:
   1. The Dataset(https://www.kaggle.com/datasets/xiaoweixumedicalai/imagetbad) consist of train, test and validation ids
   2.  Convert the annotations from PASCAL VOC to YOLO format
2. **Model Training**:
   1. Fine-tuning YOLOv12 on the annotated dataset for accurate classification and localization of surgical instruments.
   2. Save the model's best weights for validation set
3. **Inference and Evaluation**:
   1. Load the best weights for validation set
   2. Running the trained model on test data and video.

## Results
<img src="./static/results.png" width=900>

| Confusion Matrix Normalized | PR_Curve |
|:-----------------:|:-----------------:|
| <img src="./static/confusion_matrix_normalized.png" width="450"> | <img src="./static/PR_curve.png" width="450"> |



## Visualization

### Images

<img src="./static/val_batch0_pred.jpg" width=900>
<img src="./static/val_batch1_pred.jpg" width=900>
<img src="./static/val_batch2_pred.jpg" width=900>



### Videos
The model achives approximately ``50fps`` when working on videos, video link: https://drive.google.com/file/d/11sigvA18y87uZ5d-PQiZHntiQqYoN43q/view?usp=sharing

## Demo
A working demo of the entire pipeline is available in the Jupyter notebook here:
https://www.kaggle.com/code/tapaskd123/task-1-final-run-100?scriptVersionId=239962497
