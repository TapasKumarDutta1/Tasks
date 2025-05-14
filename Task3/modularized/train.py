%%writefile train.py
import os
from utils.file_utils import load_ids
from data.prepare_data import parse_annotations, save_yolo_format
from ultralytics import YOLO
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import argparse

def prepare_dataset(dataset_path):
    df = parse_annotations(dataset_path)

    train_ids = load_ids(os.path.join(dataset_path, 'ImageSets/Main/train.txt'))
    val_ids = load_ids(os.path.join(dataset_path, 'ImageSets/Main/val.txt'))
    test_ids = load_ids(os.path.join(dataset_path, 'ImageSets/Main/test.txt'))

    save_yolo_format(df[df['img_id'].isin(train_ids)], 'train')
    save_yolo_format(df[df['img_id'].isin(val_ids)], 'val')
    save_yolo_format(df[df['img_id'].isin(test_ids)], 'test')

def write_dataset_yaml():
    yaml_content = '''path: surgerical_tools
train: train/images
val: val/images
test: test/images
nc: 7
names: [Bipolar, SpecimenBag, Grasper, Irrigator, Scissors, Hook, Clipper]
'''
    with open('datasets.yaml', 'w') as f:
        f.write(yaml_content)

def train_model():
    model = YOLO('yolov8s.pt')  # Use a valid YOLOv8 model file
    os.environ['WANDB_MODE'] = 'offline'
    model.train(data='datasets.yaml', epochs=2, batch=16, device='cuda', imgsz=640, cache=True)

def plot_results():
    from utils.file_utils import extract_number_from_string
    log_dir = max(glob('runs/detect/train*'), key=extract_number_from_string)
    results = pd.read_csv(os.path.join(log_dir, 'results.csv'))

    plt.plot(results.index + 1, results['metrics/mAP50(B)'], label='mAP@0.5')
    plt.plot(results.index + 1, results['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('YOLO Training Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True, help="Path to the trained YOLO model weights")
    args = parser.parse_args()
    
    prepare_dataset(args.dataset_path)
    write_dataset_yaml()
    train_model()
    plot_results()
