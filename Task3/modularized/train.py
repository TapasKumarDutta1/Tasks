import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from ultralytics import YOLO
from utils.file_utils import load_ids, extract_number_from_string
from data.prepare_data import parse_annotations, save_yolo_format

def prepare_dataset(dataset_path):
    df = parse_annotations(dataset_path)

    train_ids = load_ids(os.path.join(dataset_path, 'ImageSets/Main/train.txt'))
    val_ids = load_ids(os.path.join(dataset_path, 'ImageSets/Main/val.txt'))
    test_ids = load_ids(os.path.join(dataset_path, 'ImageSets/Main/test.txt'))

    save_yolo_format(df[df['img_id'].isin(train_ids)], 'train')
    save_yolo_format(df[df['img_id'].isin(val_ids)], 'val')
    save_yolo_format(df[df['img_id'].isin(test_ids)], 'test')

def write_dataset_yaml(output_path='datasets.yaml'):
    yaml_content = '''path: surgerical_tools
train: train/images
val: val/images
test: test/images
nc: 7
names: [Bipolar, SpecimenBag, Grasper, Irrigator, Scissors, Hook, Clipper]
'''
    with open(output_path, 'w') as f:
        f.write(yaml_content)

def train_model(args):
    model = YOLO(args.model_weights)
    os.environ['WANDB_MODE'] = 'offline'

    model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        batch=args.batch_size,
        device=args.device,
        imgsz=args.imgsz,
        cache=args.cache
    )

def plot_results():
    log_dir = max(glob('runs/detect/train*'), key=extract_number_from_string)
    results = pd.read_csv(os.path.join(log_dir, 'results.csv'))

    plt.plot(results.index + 1, results['metrics/mAP50(B)'], label='mAP@0.5')
    plt.plot(results.index + 1, results['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('YOLO Training')
    plt.grid(True)
    plt.legend()
    plt.savefig("training_metrics.png")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train YOLO model with user-defined hyperparameters")

    parser.add_argument('--dataset-path', type=str, required=True, help="Path to dataset root directory")
    parser.add_argument('--model-weights', type=str, default='yolov8s.pt', help="Path to pretrained YOLO weights")
    parser.add_argument('--data-yaml', type=str, default='datasets.yaml', help="Path to YOLO data config YAML")
    parser.add_argument('--epochs', type=int, default=2, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=16, help="Training batch size")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use: 'cuda' or 'cpu'")
    parser.add_argument('--imgsz', type=int, default=640, help="Input image size for training")
    parser.add_argument('--cache', type=bool, default=True, help="Whether to cache images for training")

    args = parser.parse_args()

    prepare_dataset(args.dataset_path)
    write_dataset_yaml(args.data_yaml)
    train_model(args)
    plot_results()
