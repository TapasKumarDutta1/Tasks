import os
import cv2
import shutil
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from utils.file_utils import extract_number_from_string

CLASS_NAME_TO_ID = {
    "Bipolar": 0,
    "SpecimenBag": 1,
    "Grasper": 2,
    "Irrigator": 3,
    "Scissors": 4,
    "Hook": 5,
    "Clipper": 6
}

def parse_annotations(dataset_path):
    xml_files = sorted(
        [os.path.join(dataset_path, 'Annotations', f) for f in os.listdir(os.path.join(dataset_path, 'Annotations')) if f.endswith('.xml')],
        key=extract_number_from_string
    )

    labels_data = {
        'img_path': [], 'img_id': [], 'xmin': [], 'xmax': [],
        'ymin': [], 'ymax': [], 'img_w': [], 'img_h': [], 'class_id': []
    }

    for xml_file in tqdm(xml_files, desc="Parsing XML"):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        img_name = root.find('filename').text
        img_id, _ = os.path.splitext(img_name)
        img_path = os.path.join(dataset_path, 'JPEGImages', img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Image not found: {img_path}")
            continue

        height, width, _ = image.shape

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in CLASS_NAME_TO_ID:
                continue

            bndbox = obj.find('bndbox')
            labels_data['img_path'].append(img_path)
            labels_data['img_id'].append(img_id)
            labels_data['xmin'].append(int(bndbox.find('xmin').text))
            labels_data['xmax'].append(int(bndbox.find('xmax').text))
            labels_data['ymin'].append(int(bndbox.find('ymin').text))
            labels_data['ymax'].append(int(bndbox.find('ymax').text))
            labels_data['img_w'].append(width)
            labels_data['img_h'].append(height)
            labels_data['class_id'].append(CLASS_NAME_TO_ID[class_name])

    return pd.DataFrame(labels_data)

def save_yolo_format(df, split_name, base_dir='datasets/surgerical_tools'):
    labels_path = os.path.join(base_dir, split_name, 'labels')
    images_path = os.path.join(base_dir, split_name, 'images')
    os.makedirs(labels_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)

    grouped = df.groupby('img_id')
    for img_id, group in grouped:
        img_path = group.iloc[0]['img_path']
        img_name, ext = os.path.splitext(os.path.basename(img_path))
        label_lines = []

        for _, row in group.iterrows():
            xc = (row['xmin'] + row['xmax']) / 2 / row['img_w']
            yc = (row['ymin'] + row['ymax']) / 2 / row['img_h']
            w = (row['xmax'] - row['xmin']) / row['img_w']
            h = (row['ymax'] - row['ymin']) / row['img_h']
            label_lines.append(f"{row['class_id']} {xc:.4f} {yc:.4f} {w:.4f} {h:.4f}")

        with open(os.path.join(labels_path, f"{img_name}.txt"), 'w') as f:
            f.write('\n'.join(label_lines) + '\n')

        shutil.copy(img_path, os.path.join(images_path, img_name + ext))
