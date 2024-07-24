import json
import os

# Load the COCO JSON file
with open('/home/jirjeesa/Projects/RSIP-ION/FLIR_ADAS_v2/yolo_training_dataset/val/coco.json') as f:
    data = json.load(f)

images_base_path = '/home/jirjeesa/Projects/RSIP-ION/FLIR_ADAS_v2/yolo_training_dataset/val/images/'
labels_base_path = '/home/jirjeesa/Projects/RSIP-ION/FLIR_ADAS_v2/yolo_training_dataset/val/labels/'

os.makedirs(labels_base_path, exist_ok=True)

annotations = {img['id']: [] for img in data['images']}
for ann in data['annotations']:
    annotations[ann['image_id']].append(ann)

unlabeled_images = []

# Process each image
for img in data['images']:
    img_id = img['id']
    img_filename = img['file_name']
    img_width = img['width']
    img_height = img['height']

    label_file_path = os.path.join(labels_base_path, os.path.splitext(img_filename)[0] + '.txt')

    yolo_labels = []
    for ann in annotations[img_id]:
        cat_id = ann['category_id'] - 1
        x_min, y_min, width, height = ann['bbox']
        x_center = (x_min + width / 2) / img_width
        y_center = (y_min + height / 2) / img_height
        width /= img_width
        height /= img_height
        yolo_labels.append(f"{cat_id} {x_center} {y_center} {width} {height}")

    if yolo_labels:
        os.makedirs(os.path.dirname(label_file_path), exist_ok=True)
        with open(label_file_path, 'w') as file:
            file.write("\n".join(yolo_labels))
    else:
        unlabeled_images.append(img_filename)

print("Conversion complete.")
if unlabeled_images:
    print(f"Unlabeled Images ({len(unlabeled_images)}):")
    for filename in unlabeled_images:
        print(filename)
