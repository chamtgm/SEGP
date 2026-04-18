import os
import glob
import shutil
import random
import yaml

# ==========================================
# CONFIGURATION
# ==========================================
input_images_dir = r"D:\Study materials\Year 2\SEGP\Code\Datasets - Copy"
input_labels_dir = r"D:\Study materials\Year 2\SEGP\Code\Object Detection\auto_annotate_labels"
output_dataset_dir = r"D:\Study materials\Year 2\SEGP\Code\Object Detection\yolo_dataset"

train_ratio = 0.8  # 80% for training, 20% for validation

class_mapping = {
    'apple': 0,
    'banana': 1,
    'dragonfruit': 2,
    'durian': 3,
    'egg plant': 4,
    'grapes': 5,
    'orange': 6,
    'pineapple': 7,
    'strawberry': 8,
    'watermelon': 9
}

print("Setting up YOLO dataset structure...")

# 1. Create YOLO directory structure
for split in ['train', 'val']:
    os.makedirs(os.path.join(output_dataset_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_dataset_dir, 'labels', split), exist_ok=True)

# 2. Gather matching image-label pairs
all_pairs = []

for class_name in class_mapping.keys():
    folder_path = os.path.join(input_images_dir, class_name)
    if not os.path.exists(folder_path):
        continue

    images = glob.glob(os.path.join(folder_path, "*.jpg")) + \
             glob.glob(os.path.join(folder_path, "*.png")) + \
             glob.glob(os.path.join(folder_path, "*.jpeg"))
             
    for img_path in images:
        base_name = os.path.basename(img_path)
        txt_name = os.path.splitext(base_name)[0] + ".txt"
        label_path = os.path.join(input_labels_dir, txt_name)
        
        # Only add to pairs if the label exists (meaning auto_annotate successfully found the fruit)
        if os.path.exists(label_path):
            all_pairs.append((img_path, label_path))

print(f"Found {len(all_pairs)} successfully annotated image-label pairs.")

# 3. Shuffle and split into train and val
random.seed(42)  # For reproducibility
random.shuffle(all_pairs)

split_idx = int(len(all_pairs) * train_ratio)
train_pairs = all_pairs[:split_idx]
val_pairs = all_pairs[split_idx:]

# 4. Function to copy files to their new home
def copy_files(pairs, split_name):
    print(f"Copying {len(pairs)} files to {split_name}...")
    for img_path, label_path in pairs:
        # Copy image
        img_dest = os.path.join(output_dataset_dir, 'images', split_name, os.path.basename(img_path))
        shutil.copy2(img_path, img_dest)
        
        # Copy label
        label_dest = os.path.join(output_dataset_dir, 'labels', split_name, os.path.basename(label_path))
        shutil.copy2(label_path, label_dest)

copy_files(train_pairs, 'train')
copy_files(val_pairs, 'val')

# 5. Generate data.yaml file
yaml_path = os.path.join(output_dataset_dir, 'data.yaml')
yaml_content = {
    'path': output_dataset_dir.replace("\\", "/"),
    'train': 'images/train',
    'val': 'images/val',
    'names': {v: k for k, v in class_mapping.items()} # Reverses the mapping to {0: 'apple', 1: 'banana'...}
}

with open(yaml_path, 'w') as f:
    yaml.dump(yaml_content, f, sort_keys=False)

print(f"\nAll Done!")
print(f"Dataset successfully organized at: {output_dataset_dir}")
print(f"Your configuration file is at: {yaml_path}")
