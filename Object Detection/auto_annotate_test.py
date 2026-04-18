import os
import cv2
import numpy as np
import glob
from rembg import remove

def create_yolo_label(x, y, w, h, img_w, img_h, class_idx):
    """
    Convert bounding box (from OpenCV) to YOLO format.
    YOLO format: class_idx x_center y_center width height
    All values (except class_idx) must be normalized between 0 and 1.
    """
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return f"{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

# ==========================================
# CONFIGURATION
# ==========================================
input_dir = r"D:\Study materials\Year 2\SEGP\Code\Datasets - Copy"
output_preview_dir = r"D:\Study materials\Year 2\SEGP\Code\Object Detection\auto_annotate_preview"
output_labels_dir = r"D:\Study materials\Year 2\SEGP\Code\Object Detection\auto_annotate_labels"

os.makedirs(output_preview_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Define which folders correspond to which YOLO class ID
# You can expand this mapping for all your fruits
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

print(f"Starting auto-annotation test...")

for class_name, class_idx in class_mapping.items():
    folder_path = os.path.join(input_dir, class_name)
    if not os.path.exists(folder_path):
        print(f"Skipping {class_name} (folder not found)")
        continue

    # Get all images for the class
    images = glob.glob(os.path.join(folder_path, "*.jpg")) + \
             glob.glob(os.path.join(folder_path, "*.png")) + \
             glob.glob(os.path.join(folder_path, "*.jpeg"))
    
    print(f"Processing {len(images)} images for class: {class_name} ...")

    for img_path in images:
        base_name = os.path.basename(img_path)
        
        # 1. Read the image
        img = cv2.imread(img_path)
        if img is None: 
            continue
        
        img_h, img_w = img.shape[:2]

        # 2. Use rembg to isolate the fruit (removes the table, background, etc.)
        with open(img_path, 'rb') as f:
            input_bytes = f.read()
            
        out_bytes = remove(input_bytes)
        
        # 3. Decode the byte mask 
        nparr = np.frombuffer(out_bytes, np.uint8)
        out_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        # out_img comes back as RGBA. We only care about the Alpha channel (index 3)
        if out_img.shape[2] == 4:
            alpha_channel = out_img[:, :, 3]
            
            # Find all non-transparent pixels
            coords = cv2.findNonZero(alpha_channel)
            
            if coords is not None:
                # 4. Get the Bounding Box 
                x, y, w, h = cv2.boundingRect(coords)
                
                # Expand the box slightly (10 pixels padding) just in case
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img_w - x, w + padding * 2)
                h = min(img_h - y, h + padding * 2)

                # 5. Draw it on a preview image so you can visually verify
                preview = img.copy()
                cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 3) # Green box
                cv2.putText(preview, f"{class_name} (Auto)", (x, max(20, y - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                preview_path = os.path.join(output_preview_dir, f"{class_name}_{base_name}")
                cv2.imwrite(preview_path, preview)
                
                # 6. Format the coordinates into YOLO layout and save the .txt file
                yolo_label = create_yolo_label(x, y, w, h, img_w, img_h, class_idx)
                txt_name = os.path.splitext(base_name)[0] + ".txt"
                txt_path = os.path.join(output_labels_dir, txt_name)
                
                with open(txt_path, 'w') as label_file:
                    label_file.write(yolo_label + "\n")
                    
                print(f" -> Success: {base_name}. Box drawn. YOLO txt saved.")
            else:
                print(f" -> Failed: {base_name}. Background algorithm removed everything.")
        else:
            print(f" -> Failed: {base_name}. No alpha channel returned.")

print("\nAll done! Please check the Code/Object Detection/auto_annotate_preview/ folder.")
