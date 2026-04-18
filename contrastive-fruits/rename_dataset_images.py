import os

def rename_images_in_datasets(base_dir):
    # Supported image extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    
    # Check if base directory exists
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    # Iterate over each subfolder in the base directory
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        
        # Ensure it's a directory
        if os.path.isdir(folder_path):
            count = 1
            images_renamed = 0
            
            # Iterate over all files in the subfolder
            for file_name in os.listdir(folder_path):
                file_ext = os.path.splitext(file_name)[1].lower()
                
                # Check if it's an image file
                if file_ext in valid_extensions:
                    old_file_path = os.path.join(folder_path, file_name)
                    
                    # Target new name
                    new_file_name = f"{folder_name}_{count}{file_ext}"
                    new_file_path = os.path.join(folder_path, new_file_name)
                    
                    # Avoid naming conflicts if a file with the target name already exists
                    while os.path.exists(new_file_path) and old_file_path != new_file_path:
                        count += 1
                        new_file_name = f"{folder_name}_{count}{file_ext}"
                        new_file_path = os.path.join(folder_path, new_file_name)
                        
                    # Rename the file if it's not already named correctly
                    if old_file_path != new_file_path:
                        os.rename(old_file_path, new_file_path)
                        images_renamed += 1
                        
                    count += 1
                    
            print(f"Renamed {images_renamed} image(s) in folder '{folder_name}'.")

if __name__ == "__main__":
    datasets_dir = r"D:\Study materials\Year 2\SEGP\Code\Datasets - Copy"
    rename_images_in_datasets(datasets_dir)