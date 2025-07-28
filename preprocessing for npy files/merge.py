import os
import shutil

#D:\files setup\test images 2\1\{i}.npy

# Define the source and destination folders
src_folder = 'D:/files setup/FINAL DATA COLLECTION/pranav alphabet data'
dst_folder = 'D:/files setup/FINAL DATA COLLECTION/palak alphabet data'

# Check if the destination folder exists; if not, create it
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

# Iterate through each item in the source folder
for action in os.listdir(src_folder):
    src_path = os.path.join(src_folder, action)
    dst_path = os.path.join(dst_folder, action)

    # Check if the item is a directory
    if os.path.isdir(src_path):
        # Create the corresponding directory in the destination folder
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        # Iterate through each file in the source subdirectory
        for file in os.listdir(src_path):
            if file.endswith('.npy'):
                src_file = os.path.join(src_path, file)
                dst_file = os.path.join(dst_path, file)

                # Move the file from source to destination
                shutil.move(src_file, dst_file)
                print(f"{src_file} moved to {dst_file}")
    else:
        print(f"{src_path} is not a directory.")