import os

# Define the source folder
src_folder = 'D:/files setup/FINAL DATA COLLECTION/palak alphabet data'

# Iterate through each item in the source folder
for action in os.listdir(src_folder):
    src_path = os.path.join(src_folder, action)

    # Check if the item is a directory
    if os.path.isdir(src_path):
        # Get the list of files in the subfolder
        npy_files = [file for file in os.listdir(src_path) if file.endswith('.npy')]

        # Check if there are exactly 1000 .npy files
        if len(npy_files) == 1001:
            print(f"Subfolder '{action}' contains exactly 1000 .npy files.")
        else:
            print(f"Subfolder '{action}' does not contain exactly 1000 .npy files. Found {len(npy_files)} files.")
    else:
        print(f"{src_path} is not a directory.")
        