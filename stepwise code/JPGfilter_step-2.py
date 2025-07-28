import os
import shutil

# Set the source and destination directories
src_dir = '  ' # source from  where to get images
dst_dir = '  ' # destination where to save  images

# Loop through each action folder in the source directory
for action in os.listdir(src_dir):
    # Construct the full paths for source and destination
    src_path = os.path.join(src_dir, action)
    dst_path = os.path.join(dst_dir, action)

    # Check if the source is a directory
    if os.path.isdir(src_path):
        # Create the destination directory if it doesn't exist
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        # Loop through each .npy file in the action folder
        for file in os.listdir(src_path):
            if file.endswith('.jpg'):
                # Construct the full file paths
                src_file = os.path.join(src_path, file)
                dst_file = os.path.join(dst_path, file)

                # Move the file
                shutil.move(src_file, dst_file)

    else:
        print(f'Skipping non-directory: {src_path}')

print('All files have been moved to their respective action folders.')