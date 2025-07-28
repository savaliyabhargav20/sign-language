import os

# Define the main folder containing subfolders with .npy files
main_folder = 'D:/files setup/FINAL DATA COLLECTION/pranav alphabet data'

for action in os.listdir(main_folder):
    no=501
    sub_folder = os.path.join(main_folder, action)

    # Check if the item is a directory
    if os.path.isdir(sub_folder):
        # Iterate through each file in the subfolder
        for files in os.listdir(sub_folder):
            # Check if the file ends with .npy
            if files.endswith('.npy'):
                # Create a new filename (you can modify this as needed)
                new_filename = f"{no}.npy"  # Example: prefixing with 'renamed_'
                no+=1
                # Create full paths for old and new filenames
                old_file = os.path.join(sub_folder, files)  # Use subfolder path
                new_file = os.path.join(sub_folder, new_filename)  # Use subfolder path

                # Rename the file
                os.rename(old_file, new_file)
                print(f"Renamed: {old_file} to {new_file}")