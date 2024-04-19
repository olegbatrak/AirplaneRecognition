import os

def delete_rotated_images(directory_path):
    # Check if the provided path is a directory
    if not os.path.isdir(directory_path):
        print("The provided path is not a directory.")
        return

    deleted_count = 0  # Counter to track the number of deleted files

    # Traverse the directory and its subdirectories
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # Check if 'rotated' is in the file name
            if 'rotated' in file.lower() or 'flip' in file.lower():
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)  # Remove the file
                    #print(f"Deleted file: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Error while trying to delete {file_path}: {e}")

    # Print out the result
    if deleted_count > 0:
        print(f"Total deleted files: {deleted_count}")
    else:
        print("No 'rotated' files found to delete.")

# Example usage
directory_path = R'C:\Users\006ma\PycharmProjects\AirplaneRecognition\Data_updated'
delete_rotated_images(directory_path)
