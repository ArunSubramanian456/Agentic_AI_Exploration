import os
import shutil

def clean_directory(dir_path:str):
    """
    Cleans up all files and sub directories within a given path.
    If the directory exists:
    - Deletes all files within the directory
    - Recursively deletes all subdirectories and their contents
    If the directory does not exist:
    - Creates the directory
    
    Args:
        dir_path (str): Path to the directory to clean up
    """
     
    # Clean up any previously existing temp data
    if os.path.exists(dir_path):
        print(f"Directory '{dir_path}' already exists. Deleting its contents...")
        # Iterate over all items in the directory
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path) # Delete files
                print(f"  Deleted file: {item}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path) # Delete subdirectories and their contents
                print(f"  Deleted subdirectory: {item}")
        print(f"Contents of '{dir_path}' cleared.")
    else:
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created successfully.")

