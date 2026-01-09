import os
import re
from utils.helpers import alphanum_key

def rename_simple(path):
    for i in [" ", "(", ")"]:
        path = path.replace(i, '')
    return path

def list_files_structured(folder):
    """List all filenames in a folder recursively,
    keep the arborescence of the folder

    Args:
        folder (str): Path to the desired folder

    Returns:
        dict: Recursive dictionnary where each key is a folder, each folder is represented by a dictionnary
        and has a "files" key if there is any files in it
    """
    filenames = dict()

    for r, d, f in os.walk(folder):
        position = r.split(os.sep)
        current = filenames
        for p in position:
            try:
                current = current[p]
            except:
                current[p] = dict()
                current = current[p]

        current["files"] = f

    return filenames


def list_files(folder, pattern="", recursive=True):
    """List all files from a given folder

    Args:
        folder (str): Path to the desired folder
        pattern (str, optional): A pattern that is looked for in the filenames. Defaults to "".
        recursive (bool, optional): Whether to search recursively. Defaults to True.

    Returns:
        list: A list of strings, paths to every file from the folder
    """
    files = []
    compiled_pattern = re.compile(pattern)
    
    if recursive:
        for r, d, f in os.walk(folder):
            for file in f:
                if re.search(compiled_pattern, file):
                    files.append(os.path.join(r, file))
    else:
        for file in os.listdir(folder):
            full_path = os.path.join(folder, file)
            if os.path.isfile(full_path) and re.search(compiled_pattern, file):
                files.append(full_path)
    return files


def list_subfolders(folder, pattern=""):
    """List all folders from a given folder recursively

    Args:
        folder (str): Path to the desired folder
        pattern (str, optional): A pattern that is looked for in the filenames. Defaults to "".

    Returns:
        list: A list of strings, paths to every folder from the folder
    """
    folders = []
    compiled_pattern = re.compile(pattern)
    for r, d, f in os.walk(folder):
        for folder in d:
            if re.search(compiled_pattern, folder) is not None:
                folders.append(os.path.join(r, folder))

    return folders

def list_folders(folder) -> list[str]:
    """List all folders from a given folder

    Args:
        folder (str): Path to the desired folder
        pattern (str, optional): A pattern that is looked for in the filenames. Defaults to "".

    Returns:
        list: A list of strings, paths to every folder from the folder
    """
    folders = []
    for d in os.scandir(folder):
        if os.path.isdir(d):
            folders.append(d.path)

    return folders

def get_unique_folder(base_name):
    folder_name = base_name
    counter = 1
    while os.path.exists(folder_name):
        folder_name = f"{base_name}_{counter}"
        counter += 1
    return folder_name

def get_unique_file(base_name, ext=''):
    file_name = base_name+ext
    counter = 1
    while os.path.exists(file_name):
        file_name = f"{base_name}_{counter}{ext}"
        counter += 1
    return file_name

def pad_numbers_in_filenames(folder_path, width=2):

    for filename in list_files(folder_path):
        old_path = os.path.join(folder_path, filename)

        # Find numbers in filename
        new_name = re.sub(r'(\d+)', lambda m: m.group(1).zfill(width), filename)
        
        if new_name != filename:
            new_path = os.path.join(folder_path, new_name)
            print(f"Renaming: {filename} -> {new_name}")
            os.rename(old_path, new_path)
