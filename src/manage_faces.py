import os, re
import argparse
import tqdm
from pytorch3d.io import load_ply, save_ply, load_obj

def list_files(folder, pattern=""):
    """List all files from a given folder

    Args:
        folder (str): Path to the desired folder
        pattern (str, optional): A pattern that is looked for in the filenames. Defaults to "".

    Returns:
        list: A list of strings, paths to every file from the folder
    """
    files = []
    compiled_pattern = re.compile(pattern)
    
    for r, d, f in os.walk(folder):
        for file in f:
            if re.search(compiled_pattern, file):
                files.append(os.path.join(r, file))
    return files

if __name__ == '__main__':

    description = 'Script to copy faces from template to simulated meshes or remove redundant faces in dataset. Each simulated mesh should share the same face topology than the template.'
    parser = argparse.ArgumentParser(description=description, prog='face_redundancy')

    parser.add_argument('dataset_path', type=str,
                        help="Dataset path, probably '.'")
    parser.add_argument('--template_path', type=str, default="mean_shape.obj",
                        help="Template relative path in dataset")
    parser.add_argument("--remove", action=argparse.BooleanOptionalAction,
                        help="Remove faces from simulated meshes")
    parser.add_argument("--add", action=argparse.BooleanOptionalAction,
                        help="Add faces in simulated meshes")
    args = parser.parse_args()

    if bool(args.add) == bool(args.remove):
        raise Exception("Choose a single option [--remove | --add]")
    
    dataset_path = os.path.abspath(args.dataset_path)

    _, template, _ = load_obj(os.path.join(dataset_path, args.template_path), False)
    template_faces = template.verts_idx

    files = list_files(dataset_path, "out_\\d{6}.ply")

    if args.add:
        for path in tqdm.tqdm(files):
            v, _ = load_ply(path)
            save_ply(path, v, template_faces)
    
    if args.remove:
        for path in tqdm.tqdm(files):
            v, _ = load_ply(path)
            save_ply(path, v)
