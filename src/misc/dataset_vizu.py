import json
import argparse
from utils.visualization import DGarmentDatasetvisualizer

if __name__ == '__main__':
    description = 'Script to fix cloth intersecting SMPL in dataset'
    parser = argparse.ArgumentParser(description=description,
                                     prog='Vizualizer')
    parser.add_argument('config', type=str,
                        help='The config file')

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    DGarmentDatasetvisualizer(config)
