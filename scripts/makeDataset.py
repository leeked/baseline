import glob
import shutil
import os
import json
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Take JSON of ground truth and convert image file directory to ImageFolder format.')
    parser.add_argument('directory', type=str, help='filepath of input directory')
    parser.add_argument('json', type=str, help='Name of GT JSON file')

    parser.add_argument('--dest', type=str, help='optional destination directory')


    args = parser.parse_args()

    in_dir = args.directory
    gt_file = args.json

    # Create destination dir
    if args.dest:
        dest_dir = args.dest
    else:
        dest_dir = 'Datasets/TorchDataset/'
    os.makedirs(os.path.dirname(dest_dir), exist_ok=True)

    # Grab images
    imgs = {}

    for image in glob.iglob(os.path.join(in_dir, '*.jpg')):
        image_name = image[image.rfind('\\') + 1: -4]
        imgs[image_name] = image

    # Read json
    with open(gt_file, 'r') as f:
        gt = json.load(f)

        for key, val in gt.items():
            # Create subfolder and add first image
            subfolder = dest_dir + key + '/'
            os.makedirs(os.path.dirname(subfolder), exist_ok=True)
            shutil.copy(imgs[key], subfolder)

            # Add neighbors (easy, long, maybe)
            for neighbor in val['easy']:
                shutil.copy(imgs['{}'.format(neighbor)], subfolder)

            for neighbor in val['long']:
                shutil.copy(imgs['{}'.format(neighbor)], subfolder)

            for neighbor in val['maybe']:
                shutil.copy(imgs['{}'.format(neighbor)], subfolder)
