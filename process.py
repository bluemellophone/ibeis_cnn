#!/usr/bin/env python
"""

"""
from __future__ import absolute_import, division, print_function
from detecttools.directory import Directory
from os.path import join, abspath, exists, basename
import utool as ut
import cv2
import numpy as np


def process_image_directory(project_name, size, reset=True):
    # Raw folders
    raw_path = abspath(join('data', 'raw'))
    processed_path = abspath(join('data', 'processed'))
    # Project folders
    project_raw_path = join(raw_path, project_name)
    project_processed_path = join(processed_path, project_name)

    # Load raw data
    direct = Directory(project_raw_path, include_extensions='images')

    # Reset / create paths if not exist
    if exists(project_processed_path) and reset:
        ut.remove_dirs(project_processed_path)
    ut.ensuredir(project_processed_path)

    # Process by resizing the images into the desired shape
    for file_path in direct.files():
        file_name = basename(file_path)
        print('Processing %r' % (file_name, ))
        image = cv2.imread(file_path)
        image = cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
        dest_path = join(project_processed_path, file_name)
        cv2.imwrite(dest_path, image)


def numpy_processed_directory(project_name, numpy_x_file_name='X.npy',
                              numpy_y_file_name='y.npy', labels_file_name='labels.csv',
                              reset=True):
    # Raw folders
    processed_path = abspath(join('data', 'processed'))
    labels_path = abspath(join('data', 'labels'))
    numpy_path = abspath(join('data', 'numpy'))
    # Project folders
    project_processed_path = join(processed_path, project_name)
    project_labels_path = join(labels_path, project_name)
    project_numpy_path = join(numpy_path, project_name)
    # Project files
    project_numpy_x_file_name = join(project_numpy_path, numpy_x_file_name)
    project_numpy_y_file_name = join(project_numpy_path, numpy_y_file_name)
    project_numpy_labels_file_name = join(project_labels_path, labels_file_name)

    # Load raw data
    direct = Directory(project_processed_path, include_extensions='images')
    label_dict = {}
    for line in open(project_numpy_labels_file_name):
        line = line.strip().split(',')
        file_name = line[0].strip()
        label = line[1].strip()
        label_dict[file_name] = label

    # Reset / create paths if not exist
    if exists(project_numpy_path) and reset:
        ut.remove_dirs(project_numpy_path)
    ut.ensuredir(project_numpy_path)

    # Get shape for all images
    shape_x = list(cv2.imread(direct.files()[0]).shape)
    if len(shape_x) == 2:
        shape_x = shape_x + [1]
    shape_x = tuple([len(direct.files())] + shape_x[::-1])
    shape_y = shape_x[0:1]

    # Create numpy arrays
    x = np.empty(shape_x, dtype=np.float32)
    y = np.empty(shape_y, dtype=np.int32)

    # Process by loading images into the numpy array for saving
    for index, file_path in enumerate(direct.files()):
        file_name = basename(file_path)
        print('Processing %r' % (file_name, ))
        image = cv2.imread(file_path)
        try:
            label = label_dict[file_name]
            x[index] = np.array(cv2.split(image))
            y[index] = label
        except KeyError:
            print('Cannot find label')
            raw_input()

    # Save numpy array
    np.save(project_numpy_x_file_name, x)
    np.save(project_numpy_y_file_name, y)


if __name__ == '__main__':
    project_name = 'viewpoint'
    # size = (64, 64)
    # process_image_directory(project_name, size)
    numpy_processed_directory(project_name)
