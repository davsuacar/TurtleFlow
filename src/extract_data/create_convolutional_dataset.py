#!/usr/bin/python

import cv2
import sys
import os
import dlib
import glob
import csv
from skimage import io
import numpy
import util.distances as distances

numpy.set_printoptions(threshold=numpy.nan)

resize_x = 80
resize_y = 80

if len(sys.argv) != 2:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./create_convolutional_dataset.py ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

faces_folder_path = sys.argv[1]

win = dlib.image_window()

with open('../../data/convolutional/data.csv', 'wb') as csvdata:
    fieldnames = ['image', 'label']
    dataset = csv.DictWriter(csvdata, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, delimiter=':')

    dataset.writeheader()

    i = 0
    for subdir, dirs, files in os.walk(faces_folder_path):
        print "Starting from..." + subdir
        for f in glob.glob(os.path.join(subdir, "*.jpg")):
            print os.path.join(subdir, f)

            img = io.imread(f)

            img = cv2.resize(img, (resize_x, resize_y),
                             interpolation=cv2.INTER_AREA)

            img = img.reshape(-1)

            dataset.writerow({'image': numpy.array_str(img), 'label': i})

        i += 1
