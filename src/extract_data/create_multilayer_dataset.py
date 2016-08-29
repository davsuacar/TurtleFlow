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

if len(sys.argv) != 3:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./create_multilayer_dataset.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()

with open('../../data/multilayer/data.csv', 'wb') as csvdata:
    dataset = csv.writer(csvdata, delimiter=':',
                         quotechar='\t', quoting=csv.QUOTE_MINIMAL)

    i = 0
    for subdir, dirs, files in os.walk(faces_folder_path):
        print "Starting from..." + subdir
        for f in glob.glob(os.path.join(subdir, "*.jpg")):

            print os.path.join(subdir, f)

            img = io.imread(f)

            dets = detector(img, 1)

            for k, d in enumerate(dets):
                # Get the landmarks/parts for the face in box d.
                shape = predictor(img, d)

                points = [y for k in range(68) for y in [(shape.part(k).x, shape.part(k).y)]]

                input_data = distances.calculate_proportions(points)
                input_data.append(i)

                dataset.writerow(input_data)

        i += 1
