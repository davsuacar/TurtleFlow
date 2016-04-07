#!/usr/bin/python

import cv2
import sys
import os
import dlib
import glob
import csv
from skimage import io
import numpy

numpy.set_printoptions(threshold=numpy.nan)

if len(sys.argv) != 3:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()

with open('data/validation_data_david.csv', 'wb') as csvdata,\
        open('data/validation_label_david.csv', 'wb') as csvlabel:

    dataset = csv.writer(csvdata, delimiter=':',
                            quotechar='\t', quoting=csv.QUOTE_MINIMAL)
    labelset = csv.writer(csvlabel, delimiter=':',
                            quotechar='\t', quoting=csv.QUOTE_MINIMAL)

    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        print("Processing file: {}".format(f))
        img = io.imread(f)

        win.clear_overlay()
        win.set_image(img)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))

            cropped = img[d.left():d.right(), d.top():d.bottom()]
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            # Get the landmarks/parts for the face in box d.
            shape = predictor(gray, d)
            print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                      shape.part(1)))
            # Draw the face landmarks on the screen.

            win.add_overlay(shape)
            dataset.writerow([y for k in range(68) for y in [shape.part(k).x, shape.part(k).y]])
            labelset.writerow([0, 0, 1])

        win.add_overlay(dets)
