import cv2
import sys
import csv
import os

cascPath = sys.argv[1]
cascPath2 = sys.argv[2]
cascPath3 = sys.argv[3]
imagesPath = sys.argv[4]

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
eyesCascade = cv2.CascadeClassifier(cascPath2)
smileCascade = cv2.CascadeClassifier(cascPath3)

with open('data/test.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=':',
                            quotechar='\t', quoting=csv.QUOTE_MINIMAL)

    for subdir, dirs, files in os.walk(imagesPath):
        for file in files:

            print file
            # Read the image
            image = cv2.imread(imagesPath + "/" + file)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            print "Found {0} faces!".format(len(faces))


            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cropped = image[y:y+h, x:x+w]

                cropToGray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

                # Detect eyes in the image
                eyes = eyesCascade.detectMultiScale(
                    cropToGray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                # Detect smile in the image
                smiles = smileCascade.detectMultiScale(
                    cropToGray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                cv2.rectangle(cropToGray, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Draw rectangle around eyes and mouth
                for (x, y, w, h) in eyes:

                    cv2.rectangle(cropToGray, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    print (x, y, w, h)

                for (x, y, w, h) in smiles:

                    cv2.rectangle(cropToGray, (x, y), (x+w, y+h), (0, 255, 0), 2)

                cv2.imshow('Face', cropToGray)

                spamwriter.writerow(["David", gray.flatten(), faces, eyes, smiles])

cv2.waitKey(0)
