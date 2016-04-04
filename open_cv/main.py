import cv2
import sys

cascPath = sys.argv[1]
cascPath2 = sys.argv[2]
cascPath3 = sys.argv[3]

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
eyesCascade = cv2.CascadeClassifier(cascPath2)
smileCascade = cv2.CascadeClassifier(cascPath3)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        image = frame[y:y+h, x:x+w]

        cropToGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

        # Draw rectangle around eyes and mouth
        for (x, y, w, h) in eyes:

            cv2.rectangle(cropToGray, (x, y), (x+w, y+h), (0, 255, 0), 2)

            print (x, y, w, h)

        for (x, y, w, h) in smiles:

            cv2.rectangle(cropToGray, (x, y), (x+w, y+h), (0, 255, 0), 2)

            print (x, y, w, h)

        cv2.imshow('Face', cropToGray)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()



