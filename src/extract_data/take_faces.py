import cv2
import sys
import dlib
import numpy

video_capture = cv2.VideoCapture(0)

predictor_path = sys.argv[1]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()
i = 0
while True:
    i += 1
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    backtorgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    win.clear_overlay()
    win.set_image(backtorgb)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(backtorgb)

    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))

        win.add_overlay(dets)
        cropped = backtorgb[d.top(): d.top() + d.bottom(), d.left():d.right() + d.left()]
        cv2.imwrite("../img/image_" + str(i) + ".jpg", cropped)
    # Display the resulting frame
    # cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
