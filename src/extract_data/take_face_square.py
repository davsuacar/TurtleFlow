import cv2
import sys
import dlib

video_capture = cv2.VideoCapture(0)

predictor_path = sys.argv[1]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()
i = 0
while video_capture.isOpened():
    i += 1
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    win.set_image(frame)

    win.add_overlay(dlib.rectangle(long(260), long(160), long(420), long(320)))

    dets = detector(frame)

    for k, d in enumerate(dets):

        # Get the landmarks/parts for the face in box d.
        cv2.imwrite("../img/image_" + str(i) + ".png", frame[d.top():d.top() + d.height(), d.left(): d.left() + d.width()])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
