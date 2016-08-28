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

    cv2.imwrite("../img/image_" + str(i) + ".jpg", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
