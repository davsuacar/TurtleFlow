import cv2
import sys
import dlib
import tensorflow as tf
import util.distances as distances
import numpy
import time

video_capture = cv2.VideoCapture(0)

predictor_path = sys.argv[1]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()


# Parameters
learning_rate = 0.09
training_epochs = 100
display_step = 1

# Network Parameters
n_hidden_1 = 32 # 1st layer num features
n_input = 69 # Webcam landmarks data points input distances
n_classes = 3 # Three people classes(David, Cristiano, Xavi)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(_X, _weights, _biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) #Hidden layer with RELU activation
    return tf.matmul(layer_1, _weights['out']) + _biases['out']

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'out': tf.Variable(tf.truncated_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)


# defaults to saving all variables - in this case w and b
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:

    #Restore Model
    saver.restore(sess, "/tmp/model.ckpt")

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

        win.clear_overlay()
        win.set_image(backtorgb)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(backtorgb)
        #print("Number of faces detected: {}".format(len(dets)))
        for k, d in enumerate(dets):
            #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = predictor(backtorgb, d)
            #print("Part 0: {}, Part 1: {} ...".format(shape.part(0),shape.part(1)))
            # Draw the face landmarks on the screen.
            win.add_overlay(shape)

            points = [y for k in range(68) for y in [(shape.part(k).x, shape.part(k).y)]]

            input_data = distances.calculate_proportions(points)

            dict = numpy.asarray(input_data)

            feed_dict = {x: dict.reshape(1,69)}

            classification = sess.run(pred, feed_dict)
            print classification

        win.add_overlay(dets)


        # Display the resulting frame
        #cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()