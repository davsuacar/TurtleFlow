import cv2
import dlib
import tensorflow as tf
import numpy


def conv2d(x, W):
    return tf.nn.conv2d(
        x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


resize_x = 80
resize_y = 80

video_capture = cv2.VideoCapture(0)

win = dlib.image_window()

# Input and output variables

INPUTS = resize_x * resize_y * 3
OUTPUTS = 2
BATCH_SIZE = 20
NUM_EPOCHS = 51
LEARNING_RATE = 1e-04

# Interactive session because of notebook environment
sess = tf.InteractiveSession()

# Input and output placeholder
x = tf.placeholder(tf.float32, [None, INPUTS])
pkeep = tf.placeholder(tf.float32)

# First Convolutional Layer
x_image = tf.reshape(x, [-1, resize_x, resize_y, 3])
W_conv_1 = tf.Variable(tf.truncated_normal([2, 2, 3, 8], stddev=0.1))
b_conv_1 = tf.Variable(tf.constant(0.0, shape=[8]))
h_conv_1 = tf.nn.relu(conv2d(x_image, W_conv_1) + b_conv_1)
h_pool_1 = max_pool_2x2(h_conv_1)

# Second Convolutional Layer
W_conv_2 = tf.Variable(tf.truncated_normal([2, 2, 8, 32], stddev=0.1))
b_conv_2 = tf.Variable(tf.constant(0.0, shape=[32]))
h_conv_2 = tf.nn.relu(conv2d(h_pool_1, W_conv_2) + b_conv_2)
h_pool_2 = max_pool_2x2(h_conv_2)

# Third Convolutional Layer
W_conv_3 = tf.Variable(tf.truncated_normal([2, 2, 32, 128], stddev=0.1))
b_conv_3 = tf.Variable(tf.constant(0.0, shape=[128]))
h_conv_3 = tf.nn.relu(conv2d(h_pool_2, W_conv_3) + b_conv_3)
h_pool_3 = max_pool_2x2(h_conv_3)

# Fourth Convolutional Layer
W_conv_4 = tf.Variable(tf.truncated_normal([2, 2, 128, 256], stddev=0.1))
b_conv_4 = tf.Variable(tf.constant(0.0, shape=[256]))
h_conv_4 = tf.nn.relu(conv2d(h_pool_3, W_conv_4) + b_conv_4)
h_pool_4 = max_pool_2x2(h_conv_4)

# Densely connected layer
W_fc1 = tf.Variable(tf.truncated_normal([5 * 5 * 256, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.0, shape=[1024]))
h_poolfc1_flat = tf.reshape(h_pool_4, [-1, 5 * 5 * 256])
h_fc1 = tf.nn.relu(tf.matmul(h_poolfc1_flat, W_fc1) + b_fc1)

# Densely connected layer
W_fc2 = tf.Variable(tf.truncated_normal([1024, 512], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.0, shape=[512]))
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

# Dropout
h_drop = tf.nn.dropout(h_fc2, pkeep)

# Read out Layer
W_fc3 = tf.Variable(tf.truncated_normal([512, OUTPUTS], stddev=0.1))
b_fc3 = tf.Variable(tf.constant(0.0, shape=[OUTPUTS]))
y_logits = tf.matmul(h_drop, W_fc3) + b_fc3
y_softmax = tf.nn.softmax(y_logits)

#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_logits, y))
optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
#train_step = optimizer.minimize(loss)

#correct_prediction = tf.equal(tf.argmax(y_softmax, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# defaults to saving all variables - in this case w and b
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    # Restore Model
    saver.restore(sess, "/tmp/model_conv.ckpt")

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        win.clear_overlay()
        win.set_image(frame)

        img = cv2.resize(frame, (resize_x, resize_y),
                         interpolation=cv2.INTER_AREA)

        img = img.reshape(-1).reshape(1, 19200)

        img = numpy.array(img) / 255

        feed_dict = {x: img, pkeep: 1.0}

        classification = sess.run(y_softmax, feed_dict)
        print classification

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
