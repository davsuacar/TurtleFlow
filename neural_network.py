
'''
This is a tensorflow neural network structure but it still should be adapted to our project.
'''

import tensorflow as tf
from numpy import genfromtxt

faces_data_train=genfromtxt('data/total/train_data.csv', delimiter=',')
faces_label_train=genfromtxt('data/total/train_label.csv', delimiter=':')
faces_data_test=genfromtxt('data/total/test_data.csv', delimiter=',')
faces_label_test=genfromtxt('data/total/test_label.csv', delimiter=':')
faces_data_validation=genfromtxt('data/validation/validation_data_david.csv', delimiter=',').reshape(1,69)

print faces_data_train

# Parameters
learning_rate = 0.001
training_epochs = 50
display_step = 1

# Network Parameters
n_hidden_1 = 32 # 1st layer num features
n_input = 69 # Webcam landmarks data points input distances
n_classes = 3 # Three people classes(David, Pepe, Marcos)

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

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer


# Initializing the variables
init = tf.initialize_all_variables()

xs, ys = faces_data_train, faces_label_train
xs_test, ys_test = faces_data_test, faces_label_test

# Launch the graph
with tf.Session() as sess:

    # Merge all the summaries and write them out to /tmp/mnist_logs
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("/tmp/test", sess.graph_def)

    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.

        # Fit training using data
        sess.run(optimizer, feed_dict={x: xs, y: ys})

        # Compute average loss
        avg_cost += sess.run(cost, feed_dict={x: xs, y: ys})

        #print(avg_cost)
        # Display logs per epoch step
        #if epoch % display_step == 0:
            #print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    #print "Optimization Finished!"

    #print (tf.argmax(y, 1))
    # Test model
    print(tf.arg_max(y, 1))
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy:", accuracy.eval({x: faces_data_test, y: faces_label_test})

    feed_dict = {x: faces_data_validation}
    #classification = sess.run(tf.argmax(pred, 1), feed_dict)
    classification = sess.run(pred, feed_dict)
    print classification