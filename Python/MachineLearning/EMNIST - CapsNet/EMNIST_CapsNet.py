##This project is an adaptation of the Handson-ML Capsule Network (https://github.com/ageron/handson-ml/blob/master/extra_capsnets.ipynb) implementation.
##The original implementation was adapted to work on the extended MNIST dataset.
##
##The capsule network architecture is based off of CapsNet described in "Dynamic routing between capsules"(Sabour et al. 2017). 
##
##That "Dynamic Routing Between Capsules" (Sabour et al. 2017) can be found here:
##https://arxiv.org/pdf/1710.09829.pdf
##
##The process of adapting the network involved changing network layer dimensions, optimization function parameters,
##and tensorflow training information to account for the fact that the EMNIST dataset used was for 26 alphabetical characters.
##
##Dependencies :
##
##Python 3.X 64-BIT
##TensorFlow
##Numpy
##Scipy

##Based on: CapsNet MNIST - Open source Handson-ML Implementation
##CapsNet EMNIST Adaptation done by Jesse Broussard



#Imports
from __future__ import division, print_function, unicode_literals
import matplotlib
import matplotlib.pyplot
import numpy as np
import tensorflow as tf
import scipy

#EMNIST Data handler written by Jesse Broussard
import EMNIST_Extract_fast

tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/")
emnist = EMNIST_Extract_fast.emnist_ext()

#Squash and safe_norm function
def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector
    
def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)
    

#input images.
X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")
#First layer of primary capsule parameters.  
caps1_n_maps = 32
caps1_n_caps = caps1_n_maps * 6 * 6 
caps1_n_dims = 8
#convolutional layers
conv1_params = {
    "filters": 256,
    "kernel_size": 9,
    "strides": 1,
    "padding": "valid",
    "activation": tf.nn.relu,
}
conv2_params = {
    "filters": caps1_n_maps * caps1_n_dims, # 256 convolutional filters
    "kernel_size": 9,
    "strides": 2,
    "padding": "valid",
    "activation": tf.nn.relu
}
#F256-K9-S1-VP-RELU
#F(CF*CD)-K9-S1-VP-RELU
conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)
#Reshaping of the second convolutional layer's output of 256 filters, into 32 vectors and 8 dimensions.
caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims],
                       name="caps1_raw")
#How do we keep these newly formed 8 dimensional vectors within 0 to 1 range? 
#We sQUASH them using the sQUASH function we defined at the start.
caps1_output = squash(caps1_raw, name="caps1_output")
#The second set of "Capsules" are the digit capsules parameters.
caps2_n_caps = 26
caps2_n_dims = 16
#implementation of weights of the network.
init_sigma = 0.1
W_init = tf.random_normal(
    shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
    stddev=init_sigma, dtype=tf.float32, name="W_init")

W = tf.Variable(W_init,name="W")

#next the weight is tiled.

batch_size = tf.shape(X)[0]
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")

#then the caps1 output is adjusted through expansion, and being tiled.

caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                       name="caps1_output_expanded")
caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                   name="caps1_output_tile")
caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1],
                             name="caps1_output_tiled")

#finally the predicted output vectors Uj|i is computed by multiplying W_tiled and caps1_output_tiled.
#think back to WX+b, int his scenario X is the tiled caps1_output_tiled, and W is the W_tiled.

caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
                            name="caps2_predicted")

###Routing Algorithm

##Round one pass
#initialization of "routing weights" to zero.
raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],
                       dtype=np.float32, name="raw_weights")

#Routing weights through softmax.

routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")

#Calculation of weighted sums from the output of the digit capsule layer.

weighted_predictions = tf.multiply(routing_weights, caps2_predicted,
                                   name="weighted_predictions")
weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                             name="weighted_sum")

#Again we squash the outputs to make them fit within our 0 to 1 vector size.

caps2_output_round_1 = squash(weighted_sum, axis=-2,
                              name="caps2_output_round_1")

##Round two pass

caps2_output_round_1_tiled = tf.tile(
    caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1],
    name="caps2_output_round_1_tiled")

agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                      transpose_a=True, name="agreement")

raw_weights_round_2 = tf.add(raw_weights, agreement,
                             name="raw_weights_round_2")

routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,
                                        dim=2,
                                        name="routing_weights_round_2")
weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,
                                           caps2_predicted,
                                           name="weighted_predictions_round_2")
weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,
                                     axis=1, keep_dims=True,
                                     name="weighted_sum_round_2")
caps2_output_round_2 = squash(weighted_sum_round_2,
                              axis=-2,
                              name="caps2_output_round_2")

caps2_output = caps2_output_round_2

#Estimated class probabilities

y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")
y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")
y_pred = tf.squeeze(y_proba_argmax, axis=[1,2], name="y_pred")

#labels placeholder

y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

#Margin Loss

m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5

#One_hot usage to determine the digit class 0-9.

T = tf.one_hot(y, depth=caps2_n_caps, name="T")

#norm of Caps2output

caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True,
                              name="caps2_output_norm")

#Error calculations 

present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),
                              name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, 26),
                           name="present_error")


absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),
                             name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1, 26),
                          name="absent_error")

#Loss function

L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
           name="L")

margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

##End of main Network



###Decoder network preprocessing

#Placeholder for mask labels

mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                               name="mask_with_labels")

#Reconstrction "Targets" basically the goal based by label

reconstruction_targets = tf.cond(mask_with_labels, # condition
                                 lambda: y,        # if True
                                 lambda: y_pred,   # if False
                                 name="reconstruction_targets")

#Creation of the "mask" for reconstruction.

reconstruction_mask = tf.one_hot(reconstruction_targets,
                                 depth=caps2_n_caps,
                                 name="reconstruction_mask")

#Mask is then reshaped to multiply caps2_output and the mask.

reconstruction_mask_reshaped = tf.reshape(
    reconstruction_mask, [-1, 1, caps2_n_caps, 1, 1],
    name="reconstruction_mask_reshaped")

#The multiplication


caps2_output_masked = tf.multiply(
    caps2_output, reconstruction_mask_reshaped,
    name="caps2_output_masked")


#Lastly we must flatten the decoder's inputs, which is done in the next line


decoder_input = tf.reshape(caps2_output_masked,
                           [-1, caps2_n_caps * caps2_n_dims],
                           name="decoder_input")

###Decoder network


n_hidden1 = 512
n_hidden2 = 1024
n_hidden3 = 1024
n_output = 28 * 28

with tf.name_scope("decoder"):
    hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                              activation=tf.nn.relu,
                              name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2,
                              activation=tf.nn.relu,
                              name="hidden2")
    decoder_output = tf.layers.dense(hidden2, n_output,
                                     activation=tf.nn.sigmoid,
                                     name="decoder_output")
    
#reconstruction loss

X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
squared_difference = tf.square(X_flat - decoder_output,
                               name="squared_difference")
reconstruction_loss = tf.reduce_mean(squared_difference,
                                    name="reconstruction_loss")
#accuracy

correct = tf.equal(y, y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

#final loss

alpha = 0.0005

loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")

#optimizer and training operations

optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss, name="training_op")

#initializations and Saver load in

init = tf.global_variables_initializer()
saver = tf.train.Saver()

print('Done...')

########### Model Fitting ################

n_epochs = 10
batch_size = 50
restore_checkpoint = True

n_iterations_per_epoch = 88800 // batch_size
n_iterations_validation = 14800 // batch_size
best_loss_val = np.infty
checkpoint_path = "./my_capsule_network"

config = tf.ConfigProto(allow_soft_placement = True, log_device_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()

    for epoch in range(n_epochs):
        for iteration in range(1, n_iterations_per_epoch + 1):
            X_batch, y_batch = emnist.train_batch(batch_size)
            # Run the training operation and measure the loss:
            _, loss_train = sess.run(
                [training_op, loss],
                feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),
                           y: y_batch,
                           mask_with_labels: True})
            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                      iteration, n_iterations_per_epoch,
                      iteration * 100 / n_iterations_per_epoch,
                      loss_train),
                  end="")

        # At the end of each epoch,
        # measure the validation loss and accuracy:
        loss_vals = []
        acc_vals = []
        for iteration in range(1, n_iterations_validation + 1):
            X_batch, y_batch = emnist.test_batch(batch_size)
            loss_val, acc_val = sess.run(
                    [loss, accuracy],
                    feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),
                               y: y_batch})
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                      iteration, n_iterations_validation,
                      iteration * 100 / n_iterations_validation),
                  end=" " * 10)
        loss_val = np.mean(loss_vals)
        acc_val = np.mean(acc_vals)
        print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
            epoch + 1, acc_val * 100, loss_val,
            " (improved)" if loss_val < best_loss_val else ""))

        # And save the model if it improved:
        if loss_val < best_loss_val:
            save_path = saver.save(sess, checkpoint_path)
            best_loss_val = loss_val
