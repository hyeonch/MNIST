import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
np.random.seed(0)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

y_train, y_test = tf.raw_ops.OneHot(indices=y_train, depth=10, on_value=1, off_value=0, axis=1), \
                  tf.raw_ops.OneHot(indices=y_test, depth=10, on_value=1, off_value=0, axis=1)
import numpy as np

# MLP
# dataset reshape
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# params modification
epochs = 10
dense_nodes = 512
hidden_layers = 1
batch_size = 512
learning_rate = 0.1
iter_in_an_epoch = int(y_train.shape[0] // batch_size)

keep_prob = tf.placeholder(tf.float32)
X = tf.placeholder(tf.float32, [None, 28 * 28])
Y = tf.placeholder(tf.float32, [None, 10])
Weights = [tf.Variable(tf.random_normal([28 * 28, dense_nodes])),
           tf.Variable(tf.random_normal([dense_nodes, 10]))]

Biases = [tf.Variable(tf.random_normal([dense_nodes])),
          tf.Variable(tf.random_normal([10]))]

Hypotheses = [tf.nn.dropout(tf.nn.relu(tf.matmul(X, Weights[0]) + Biases[0]), keep_prob)]

for i in range(hidden_layers):
    Weights.insert(-1, tf.Variable(tf.random_normal([dense_nodes, dense_nodes])))
    Biases.insert(-1, tf.Variable(tf.random_normal([dense_nodes])))
    Hypotheses.append(tf.nn.dropout(tf.nn.relu(tf.matmul(Hypotheses[i], Weights[i+1])), keep_prob))
Hypotheses.append(tf.matmul(Hypotheses[-1], Weights[-1]) + Biases[-1])


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Hypotheses[-1], labels=Y))
train = tf.train.AdagradOptimizer(learning_rate).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Hypotheses[-1], 1), tf.argmax(Y, 1)), tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(iter_in_an_epoch):
            batch_x = x_train[batch_size * i:batch_size * (i + 1)]
            batch_y = y_train[batch_size * i:batch_size * (i + 1)].eval(session=sess)
            _, c, val_acc = sess.run((train, cost, accuracy), feed_dict={X: batch_x, Y: batch_y, keep_prob: 1})
            avg_cost += c / iter_in_an_epoch
        print(f"Epoch: {epoch + 1}, " + f"Cost: {avg_cost:.3f}, " + f"accuracy: {val_acc:.4f}")

    print("Cost:", sess.run(cost, feed_dict={X:x_test, Y: y_test.eval(session=sess), keep_prob: 1}))
    print("Accuracy:", sess.run(accuracy, feed_dict={X: x_test, Y: y_test.eval(session=sess), keep_prob: 1}))

# CNN
# dataset reshape
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# params modification
kernels = 256
epochs = 10
batch_size = 512
learning_rate = 0.1
num_layers = 2
iter_in_an_epoch = int(y_train.shape[0] // batch_size)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

Kernels = [tf.Variable(tf.random_normal([3, 3, 1, kernels]))] + \
          [tf.Variable(tf.random_normal([3, 3, kernels, kernels]))] * (num_layers-1)
# Biases = [tf.Variable(tf.random_normal([kernels]))] * num_layers

W = tf.Variable(tf.random_normal([28 * 28 * kernels // (4 ** num_layers) if num_layers<3 else 1152, 7 *7 *kernels//4]))
B = tf.Variable(tf.random_normal([7*7*kernels//4]))

W2 = tf.Variable(tf.random_normal([7*7*kernels//4, 10]))
B2 = tf.Variable(tf.random_normal([10]))


ConvLayers = [tf.nn.relu(tf.nn.conv2d(X, filter=Kernels[0], strides=1, padding='SAME'))]
PoolLayers = [tf.nn.dropout(tf.nn.max_pool(ConvLayers[0], 2, 2, padding='VALID'),keep_prob=keep_prob)]

for i in range(num_layers - 1):
    ConvLayers.append(tf.nn.relu(tf.nn.conv2d(PoolLayers[i], Kernels[i + 1], 1, padding='SAME')))
    PoolLayers.append(tf.nn.dropout(tf.nn.max_pool(ConvLayers[i + 1], 2, 1 + (i < 1), padding='VALID'),keep_prob=keep_prob))

Flatten = tf.layers.flatten(PoolLayers[-1])
Dense = tf.nn.relu(tf.matmul(Flatten, W) + B)
Dense2 = tf.matmul(Dense, W2) + B2

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Dense2, labels=Y))
train = tf.train.AdagradOptimizer(learning_rate).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Dense2, 1), tf.argmax(Y, 1)), tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(iter_in_an_epoch):
            batch_x = x_train[batch_size * i:batch_size * (i + 1)]
            batch_y = y_train[batch_size * i:batch_size * (i + 1)].eval(session=sess)
            _, c, val_acc = sess.run((train, cost, accuracy), feed_dict={X: batch_x, Y: batch_y, keep_prob:0.9})
            avg_cost += c / iter_in_an_epoch
        print(f"Epoch: {epoch + 1}, " + f"Cost: {avg_cost:.3f}, " + f"accuracy: {val_acc:.4f}")

    print("Cost:", sess.run(cost, feed_dict={X: x_test, Y: y_test.eval(session=sess), keep_prob: 1}))
    print("Accuracy:", sess.run(accuracy, feed_dict={X: x_test, Y: y_test.eval(session=sess), keep_prob: 1}))




# rnn
# params modification

seq_block = 4
epochs = 10
batch_size = 512
learning_rate = 0.1
num_seq = 28 * seq_block
num_layers = 3
iter_in_an_epoch = int(y_train.shape[0] // batch_size)

x_train = x_train.reshape(x_train.shape[0], num_seq, -1)
x_test = x_test.reshape(x_test.shape[0],num_seq, -1)

X = tf.placeholder(tf.float32, [None, num_seq, x_train.shape[-1]])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# Wxh = tf.Variable(tf.random_normal([28 * sequence_block * 2, 28 * sequence_block]))  # [h_t-1,x_t] 곱해지는 Weight
# bh = tf.Variable(tf.random_normal([28 * sequence_block * 2]))
#
Wyh = tf.Variable(tf.random_normal([x_train.shape[-1], 10]))
by = tf.Variable(tf.random_normal([10]))
#
# h_prev = tf.zeros([28 * sequence_block])
# for x_t in tf.unstack(tf.transpose(X, perm=[1,0,2])):
#     h_t = tf.math.tanh(tf.matmul(tf.concat(0, [h_prev, x_t]), Wxh))
#     h_prev = h_t
#
# y_t = tf.nn.relu(tf.matmul(h_t, Wyh) + by)

layers = [tf.nn.rnn_cell.BasicLSTMCell(x_train.shape[-1],forget_bias=1)] * num_layers
drops = [tf.nn.rnn_cell.DropoutWrapper(layer, output_keep_prob = keep_prob) for layer in layers]
multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(drops)
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]

rnn_logits = tf.matmul(outputs, Wyh) + by
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=rnn_logits, labels=Y))
train = tf.train.AdagradOptimizer(learning_rate).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(rnn_logits, 1), tf.argmax(Y, 1)), tf.float32))



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(iter_in_an_epoch):
            batch_x = x_train[batch_size * i:batch_size * (i + 1)]
            batch_y = y_train[batch_size * i:batch_size * (i + 1)].eval(session=sess)
            _, c, val_acc = sess.run((train, cost, accuracy), feed_dict={X: batch_x, Y: batch_y, keep_prob: 1})
            avg_cost += c / iter_in_an_epoch
        print(f"Epoch: {epoch + 1}, " + f"Cost: {avg_cost:.3f}, " + f"accuracy: {val_acc:.4f}")

    print("Cost:", sess.run(cost, feed_dict={X: x_test, Y: y_test.eval(session=sess), keep_prob: 1}))
    print("Accuracy:", sess.run(accuracy, feed_dict={X: x_test, Y: y_test.eval(session=sess), keep_prob: 1}))