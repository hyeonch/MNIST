import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import os

os.environ["CUDA_VISIBLE_DEVICES"]='3'


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test,y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)


print(y_train.shape)
# mlp
model_mlp = tf.keras.Sequential()
model_mlp.add(layers.Flatten())
model_mlp.add(layers.Dense(256, activation='relu'))
model_mlp.add(layers.Dense(256, activation='relu'))
model_mlp.add(layers.Dense(10, activation='softmax'))
model_mlp.compile(
    optimizer = tf.keras.optimizers.Adagrad(learning_rate = 0.1),
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy'] )
model_mlp.fit(x_train, y_train, batch_size = 256, epochs = 10)
model_mlp.evaluate(x_test,y_test)


#cnn
# x_train_cnn, x_test_cnn = x_train.reshape(x_train.shape[0], 28, 28, 1), x_test.reshape(x_test.shape[0], 28, 28, 1)
# model_cnn = tf.keras.Sequential()
# model_cnn.add(layers.Conv2D(32,(3,3),padding='same',activation='relu'))
# model_cnn.add(layers.MaxPool2D(strides=(2,2)))
# model_cnn.add(layers.Conv2D(32,(3,3),padding='same',activation='relu'))
# model_cnn.add(layers.MaxPool2D(strides=(2,2)))
# model_cnn.add(layers.Flatten())
# model_cnn.add(layers.Dense(10))
#
# model_cnn.compile(
#     optimizer = tf.keras.optimizers.Adagrad(learning_rate = 0.1),
#     loss=tf.keras.losses.CategoricalCrossentropy(from_logits = True),
#     metrics = ['accuracy'])
# model_cnn.fit(x_train_cnn, y_train, batch_size = 64, epochs = 10)
# model_cnn.evaluate(x_test_cnn,y_test)
# print(model_cnn.summary())


#rnn
# x_train_cnn, x_test_cnn = x_train.reshape(x_train.shape[0], 28, 28), x_test.reshape(x_test.shape[0], 28, 28)
# model_rnn = tf.keras.Sequential()
# model_rnn.add(layers.LSTM(64))
# model_rnn.add(layers.Dense(10))
#
# model_rnn.compile(
#     optimizer = tf.keras.optimizers.Adagrad(learning_rate = 0.1),
#     loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True),
#     metrics = ['accuracy'])
# model_rnn.fit(x_train_cnn,y_train,batch_size= 64, epochs = 10)
# model_rnn.evaluate(x_test_cnn,y_test)