import tensorflow.compat.v1 as tf
from numba import cuda

a=tf.constant([1.0,2.0,3.0],shape=[3],name='a')
b=tf.constant([1.0,2.0,3.0],shape=[3],name='b')
with tf.device('/gpu:1'):
    c=a+b

TF_CONFIG = tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1),
    allow_soft_placement=True
)

sess = tf.Session(config=TF_CONFIG)
sess.run(tf.global_variables_initializer())
i=1
while(i<1000):
    i=i+1
    print(sess.run(c))

sess.close()
cuda.select_device(1)
cuda.close()
with tf.device('/gpu:1'):
    c=a+b

TF_CONFIG = tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5),
    allow_soft_placement=True
)
sess = tf.Session(config=TF_CONFIG)
sess.run(tf.global_variables_initializer())
while(1):
    print(sess.run(c))