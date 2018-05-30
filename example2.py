import tensorflow as tf

a = tf.placeholder(dtype=tf.bool)

b = tf.cond(a, lambda: tf.constant(2.0), lambda: tf.constant(3.0))

feed_dict = {a: False}

# with tf.Session() as sess:
#     c = sess.run(b,feed_dict=feed_dict)
#     print(c)


d = tf.random_uniform([2, 3, 4],dtype=tf.float32)

if len(d.shape) == 2:
    print(d.shape)

