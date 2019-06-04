import tensorflow as tf

# placeholder for input to the computation
x = tf.placeholder(dtype=tf.float32, name="x")

# bias variable for the affine weight transformation
b = tf.Variable(tf.zeros(100))

# weight variable for the affine wegiht transformation with random values
W = tf.Variable(tf.random_uniform([784, 100]), tf.float32)

# activation as a function of the weight transformation
a = tf.relu(tf.matmul(W, x) + b)

# cost computed as a function of the activation
# and the target optimization task
C = [...]

# Start session to run the computational graph
session = tf.InteractiveSession()

# Initialize all variables, in this example only the weight
# matrix depends on an initialization
tf.global_variables_initializer()

for i in range(epochs):
    result = session.run(C, feed_dict={x: data[batch_indices]})
    print(i, result)
