import cv2
import tensorflow as tf
import mobilenet_classifier as mc
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.1
training_epochs = 15
batch_size = 100
num_classes = 10

X = tf.placeholder(tf.float32, [None, 784])
X_img_reshaped = tf.reshape(X, [-1, 28, 28, 1])  # img 28x28x1 (black/white)
X_img = tf.image.resize_images(X_img_reshaped, [224, 224])
Y = tf.placeholder(tf.float32, [None, num_classes])

hypothesis, end_points = mc.mobilenet(X_img)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,
                                                              labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

# train my model
print('Learning started.')
for epoch in range(training_epochs):
    avg_cost = 0
    num_batch = int(mnist.train.num_examples / batch_size)

    for i in range(num_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
        avg_cost += c / num_batch
        print('One iteration done')

    print('Epoch:', '%04d' % (epoch + 1),
          'cost =', '{:.9f}'.format(avg_cost))
print("Learning Done!")

# # Test model and check accuracy
# correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print('Accuracy:', sess.run(accuracy, feed_dict={
#     X: mnist.test.images, Y:mnist.test.labels}))

# # Get one and predict
# r = random.randint(0, mnist.test.num_examples -1)
# print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
# print("Prediction: ", sess.run(tf.argmax(logits, 1), feed_dict={X:mnist.test.images[r:r + 1]}))