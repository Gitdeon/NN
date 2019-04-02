import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

training_set = np.loadtxt('data/fashion-mnist_train.csv', delimiter=',', skiprows=1)[:,1:]
test_set = np.loadtxt('data/fashion-mnist_test.csv', delimiter=',', skiprows=1)[:,1:]

#encoding
n_nodes_input = 784 
n_nodes_hidden1  = 64

#decoding
n_nodes_hidden2  = 64
n_nodes_output = 784 

hidden1_values = {
      'weights':tf.Variable(tf.random_normal([n_nodes_input,n_nodes_hidden1])),
      'biases':tf.Variable(tf.random_normal([n_nodes_hidden1]))  }
hidden2_values = {
      'weights':tf.Variable(tf.random_normal([n_nodes_hidden1, n_nodes_hidden2])),
   'biases':tf.Variable(tf.random_normal([n_nodes_hidden2]))  }
output_values = {
      'weights':tf.Variable(tf.random_normal([n_nodes_hidden2,n_nodes_output])),               
      'biases':tf.Variable(tf.random_normal([n_nodes_output])) }

# image with shape 784 goes in
input_layer = tf.placeholder('float', [None, 784])

# multiply output of input_layer wth a weight matrix and add biases
layer_1 = tf.nn.sigmoid(
       tf.add(tf.matmul(input_layer,hidden1_values['weights']),
       hidden1_values['biases']))

# multiply output of layer_1 wth a weight matrix and add biases
layer_2 = tf.nn.sigmoid(
       tf.add(tf.matmul(layer_1,hidden2_values['weights']),
       hidden2_values['biases']))

# multiply output of layer_2 wth a weight matrix and add biases
output_layer = tf.matmul(layer_2,output_values['weights']) + output_values['biases']

# output_true shall have the original image for error calculations
output_true = tf.placeholder('float', [None, 784])

# define our cost function
meansq = tf.reduce_mean(tf.square(output_layer - output_true))

# define our optimizer
learn_rate = 0.1   # how fast the model should learn
optimizer = tf.train.AdagradOptimizer(learn_rate).minimize(meansq)

# initializing and starting the session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# defining batch size, number of epochs and learning rate
batch_size = 100  # how many images to use together for training
hm_epochs =100    # how many times to go through the entire dataset
tot_images = 60000 # total number of images

# running the model for a 1000 epochs taking 100 images in batches
# total improvement is printed out after each epoch
for epoch in range(hm_epochs):
    epoch_loss = 0    # initializing error as 0
    for i in range(int(tot_images/batch_size)):
        epoch_x = training_set[ i*batch_size : (i+1)*batch_size ]
        _, c = sess.run([optimizer, meansq],\
               feed_dict={input_layer: epoch_x, \
               output_true: epoch_x})
        epoch_loss += c
    print('Epoch', epoch, '/', hm_epochs, 'loss:',epoch_loss)


encoded_any_image = sess.run(layer_1, feed_dict={input_layer:[any_image]})

any_image = training_set[0]
plt.imshow(any_image.reshape(28,28),  cmap='Greys')
plt.show()

output_any_image = sess.run(output_layer, feed_dict={input_layer:[any_image]})
plt.imshow(output_any_image.reshape(28,28),  cmap='Greys')
plt.show()

print(encoded_any_image)
