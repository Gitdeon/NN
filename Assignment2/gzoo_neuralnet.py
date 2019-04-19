import time
import glob
import numpy as np
#import pandas as pd
#from PIL import Image

'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID = 0
import matplotlib.pyplot as plt
'''

'''
train_jpgs_gzoo = glob.glob('/data/s1630784/NN/Challenge_2/data/images_training_rev1/*.jpg')
test_jpgs_gzoo = glob.glob('/data/s1630784/NN/Challenge_2/data/images_test_rev1/*.jpg')

print('len(train_jpgs_gzoo) is', len(train_jpgs_gzoo))
print('len(test_jpgs_gzoo) is', len(test_jpgs_gzoo))

def jpg_to_array(img):
return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)

convert_training_jpgs_to_arrays = [jpg_to_array(Image.open(jpg)) for jpg in train_jpgs_gzoo]
convert_testing_jpgs_to_arrays = [jpg_to_array(Image.open(jpg)) for jpg in test_jpgs_gzoo]
'''

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

#Normalize pixel values to be between 0 and 1
train_images, test_images = train_images/255.0, test_images/255.0

'''
input types must be string and integer, respectively
activation functions available are listed on Keras documentation
example activation functions:  'relu', 'tanh', 'sigmoid', 'exponential', 'linear'
'''

def CNN(activation_fn, n_layers):

        '''
        returns a CNN with the input number of convolution/pooling layers
        '''

        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation=activation_fn, input_shape=(28,28,1)))
        model.add(layers.MaxPooling2D((2,2)))

        n = 2
        while n < n_layers - 4:

                model.add(layers.Conv2D(64, (3, 3), activation=activation_fn))
                n += 1

        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation=activation_fn))
        model.add(layers.Dense(10, activation='softplus'))

        model.summary()

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_images, train_labels, epochs=5)

        return model

def CNN_performance(activation_fn, n_layers):

        '''
        returns loss, accuracy, and runtime when evaluated on the testing data
        '''

        A = time.time()

        model = CNN(activation_fn, n_layers)

        test_loss, test_acc = model.evaluate(test_images, test_labels)

        B = time.time()
        runtime = B-A

        return test_loss, test_acc, runtime

'''
creating data necessary for plots
each element of the data list is activation_fn, n_layers, loss, acc, runtime
'''

activation_fns = ['relu', 'tanh', 'sigmoid', 'exponential', 'linear']
n_layer_possibilities = [4, 6, 10, 16]

information_for_plots = []

for activation_fn in activation_fns:
        for n_layers in n_layer_possibilities:
                loss, acc, runtime = CNN_performance(activation_fn, n_layers)
                information_for_plots.append([activation_fn, n_layers, loss, acc, runtime])

print('info_for_plots list is', information_for_plots)

activation_fns_plotting = [ifp[0] for ifp in information_for_plots]
n_layers_plotting = [ifp[1] for ifp in information_for_plots]
losses_plotting = [ifp[2] for ifp in information_for_plots]
accuracies_fns_plotting = [ifp[3] for ifp in information_for_plots]
runtimes_fns_plotting = [ifp[4] for ifp in information_for_plots]

'''
use activation_fn = 'relu' and n_layers = 5 as the controls for each of the panels 
('relu', n_layers = 5, 7, 11), (activation_fn = 'relu',... , n_layers = 5)
'''

plt.rc('test', usetex = True)
plt.rc('font', family = 'serif')

divider, eps = 3.3, 0.09 #for plotting using fig.add_axes

fig = plt.figure()
xvals, yvals = [eps, eps+(1-eps)/divider], [eps, eps+(1-eps)/divider]
plots = [fig.add_axes([x0, y0, (1-eps)/divider, (1-eps)/divider]) for x0 in xvals for y0 in yvals]

for i in range(len(plots)):
        ax = plots[i]
        ax.tick_params(left = True, bottom = True, right = False, top = False, labelsize = 'small')
        if i == 0:
                ax.scatter(range(len(accuracies_fns_plotting), accuracies_fns_plotting)
                ax.set_xticklabels(activation_fns_plotting)
                ax.set_ylabel(r'$f$(activation function) [%/100]', fontsize = 10)
                ax.set_xlabel(r'$f \equiv$ accuracy', fontsize = 10)
                ax.xaxis.set_label_position('top')
        if i == 1:
                ax.scatter(range(len(accuracies_fns_plotting), runtimes_fns_plotting)
                ax.set_xticklabels(activation_fns_plotting)
                ax.set_ylabel(r'$g$(activation function) [s]', fontsize = 10)
                ax.set_xlabel(r'$g \equiv$ runtime', fontsize = 10)
                ax.xaxis.set_label_position('top')
        if i == 2:
                ax.plot(n_layers_plotting, accuracies_fns_plotting)
                ax.set_xticks(n_layers_plotting)
                ax.set_ylabel(r'$f(N_{layers})$', fontsize = 10)
                ax.set_xlabel(r'$N_{layers}$', fontsize = 10)
        if i == 3:
                ax.plot(n_layers_plotting, runtimes_fns_plotting)
                ax.set_xticks(n_layers_fns_plotting)
                ax.set_ylabel(r'$g(N_{layers})$', fontsize = 10)
                ax.set_xlabel(r'$N_{layers}$', fontsize = 10)
                                                                                                                                                                                                                            115,1         Bot
