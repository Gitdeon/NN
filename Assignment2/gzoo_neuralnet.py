import glob
import os
import numpy as np
import matplotlib as plt
import cv2 #pip3 install opencv-python
import tensorflow as tf
import keras.backend as K
from keras import datasets, layers, models
from sklearn.model_selection import train_test_split

K.set_image_dim_ordering('tf')
#os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID = 0
#doesn't need the above line, taken care of in bashrc I think

jpg_paths = glob.glob('data/images_training_rev1/*.jpg')
jpg_paths = np.sort(jpg_paths)

#Loading all images
'''
galaxy_images = []
for i in range(8000): #test with small sample; replace with range(len(jpg_paths))
    jpg = cv2.imread(jpg_paths[i], 0) #second argument reads in the jpg as grayscale
    galaxy_images.append(cv2.resize(jpg, dsize=(60, 60), interpolation=cv2.INTER_CUBIC))
    if i % 1000 == 0: 
        print('Loaded ', i, 'images.')
'''

#list comprehension for commented block above
galaxy_images = [cv2.resize(cv2.imread(jpg_paths[i], 0), dsize = (60, 60), interpolation = cv2.INTER_CUBIC)) for i in range(8000)]
        
#Get predicitions
solutions = np.loadtxt('data/training_solutions_rev1.csv', delimiter = ',', skiprows=1)
classification_solutions = []
for i in range(8000): #range(len(solutions)
    classification_solutions.append(np.argmax(solutions[i][1:]))

images_train, images_test, solutions_train, solutions_test = train_test_split(galaxy_images, classification_solutions, test_size=0.2, random_state=42)

#Normalize pixel values to be between 0 and 1
#not sure the maximum value would be 255 still

maxs_train = [max(it.flatten()) for it in images_train]
maxs_test = [max(it.flatten()) for it in images_test]

max_train = max(maxs_train)
max_test = max(maxs_test)

'''
images_train = [np.array(images_train[i]/max_train) for i in range(len(images_train))]
images_test = [np.array(images_test[i]/max_train) for i in range(len(images_test))]
'''
    
'''
    CNN OUTLINE:
    
activation_fn = 'relu'
model = models.Sequential()
model.add(layers.Conv2D(filters=96, kernel_size=11, strides=4, padding='same', activation=activation_fn, input_shape=(60,60,3)))
model.add(layers.MaxPooling2D(pool_size=3, strides=2,padding='same'))
model.add(layers.Conv2D(filters=265, kernel_size=5, padding='same', activation=activation_fn))
model.add(layers.MaxPooling2D(pool_size=3, strides=2,padding='same'))

model.add(layers.Conv2D(filters=384, kernel_size=3, padding='same', activation=activation_fn))
model.add(layers.Conv2D(filters=384, kernel_size=3, padding='same', activation=activation_fn))
model.add(layers.Conv2D(filters=256, kernel_size=3, padding='same', activation=activation_fn))
model.add(layers.MaxPooling2D(pool_size=2, strides=2, padding='same'))
model.add(layers.Flatten())

model_shape = model.output_shape[1]
model.add(layers.Dense(model_shape, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(images_train, solutions_train, epochs=5)
'''

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
            model.add(layers.Conv2D(filters=96, kernel_size=11, strides=4, padding='same', activation=activation_fn, input_shape=(60,60,3)))
            model.add(layers.MaxPooling2D(pool_size=3, strides=2,padding='same'))
            model.add(layers.Conv2D(filters=265, kernel_size=5, padding='same', activation=activation_fn))
            model.add(layers.MaxPooling2D(pool_size=3, strides=2,padding='same'))
            
            n = 2
            while n < n_layers - 4:
                model.add(layers.Conv2D(filters=384, kernel_size=3, padding='same', activation=activation_fn))
                n += 1

            model.add(layers.Conv2D(filters=256, kernel_size=3, padding='same', activation=activation_fn))
            model.add(layers.MaxPooling2D(pool_size=3, strides=2))
            model.add(layers.Flatten())
                      
            model_shape = model.output_shape[1]
            model.add(layers.Dense(shape, activation='relu'))
            model.add(layers.Dense(512, activation='relu'))
            model.add(layers.Dense(5, activation='softmax'))
                    
            model.summary()
                    
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(images_train, solutions_train, epochs=5)
                    
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
        ax.scatter(range(len(accuracies_fns_plotting)), accuracies_fns_plotting)
        ax.set_xticklabels(activation_fns_plotting)
        ax.set_ylabel(r'$f$(activation function) [%/100]', fontsize = 10)
        ax.set_xlabel(r'$f \equiv$ accuracy', fontsize = 10)
        ax.xaxis.set_label_position('top')
        if i == 1:
            ax.scatter(range(len(accuracies_fns_plotting)), runtimes_fns_plotting)
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
