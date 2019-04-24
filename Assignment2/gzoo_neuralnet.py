import glob
import os
import time
import numpy as np
import matplotlib as plt
plt.use('agg')
import matplotlib.pyplot as pplt
import cv2 #pip3 install opencv-python
import tensorflow as tf
import keras.backend as K
from keras import datasets, layers, models
from sklearn.model_selection import train_test_split

#K.set_image_dim_ordering('tf')
#os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID = 0
#doesn't need the above line, taken care of in bashrc I think

jpg_paths = glob.glob('data/images_training_rev1/*.jpg')
jpg_paths = np.sort(jpg_paths)

#Loading all images
#list comprehension for commented block above
galaxy_images = [cv2.resize(cv2.imread(jpg_paths[i], 0), dsize = (60, 60), interpolation = cv2.INTER_CUBIC) for i in range(len(jpg_paths))]
        
#Get predicitions
solutions = np.loadtxt('data/training_solutions_rev1.csv', delimiter = ',', skiprows=1)
classification_solutions = []
for i in range(len(solutions)):
    classification_solutions.append(np.argmax(solutions[i][1:]))

images_train, images_test, solutions_train, solutions_test = train_test_split(galaxy_images, classification_solutions, test_size=0.3, random_state=42)


#Normalize pixel values to be between 0 and 1
#not sure the maximum value would be 255 still

maxs_train = [max(it.flatten()) for it in images_train]
maxs_test = [max(it.flatten()) for it in images_test]

max_train = max(maxs_train)
max_test = max(maxs_test)

images_train = [np.array(images_train[i]/max_train) for i in range(len(images_train))]
images_test = [np.array(images_test[i]/max_test) for i in range(len(images_test))]

#Reshaping input and labels to make them compatible with the CNN
images_train = np.array(images_train).reshape(len(images_train), 60, 60, 1)
images_test = np.array(images_test).reshape(len(images_test), 60, 60, 1)
solutions_train = np.array(solutions_train)
solutions_test = np.array(solutions_test)
solutions_train[solutions_train == 13] = 3
solutions_train[solutions_train == 14] = 4
solutions_test[solutions_test == 13] = 3
solutions_test[solutions_test == 14] = 4

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
    model.add(layers.Conv2D(filters=96, kernel_size=11, strides=4, padding='same', activation=activation_fn, input_shape=(60,60,1)))
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
    model.add(layers.Dense(model_shape, activation='relu'))
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
    test_loss, test_acc = model.evaluate(images_test, solutions_test)
    B = time.time()
    runtime = B-A
    return test_loss, test_acc, runtime

'''
    creating data necessary for plots
    each element of the data list is activation_fn, n_layers, loss, acc, runtime
    '''

activation_fns = ['relu', 'tanh', 'exponential', 'sigmoid']
n_layer_possibilities = [5, 7, 9]

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
    ('relu', n_layers = 5, 7, 9), (activation_fn = 'relu',... , n_layers = 5)
    '''

pplt.rc('text', usetex = True)
pplt.rc('font', family = 'serif')
fig = pplt.figure()

for_second_plot, Xs, Ys = [], [], []

for ifp in information_for_plots:
    label = str(ifp[0]) + ', ' + str(ifp[1])
    f = ifp[3]
    g = ifp[4]
    
    Xs.append(g)
    Ys.append(f)
    
    for_second_plot.append([label, f, g])

fig = pplt.figure()
for element in for_second_plot:
    label, f, g = element[:]
    pplt.scatter(g, f, s = 1, color='k')
    pplt.annotate(label, xy = (g, f), xycoords = 'data', color='r', fontsize = 8)

pplt.annotate('good', xy = (0.05, 0.85), xycoords = 'axes fraction', fontsize = 10)
pplt.annotate('bad', xy = (0.85, 0.05), xycoords = 'axes fraction', fontsize = 10)
pplt.xlabel('Runtime [s]', fontsize = 12)
pplt.ylabel('Accuracy [$\%$/100]', fontsize = 12)
pplt.title('Evaluating CNN Performance', fontsize = 12)
pplt.tight_layout()
pplt.savefig('homework2_secondplot.pdf')
pplt.close()
