import glob
import os
import numpy as np
import matplotlib as plt
import cv2 #pip3 install opencv-python
import tensorflow as tf
from keras import datasets, layers, models
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID = 0

jpg_paths = glob.glob('data/images_training_rev1/*.jpg')
jpg_paths = np.sort(jpg_paths)

#Loading all images
galaxy_images = []
for i in range(8000): #test with small sample; replace with range(len(jpg_paths))
    jpg = cv2.imread(jpg_paths[i])
    galaxy_images.append(cv2.resize(jpg, dsize=(128, 128), interpolation=cv2.INTER_CUBIC))
    if i % 1000 == 0: print('Loaded ', i, 'images.')

#Get predicitions
solutions = np.loadtxt('data/training_solutions_rev1.csv', delimiter = ',', skiprows=1)
classification_solutions = []
for i in range(8000): #range(len(solutions)
    classification_solutions.append(np.argmax(solutions[i][1:]))

images_train, images_test, solutions_train, solutions_test = train_test_split(galaxy_images, classification_solutions, test_size=0.2, random_state=42)

#Normalize pixel values to be between 0 and 1
for i in range(len(images_train)):
    images_train[i] = images_train[i]/255.0
for i in range(len(images_test)):
    images_test[i] = images_test[i]/255.0

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
            model.add(layers.Conv2D(45, (6, 6), activation=activation_fn, input_shape=(45,45,3)))
            model.add(layers.Conv2D(40, (5, 5), activation=activation_fn))
            model.add(layers.MaxPooling2D((20,20)))
            model.add(layers.Conv2D(16, (3, 3), activation=activation_fn))
            model.add(layers.MaxPooling2D((8,8)))
            
            n = 2
                while n < n_layers - 4:
                
                model.add(layers.Conv2D(6, (3, 3), activation=activation_fn))
                n += 1
                    
                    model.add(layers.Conv2D(4, (2, 2), activation=activation_fn))
                    model.add(layers.MaxPooling2D((2,2)))
                    model.add(layers.Flatten())
                    model.add(layers.Dense(64, activation='relu'))
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
