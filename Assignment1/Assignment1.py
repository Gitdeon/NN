##############################################
#                                            #
#        Neural Networks: Assignment 1       #
#        Spring Semester                     #
#                                            #
#        Please find the solutions to        #
#        all five tasks below.               #
#                                            #
##############################################


# In[8]:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances

get_ipython().magic(u'matplotlib inline')


# In[9]:

train_in = np.genfromtxt('data/train_in.csv', delimiter=',')
train_out = np.genfromtxt('data/train_out.csv', delimiter=',')
test_in = np.genfromtxt('data/test_in.csv', delimiter=',')
test_out = np.genfromtxt('data/test_out.csv', delimiter=',')


# <b>Task 1</b>

# In[10]:

rows, columns = train_in.shape

train_in_vectors = [train_in[i,:] for i in range(rows)]
actual_numbers = [int(train_out[i]) for i in range(rows)]

storing_vectors = [[] for j in range(10)]
centers = [[] for j in range(10)]

#matching train_in and train_out
for i in range(rows):
    j = actual_numbers[i]
    into_j_bin = storing_vectors[j]
    into_j_bin.append(train_in_vectors[i])
    
#determining how many points there are in each cloud C_{d}
instances_of_each_number = [len(storing_vectors[j]) for j in range(10)]
    
#find the cloud centers for d = 0, 1, ..., 9
for j in range(10):
    set_of_vectors = storing_vectors[j]
    N = len(set_of_vectors)
    center = np.zeros((columns))
    
    for vector in set_of_vectors:
        center += vector
    center = center/float(N)
    
    centers[j] = center
    
distances_storage = [[] for j in range(10)]
    
def distance(a,b):
    distancesquared = 0
    for k in range(len(a)):
        distancesquared += (a[k]-b[k])**2
    return np.sqrt(distancesquared)
    
#finding the distances from each cloud center
for j in range(10):
    center = centers[j]
    distance_storage = distances_storage[j]
    for vector in storing_vectors[j]:
        dist = distance(vector, center)
        distance_storage.append(dist)

#finding the radii of each cloud

radii = [[] for j in range(10)]

for j in range(10):
    radii[j] = max(distances_storage[j])
    
#calculating the distance between cloud centers (0,1 and 0,2 and so on)
indexpairs = []
separations = []
for j in range(9):
    for k in range(j+1, 10):
        center1, center2 = centers[j], centers[k]
        dist = distance(center1, center2)
        indexpairs.append((j,k))
        separations.append(dist)
        
minindex = separations.index(min(separations))
maxindex = separations.index(max(separations))

print ('hardest to separate are', indexpairs[minindex])
print ('easiest to separate are', indexpairs[maxindex])


# <b>Task 2, Training Data</b>

# In[11]:

mets = ['euclidean', 'correlation', 'seuclidean', 'cosine', 'manhattan']

for met in mets:
    classifications = []

    D = pairwise_distances(train_in_vectors, centers, metric=met)
    
    for k in range(rows):
        v = D[k,:]
        classifications.append(np.where(v == v.min()))
    
    c = np.zeros([10,10])
    
    for k in range(rows):
        i, j = actual_numbers[k], classifications[k]
        c[i,j] += 1

    got_it_right = 0
    for j in range(10):
        got_it_right += c[j,j] 

    perc_error = str(round(int(rows - got_it_right)/float(rows)*100, 2))

    ticks = [0,1,2,3,4,5,6,7,8,9]

    plt.figure()
    plt.title('Training Data (%s), Error = %s percent'%(met,perc_error))
    plt.xlabel('Training Output')
    plt.ylabel('Classification')
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.imshow(c, origin = 'lower', interpolation='nearest', aspect = 'equal', cmap='binary')
    plt.colorbar()

    plt.savefig('Task2Training_%s.pdf'%met)


# <b>Task 2, Testing Data</b>

# In[13]:

rows, columns = test_in.shape
test_in_vectors = [test_in[i,:] for i in range(rows)]
actual_numbers = [int(test_out[i]) for i in range(rows)]

mets = ['euclidean', 'correlation', 'seuclidean', 'cosine', 'manhattan']

for met in mets:
    classifications = []

    D = pairwise_distances(test_in_vectors, centers, metric=met)
    
    for k in range(rows):
        v = D[k,:]
        classifications.append(np.where(v == v.min()))
    
    c = np.zeros([10,10])
    
    for k in range(rows):
        i, j = actual_numbers[k], classifications[k]
        c[i,j] += 1

    got_it_right = 0
    for j in range(10):
        got_it_right += c[j,j] 

    perc_error = str(round(int(rows - got_it_right)/float(rows)*100, 2))

    ticks = [0,1,2,3,4,5,6,7,8,9]

    plt.figure()
    plt.title('Training Data (%s), Error = %s percent'%(met,perc_error))
    plt.xlabel('Training Output')
    plt.ylabel('Classification')
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.imshow(c, origin = 'lower', interpolation='nearest', aspect = 'equal', cmap='binary')
    plt.colorbar()

    plt.savefig('Task2Testing_%s.pdf'%met)


# In[ ]:

# <b> Task 3, Bayes classifier: Asymmetry feature </b>

rows, columns = train_in.shape

train_in_vectors = [train_in[i,:] for i in range(rows)]
actual_numbers = [int(train_out[i]) for i in range(rows)]

storing_vectors = [[] for j in range(10)]

#matching train_in and train_out
for i in range(rows):
    j = actual_numbers[i]
    into_j_bin = storing_vectors[j]
    into_j_bin.append(train_in_vectors[i])

# Calculate asymmetry values for 1's (Average difference between right/left side of images) and add bins
amount_ones = len(storing_vectors[1])
Asymmetry_ones_bin = np.array([])
for x in range(amount_ones):
   asymmetry = 0;
   for y in range(0, 256,16):
      asymmetry = asymmetry + sum(abs(storing_vectors[1][x][range(y-8,y)] - list(reversed(storing_vectors[1][x][range(y, y+8)]))))
   Asymmetry_ones_bin = np.append(Asymmetry_ones_bin, asymmetry)
print('Average asymmetry value for ones:', sum(Asymmetry_ones_bin)/amount_ones)

# Calculate asymmetry values for 5's (Average difference between right/left side of images) and add bins
amount_fives = len(storing_vectors[5])
Asymmetry_fives_bin = np.array([])
for x in range(amount_fives):
   asymmetry = 0
   for y in range(0, 256,16):
      asymmetry = asymmetry + sum(abs(storing_vectors[5][x][range(y-8,y)] - list(reversed(storing_vectors[5][x][range(y, y+8)]))))
   Asymmetry_fives_bin = np.append(Asymmetry_fives_bin, asymmetry)
print('Average asymmetry value for fives:', sum(Asymmetry_fives_bin)/amount_fives)

#plot histograms 
plt.hist([Asymmetry_ones_bin, Asymmetry_fives_bin], bins = 10, label=['Ones','Fives'])
plt.title("Histograms")
plt.legend(loc ='upper right')
plt.show()

#Estimate probabilities 
P_one = amount_ones / rows #P(one)
P_five = amount_fives / rows #P(five)
bins = np.array(range(0,100,10)) #use 10 bins
one_bins = np.digitize(Asymmetry_ones_bin, bins)
five_bins = np.digitize(Asymmetry_fives_bin, bins)

bayesian_out = np.array([])
for i in range(len(test_in)):
   #calculate asymmetry of digit
   X = 0
   for x in range(0,256,16):
      X = X + sum(abs(test_in[i][range(x-8,x)] - list(reversed(test_in[i][range(x, x+8)]))))
      
   X = np.digitize(X, bins)   
   
   #Calculate P(X=x|one)
   one_count = np.count_nonzero(one_bins == X)
   P_X_given_one = one_count / amount_ones
      
   #Calculate P(X=x|five)
   five_count = np.count_nonzero(five_bins == X)
   P_X_given_five = five_count / amount_fives
      
   P_X = (one_count + five_count) / (amount_ones + amount_fives)
      
   P_one_given_X = P_X_given_one * P_one / P_X #Calculate P(one|X=x)
   P_five_given_X = P_X_given_five * P_five / P_X #Calculate P(five|X=x)
      
   if P_one_given_X > P_five_given_X:
      bayesian_out = np.append(bayesian_out, 1)
   else: bayesian_out = np.append(bayesian_out, 5)

ones_array = bayesian_out[np.where(test_out == 1)] #obtain indices
fives_array = bayesian_out[np.where(test_out == 5)] 

print("Correctly classified ones:", np.count_nonzero(ones_array == 1), "/", len(ones_array))
print("Correctly classified fives:", np.count_nonzero(fives_array == 5), "/", len(fives_array))
print("Total accuracy:", (np.count_nonzero(ones_array == 1) + np.count_nonzero(fives_array == 5)) / (len(ones_array)+ len(fives_array)))
   

#EXTRA -- Calculate all symmetry values to get an idea of how it performs:
#for i in range(10):
#   amount = len(storing_vectors[i])
#   Asymmetry_value = 0
#   for x in range(amount):
#      for y in range(0, 256,16):
#         Asymmetry_value = Asymmetry_value + sum(abs(storing_vectors[i][x][range(y-8,y)] - list(reversed(storing_vectors[i][x][range(y, y+8)]))))
#   print('Average asymmetry value for', i, ': ', Asymmetry_value/amount)

# In[ ]:

# <b> Task 4: Implement a multi-class perceptron algorithm </b>

def activation(input_vec, weight_vec): #uses 
    summation = np.dot(input_vec, weight_vec[1:]) + weight_vec[0]
    #step function
    if summation > 0:
        return 0 #correctly classified
    else:
        return 1 #incorrectly classified

class perceptron(object):
    
    def __init__(self, len_input, threshold, learning_rate):
        self.len_input = len_input
        self.threshold = threshold
        self.learning_rate = learning_rate
        #need a vector of weights for each digit 0-9, initialize randomly
        self.weights = np.random.rand(len_input+1, 10) 
        
    def train(self, inputs, outputs):
        count = 0
        while count < self.threshold:
            for input_vec, output in zip(inputs, outputs):
                weight_vec = self.weights[:,output]
                #determine if weights need to be adjusted
                activation_w_x = activation(input_vec, weight_vec) 
                self.weights[1:, output] += learning_rate * activation_w_x * input_vec
                self.weights[0, output] += learning_rate * activation_w_x
            count += 1
        
        return self.weights

len_input, threshold, learning_rate = len(train_in[0]), 100, 0.01

#sets up weights matrix on which the MNIST data can be applied
W = perceptron(len_input, threshold, learning_rate)

#applies the MNIST training data to find the weights
W_trained = W.train(train_in, train_out)

def get_output(inputs):
    outputs_from_W = []
    for x in inputs:
        w_dot_x = np.zeros(10) #finding the maximum value of \vec{w}\cdot\vec{x}
        for i in range(10): #iterating through the digits
            weight_vec = W_trained[:,i]
            w_dot_x[i] = (np.dot(x, weight_vec[1:]) + weight_vec[0])
        classification = np.argmax(w_dot_x)
        outputs_from_W.append(classification)
    return outputs_from_W

def confusion(actual, predicted, title):
    
    c = np.zeros([10,10])

    rows = len(actual)

    for k in range(rows):
        i, j = actual[k], predicted[k]
        c[i,j] += 1

    got_it_right = 0
    for j in range(10):
        got_it_right += c[j,j] 

    perc_error = str(round(int(rows - got_it_right)/float(rows)*100, 2))

    ticks = [0,1,2,3,4,5,6,7,8,9]

    plt.figure()
    plt.title('%s Data, Error = %s percent'%(title, perc_error))
    plt.xlabel('%s Output'%title)
    plt.ylabel('Classification')
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.imshow(c, origin = 'lower', interpolation='nearest', aspect = 'equal', cmap='binary')
    plt.colorbar()

    plt.savefig('Task4_%s.pdf'%title)

train_outputs_from_W = get_output(train_in)
test_outputs_from_W = get_output(test_in)

confusion(train_out, train_outputs_from_W, 'Training')
confusion(test_out, test_outputs_from_W, 'Testing')

# In[ ]:

# <b> Task 5: Implement the gradient descent algorithm </b>

#Activation functions definition
def sigmoid(x):
   return 1 / (1 + np.exp(-x))
def relu(x):
   return np.maximum(0,x)

#Simulate NN output for nodes & weights input, comment/uncomment to switch activation functions
def xor_net(x1,x2,weights):
   y = np.dot(np.array([weights[0],weights[1]]),np.array([1,x1,x2]))
   
   #Sigmoid function
   y = list(map(lambda y: sigmoid(y), y))
   z = sigmoid(np.dot(np.array(weights[2]),np.array([1,y[0],y[1]])))
   
   #Relu function
   #y = list(map(lambda y: relu(y), y))
   #z = relu(np.dot(np.array(weights[2]),np.array([1,y[0],y[1]])))
   
   #Hyperbolic tangent function
   #y = list(map(lambda y: np.tanh(y), y))
   #z = np.tanh(np.dot(np.array(weights[2]),np.array([1,y[0],y[1]])))
   return z

#Calculate MSE for given weights
def mse(weights):
   mse_11 = (0-xor_net(1,1,weights))**2
   mse_10 = (1-xor_net(1,0,weights))**2
   mse_01 = (1-xor_net(0,1,weights))**2
   mse_00 = (0-xor_net(0,0,weights))**2
   total_mse = mse_11 + mse_10 + mse_01 + mse_00
   return total_mse

#Count amount of misclassified XOR inputs with given weights
def missclassified(weights):
   missclassified = 0
   if xor_net(0,0,weights) > 0.5:
      missclassified += 1
   if xor_net(0,1,weights) <= 0.5:
      missclassified += 1
   if xor_net(1,0,weights) <= 0.5:
      missclassified += 1
   if xor_net(1,1,weights) > 0.5:
      missclassified += 1
   return missclassified

#Calculate MSE gradient with given weights
def grdmse(weights):
   error = pow(10, -3) #avoid division by zero
   columns = weights.shape[1]
   rows = weights.shape[0]
   grdmse = np.zeros((3,3))
   for i in range(rows):
      for j in range(columns):
         a = np.zeros((3,3))
         a[i][j] = error
         grdmse[i][j] = (mse(weights + a) - mse(weights)) / error
   return grdmse

#Train XOR network with 4000 iterations
def train_xor_net(learningrate, net_input):
   np.random.seed(42)
   if net_input == 'normal':
      weights = np.random.randn(3,3) #normal dist with mean = 0 and variance = 1
   if net_input == 'uniform':
      weights = np.random.uniform(-1,1,9).reshape(3,3) #uniform dist from -1 to 1
   iterator = 0
   mse_list = []
   missclassifiedlist = []
   while iterator < 4000:
      weights = weights-learningrate * grdmse(weights)
      mse_list.append(mse(weights))
      missclassifiedlist.append(missclassified(weights))
      iterator += 1
   return weights, mse_list, missclassifiedlist


#Calculate intermediate results for various initialized weights and learning rates
weights, mse_list, missclassifiedlist = np.full((2,3,3,3), 0),np.zeros((2,3,4000)),np.zeros((2,3,4000))
learningrate = [0.1,0.25,0.5]
net_input = ['normal','uniform']
for i in range (2):
   for j in range(3):
      weights[i][j], mse_list[i][j], missclassifiedlist[i][j] = train_xor_net(learningrate[j],net_input[i])

#Plot MSE and amount of misclassified cases for various initialized weights and learning rates
plot = plt.figure()
for i in range(2):
    for j in range(3):
        axis = plot.add_subplot(2,3,3*i+j+1, label="1")
        axis_2 = plot.add_subplot(2,3,3*i+j+1, label="2", frame_on=False)

        axis.plot(range(len(mse_list[i][j])), mse_list[i][j], color="blue")

        axis.set_title('%s, $\eta = %s$'%(net_input[i], str(learningrate[j])), fontsize = 8)
        axis.set_xticks([0,1000,2000,3000,4000])
        axis.set_xticklabels(['0','1','2','3','4'], fontsize = 8)
        axis.set_xlabel("Iterations (x 1000)", color="k", fontsize = 8)
        axis.set_ylabel("Mean squared error", color="blue", fontsize = 8)
        axis.set_ylim([0,1])
        axis.tick_params(axis='y', colors="blue")

        axis_2.plot(range(len(missclassifiedlist[i][j])), missclassifiedlist[i][j], color="green")
        axis_2.set_xticks([0,1000,2000,3000,4000])
        axis_2.set_xticklabels(['0','1','2','3','4'], fontsize = 8)
        axis_2.yaxis.tick_right()
        axis_2.set_ylabel('Missclassified units', color="green", fontsize = 8)
        axis_2.set_ylim([0,4])
        axis_2.yaxis.set_label_position('right')
        axis_2.tick_params(axis='y', colors="green")

plt.subplots_adjust(hspace=0.5, wspace=1)
plt.savefig('sigmoid_activation_function.pdf')






















