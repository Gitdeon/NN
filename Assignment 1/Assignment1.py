
# coding: utf-8

# In[8]:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import itertools
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

# <b> Task 3, Bayes classifier: Asymmetry idea </b>

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
plt.hist([Asymmetry_ones_bin, Asymmetry_fives_bin], bins = 20, label=['Ones','Fives'])
plt.title("Asymmetry histogram for the digit five")
plt.legend(loc ='upper right')
plt.show()

#Estimate probabilities 
P_one = amount_ones / rows #P(one)
P_five = amount_fives / rows #P(five)












'''
# EXTRA -- Calculate all symmetry values to get an idea of how it performs:
for i in range(10):
   amount = len(storing_vectors[i])
   Asymmetry_value = 0
   for x in range(amount):
      for y in range(0, 256,16):
         Asymmetry_value = Asymmetry_value + sum(abs(storing_vectors[i][x][range(y-8,y)] - list(reversed(storing_vectors[i][x][range(y, y+8)]))))
   print('Average asymmetry value for', i, ': ', Asymmetry_value/amount)
   
'''


















