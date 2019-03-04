class perceptron(object):
    
    def __init__(self, len_input, threshold, learning_rate):
        self.len_input = len_input
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros((len_input+1, 10)) #need a vector of weights for each digit 0-9

    def predict(input_vec, weight_vec): #len(weight_vec) = len(input_vec) + 1
        summation = np.dot(input_vec, weight_vec[1:]) + weight_vec[0]
        if summation > 0:
            return 1
        else:
            return 0

    def train(self, inputs, outputs):
        count = 0
        while count < self.threshold:
            for input_vec, output in zip(inputs, outputs):
                weight_vec = self.weights[:,output]
                prediction = predict(input_vec, weight_vec)
                self.weights[1:, output] += learning_rate * (output-prediction)*input_vec
                self.weights[0, output] += learning_rate * (output-prediction)
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
    
    return 1

train_outputs_from_W = get_output(train_in)
test_outputs_from_W = get_output(test_in)

confusion(train_out, train_outputs_from_W, 'Training')
confusion(test_out, test_outputs_from_W, 'Testing')
