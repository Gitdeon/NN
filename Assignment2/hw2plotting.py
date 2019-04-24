import numpy as np
import matplotlib.pyplot as plt
import random
import itertools

activation_fns = ['relu', 'tanh', 'exponential', 'sigmoid']
n_extralayer_possibilities = [0, 1, 3]

information_for_plots = information_for_plots = [['relu', 0, 0.7890431565279257, 0.6738118436850936, 318.3303031921387], ['relu', 1, 0.8103563215124203, 0.6661794955201049, 449.190043926239], ['relu', 3, 1.0394622508202123, 0.5945653350687239, 769.817830324173], ['tanh', 0, 0.7569174698979563, 0.6895637111551798, 325.73764395713806], ['tanh', 1, 0.9466547722603267, 0.5945653350687239, 464.07766366004944], ['tanh', 3, 1.040533327160685, 0.5945653350687239, 799.8295550346375], ['exponential', 0, 0.0, 0.08861102089422973, 350.4982192516327], ['exponential', 1, 0.0, 0.08861102089422973, 484.5091931819916], ['exponential', 3, 0.0, 0.08861102089422973, 967.0292460918427], ['sigmoid', 0, 1.0411327193796964, 0.5945653350687239, 417.04072070121765], ['sigmoid', 1, 1.0411631583367331, 0.5945653350687239, 527.8208146095276], ['sigmoid', 3, 1.0395500052603754, 0.5945653350687239, 821.9365644454956]]

'''
FIRST PLOT
use activation_fn = 'relu' and n_layers = 5 as the controls for each of the panels
('relu', n_layers = 5, 7, 11), (activation_fn = 'relu',... , n_layers = 5)
n_layers_relu_plotting, accuracies_relu_plotting, runtimes_relu_plotting = [], [], []
activation_fns_five_plotting, accuracies_five_plotting, runtimes_five_plotting = [], [], []
for ifp in information_for_plots:
if ifp[0] == activation_fns[0]:
n_layers_relu_plotting.append(ifp[1])
accuracies_relu_plotting.append(ifp[3])
runtimes_relu_plotting.append(ifp[4])
for ifp in information_for_plots:
if ifp[1] == n_layer_possibilities[0]:
activation_fns_five_plotting.append(ifp[0])
accuracies_five_plotting.append(ifp[3])
runtimes_five_plotting.append(ifp[4])
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
divider, eps = 2.5, 0.1 #for plotting using fig.add_axes
fig = plt.figure()
xvals, yvals = [eps, eps+eps+(1-eps)/divider], [eps, eps+eps+(1-eps)/divider]
plots = [fig.add_axes([x0, y0, (1-eps)/divider, (1-eps)/divider]) for x0 in xvals for y0 in yvals]
fs = 8
for i in range(len(plots)):
ax = plots[i]
ax.tick_params(left = True, bottom = True, right = False, top = False, labelsize = 'x-small')
if i == 3:
ax.scatter(range(1,len(accuracies_five_plotting)+1), accuracies_five_plotting, s = 1, color='k')
ax.set_xticklabels(activation_fns_five_plotting)
ax.set_ylabel(r'$f$(activation function) [\%/100]', fontsize = fs)
ax.set_xlabel(r'$f \equiv$ accuracy', fontsize = fs)
ax.xaxis.set_label_position('top')
ax.annotate(r'$N_{layers} = 5$', xy = (0.55, 0.15), xycoords = 'axes fraction', fontsize = fs)
if i == 1:
ax.scatter(range(1,len(runtimes_five_plotting)+1), runtimes_five_plotting, s = 1, color='k')
ax.set_xticklabels(activation_fns_five_plotting)
ax.set_ylabel(r'$g$(activation function) [s]', fontsize = fs)
ax.set_xlabel(r'$g \equiv$ runtime', fontsize = fs)
ax.xaxis.set_label_position('top')
ax.annotate(r'$N_{layers} = 5$', xy = (0.55, 0.15), xycoords = 'axes fraction', fontsize = fs)
if i == 2:
ax.scatter(n_layers_relu_plotting, accuracies_relu_plotting, s = 1, color='k')
ax.set_xticks(n_layers_relu_plotting)
ax.set_ylabel(r'$f(N_{layers})$', fontsize = fs)
ax.set_xlabel(r'$N_{layers}$', fontsize = fs)
ax.annotate('activation = relu', xy = (0.55, 0.15), xycoords = 'axes fraction', fontsize = fs)
if i == 0:
ax.scatter(n_layers_relu_plotting, runtimes_relu_plotting,  s = 1, color='k')
ax.set_xticks(n_layers_relu_plotting)
ax.set_ylabel(r'$g(N_{layers})$', fontsize = fs)
ax.set_xlabel(r'$N_{layers}$', fontsize = fs)
ax.annotate('activation = relu', xy = (0.55, 0.15), xycoords = 'axes fraction', fontsize = 8)
plt.savefig('homework2_firstplot.pdf')
'''

'''
SECOND PLOT
'''

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
fig = plt.figure()

for_second_plot, Xs, Ys = [], [], []

for ifp in information_for_plots:
    label = str(ifp[0]) + ', ' + str(ifp[1])
    f = ifp[3]
    g = ifp[4]

    Xs.append(g)
    Ys.append(f)

    for_second_plot.append([label, f, g])

fig = plt.figure()
for element in for_second_plot:
    label, f, g = element[:]
    
    if 'tanh' in label:
        m = '+'
    if 'sigmoid' in label:
        m = '*'
    if 'exponential' in label:
        m = '^'
    if 'relu' in label:
        m = 'o'

    if '0' in label:
        c = 'r'
    if '1' in label:
        c = 'g'
    if '3' in label:
        c = 'b'

    plt.scatter(g, f, s = 36, color = c, marker = m, label=label)

plt.annotate('good performance', xy = (0.05, 0.95), xycoords = 'axes fraction', fontsize = 14)
plt.annotate('bad performance', xy = (0.75, 0.05), xycoords = 'axes fraction', fontsize = 14)
plt.xlabel('Runtime [s]', fontsize = 16)
plt.ylabel('Accuracy [$\%$/100]', fontsize = 16)
plt.title('Evaluating CNN Performance', fontsize = 16)
plt.legend(loc='best', fontsize=12)
plt.tight_layout()
plt.savefig('homework2_secondplot.pdf')
plt.close()
