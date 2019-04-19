import numpy as np
import matplotlib.pyplot as plt
import random

activation_fns = ['relu', 'tanh', 'sigmoid', 'exponential', 'linear']
n_layer_possibilities = [5, 7, 11]

information_for_plots = []

for activation_fn in activation_fns:
        for n_layers in n_layer_possibilities:
                loss, acc, runtime = random.random(), 1e-4 + (1e-1-1e-4)*random.random(), 10**(4*random.random())
                information_for_plots.append([activation_fn, n_layers, loss, acc, runtime])

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
	f = 1-ifp[3]
	g = ifp[4]

	Xs.append(g)
	Ys.append(f)

	for_second_plot.append([label, f, g])

fig = plt.figure()
for element in for_second_plot:
	label, f, g = element[:]
	plt.scatter(g, f, s = 1, color='k')
	plt.annotate(label, xy = (g, f), xycoords = 'data', color='r', fontsize = 8)

plt.annotate('good', xy = (0.05, 0.05), xycoords = 'axes fraction', fontsize = 10)
plt.annotate('bad', xy = (0.85, 0.85), xycoords = 'axes fraction', fontsize = 10)
plt.xlabel('Runtime [s]', fontsize = 12)
plt.xscale('log')
plt.yscale('log')
plt.ylabel('(1 - Accuracy) [$\%$/100]', fontsize = 12)
plt.title('Evaluating CNN Performance', fontsize = 12)
plt.tight_layout()
plt.savefig('homework2_secondplot.pdf')
plt.close()

