#Plot MSE and amount of misclassified cases for various initialized weights and learning rates
plot = plt.figure()
for i in range(2):
    for j in range(3):
        axis = plot.add_subplot(2,3,3*i+j+1, label="1")
        axis_2 = plot.add_subplot(2,3,3*i+j+1, label="2", frame_on=False)

        axis.plot(range(len(mse_list[i][j])), mse_list[i][j], color="blue")

        axis.set_title('%s, \eta = %s'%(net_input[i], str(learningrate[j])), fontsize = 8)
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
plt.savefig('Sigmoid_activation_function.pdf')
