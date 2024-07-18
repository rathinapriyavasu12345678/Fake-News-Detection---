import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


def plot_results():
    matplotlib.use('TkAgg')
    eval1 = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Terms = [0, 3, 4, 7]
    Algorithm = ['TERMS', 'HHO-OMCNLSTM [36]', 'DHOA-OMCNLSTM [37]', 'SSA-OMCNLSTM [34]', 'ARO-OMCNLSTM [27]', 'AARO-OMCNLSTM']
    Classifier = ['TERMS', 'GRU [35]', 'CNN [5]', 'LSTM [7]', 'CNN-LSTM [33]', 'AARO-OMCNLSTM']
    for i in range(eval1.shape[0]):
        value1 = eval1[i, 4, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value1[j, :])
        print('-------------------------------------------------- Dataset', i + 1, 'Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- Dataset', i + 1, ' Classifier Comparison',
              '--------------------------------------------------')
        print(Table)

    learnper = [35, 45, 55, 65, 75, 85]
    for i in range(eval1.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros((eval1.shape[1], eval1.shape[2] + 1))
            for k in range(eval1.shape[1]):
                for l in range(eval1.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval1[i, k, l, Graph_Terms[j] + 4]
                    else:
                        Graph[k, l] = eval1[i, k, l, Graph_Terms[j] + 4]

            plt.plot(learnper, Graph[:, 0], color='r', linewidth=3, marker='o', markerfacecolor='blue', markersize=12,
                     label="HHO-OMCNLSTM [36]")
            plt.plot(learnper, Graph[:, 1], color='g', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
                     label="DHOA-OMCNLSTM [37]")
            plt.plot(learnper, Graph[:, 2], color='b', linewidth=3, marker='o', markerfacecolor='green', markersize=12,
                     label="SSA-OMCNLSTM [34]")
            plt.plot(learnper, Graph[:, 3], color='m', linewidth=3, marker='o', markerfacecolor='yellow', markersize=12,
                     label="ARO-OMCNLSTM [27]")
            plt.plot(learnper, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='magenta',
                     markersize=12,
                     label="AARO-OMCNLSTM")
            plt.xlabel('Learning Percentage (%)')
            plt.ylabel(Terms[Graph_Terms[j]])
            plt.legend(loc=4)
            path1 = "./Results/Dataset_%s_%s_line_lrean.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(6)

            ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="GRU [35]")
            ax.bar(X + 0.10, Graph[:, 6], color='g', width=0.10, label="CNN [5]")
            ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="LSTM [7]")
            ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="CNN-LSTM [33]")
            ax.bar(X + 0.40, Graph[:, 9], color='k', width=0.10, label="AARO-OMCNLSTM")
            plt.xticks(X + 0.10, ('35', '45', '55', '65', '75', '85'))
            plt.xlabel('Learning Percentage (%)')
            plt.ylabel(Terms[Graph_Terms[j]])
            plt.legend(loc=4)
            path1 = "./Results/Dataset_%s_%s_bar_lrean.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path1)
            plt.show()


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v

def plot_results_conv():
    # matplotlib.use('TkAgg')
    Result = np.load('Fitness.npy', allow_pickle=True)
    Fitness = np.load('Fitness.npy', allow_pickle=True)

    Algorithm = ['TERMS', 'HHO-OMCNLSTM [36]', 'DHOA-OMCNLSTM [37]', 'SSA-OMCNLSTM [34]', 'ARO-OMCNLSTM [27]', 'AARO-OMCNLSTM']
    Classifier = ['TERMS', 'GRU [35]', 'CNN [5]', 'LSTM [7]', 'CNN-LSTM [33]', 'AARO-OMCNLSTM']

    for i in range(Result.shape[0]):
        Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = stats(Fitness[i, j, :])
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Report Dataset ', 0 + 1,
              '--------------------------------------------------')
        print(Table)

        length = np.arange(10)
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red', markersize=12,
                 label='HHO-OMCNLSTM [36]')
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12,
                 label='DHOA-OMCNLSTM [37]')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12,
                 label='SSA-OMCNLSTM [34]')
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12,
                 label='ARO-OMCNLSTM [27]')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12,
                 label='AARO-OMCNLSTM')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/User Variation_%s_%s_Conv.png" % (i + 1, 1))
        plt.show()


def plot_results_Abliation():
    eval1 = np.load('Eval_all_ABLIATION.npy', allow_pickle=True)
    Terms = ['Accuracy']

    Classifier = ['TERMS', 'Fast R_CNN', 'Mask R_CNN', 'CNN [5]', '1D-CNN', 'BI-LSTM', 'LSTM [7]', 'OMCN-LSTM', 'AARO-OMCNLSTM']
    for i in range(eval1.shape[0]):
        value1 = eval1[i, 4, :, :]

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value1[j, :])
        print('-------------------------------------------------- Dataset', i + 1, ' Abliation Experiment',
              '--------------------------------------------------')
        print(Table)


if __name__ == '__main__':
    # plot_results()
    plot_results_conv()
    # plot_results_Abliation()
