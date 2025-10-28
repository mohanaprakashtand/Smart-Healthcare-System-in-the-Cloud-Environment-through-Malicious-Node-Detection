from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import roc_curve
from Plot_BlockChain_Final import *

No_of_Dataset = 2

def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def Plot_ROC_Curve():
    lw = 2
    cls = ['SS-RBFN', 'MS-BNN', 'ODNN-GRU', 'MLSTM-AM', 'IRV-SOA-Ada-MLSTM-AM']
    for a in range(2):  # For 2 Datasets
        Actual = np.load('Target_'+str(a+1)+'.npy', allow_pickle=True).astype('int')
        per = round(Actual.shape[0] * 0.75)
        Actual = Actual[per:, :]

        colors = cycle(["blue", "darkorange", "cornflowerblue", "deeppink", "black"])  # "aqua",
        for i, color in zip(range(len(cls)), colors):  # For all classifiers
            Predicted = np.load('Y_Score_'+str(a+1)+'.npy', allow_pickle=True)[i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i], )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Accuracy')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = ("./Results/ROC_%s.png" % (a + 1))
        plt.savefig(path1)
        plt.show()


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'AVOA-Ada-MLSTM-AM', 'WWO-Ada-MLSTM-AM', 'KOA-Ada-MLSTM-AM', 'SOA-Ada-MLSTM-AM', 'IRV-SOA-Ada-MLSTM-AM']

    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']

    for i in range(Fitness.shape[0]):
        Conv_Graph = np.zeros((len(Algorithm) - 1, 5))
        for j in range(len(Algorithm) - 1):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('--------------------------------------------------'' Statistical Report '
              '--------------------------------------------------')
        print(Table)

        Conv_Graph = Fitness[i]
        length = np.arange((Conv_Graph.shape[1]))
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label='AVOA-Ada-MLSTM-AM')
        plt.plot(length, Conv_Graph[1, :], color=[0, 0.5, 0.5], linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label='WWO-Ada-MLSTM-AM')
        plt.plot(length, Conv_Graph[2, :], color=[0.5, 0, 0.5], linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label='KOA-Ada-MLSTM-AM')
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label='SOA-Ada-MLSTM-AM')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label='IRV-SOA-Ada-MLSTM-AM')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()



def plot_results_Epoch():
    # matplotlib.use('TkAgg')
    eval = np.load('Eval_all_Epoch.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Classfier = ['SS-RBFN', 'MS-BNN', 'ODNN-GRU', 'MLSTM-AM', 'IRV-SOA-Ada-MLSTM-AM']
    Graph_Term = [0, 1, 2,]


    for i in range(No_of_Dataset):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4]

            Epoch = [50, 100, 150, 200, 250]
            fig = plt.figure(facecolor='#e4f5e8')
            ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
            ax.set_facecolor("#e4f5e8")
            plt.plot(Epoch, Graph[:, 0], color='#5e606c', linestyle='solid', linewidth=2, marker='o',
                     markerfacecolor='#ffd600', markeredgecolor="k", markersize=15, label="AVOA-Ada-MLSTM-AM")

            plt.plot(Epoch, Graph[:, 1], color='#a7cbd9', linestyle='solid', linewidth=2, marker='o',
                     markerfacecolor='#ff7a00', markeredgecolor="k", markersize=15, label="WWO-Ada-MLSTM-AM")

            plt.plot(Epoch, Graph[:, 2], color='#68a0a6', linestyle='solid', linewidth=2, marker='o',
                     markerfacecolor='#ff0069', markeredgecolor="k", markersize=15, label="KOA-Ada-MLSTM-AM")

            plt.plot(Epoch, Graph[:, 3], color='#d69e49', linestyle='solid', linewidth=2, marker='o',
                     markerfacecolor='#d300c5', markeredgecolor="k", markersize=15, label="SOA-Ada-MLSTM-AM")

            plt.plot(Epoch, Graph[:, 4], color='#a66f6f', linestyle='solid', linewidth=2, marker='o',
                     markerfacecolor='#7638fa', markeredgecolor="k", markersize=15, label="IRV-SOA-Ada-MLSTM-AM")
            plt.xticks(Epoch, (50, 100, 150, 200, 250))
            # plt.style.use("dark_background")
            plt.xlabel('Epoch')
            plt.ylabel(Terms[Graph_Term[j]])
            # plt.tick_params(axis='x', labelrotation=25)
            # plt.ylim([60, 100])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True, facecolor='#e4f5e8')
            path1 = "./Results/Dataset_%s_Epoch_Alg_Line_%s.png" % ((i + 1), Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

            # -----------------------------------------------Classifier------------------------------------------------


            # ax = plt.axes(projection="3d")
            fig = plt.figure(facecolor='#fefee3')
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.set_facecolor("#fefee3")
            X = np.arange(Graph.shape[0])
            barWidth = 0.15
            color = ['#1a0f8b', '#c6f40c', '#efa9fc', '#e268a5', '#98f47f']
            for l in range(5, Graph.shape[1]):
                ax.bar(X + (l * barWidth), Graph[:, l], color=color[l - 5], width=barWidth, edgecolor='k',
                       label=Classfier[l - 5])

            plt.xticks(X + (((len(Classfier)) * barWidth)*1.4), (50, 100, 150, 200, 250))
            plt.xlabel('Epoch')
            # ax.tick_params(axis='x', labelrotation=45)
            plt.ylabel(Terms[Graph_Term[j]])
            # plt.ylim([60, 100])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True, facecolor ='#fefee3')
            path1 = "./Results/Dataset_%s_Met_bar_%s.png" % ((i + 1), Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()


def Plot_Results_Batch():  # Table Results
    # matplotlib.use('TkAgg')
    eval = np.load('Eval_all_Batch.npy', allow_pickle=True)
    Terms = ['accuracy', 'sensitivity', 'specificity', 'precision', 'FPR', 'FNR',  'NPV', 'FDR',
                         'f1score', 'mcc', 'For', 'pt', 'csi', 'ba', 'fm', 'bm', 'mk', 'lrplus', 'lrminus', 'dor', 'prevalence']
    Algorithm = ['Batch Size\Algorithm', 'AVOA-Ada-MLSTM-AM', 'WWO-Ada-MLSTM-AM', 'KOA-Ada-MLSTM-AM', 'SOA-Ada-MLSTM-AM', 'IRV-SOA-Ada-MLSTM-AM']
    Classifier = ['Batch Size\Methods ', 'SS-RBFN', 'MS-BNN', 'ODNN-GRU', 'MLSTM-AM', 'IRV-SOA-Ada-MLSTM-AM']

    Batch = [4, 8, 16, 32, 64]
    for i in range(eval.shape[0]):
        for k in range(eval.shape[2]-3):
            value = eval[i, :, :, k + 4]
            Table = PrettyTable()
            Table.add_column(Algorithm[0], Batch[:])
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j])

            print('--------------------------------------------------Dataset-' + str(i + 1) + Terms[k],
                  '-Algorithm Comparison ',
                  '--------------------------------------------------')
            print(Table)

            value2 = eval[i, :, :, k + 4]
            Table = PrettyTable()
            Table.add_column(Classifier[0], Batch[:])
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value2[:, j+5])
            print('--------------------------------------------------Dataset-' + str(i + 1) + Terms[k],
                  '-Classifier Comparison',
                  '--------------------------------------------------')
            print(Table)


if __name__ == '__main__':
    plot_CPA_KPA()
    plot_KEY_RE()
    plot_ENC_DEC_CONS_MEM()
    Plot_ROC_Curve()
    plotConvResults()
    plot_results_Epoch()
    Plot_Results_Batch()