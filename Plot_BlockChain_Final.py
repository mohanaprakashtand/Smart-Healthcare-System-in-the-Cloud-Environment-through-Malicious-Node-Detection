import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def statistical_analysis(v):
    a = np.zeros((5))
    a[0] = np.min(v)
    a[1] = np.max(v)
    a[2] = np.mean(v)
    a[3] = np.median(v)
    a[4] = np.std(v)
    return a


def plot_CPA_KPA():
    for a in range(1):

        Eval = np.load('CPA_attack.npy', allow_pickle=True)[a]
        learnper = [10, 20, 30, 40, 50]
        X = np.arange(5)
        plt.plot(learnper, Eval[:, 5], color='#aaff32', linewidth=3, marker='H', markerfacecolor='#aaff32', markersize=14,
                 label="RSA")
        plt.plot(learnper, Eval[:, 6], color='#8c564b', linewidth=3, marker='H', markerfacecolor='#8c564b', markersize=14,
                 label="Trust-based encryption")
        plt.plot(learnper, Eval[:, 7], color='#017b92', linewidth=3, marker='H', markerfacecolor='#017b92', markersize=14,
                 label="Homomorphic encryption")
        plt.plot(learnper, Eval[:, 8], color='#ff000d', linewidth=3, marker='H', markerfacecolor='#ff000d', markersize=14,
                 label="Elgamal Encryption")
        plt.plot(learnper, Eval[:, 9], color='k', linewidth=3, marker='H', markerfacecolor='k', markersize=14,
                 label="HE-RSA")
        x = [10, 20, 30, 40, 50]
        labels = ['1', '2', '3', '4', '5']
        plt.xticks(x, labels)

        plt.xlabel('Case')
        plt.ylabel('CPA')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        path1 = "./Results/CPA_Encryption_Met_%s.png"  %(str(a+1))
        plt.savefig(path1)
        plt.show()

        Eval = np.load('KPA_attack.npy', allow_pickle=True)[a]
        learnper = [10, 20, 30, 40, 50]
        X = np.arange(5)
        plt.plot(learnper, Eval[:, 5], color='#aaff32', linewidth=3, marker='s', markerfacecolor='#aaff32', markersize=10,
                 label="RSA")
        plt.plot(learnper, Eval[:, 6], color='#8c564b', linewidth=3, marker='s', markerfacecolor='#8c564b', markersize=10,
                 label="Trust-based encryption")
        plt.plot(learnper, Eval[:, 7], color='#017b92', linewidth=3, marker='s', markerfacecolor='#017b92', markersize=10,
                 label="Homomorphic encryption")
        plt.plot(learnper, Eval[:, 8], color='#ff000d', linewidth=3, marker='s', markerfacecolor='#ff000d', markersize=10,
                 label="Elgamal Encryption")
        plt.plot(learnper, Eval[:, 9], color='k', linewidth=3, marker='s', markerfacecolor='k', markersize=10,
                 label="HE-RSA")
        x = [10, 20, 30, 40, 50]
        labels = ['1', '2', '3', '4', '5']
        plt.xticks(x, labels)

        plt.xlabel('Case')
        plt.ylabel('KPA')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        path1 = "./Results/KPA_Method_%s.png"  %str(a+1)
        plt.savefig(path1)
        plt.show()


def plot_KEY_RE():
    for a in range(1):
        Eval = np.load('key.npy', allow_pickle=True)[a]
        learnper = [10, 20, 30, 40, 50]
        X = np.arange(5)
        plt.plot(learnper, Eval[:, 5], color='#aaff32', linewidth=3, marker='*', markerfacecolor='#aaff32', markersize=14,
                 label="RSA")
        plt.plot(learnper, Eval[:, 6], color='#ad03de', linewidth=3, marker='*', markerfacecolor='#ad03de', markersize=14,
                 label="Trust-based encryption")
        plt.plot(learnper, Eval[:, 7], color='#8c564b', linewidth=3, marker='*', markerfacecolor='#8c564b', markersize=14,
                 label="Homomorphic encryption")
        plt.plot(learnper, Eval[:, 8], color='#ff000d', linewidth=3, marker='*', markerfacecolor='#ff000d', markersize=14,
                 label="Elgamal Encryption")
        plt.plot(learnper, Eval[:, 9], color='k', linewidth=3, marker='*', markerfacecolor='k', markersize=14,
                 label="HE-RSA")
        # plt.xticks(X + 1, ('1', '2', '3', '4', '5'))

        plt.xlabel('Case')
        plt.ylabel('Key Sensitivity Analysis')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        path1 = "./Results/KEY_Method_%s.png" % str(a + 1)
        plt.savefig(path1)
        plt.show()


def plot_ENC_DEC_CONS_MEM():
    for a in range(1):
        Encryp = ['RSA', 'Trust-based encryption', 'Homomorphic encryption', 'Elgamal Encryption', 'HE-RSA']
        Eval = np.load('Encryption_time.npy', allow_pickle=True)[a]
        learnper = [1, 2, 3, 4, 5]
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(5)
        ax.bar(X + 0.00, Eval[:, 5], color='#aaff32', width=0.10, edgecolor='k', label=Encryp[0])
        ax.bar(X + 0.10, Eval[:, 6], color='#ad03de', width=0.10, edgecolor='k', label=Encryp[1])
        ax.bar(X + 0.20, Eval[:, 7], color='#8c564b', width=0.10, edgecolor='k', label=Encryp[2])
        ax.bar(X + 0.30, Eval[:, 8], color='#ff000d', width=0.10, edgecolor='k', label=Encryp[3])
        ax.bar(X + 0.40, Eval[:, 9], color='k', width=0.10, edgecolor='k', label=Encryp[4])
        # plt.xticks(X + 0.25, ('5', '10', '15', '20', '25'))
        labels = ['5', '10', '15', '20', '25']
        plt.xticks(X, labels)
        plt.xlabel('BLOCK SIZE')
        plt.ylabel('ENCRYPTION_TIME(Seconds)')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        path1 = "./Results/Dataset_%s_ENCRYPTION_TIME.png" % (a + 1)
        plt.savefig(path1)
        plt.show()

        Eval = np.load('Decryption_time.npy', allow_pickle=True)[a]
        learnper = [1, 2, 3, 4, 5]
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(5)
        ax.bar(X + 0.00, Eval[:, 5], color='#aaff32', width=0.10, edgecolor='k', label=Encryp[0])
        ax.bar(X + 0.10, Eval[:, 6], color='#ad03de', width=0.10, edgecolor='k', label=Encryp[1])
        ax.bar(X + 0.20, Eval[:, 7], color='#8c564b', width=0.10, edgecolor='k', label=Encryp[2])
        ax.bar(X + 0.30, Eval[:, 8], color='#ff000d', width=0.10, edgecolor='k', label=Encryp[3])
        ax.bar(X + 0.40, Eval[:, 9], color='k', width=0.10, edgecolor='k', label=Encryp[4])
        # plt.xticks(X + 0.25, ('5', '10', '15', '20', '25'))
        labels = ['5', '10', '15', '20', '25']
        plt.xticks(X, labels)
        plt.xlabel('BLOCK SIZE')
        plt.ylabel('DECRYPTION Time (Seconds)')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        path = "./Results/Dataset %s DECRYPTION Time.png" % (a + 1)
        plt.savefig(path)
        plt.show()

        Eval = np.load('Total Consumption time.npy', allow_pickle=True)[a]
        learnper = [1, 2, 3, 4, 5]
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(5)
        ax.bar(X + 0.00, Eval[:, 5], color='#aaff32', width=0.10, edgecolor='k', label=Encryp[0])
        ax.bar(X + 0.10, Eval[:, 6], color='#ad03de', width=0.10, edgecolor='k', label=Encryp[1])
        ax.bar(X + 0.20, Eval[:, 7], color='#8c564b', width=0.10, edgecolor='k', label=Encryp[2])
        ax.bar(X + 0.30, Eval[:, 8], color='#ff000d', width=0.10, edgecolor='k', label=Encryp[3])
        ax.bar(X + 0.40, Eval[:, 9], color='k', width=0.10, edgecolor='k', label=Encryp[4])
        # plt.xticks(X + 0.25, ('5', '10', '15', '20', '25'))
        x = [1, 2, 3, 4, 5]
        labels = ['5', '10', '15', '20', '25']
        plt.xticks(X, labels)
        plt.xlabel('BLOCK SIZE')
        plt.ylabel('TOTAL COMPUTATION TIME(Seconds)')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        path1 = "./Results/Dataset_%s_COMPUTATION TIME.png" % (a + 1)
        plt.savefig(path1)
        plt.show()

        Eval = np.load('Memory Size.npy', allow_pickle=True)[a]
        learnper = [1, 2, 3, 4, 5]
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(5)
        ax.bar(X + 0.00, Eval[:, 5], color='#aaff32', edgecolor='k', width=0.10, label=Encryp[0])
        ax.bar(X + 0.10, Eval[:, 6], color='#ad03de', edgecolor='k', width=0.10, label=Encryp[1])
        ax.bar(X + 0.20, Eval[:, 7], color='#017b92', edgecolor='k', width=0.10, label=Encryp[2])
        ax.bar(X + 0.30, Eval[:, 8], color='#ff000d', edgecolor='k', width=0.10, label=Encryp[3])
        ax.bar(X + 0.40, Eval[:, 9], color='#0bf9ea', edgecolor='k', width=0.10, label=Encryp[4])
        # plt.xticks(X + 0.25, ('5', '10', '15', '20', '25'))
        x = [1, 2, 3, 4, 5]
        labels = ['5', '10', '15', '20', '25']
        plt.xticks(X, labels)
        plt.xlabel('BLOCK SIZE')
        plt.ylabel('MEMORY SIZE(KB)')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        path1 = "./Results/Dataset_%s_MEMORY_SIZE.png" % (a + 1)
        plt.savefig(path1)
        plt.show()


if __name__ == '__main__':
    plot_CPA_KPA()
    plot_KEY_RE()
    plot_ENC_DEC_CONS_MEM()
