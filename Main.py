import os
import numpy as np
import pandas as pd
from numpy import matlib
from AES import AES
from AVOA import AVOA
from DES import DES
from ElGamal_encryption import ElGamal_encryption
from Elgamal_RSA import ElGamal_RSA
from Global_vars import Global_vars
from KOA import KOA
from Model_Ada_MLSTM_AM import Model_Ada_MLSTM_AM
from Model_DBN import Model_DBN
from Model_LSTM import Model_LSTM
from Model_RNN import Model_RNN
from PROPOSED import PROPOSED
from RSA import RSA
from SOA import SOA
from WWO import WWO
from objfun import objfun

no_of_dataset = 2

def convert_to_numeric(df, columns):
    for col in columns:
        unique_values = df[col].unique()
        value_map = {val: idx for idx, val in enumerate(unique_values)}
        df[col] = df[col].map(value_map)
    return df


# Read the dataset
an = 0
if an == 1:
    Dataset = './Dataset/'
    Dataset_path = os.listdir(Dataset)
    for i in range(len(Dataset_path)):
        Data_file = Dataset + Dataset_path[i]
        if i == 0:  # './Dataset/diabetes.csv'
            print(i)
            Data = pd.read_csv(Data_file)
            Data.drop('Outcome', inplace=True, axis=1)
            data_1 = np.asarray(Data)
            targ = pd.read_csv(Data_file, usecols=['Outcome'])
            data_1 = data_1.astype('float')
            tar = np.asarray(targ).reshape(-1, 1)
            np.save('Datas_1.npy', data_1)
            np.save('Targets_1.npy', tar)

        elif i == 1:  # './Dataset/heart.csv'
            print(i)
            Data_2 = pd.read_csv(Data_file)
            Data_2.drop('target', inplace=True, axis=1)
            data_2 = np.asarray(Data_2)
            targ = pd.read_csv(Data_file, usecols=['target'])
            data_2 = data_2.astype('float')
            tar = np.asarray(targ).reshape(-1, 1)
            np.save('Datas_2.npy', data_2)
            np.save('Targets_2.npy', tar)

    num_of_node = 50
    Net = []
    Tar = []
    Node_name = np.random.randint(1, num_of_node, size=(num_of_node, 1))
    Node_Pos = np.random.uniform(0, num_of_node, size=(num_of_node, num_of_node))
    Topology = np.random.uniform(0, 1, size=(50, 50))
    Traversal = np.random.uniform(0, 1, size=(50, 50))
    Response_Time = np.random.randint(1, 100, size=(num_of_node, 1))
    Packet_Delivery_Rate = np.random.randint(1, 100, size=(num_of_node, 1))
    squared_dists = np.sum((Topology[:, None] - Traversal) ** 2, axis=2)
    distance_matrix = np.sqrt(squared_dists)
    Data = [Node_name, Node_Pos, Topology, Traversal, distance_matrix, Response_Time, Packet_Delivery_Rate]
    Target = np.random.uniform(0, 1, size=(num_of_node, 1))

    np.save('Init_Network.npy', Data)
    np.save('Target.npy', Target)


# Optimization for Authentication of User
an = 0
if an == 1:
    BestSOL = []
    FITNESS = []
    for n in range(no_of_dataset):
        Data = np.load('Datas_'+str(n + 1)+'.npy', allow_pickle=True)
        Feature = np.load('Init_Network.npy', allow_pickle=True)  # Load the Feature
        Target = np.load('Target.npy', allow_pickle=True)  # Load the Target
        Global_vars.Data = Data
        Global_vars.Feat = Feature
        Global_vars.Target = Target
        Npop = 10
        Chlen = 3  # hidden neuron count, Epoch, Step per epoch in Ada-MLSTM-AM
        xmin = matlib.repmat(np.asarray([5, 5, 50]), Npop, 1)
        xmax = matlib.repmat(np.asarray([255, 50, 250]), Npop, 1)
        fname = objfun
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("AVOA...")
        [bestfit1, fitness1, bestsol1, time1] = AVOA(initsol, fname, xmin, xmax, Max_iter)  # AVOA

        print("WWO...")
        [bestfit2, fitness2, bestsol2, time2] = WWO(initsol, fname, xmin, xmax, Max_iter)  # WWO

        print("KOA...")
        [bestfit3, fitness3, bestsol3, time3] = KOA(initsol, fname, xmin, xmax, Max_iter)  # KOA

        print("SOA...")
        [bestfit4, fitness4, bestsol4, time4] = SOA(initsol, fname, xmin, xmax, Max_iter)  # SOA

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

        BestSol_CLS = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(),
                       bestsol5.squeeze()]
        fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(),
                   fitness5.squeeze()]

        BestSOL.append(BestSol_CLS)
        FITNESS.append(fitness)
    np.save('Fitness.npy', np.asarray(FITNESS))
    np.save('BestSol.npy', np.asarray(BestSOL))  # Save the Bestsol


# KFOLD - Classification
an = 0
if an == 1:
    Eval_ALl = []
    for n in range(no_of_dataset):
        Data = np.load('Datas_'+str(n + 1)+'.npy', allow_pickle=True)
        BestSol = np.load('BestSol.npy', allow_pickle=True)
        Feat = np.load('Init_Network.npy', allow_pickle=True)
        Target = np.load('Target.npy', allow_pickle=True)
        K = 5
        Per = 1 / 5
        Perc = round(Feat.shape[0] * Per)
        eval = []
        for i in range(K):
            Eval = np.zeros((10, 14))
            Test_Data = Feat[i * Perc: ((i + 1) * Perc), :]
            Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
            test_index = np.arange(i * Perc, ((i + 1) * Perc))
            total_index = np.arange(Feat.shape[0])
            train_index = np.setdiff1d(total_index, test_index)
            Train_Data = Feat[train_index, :]
            Train_Target = Target[train_index, :]
            for j in range(BestSol.shape[0]):
                sol = BestSol[j, :]
                Eval[j, :], pred_0 = Model_Ada_MLSTM_AM(Train_Data, Train_Target, Test_Data, Test_Target, sol=sol)
            Eval[5, :], pred_1 = Model_RNN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[6, :], pred_2 = Model_DBN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[7, :], pred_3 = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[8, :], pred_4 = Model_Ada_MLSTM_AM(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[9, :] = Eval[4, :]
            eval.append(Eval)
        Eval_ALl.append(Eval_ALl)
    np.save('Eval_all_Fold.npy', np.asarray(Eval_ALl))


# Cryptography
an = 0
if an == 1:
    Data = np.load('Init_Network.npy', allow_pickle=True)
    EVAL = []
    NoOfBlocks = [5, 10, 15, 20, 25]
    for i in range(len(NoOfBlocks)):
        Eval = np.zeros((5, 4))
        Eval[0, :] = DES(Data, NoOfBlocks[i])
        Eval[1, :] = AES(Data, NoOfBlocks[i])
        Eval[2, :] = ElGamal_encryption(Data, NoOfBlocks[i])
        Eval[3, :] = RSA(Data, NoOfBlocks[i])
        Eval[4, :] = ElGamal_RSA(Data, NoOfBlocks[i])
        EVAL.append(Eval)
    Encryption_Time = EVAL[:, 0]
    Decryption_Time = EVAL[:, 1]
    Memory_size = EVAL[:, 2]
    Computational_time = EVAL[:, 3]
    np.save('Time_encryp.npy', Encryption_Time)
    np.save('Time_decryp.npy', Decryption_Time)
    np.save('Memory_size.npy', Memory_size)
    np.save('Total_comput_time.npy', Computational_time)

