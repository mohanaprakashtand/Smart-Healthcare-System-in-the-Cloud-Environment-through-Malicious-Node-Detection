import numpy as np
from Classificaltion_Evaluation import ClassificationEvaluation
from Global_vars import Global_vars
from Model_Ada_MLSTM_AM import Model_Ada_MLSTM_AM


def objfun(Soln):
    Feat = Global_vars.Feat
    Tar = Global_vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(Feat.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = Feat[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = Feat[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, pred = Model_Ada_MLSTM_AM(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval = ClassificationEvaluation(Test_Target, pred)
            Fitn[i] = 1 / (Eval[4] + Eval[10]) + Eval[8]  # 1 / (Accuracy + NPV) + FPR
        return Fitn
    else:
        learnper = round(Feat.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = Feat[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = Feat[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, pred = Model_Ada_MLSTM_AM(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Eval = ClassificationEvaluation(Test_Target, pred)
        Fitn[i] = 1 / (Eval[4] + Eval[10]) + Eval[8]  # 1 / (Accuracy + NPV) + FPR
        return Fitn


