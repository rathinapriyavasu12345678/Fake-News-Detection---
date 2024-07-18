import numpy as np
from Evaluation import evaluation
from Global_Vars import Global_Vars
from Model_OMCN_LSTM import Model_OMCN_LSTM


def objfun_feat(Soln):
    Feat = Global_Vars.Feat
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            feat = Feat[:, sol]
            corr_coef = np.corrcoef(feat)
            Fitn = 1 / corr_coef  # Correlation Coefficient
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        feat = Feat[:, sol]
        corr_coef = np.corrcoef(feat)
        Fitn = 1 / corr_coef  # Correlation Coefficient
        return Fitn


def objfun_feat_1(Soln):
    Feat = Global_Vars.Feat
    Tar = Global_Vars.Target
    Feat_1 = Global_Vars.Feat_1
    Feat_2 = Global_Vars.Feat_2
    Feat_3 = Global_Vars.Feat_3
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            for j in range(Feat_1.shape[0]):
                F1 = Feat_1[j]
                F2 = Feat_2[0]
                F3 = Feat_3[j]
                w1 = Soln[27]
                w2 = Soln[28]
                Featu = w1 * F1[0:9] + (1 - w1) * F2[1, 0:9]
                Featur = w2 * Featu + (1 - w2) * F3[0:9]
                corr_coef = np.corrcoef(Featur)
                Fitn = 1 / corr_coef  # Correlation Coefficient
        return Fitn
    else:
        for j in range(Feat_1.shape[0]):
            F1 = Feat_1[j]
            F2 = Feat_2[0]
            F3 = Feat_3[j]
            w1 = Soln[27]
            w2 = Soln[28]
            Featu = w1 * F1[0:9] + (1 - w1) * F2[1, 0:9]
            Featur = w2 * Featu + (1 - w2) * F3[0:9]
            corr_coef = np.corrcoef(Featur)
            Fitn = 1 / corr_coef  # Correlation Coefficient
        return Fitn


def objfun_cls(Soln):
    Feat = Global_Vars.Feat
    Tar = Global_Vars.Target
    Tar = np.reshape(Tar, (-1, 1))
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
            Eval, pred = Model_OMCN_LSTM(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            predict = pred
            Eval = evaluation(predict, Test_Target)
            Fitn[i] = 1/(Eval[4] + Eval[7])
        return Fitn
    else:
        learnper = round(Feat.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = Feat[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = Feat[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, pred = Model_OMCN_LSTM(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        predict = pred
        Eval = evaluation(predict, Test_Target)
        Fitn = 1 / (Eval[4] + Eval[7])
        return Fitn
