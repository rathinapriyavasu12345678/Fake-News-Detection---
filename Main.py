import numpy as np
import pandas as pd
from numpy import matlib
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
import random as rn
from Model_OMCN_LSTM import Model_OMCN_LSTM
from Objective_Function import objfun_feat, objfun_cls, objfun_feat_1
from ARO import ARO
from DHOA import DHOA
from Global_Vars import Global_Vars
from HHO import HHO
from Model_CNN import Model_CNN
from Model_GRU import Model_GRU
from Model_LSTM import Model_LSTM
from Plot_Results import plot_results, plot_results_Abliation, plot_results_conv
from Proposed import PROPOSED
from SSA import SSA
from Tfidf import TF_IDF
import warnings
from tensorflow.keras.preprocessing.text import Tokenizer
from embedding4bert import Embedding4BERT

warnings.filterwarnings(action='ignore')

no_of_dataset = 3


# Removing puctuations
def rem_punct(my_str):
    # define punctuation
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~â€™â€˜'''
    # remove punctuation from the string
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct = no_punct + char + " "
    # display the unpunctuated string
    return no_punct


def embedding_for_vocab(filepath, word_index,
                        embedding_dim):
    vocab_size = len(word_index) + 1
    # Adding again 1 because of reserved 0 index
    embedding_matrix_vocab = np.zeros((vocab_size, embedding_dim))
    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix_vocab[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix_vocab


#### Read the dataset 1 #############
an = 0
if an == 1:
    Data1 = np.asarray(pd.read_excel('./Dataset/gossipcop_fake.xlsx'))  # Path of the dataset 1
    Data2 = np.asarray(pd.read_excel('./Dataset/gossipcop_real.xlsx'))  # Path of the Target 1
    Data = np.concatenate((Data1, Data2), axis=0)  # Data1 - 5334, Data2 - 16817
    Target = np.concatenate((np.zeros((len(Data1), 1)), (np.ones((len(Data2), 1)))), axis=0)
    np.save('Data_1.npy', Data.ravel())  # Save the Dataset 1
    np.save('Target_1.npy', Target)  # Save the Target 1

#### Read the dataset 2 #############
an = 0
if an == 1:
    Data1 = np.asarray(pd.read_excel('./Dataset/politifact_fake.xlsx'))  # Path of the dataset 2
    Data2 = np.asarray(pd.read_excel('./Dataset/politifact_real.xlsx'))  # Path of the target 2
    Data = np.concatenate((Data1, Data2), axis=0)  # Data1 - 472, Data2 - 795
    Target = np.concatenate((np.zeros((len(Data1), 1)), (np.ones((len(Data2), 1)))), axis=0)
    np.save('Data_2.npy', Data.ravel())  # Save the Dataset 2
    np.save('Target_2.npy', Target)  # Save the Target 2

#### Read the dataset 3 #############
an = 0
if an == 1:
    Data1 = np.asarray(pd.read_excel('./Dataset/train_stances.xlsx'))  # (Data1 - 49971) Path of the dataset 3
    tar = np.zeros((len(Data1)))
    for i in range(len(Data1)):
        if Data1[i, 1] == 'agree':
            tar[i] = 1
        else:
            tar[i] = 0
    np.save('Data_3.npy', Data1.ravel())  # Save the Dataset 3
    np.save('Target_3.npy', tar)  # Save the target 3

# Pre-Processing
an = 0
if an == 1:
    ps = PorterStemmer()
    for n in range(no_of_dataset):
        data = np.load('Data_' + str(n + 1) + '.npy', allow_pickle=True)  # Load Datas
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)  # Load Targets
        # data = data.decode('utf-8')
        Prep = []
        for k in range(len(data)):
            print(n, k)
            if isinstance(data[k], str):
                text_tokens = word_tokenize(data[k])
                # convert it in to tokens
                stem = []
                for w in text_tokens:  # Stemming with Lower case conversion
                    stem_tokens = ps.stem(w)
                    stem.append(stem_tokens)
                words = [word for word in stem if
                         not word in stopwords.words()]  # tokens without stop words
                pun = rem_punct(words)  # remove punctuations
                Prep.append(pun)
        np.save('Pre_Processing_' + str(n + 1) + '.npy', Prep)  # Save the Preprocessing data

#################### Feature extraction ##################
### GLove Embedding ###
an = 0
if an == 1:
    glove = []
    for n in range(no_of_dataset):
        x = np.load('Pre_Processing_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Preprocessing data
        for i in range(x.shape[0]):
            print(n, i, x.shape[0])
            # code for Glove word embedding
            # create the dict.
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(x[i])
            # number of unique words in dict.
            len_t_index = len(tokenizer.word_index)
            t_index = tokenizer.word_index
            embedding_dim = 50  # Four varieties are: 50d, 100d, 200d and 300d.
            embedding_matrix_vocab = embedding_for_vocab('./glove.6B.50d.txt', tokenizer.word_index, embedding_dim)
            embed = embedding_matrix_vocab[1]
        glove.append(embed)
        np.save('Glove_' + str(n + 1) + '.npy', glove)  # Save the Glove Data

### Bidirectional Encoder Representations from Transformers (BERT) ###
an = 0
if an == 1:
    for n in range(no_of_dataset):
        prep = np.load('Pre_Processing_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Preprocessing Data
        BERT = []
        for i in range(prep.shape[0]):
            print(n, i)
            emb4bert = Embedding4BERT("bert-base-cased")  # bert-base-uncased
            tokens, embeddings = emb4bert.extract_word_embeddings(prep[i])
            BERT.append(embeddings)
        np.save('BERT_' + str(n + 1) + '.npy', BERT)  # Save the BERT data

######## TFIDF ########
an = 0
if an == 1:
    for n in range(no_of_dataset):
        prep = np.load('Pre_Processing_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Preprocessing data
        TFIDF = []
        for i in range(prep.shape[0]):  # For all datasets
            print(n, i)
            tfidf = TF_IDF(prep[i])
            TFIDF.append(tfidf)
        np.save('TFIDF_' + str(n + 1) + '.npy', TFIDF)  ## Save the TFIDF

# Optimization for Feature Selection And Weight Optimization
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Feat = np.load('Pre_Processing_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Segmented_Image
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Target Image

        Feat_1 = np.load('Glove_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Feat 1
        Feat_2 = np.load('BERT_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Feat 2
        Feat_3 = np.load('TFIDF_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Feat 3

        Global_Vars.Feat = Feat
        Global_Vars.Target = Target
        Global_Vars.Feat_1 = Feat_1
        Global_Vars.Feat_2 = Feat_2
        Global_Vars.Feat_3 = Feat_3

        Npop = 10
        Chlen = 29  # 9+9+9 = 27 for feature and 2 for weights
        xmin = matlib.repmat(np.append(np.ones(27), np.append(0.01, 0.01), axis=0), Npop, 1)
        xmax = matlib.repmat(np.append(Feat.shape[0] - 1 * np.ones(27), np.append(0.99, 0.99), axis=0), Npop, 1)
        fname = objfun_feat_1
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = rn.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 25

        print("HHO...")
        [bestfit1, fitness1, bestsol1, time1] = HHO(initsol, fname, xmin, xmax, Max_iter)  # HHO

        print("DHOA...")
        [bestfit2, fitness2, bestsol2, time2] = DHOA(initsol, fname, xmin, xmax, Max_iter)  # DHOA

        print("SSA...")
        [bestfit4, fitness4, bestsol4, time3] = SSA(initsol, fname, xmin, xmax, Max_iter)  # SSA

        print("ARO...")
        [bestfit3, fitness3, bestsol3, time4] = ARO(initsol, fname, xmin, xmax, Max_iter)  # ARO

        print("Proposed...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # Improved ARO
        BestSol_feat = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        np.save('BestSol_FEAT_' + str(n + 1) + '.npy', BestSol_feat)  # Save the BestSol_FEAT


## feature Concatenation for Feature Fusion
an = 0
if an == 1:
    for i in range(no_of_dataset):
        Feat_1 = np.load('Glove_' + str(i + 1) + '.npy', allow_pickle=True)  # Load the Feat 1
        Feat_2 = np.load('BERT_' + str(i + 1) + '.npy', allow_pickle=True)  # Load the Feat 2
        Feat_3 = np.load('TFIDF_' + str(i + 1) + '.npy', allow_pickle=True)  # Load the Feat 3
        bests = np.load('BestSol_FEAT_' + str(i + 1) + '.npy', allow_pickle=True)  # Load the Feat 4
        Feature = []
        for j in range(Feat_1.shape[0]):
            F1 = Feat_1[j]
            F2 = Feat_2[j]
            F3 = Feat_3[j]
            for n in range(len(F1)):
                for k in range(bests.shape[0]):
                    print(i, j, k)
                    w1 = bests[k, 27]
                    w2 = bests[k, 28]
                    Feat = w1 * F1[0:9] + (1 - w1) * F2[k, 0:9]
                    Featur = w2 * Feat + (1 - w2) * F3[0:9]
            Feature.append(Featur)
            np.save('Feature_' + str(i + 1) + '.npy', Feature)  # Save the Feature

# optimization for Classification
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Feat = np.load('Feature_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Selected features
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)[:Feat.shape[0]]  # Load the Target
        Global_Vars.Feat = Feat
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 4  # 1 for activation of OMCN_LSTM, 1 for optimizer of OMCN_LSTM, 1 for epoch of OMCN_LSTM,
        # 1 for leaning rate of OMCN_LSTM
        xmin = matlib.repmat([0, 0, 50, 0.01], Npop, 1)
        xmax = matlib.repmat([3, 4, 100, 0.99], Npop, 1)
        fname = objfun_cls
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = rn.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 10

        print("HHO...")
        [bestfit1, fitness1, bestsol1, time1] = HHO(initsol, fname, xmin, xmax, Max_iter)  # HHO

        print("DHOA...")
        [bestfit2, fitness2, bestsol2, time2] = DHOA(initsol, fname, xmin, xmax, Max_iter)  # DHOA

        print("SSA...")
        [bestfit4, fitness4, bestsol4, time3] = SSA(initsol, fname, xmin, xmax, Max_iter)  # SSA

        print("ARO...")
        [bestfit3, fitness3, bestsol3, time4] = ARO(initsol, fname, xmin, xmax, Max_iter)  # ARO

        print("Proposed...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # Improved ARO

        BestSol_CLS = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        np.save('BestSol_CLS_' + str(n + 1) + '.npy', BestSol_CLS)  # Bestsol classification

# Classification
an = 0
if an == 1:
    Eval_all = []
    for n in range(no_of_dataset):
        Feature = np.load('Feature_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Selected features
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)[:Feature.shape[0]] # Load the Target
        BestSol = np.load('BestSol_CLS_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Bestsol Classification
        Feat = Feature
        EVAL = []
        Learnper = [0.35, 0.45, 0.55, 0.65, 0.754, 0.85]
        for learn in range(len(Learnper)):
            learnperc = round(Feat.shape[0] * Learnper[learn])
            Train_Data = Feat[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Feat[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            Eval = np.zeros((10, 14))
            for j in range(BestSol.shape[0]):
                print(learn, j)
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Eval[j, :], pred0 = Model_OMCN_LSTM(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval[5, :], pred1 = Model_GRU(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[6, :], pred2 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[7, :], pred3 = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[8, :], pred4 = Model_OMCN_LSTM(Train_Data, Train_Target, Test_Data, Test_Target, [0, 0, 50, 0.01])
            Eval[9, :], pred5 = Eval[4, :]
            EVAL.append(Eval)
        Eval_all.append(EVAL)
        np.save('Eval_all.npy', Eval_all)  # Save the Eval all

plot_results()
plot_results_conv()
plot_results_Abliation()