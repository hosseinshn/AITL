import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import sklearn.preprocessing as sk
import seaborn as sns
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
import random
from random import randint
from sklearn.model_selection import StratifiedKFold
import itertools
from itertools import cycle
from torch.autograd import Function
import os
from plot_loss_acc import plot_learning_curve
from aitl import FX, MTL, GradReverse, Discriminator

#######################################################
#                 DRUG, SAVE, LOAD                    #          
#######################################################
## Note: For Paclitaxel PDX + Patient, use DRUG = 'Paclitaxel/PDXPatientCombined' ## 

DRUG = 'Docetaxel'
MAX_ITER = 2
MODE = "cleaning_up_aitl_test"
GPU = True

if GPU:
    device = "cuda"
else:
    device = "cpu"

SAVE_RESULTS_TO = "./results/" + DRUG + "/" + MODE + "/"
SAVE_TRACE_TO = "./results/" + DRUG + "/" + MODE + "/trace/"
TARGET_DIR = 'target_3_folds'
SOURCE_DIR = 'source_3_folds'
LOAD_DATA_FROM = '../../cancer-genomics-data-preprocessing/data/split/' + DRUG + '/stratified/'
torch.manual_seed(42)

dirName = SAVE_RESULTS_TO + 'model/'
if not os.path.exists(dirName):
    os.makedirs(dirName)
    print("Directory ", dirName, " Created ")
else:
    print("Directory ", dirName, " already exists")

dirName = SAVE_RESULTS_TO + 'test/'
if not os.path.exists(dirName):
    os.makedirs(dirName)
    print("Directory ", dirName, " Created ")
else:
    print("Directory ", dirName, " already exists")



#######################################################
#                    FUNCTIONS                        #          
#######################################################

def predict_label(XTestPatients, gen_model, map_model):    
    """ 
    Inputs: 
    :param XTestPatients - X_target_test 
    :param gen_model    - current FX model
    :param map_model    - current MTL model

    Output:
        - Predicted (binary) labels (Y for input target test data) 
    """ 

    gen_model.eval()
    gen_model.to(device)
    map_model.eval()
    map_model.to(device)

    XTestPatients = XTestPatients.to(device)
    F_xt_test = gen_model(XTestPatients)

    _, yhatt_test = map_model(None, F_xt_test)
    return yhatt_test
   
def evaluate_model(XTestPatients, YTestPatients, gen_model, map_model):
    """
    Inputs:
    :param XTestPatients - patient test data 
    :param YTestPatients - true class labels (binary) for patient test data 
    :param path_to_models - path to the saved models from training 

    Outputs:
        - test loss 
        - test accuracy (AUC)
    """
    XTestPatients = XTestPatients.to(device)
    YTestPatients = YTestPatients.to(device)
    y_predicted = predict_label(XTestPatients, gen_model, map_model)

    # #LOSSES
    C_loss_eval = torch.nn.BCELoss()
    closs_test = C_loss_eval(y_predicted, YTestPatients)

    if device == "cuda":
        YTestPatients = YTestPatients.to("cpu")
    yt_true_test = YTestPatients.view(-1,1)
    yt_true_test = yt_true_test.cpu()
    y_predicted = y_predicted.cpu()
    AUC_test = roc_auc_score(yt_true_test.detach().numpy(), y_predicted.detach().numpy())        
    return closs_test, AUC_test

def roc_auc_score_trainval(y_true, y_predicted):
    # To handle the case where we only have training samples of one class 
    # in our mini-batch when training since roc_auc_score 'breaks' when 
    # there is only one class present in y_true:
    # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.

    # The following code is taken from 
    # https://stackoverflow.com/questions/45139163/roc-auc-score-only-one-class-present-in-y-true?rq=1 # 
    if len(np.unique(y_true)) == 1: 
        return accuracy_score(y_true, np.rint(y_predicted))
    return roc_auc_score(y_true, y_predicted)




#######################################################
#                Hyper-Parameter Lists                #          
#######################################################


ls_splits = ['split1', 'split2', 'split3']
ls_ftsplits = ['ftsplit1', 'ftsplit2', 'ftsplit3']
ls_mb_size = [  {'mbS': 8, 'mbT': 16}, {'mbS': 8, 'mbT': 32}, \
                    {'mbS': 16, 'mbT': 8}, {'mbS': 16, 'mbT': 16}, {'mbS': 16, 'mbT': 32}, \
                    {'mbS': 32, 'mbT': 16}, {'mbS': 32, 'mbT': 32} ]
      
ls_h_dim = [1024, 512, 256, 128, 64, 32, 16]
ls_z_dim = [1024, 512, 256, 128, 64, 32, 16]
ls_lr = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
ls_epoch = [10, 15, 20, 25, 30, 35, 40, 45, 50]
ls_lam = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ls_dropout_gen = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
ls_dropout_mtl = ls_dropout_gen
ls_dropout_dg = ls_dropout_gen
ls_dropout_ds = ls_dropout_gen
ls_dropout_dr = ls_dropout_gen

skf_trainval = StratifiedKFold(n_splits=3, random_state=42)
skf_train = StratifiedKFold(n_splits=3, random_state=42)



#######################################################
#         AITL Model Training Starts Here             #          
#######################################################

for index, mbsize in enumerate(ls_mb_size):
    mbS = mbsize['mbS']
    mbT = mbsize['mbT']
    # random.seed(42)
    for iters in range(MAX_ITER):
        print("\n\n\nITERATION # {}".format(iters))
        print("-----------------------------\n\n")

        # Randomly selecting hyper-parameter values #
        hdm = random.choice(ls_h_dim)
        zdm = random.choice(ls_z_dim)
        lrs = random.choice(ls_lr)
        epch = random.choice(ls_epoch)   
        lambd1 = random.choice(ls_lam)   
        lambd2 = random.choice(ls_lam)
        drop_gen = random.choice(ls_dropout_gen)
        drop_mtl = random.choice(ls_dropout_mtl)
        drop_dg = random.choice(ls_dropout_dg)
        drop_ds = random.choice(ls_dropout_ds)
        drop_dr = random.choice(ls_dropout_dr)
        
        h_dim = hdm
        z_dim = zdm
        lr = lrs
        epoch = epch
        lam1 = lambd1
        lam2 = lambd2
        dropout_gen = drop_gen
        dropout_mtl = drop_gen
        dropout_dg = drop_gen
        dropout_ds = drop_gen
        dropout_dr = drop_gen


        print("-- Parameters used: --")
        print("h_dim: {}\nz_dim: {}\nlr: {}\nepoch: {}\nlambda1: {}\nlambda2: {}".format(h_dim,
                                                                                            z_dim,
                                                                                            lr,
                                                                                            epoch,
                                                                                            lam1,
                                                                                            lam2))
                                                                                        
        print("mbS: {}\nmbT: {}\ndropout_gen: {}\ndropout_mtl: {}\ndropout_dg: {}\ndropout_ds: {}\ndropout_dr: {}\n".format(mbS,
                                                                                                                            mbT,
                                                                                                                            dropout_gen,
                                                                                                                            dropout_mtl,
                                                                                                                            dropout_dg,
                                                                                                                            dropout_ds,
                                                                                                                            dropout_dr))

        batch_sizes = 'mbS' + str(mbS) + '_mbT' + str(mbT)
        test_results_name = 'hdim' + str(h_dim) + '_zdim' + str(z_dim) + '_lr' + str(lr) + '_epoch' + str(epoch) + '_lambda1' + str(lam1) + '_lambda2' + str(lam2) + '_dropouts' + str(dropout_gen) \
                            + '_' + str(dropout_mtl) + '_' + str(dropout_dg) + '_' + str(dropout_ds) + '_' + str(dropout_dr) +  '_mbS' + str(mbS) + '_mbT' + str(mbT) + '.tsv'
        test_results_dir = SAVE_RESULTS_TO + 'test/' + batch_sizes + '/'
        dirName = test_results_dir
        if not os.path.exists(dirName):
            os.makedirs(dirName)
            print("Directory ", dirName, " Created ")
        else:
            print("Directory ", dirName, " already exists")
        test_results_file = os.path.join(test_results_dir, test_results_name)
        if os.path.isfile(test_results_file):
            os.remove(test_results_file)   
        with open(test_results_file, 'a') as f: 
            f.write("-- Parameters --\n\n")
            f.write("h_dim: {}\nz_dim: {}\nlr: {}\nepoch: {}\nlambda1: {}\nlambda2: {}\nmbS: {}\nmbT: {}\n".format(h_dim,
                                                                                                                    z_dim,
                                                                                                                    lr,
                                                                                                                    epoch,
                                                                                                                    lam1,
                                                                                                                    lam2,
                                                                                                                    mbS,
                                                                                                                    mbT))
            f.write("dropout_gen: {}\ndropout_mtl: {}\ndropout_dg: {}\ndropout_ds: {}\ndropout_dr: {}\n\n".format(dropout_gen,
                                                                                                                    dropout_mtl,
                                                                                                                    dropout_dg,
                                                                                                                    dropout_ds,
                                                                                                                    dropout_dr))

        AUCtest_splits_total = []

        for split in ls_splits:		# for each split 
            print("\n\nReading data for {} ...\n".format(split))
            # Loading Source Data # 
            XTrainGDSC = pd.read_csv(LOAD_DATA_FROM + SOURCE_DIR + '/' + split + '/X_train_source.tsv',
                                        sep='\t', index_col=0, decimal='.')
            YTrainGDSC = pd.read_csv(LOAD_DATA_FROM + SOURCE_DIR + '/' + split + '/Y_logIC50train_source.tsv',
                                        sep='\t', index_col=0, decimal='.')
            XValGDSC = pd.read_csv(LOAD_DATA_FROM + SOURCE_DIR + '/' + split + '/X_val_source.tsv',
                                        sep='\t', index_col=0, decimal='.')
            YValGDSC = pd.read_csv(LOAD_DATA_FROM + SOURCE_DIR + '/' + split + '/Y_logIC50val_source.tsv',
                                        sep='\t', index_col=0, decimal='.')

            
            # Loading Target (Patient) Data # 
            XTestPatients = pd.read_csv(LOAD_DATA_FROM + TARGET_DIR + '/' + split + '/X_test_target.tsv',
                                            sep='\t', index_col=0, decimal='.')
            YTestPatients = pd.read_csv(LOAD_DATA_FROM + TARGET_DIR + '/' + split + '/Y_test_target.tsv',
                                            sep='\t', index_col=0, decimal='.')

            for ftsplit in ls_ftsplits:  # target train/val splits for finetuning 

                model_params = 'hdim' + str(h_dim) + '_zdim' + str(z_dim) + '_lr' + str(lr) + '_epoch' + str(epoch) + '_lamb1' + str(lam1) + '_lamb2' + str(lam2) \
                                + '_dropouts' + str(dropout_gen) + '_' + str(dropout_mtl) + '_' + str(dropout_dg) + '_' + str(dropout_ds) + '_' + str(dropout_dr) \
                                + '_mbS' + str(mbS) + '_mbT' + str(mbT)
                dirName = SAVE_TRACE_TO + batch_sizes + '/' + model_params + '/'
                if not os.path.exists(dirName):
                    os.makedirs(dirName)
                    print("Directory ", dirName, " Created ")
                else:
                    print("Directory ", dirName, " already exists")
                trace_file_tsv = os.path.join(dirName, split + '_' + ftsplit + '_trace.tsv')
                trace_file_txt = os.path.join(dirName, split + '_' + ftsplit + '_trace.txt')
                if os.path.isfile(trace_file_tsv):
                    os.remove(trace_file_tsv)   
                if os.path.isfile(trace_file_txt):
                    os.remove(trace_file_txt)   
                with open(trace_file_txt, 'a') as f: 
                    f.write("-- Parameters --\n\n")
                    f.write("h_dim: {}\nz_dim: {}\nlr: {}\nepoch: {}\nlambda1: {}\nlambda2: {}\nmbS: {}\nmbT: {}\n".format(h_dim,
                                                                                                                            z_dim,
                                                                                                                            lr,
                                                                                                                            epoch,
                                                                                                                            lam1,
                                                                                                                            lam2,
                                                                                                                            mbS,
                                                                                                                            mbT))
                    f.write("dropout_gen: {}\ndropout_mtl: {}\ndropout_dg: {}\ndropout_ds: {}\ndropout_dr: {}\n".format(dropout_gen,
                                                                                                                        dropout_mtl,
                                                                                                                        dropout_dg,
                                                                                                                        dropout_ds,
                                                                                                                        dropout_dr))

                with open(trace_file_tsv, 'a') as f: 
                    f.write("-- Parameters --\n\n")
                    f.write("h_dim: {}\nz_dim: {}\nlr: {}\nepoch: {}\nlambda1: {}\nlambda2: {}\nmbS: {}\nmbT: {}\n".format(h_dim,
                                                                                                                            z_dim,
                                                                                                                            lr,
                                                                                                                            epoch,
                                                                                                                            lam1,
                                                                                                                            lam2,
                                                                                                                            mbS,
                                                                                                                            mbT))
                    f.write("dropout_gen: {}\ndropout_mtl: {}\ndropout_dg: {}\ndropout_ds: {}\ndropout_dr: {}\n".format(dropout_gen,
                                                                                                                        dropout_mtl,
                                                                                                                        dropout_dg,
                                                                                                                        dropout_ds,
                                                                                                                        dropout_dr))
                    f.write("\n\n#\n")
                    # Dataframe header # 
                    f.write("epoch\ttrain_loss1\ttrain_loss2\ttrain_losstotal\ttrain_regloss\ttrain_closs\ttrain_DGloss\ttrain_DRloss\ttrain_DSloss\ttrain_AUC\ttrain_DGauc\ttrain_DRauc\ttrain_DSauc\tval_loss1\tval_loss2\tval_losstotal\tval_regloss\tval_closs\tval_DGloss\tval_DRloss\tval_DSloss\tval_AUC\tval_DGauc\tval_DRauc\tval_DSauc\n")

                print("\n\n-- Reading data for {} of {} ... --".format(ftsplit, split))
                XTrainPatients = pd.read_csv(LOAD_DATA_FROM + TARGET_DIR + '/' + split + '/' + ftsplit + '/X_train_ft_target.tsv',
                                            sep='\t', index_col=0, decimal='.')
                YTrainPatients = pd.read_csv(LOAD_DATA_FROM + TARGET_DIR + '/' + split + '/' + ftsplit + '/Y_train_ft_target.tsv',
                                            sep='\t', index_col=0, decimal='.')
                XValPatients = pd.read_csv(LOAD_DATA_FROM + TARGET_DIR + '/' + split + '/' + ftsplit + '/X_val_ft_target.tsv',
                                            sep='\t', index_col=0, decimal='.')
                YValPatients = pd.read_csv(LOAD_DATA_FROM + TARGET_DIR + '/' + split + '/' + ftsplit + '/Y_val_ft_target.tsv',
                                            sep='\t', index_col=0, decimal='.')
                print("Data successfully read!")
                
                # Temporarily combine Source training data and Target training data 
                # to fit standard scaler on gene expression of combined training data.
                # Then, apply fitted scaler to (and transform) Source validation, 
                # Target validation, and Target test (e.g. normalize validation and test
                # data of source and target with respect to source and target train)r
                XTrainCombined = pd.concat([XTrainGDSC, XTrainPatients])                              
                scalerTrain = sk.StandardScaler()
                scalerTrain.fit(XTrainCombined.values)
                XTrainGDSC_N = scalerTrain.transform(XTrainGDSC.values)
                XTrainPatients_N = scalerTrain.transform(XTrainPatients.values)
                XValGDSC_N = scalerTrain.transform(XValGDSC.values)
                XValPatients_N = scalerTrain.transform(XValPatients.values)
                XTestPatients_N = scalerTrain.transform(XTestPatients.values)

                TXValGDSC_N = torch.FloatTensor(XValGDSC_N)
                TXValPatients_N = torch.FloatTensor(XValPatients_N)
                TYValGDSC = torch.FloatTensor(YValGDSC.values)
                TYValPatients = torch.FloatTensor(YValPatients.values.astype(int))
                TYValPatients = TYValPatients.to(device)
                TXValGDSC_N = TXValGDSC_N.to(device)
                TXValPatients_N = TXValPatients_N.to(device)
                TYValGDSC = TYValGDSC.to(device)
                
                TXTestPatients_N = torch.FloatTensor(XTestPatients_N)
                TYTestPatients = torch.FloatTensor(YTestPatients.values.astype(int))
                TXTestPatients_N = TXTestPatients_N.to(device)
                TYTestPatients = TYTestPatients.to(device)

                class_sample_count = np.array([len(np.where(YTrainPatients.values==t)[0]) for t in np.unique(YTrainPatients.values)])
                print("\nclass_sample_count: {}\n".format(class_sample_count))
                weight = 1. / class_sample_count
                samples_weight = np.array([weight[t] for t in YTrainPatients.values])
                # samples_weight = np.array(weight[t] for t in YTrainPatients.values)

                samples_weight = torch.from_numpy(samples_weight)
                samples_weight = samples_weight.reshape(-1) # Flatten out the weights so it's a 1-D tensor of weights
                # print("\nsamples_weight: {}\n".format(samples_weight))
                sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)

                # Apply sampler on XTrainPatients_N
                # print("\n\nsampler:\n{}\n\n".format(list(sampler)))
                PDataset = torch.utils.data.TensorDataset(torch.FloatTensor(XTrainPatients_N), torch.FloatTensor(YTrainPatients.values.astype(int)))
                # print("PDataset: {}\n".format(PDataset.tensors))
                PLoader = torch.utils.data.DataLoader(dataset = PDataset, batch_size= mbT, shuffle=False, sampler = sampler)
                # PLoader = torch.utils.data.DataLoader(dataset = PDataset, batch_size= mbT, shuffle=False)


                CDataset = torch.utils.data.TensorDataset(torch.FloatTensor(XTrainGDSC_N), torch.FloatTensor(YTrainGDSC.values))
                CLoader = torch.utils.data.DataLoader(dataset = CDataset, batch_size= mbS, shuffle=True)

                n_sample, IE_dim = XTrainGDSC_N.shape

                AUCvals = []



                Gen = FX(dropout_gen, IE_dim, h_dim, z_dim)
                Map = MTL(dropout_mtl, h_dim, z_dim)
                DG = Discriminator(dropout_dg, h_dim, z_dim)
                DS = Discriminator(dropout_ds, h_dim, z_dim)
                DR = Discriminator(dropout_dr, h_dim, z_dim)
                Gen.to(device)
                Map.to(device)
                DG.to(device)
                DS.to(device)
                DR.to(device)

                optimizer_2 = torch.optim.Adagrad(itertools.chain(Gen.parameters(), Map.parameters(), DG.parameters(), DS.parameters(), DR.parameters()), 
                                                    lr = lr)

                C_loss = torch.nn.BCELoss()
                R_loss = torch.nn.MSELoss()

                l1 = []
                l2 = []
                regs = []
                classif = []
                aucs = []
                L = [] # total loss
                DG_losstr = []
                DR_losstr = []
                DS_losstr = []
                DG_auctr = []
                DR_auctr = []
                DS_auctr = []
            
                AUCtests = []
                Losstest = []

                for it in range(epoch):
                    epoch_cost1 = 0
                    epoch_cost2 = 0
                    epoch_cost1ls = []
                    epoch_cost2ls = []
                    epoch_auc = []
                    epoch_reg = 0
                    epoch_regls = []
                    epoch_classifls = []
                    epoch_DGloss = []
                    epoch_DRloss = []
                    epoch_DSloss = []
                    epoch_DGauc = []
                    epoch_DRauc = []
                    epoch_DSauc = []

                    AUCvals = []
                    loss2_vals = []
                    loss1_vals = []
                    totloss_vals = []
                    DG_lossval = []
                    DR_lossval = []
                    DS_lossval = []
                    reg_lossval = []
                    classif_lossval = []
                    DG_aucval = []
                    DR_aucval = []
                    DS_aucval = []

                    epoch_losstotal = 0 
                    epoch_loss = []

                    for i, data in enumerate(zip(CLoader, cycle(PLoader))):
                        DataS = data[0]
                        DataT = data[1]
                        ## Sending data to device = cuda/cpu
                        xs = DataS[0].to(device)
                        ys = DataS[1].view(-1,1).to(device)
                        xt = DataT[0].to(device)
                        yt = DataT[1].view(-1,1).to(device)


                        # Skip to next set of training batch if any of xs or xt has less
                        # than a certain threshold of training examples. Let such threshold 
                        # be 5 for now 
                        if xs.size()[0] < 5 or xt.size()[0] < 5: 
                            continue

                        Gen.train()
                        Map.train()
                        DG.train()
                        DS.train()
                        DR.train()         
                    
                        F_xs = Gen(xs)
                        F_xt = Gen(xt)  
                        yhat_xs, yhat_xt = Map(F_xs, F_xt)
                        _, yhat_xsB0 = Map(None, F_xs)

                        closs = C_loss(yhat_xt, yt)
                        rloss = R_loss(yhat_xs, ys)
                        loss1 = closs + rloss

                        t = torch.Tensor([torch.mean(yhat_xsB0)]).to(device) 
                        yhat_sxB = (yhat_xsB0 > t).float()

                        Labels = torch.ones(F_xs.size(0), 1)
                        Labelt = torch.zeros(F_xt.size(0), 1)
                        Lst = torch.cat([Labels, Labelt],0).to(device)
                        Xst = torch.cat([F_xs, F_xt], 0).to(device)   
                        Yst = torch.cat([yhat_sxB, yt],0).to(device)
                        try:
                            locR = (Yst==0).nonzero()[:, 0] # Proper way to obtain location indices 
                        except ValueError:
                            print("Error in 'locR = (Yst==0).nonzero()[:, 0]'")
                            print("(Yst==0).nonzero(): {}\n\n".format((Yst==0).nonzero()))
                        try:
                            locS = (Yst).nonzero()[:, 0]    # Proper way to obtain location indices
                        except ValueError:
                            print("Error in 'locS = (Yst).nonzero()[:, 0]'")
                            print("(Yst).nonzero(): {}\n\n".format((Yst).nonzero()))


                        XDS = Xst[locS].to(device)
                        LabDS = Lst[locS].to(device)
                        XDR = Xst[locR].to(device)
                        LabDR = Lst[locR].to(device)

                        yhat_DG = DG(Xst)
                        yhat_DS = DS(XDS)
                        yhat_DR = DR(XDR)
                        DG_loss = C_loss(yhat_DG, Lst)
                        DS_loss = C_loss(yhat_DS, LabDS)
                        DR_loss = C_loss(yhat_DR, LabDR)

                        loss2 = lam1*DG_loss + lam2*DS_loss + lam2*DR_loss
                        Loss = loss1 + loss2
                        optimizer_2.zero_grad()
                        Loss.backward()
                        optimizer_2.step()

                        epoch_cost1ls.append(loss1)
                        epoch_cost2ls.append(loss2)
                        epoch_regls.append(rloss)
                        epoch_classifls.append(closs)
                        epoch_loss.append(Loss)
                        epoch_DGloss.append(DG_loss)
                        epoch_DSloss.append(DS_loss)
                        epoch_DRloss.append(DR_loss)

                        y_true = yt.view(-1,1)
                        y_pred = yhat_xt
                        y_true = y_true.cpu()
                        y_pred = y_pred.cpu()
                        AUC = roc_auc_score_trainval(y_true.detach().numpy(), y_pred.detach().numpy()) 
                        epoch_auc.append(AUC)

                        y_trueDG = Lst.view(-1,1)
                        y_predDG = yhat_DG
                        y_trueDR = LabDR.view(-1,1)
                        y_predDR = yhat_DR
                        y_trueDS = LabDS.view(-1,1)
                        y_predDS = yhat_DS

                        y_trueDG = y_trueDG.cpu()
                        y_predDG = y_predDG.cpu()
                        y_trueDR = y_trueDR.cpu()
                        y_predDR = y_predDR.cpu()
                        y_trueDS = y_trueDS.cpu()
                        y_predDS = y_predDS.cpu()
                        AUCDG = roc_auc_score_trainval(y_trueDG.detach().numpy(), y_predDG.detach().numpy()) 
                        AUCDR = roc_auc_score_trainval(y_trueDR.detach().numpy(), y_predDR.detach().numpy()) 
                        AUCDS = roc_auc_score_trainval(y_trueDS.detach().numpy(), y_predDS.detach().numpy()) 
                        epoch_DGauc.append(AUCDG)
                        epoch_DRauc.append(AUCDR)
                        epoch_DSauc.append(AUCDS)

                    l1.append(torch.mean(torch.Tensor(epoch_cost1ls)))
                    l2.append(torch.mean(torch.Tensor(epoch_cost2ls)))
                    regs.append(torch.mean(torch.Tensor(epoch_regls)))
                    classif.append(torch.mean(torch.Tensor(epoch_classifls)))
                    aucs.append(np.mean(epoch_auc))
                    L.append(torch.mean(torch.FloatTensor(epoch_loss)))
                    DG_losstr.append(torch.mean(torch.Tensor(epoch_DGloss)))
                    DR_losstr.append(torch.mean(torch.Tensor(epoch_DRloss)))
                    DS_losstr.append(torch.mean(torch.Tensor(epoch_DSloss)))
                    DG_auctr.append(torch.mean(torch.Tensor(epoch_DGauc)))
                    DR_auctr.append(torch.mean(torch.Tensor(epoch_DRauc)))
                    DS_auctr.append(torch.mean(torch.Tensor(epoch_DSauc)))


                    with torch.no_grad():

                        Gen.eval()
                        Gen.to(device)
                        Map.eval()
                        Map.to(device)
                        DG.eval()
                        DG.to(device)
                        DS.eval()
                        DS.to(device)
                        DR.eval()    
                        DR.to(device)       

                        TXValGDSC_N = TXValGDSC_N.to(device)
                        TXValPatients_N = TXValPatients_N.to(device)
                        F_xs_val = Gen(TXValGDSC_N)
                        F_xt_val = Gen(TXValPatients_N)

                        yhats_val, yhatt_val = Map(F_xs_val, F_xt_val)
                        _, yhats_valB0 = Map(None, F_xs_val)

                        # Discriminators
                        t_val = torch.Tensor([torch.mean(yhats_valB0)]) 
                        t_val = t_val.to(device)
                        yhats_valB = (yhats_valB0 > t_val).float()

                        Labels_val = torch.ones(F_xs_val.size(0), 1)
                        Labelt_val = torch.zeros(F_xt_val.size(0), 1)
                        # print("\n\n-- Validation test-- \n\n")
                        # print("Labels_val: {}\n".format(Labels_val))
                        # print("Labelt_val: {}\n".format(Labelt_val))
                        Lst_val = torch.cat([Labels_val, Labelt_val],0)
                        Lst_val = Lst_val.to(device)
                        # print("Lst_val: {}\n".format(Lst_val))
                        Xst_val = torch.cat([F_xs_val, F_xt_val], 0)    
                        Yst_val = torch.cat([yhats_valB, TYValPatients],0)
                        locR_val = (Yst_val==0).nonzero()[:, 0]     # Proper way to obtain location indices is with '[:, 0]'
                        locS_val = (Yst_val).nonzero()[:, 0]        # Proper way to obtain location indices is with '[:, 0]'

                        XDS_val = Xst_val[locS_val]
                        LabDS_val = Lst_val[locS_val]
                        XDR_val = Xst_val[locR_val]
                        LabDR_val = Lst_val[locR_val]

                        yhat_DG_val = DG(Xst_val)
                        yhat_DS_val = DS(XDS_val)
                        yhat_DR_val = DR(XDR_val)

                        DG_loss_val = C_loss(yhat_DG_val, Lst_val)
                        DS_loss_val = C_loss(yhat_DS_val, LabDS_val)
                        DR_loss_val = C_loss(yhat_DR_val, LabDR_val)

                        loss2_val = lam1*DG_loss_val + lam2*DS_loss_val + lam2*DR_loss_val

                        #LOSSES
                        closs_val = C_loss(yhatt_val, TYValPatients)
                        rloss_val = R_loss(yhats_val, TYValGDSC)

                        loss1_val = closs_val + rloss_val

                        yt_true_val = TYValPatients.view(-1,1)
                        yt_true_val = yt_true_val.cpu()
                        yhatt_val = yhatt_val.cpu()
                        AUC_val = roc_auc_score_trainval(yt_true_val.detach().numpy(), yhatt_val.detach().numpy())        

                        AUCvals.append(AUC_val)
                        loss2_vals.append(loss2_val)
                        loss1_vals.append(loss1_val)
                        totloss_vals.append(loss1_val + loss2_val) # Total loss = loss 1 + loss 2
                        DG_lossval.append(DG_loss_val)
                        DS_lossval.append(DS_loss_val)
                        DR_lossval.append(DR_loss_val)
                        reg_lossval.append(rloss_val)
                        classif_lossval.append(closs_val)

                        y_trueDG_val = Lst_val.view(-1,1)
                        y_predDG_val = yhat_DG_val
                        y_trueDR_val = LabDR_val.view(-1,1)
                        y_predDR_val = yhat_DR_val
                        y_trueDS_val = LabDS_val.view(-1,1)
                        y_predDS_val = yhat_DS_val

                        y_trueDG_val = y_trueDG_val.cpu()
                        y_predDG_val = y_predDG_val.cpu()
                        y_trueDR_val = y_trueDR_val.cpu()
                        y_predDR_val = y_predDR_val.cpu()
                        y_trueDS_val = y_trueDS_val.cpu()
                        y_predDS_val = y_predDS_val.cpu()
                        AUCDG_val = roc_auc_score_trainval(y_trueDG_val.detach().numpy(), y_predDG_val.detach().numpy()) 
                        AUCDR_val = roc_auc_score_trainval(y_trueDR_val.detach().numpy(), y_predDR_val.detach().numpy()) 
                        AUCDS_val = roc_auc_score_trainval(y_trueDS_val.detach().numpy(), y_predDS_val.detach().numpy()) 
                        # print("AUC DG val: {}\n".format(AUCDG_val))
                        # print("AUC DR val: {}\n".format(AUCDR_val))
                        # print("AUC DS val: {}\n".format(AUCDS_val))
                        # print("y_predDG_val: {}\n".format(y_predDG_val))
                        # print("y_predDR_val: {}\n".format(y_predDR_val))
                        # print("y_predDS_val: {}\n".format(y_predDS_val))
                        DG_aucval.append(AUCDG_val)
                        DR_aucval.append(AUCDR_val)
                        DS_aucval.append(AUCDS_val)

                    # Take average across all training batches
                    loss1tr_mean = l1[it]
                    loss2tr_mean = l2[it]
                    rlosstr_mean = regs[it]
                    AUCtr_mean = aucs[it]
                    totlosstr_mean = L[it]
                    DRlosstr_mean = DR_losstr[it]
                    DGlosstr_mean = DG_losstr[it]
                    DSlosstr_mean = DS_losstr[it]
                    DGauctr_mean = DG_auctr[it]
                    DRauctr_mean = DR_auctr[it]
                    DSauctr_mean = DS_auctr[it]
                    closstr_mean = classif[it]


                    loss1val_mean = np.mean(np.array(loss1_vals))
                    loss2val_mean = np.mean(np.array(loss2_vals))
                    totlossval_mean = np.mean(np.array(totloss_vals))
                    AUCval_mean = np.mean(np.array(AUCvals))
                    DRlossval_mean = np.mean(np.array(DR_lossval))
                    DGlossval_mean = np.mean(np.array(DG_lossval))
                    DSlossval_mean = np.mean(np.array(DS_lossval))
                    reglossval_mean = np.mean(np.array(reg_lossval))
                    DGaucval_mean = np.mean(np.array(DG_aucval))
                    DRaucval_mean = np.mean(np.array(DR_aucval))
                    DSaucval_mean = np.mean(np.array(DS_aucval))
                    clossval_mean = np.mean(np.array(classif_lossval))
                    print("\n\nEpoch: {}".format(it))
                    print("(tr) loss1 mean: {}".format(loss1tr_mean))
                    print("(tr) loss2 mean: {}".format(loss2tr_mean))
                    print("(tr) total loss mean: {}".format(totlosstr_mean))
                    print("(tr) DG loss mean: {}".format(DGlosstr_mean))
                    print("(tr) DR loss mean: {}".format(DRlosstr_mean))
                    print("(tr) DS loss mean: {}".format(DSlosstr_mean))
                    print("(tr) AUC mean: {}".format(AUCtr_mean))
                    print("\n(val) loss1 mean: {}".format(loss1val_mean))
                    print("(val) loss2 mean: {}".format(loss2val_mean))
                    print("(val) total loss mean: {}".format(totlossval_mean))
                    print("(val) DG loss mean: {}".format(DGlossval_mean))
                    print("(val) DR loss mean: {}".format(DRlossval_mean))
                    print("(val) DS loss mean: {}".format(DSlossval_mean))
                    print("(val) AUC mean: {}".format(AUCval_mean))
                    # Write to file
                    # Take avg
                    with open(trace_file_txt, 'a') as f: 
                        f.write("\nepoch: {}\ttrain_loss1: {}\ttrain_loss2: {}\ttrain_losstotal: {}\ttrain_regloss: {}\ttrain_closs\ttrain_DGloss: {}\ttrain_DRloss: {}\ttrain_DSloss: {}\ttrain_AUC: {}\ttrain_DGauc: {}\ttrain_DRauc: {}\ttrain_DSauc: {}\n \
                                \tval_loss1: {}\tval_loss2: {}\tval_losstotal: {}\tval_regloss: {}\tval_closs\tval_DGloss: {}\tval_DRloss: {}\tval_DSloss: {}\tval_AUC: {}\tval_DGauc: {}\tval_DRauc: {}\tval_DSauc: {}\n".format(it,
                                                                                                                                                                                                                                loss1tr_mean,
                                                                                                                                                                                                                                loss2tr_mean,
                                                                                                                                                                                                                                totlosstr_mean,
                                                                                                                                                                                                                                rlosstr_mean,
                                                                                                                                                                                                                                closstr_mean,
                                                                                                                                                                                                                                DGlosstr_mean,
                                                                                                                                                                                                                                DRlosstr_mean,
                                                                                                                                                                                                                                DSlosstr_mean,
                                                                                                                                                                                                                                AUCtr_mean,
                                                                                                                                                                                                                                DGauctr_mean,
                                                                                                                                                                                                                                DRauctr_mean,
                                                                                                                                                                                                                                DSauctr_mean,
                                                                                                                                                                                                                                loss1val_mean,
                                                                                                                                                                                                                                loss2val_mean,
                                                                                                                                                                                                                                totlossval_mean,
                                                                                                                                                                                                                                reglossval_mean,
                                                                                                                                                                                                                                clossval_mean,
                                                                                                                                                                                                                                DGlossval_mean,
                                                                                                                                                                                                                                DRlossval_mean,
                                                                                                                                                                                                                                DSlossval_mean,
                                                                                                                                                                                                                                AUCval_mean,
                                                                                                                                                                                                                                DGaucval_mean,
                                                                                                                                                                                                                                DRaucval_mean,
                                                                                                                                                                                                                                DSaucval_mean))
                    with open(trace_file_tsv, 'a') as f: 
                        f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(it,
                                                                                                                                            loss1tr_mean,
                                                                                                                                            loss2tr_mean,
                                                                                                                                            totlosstr_mean,
                                                                                                                                            rlosstr_mean,
                                                                                                                                            closstr_mean,
                                                                                                                                            DGlosstr_mean,
                                                                                                                                            DRlosstr_mean,
                                                                                                                                            DSlosstr_mean,
                                                                                                                                            AUCtr_mean,
                                                                                                                                            DGauctr_mean,
                                                                                                                                            DRauctr_mean,
                                                                                                                                            DSauctr_mean,
                                                                                                                                            loss1val_mean,
                                                                                                                                            loss2val_mean,
                                                                                                                                            totlossval_mean,
                                                                                                                                            reglossval_mean,
                                                                                                                                            clossval_mean,
                                                                                                                                            DGlossval_mean,
                                                                                                                                            DRlossval_mean,
                                                                                                                                            DSlossval_mean,
                                                                                                                                            AUCval_mean,
                                                                                                                                            DGaucval_mean,
                                                                                                                                            DRaucval_mean,
                                                                                                                                            DSaucval_mean)) 
                                                                                        

                    # Save the current model # 
                    print("totlossval_mean: {}".format(totlossval_mean))
                    save_best_model_to = os.path.join(SAVE_RESULTS_TO + 'model/', split + '_' + ftsplit + '_best_model.pt')
                    print("==> saving current model (loss = {:0.6f}) ...".format(totlossval_mean))
                    ## Saving multiple models in one file (Feature extractors, mapper, discriminators) ## 
                    ##  https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-multiple-models-in-one-file ## 
                    torch.save({
                                'Gen_state_dict': Gen.state_dict(),
                                'Map_state_dict': Map.state_dict(),
                                'DG_state_dict': DG.state_dict(),
                                'DS_state_dict': DS.state_dict(),
                                'DR_state_dict': DR.state_dict(),
                                'optimizer_2_state_dict': optimizer_2.state_dict(),
                                }, save_best_model_to)

                
                ## Evaluate model ## 
                print("\n\n-- Evaluation -- \n\n")
                print("TXTestPatients_N shape: {}\n".format(TXTestPatients_N.size()))
                print("TYTestPatients shape: {}\n".format(TYTestPatients.size()))
                save_best_model_to = os.path.join(SAVE_RESULTS_TO + 'model/', split + '_' + ftsplit + '_best_model.pt')
                test_loss, test_auc = evaluate_model(TXTestPatients_N, TYTestPatients, Gen, Map)
                print("\n\n-- Test Results --\n\n")
                print("test loss: {}".format(test_loss))
                print("test auc: {}".format(test_auc))
                print("\n ----------------- \n\n\n")
                with open(test_results_file, 'a') as f: 
                    f.write("-- Split {} - ftsplit {} --\n".format(split, ftsplit))
                    f.write("Test loss: {}\t Test AUC: {}\n\n\n".format(test_loss, test_auc))
                AUCtest_splits_total.append(test_auc)

                
                ## Plot Learning Curves for 9 models trained using this specific param setting ## 
                print("Plotting learning curves ... ")
                plot_opts = {}
                plot_opts['model_params'] = 'hdim' + str(h_dim) + '_zdim' + str(z_dim) + '_lr' + str(lr) + '_epoch' + str(epoch) + '_lamb1' + str(lam1) + '_lamb2' + str(lam2) \
                                                + '_dropouts' + str(dropout_gen) + '_' + str(dropout_mtl) + '_' + str(dropout_dg) + '_' + str(dropout_ds) + '_' + str(dropout_dr) \
                                                + '_mbS' + str(mbS) + '_mbT' + str(mbT) 
                plot_opts['base_trace_dir'] = SAVE_TRACE_TO + batch_sizes + '/'
                plot_opts['split'] = split
                plot_opts['ftsplit'] = ftsplit
                plot_learning_curve(plot_opts)

            
        ## Calculate (held out) test set's avg AUC across different splits
        AUCtest_splits_total = np.array(AUCtest_splits_total)
        avgAUC = np.mean(AUCtest_splits_total)
        stdAUC = np.std(AUCtest_splits_total)
        with open(test_results_file, 'a') as f: 
            f.write("\n\n-- Average Test AUC --\n\n")
            f.write("Mean: {}\tStandard Deviation: {}\n".format(avgAUC, stdAUC))


