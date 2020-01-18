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
from sklearn.metrics import roc_auc_score, average_precision_score
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

DRUG = 'Bortezomib'
max_iter = 1
MODE = "cleaning_up_aitl_test/best_models"
SAVE_RESULTS_TO = "./results/" + DRUG + "/" + MODE + "/"
SAVE_TRACE_TO = "./results/" + DRUG + "/" + MODE + "/trace/"
TARGET_DIR = 'target_3_folds'
SOURCE_DIR = 'source_3_folds'
LOAD_DATA_FROM = '../../cancer-genomics-data-preprocessing/data/split/' + DRUG + '/stratified/'
torch.manual_seed(42)


dirName = SAVE_RESULTS_TO + 'test/'
if not os.path.exists(dirName):
    os.makedirs(dirName)
    print("Directory ", dirName, " Created ")
else:
    print("Directory ", dirName, " already exists")


#######################################################
#                      Functions                      #          
#######################################################

def predict_label(XTestPatients, gen_model, map_model):    
    """ 
    Inputs: 
    :param XTestPatients - X_target_test 
    :param gen_model 
    :param map_model

    Output:
        - Predicted (binary) labels (Y for input target test data) 
    """ 

    gen_model.eval()
    map_model.eval()

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

    y_predicted = predict_label(XTestPatients, gen_model, map_model)

    # #LOSSES
    C_loss_eval = torch.nn.BCELoss()
    closs_test = C_loss_eval(y_predicted, YTestPatients)

    yt_true_test = YTestPatients.view(-1,1)
    AUC_test = roc_auc_score(yt_true_test.detach().numpy(), y_predicted.detach().numpy())  

    # Precision Recall
    APR_test = average_precision_score(yt_true_test.detach().numpy(), y_predicted.detach().numpy())     
    return closs_test, AUC_test, APR_test


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
ls_mb_size = [{'mbS': 16, 'mbT': 16}]      # Set the batch size for the best model manually
 

#######################################################
#         AITL Model Training Starts Here             #          
#######################################################

for index, mbsize in enumerate(ls_mb_size):
    mbS = mbsize['mbS']
    mbT = mbsize['mbT']

    ## Hard-code the values for testing and training 'best model' ## 
    h_dim = 1024
    z_dim = 1024
    lr = 0.0005
    epoch = 15
    lam1 = 0.2
    lam2 = 0.4
    dropout_gen = 0.4
    dropout_mtl = dropout_gen
    dropout_dg = dropout_gen
    dropout_ds = dropout_gen
    dropout_dr = dropout_gen

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
                        + '_' + str(dropout_mtl) + '_' + str(dropout_dg) + '_' + str(dropout_ds) + '_' + str(dropout_dr) + '_mbS' + str(mbS) + '_mbT' + str(mbT) + '.tsv'
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
    APRtest_splits_total = []

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

        # Combine train + val to form joint training set (no validation needed) # 
        XTrainGDSC = pd.concat([XTrainGDSC, XValGDSC])
        YTrainGDSC = pd.concat([YTrainGDSC, YValGDSC])

        # Loading Target (Patient) Data # 
        XTestPatients = pd.read_csv(LOAD_DATA_FROM + TARGET_DIR + '/' + split + '/X_test_target.tsv',
                                        sep='\t', index_col=0, decimal='.')
        YTestPatients = pd.read_csv(LOAD_DATA_FROM + TARGET_DIR + '/' + split + '/Y_test_target.tsv',
                                        sep='\t', index_col=0, decimal='.')
        XTrainPatients = pd.read_csv(LOAD_DATA_FROM + TARGET_DIR + '/' + split + '/X_train_target.tsv',
                                    sep='\t', index_col=0, decimal='.')
        YTrainPatients = pd.read_csv(LOAD_DATA_FROM + TARGET_DIR + '/' + split + '/Y_train_target.tsv',
                                    sep='\t', index_col=0, decimal='.')
        print("Data successfully read!")

    ####
        model_params = 'hdim' + str(h_dim) + '_zdim' + str(z_dim) + '_lr' + str(lr) + '_epoch' + str(epoch) + '_lamb1' + str(lam1) + '_lamb2' + str(lam2) \
                        + '_dropouts' + str(dropout_gen) + '_' + str(dropout_mtl) + '_' + str(dropout_dg) + '_' + str(dropout_ds) + '_' + str(dropout_dr) \
                        + '_mbS' + str(mbS) + '_mbT' + str(mbT)
        dirName = SAVE_TRACE_TO + batch_sizes + '/' + model_params + '/'
        if not os.path.exists(dirName):
            os.makedirs(dirName)
            print("Directory ", dirName, " Created ")
        else:
            print("Directory ", dirName, " already exists")
        trace_file_tsv = os.path.join(dirName, split + '_trace.tsv')
        trace_file_txt = os.path.join(dirName, split + '_trace.txt')
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
            # Dataframe header 
            f.write("epoch\ttrain_loss1\ttrain_loss2\ttrain_losstotal\ttrain_regloss\ttrain_closs\ttrain_DGloss\ttrain_DRloss\ttrain_DSloss\ttrain_AUC\ttrain_DGauc\ttrain_DRauc\ttrain_DSauc\n")

    ###

                
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
        XTestPatients_N = scalerTrain.transform(XTestPatients.values)
                
        TXTestPatients_N = torch.FloatTensor(XTestPatients_N)
        TYTestPatients = torch.FloatTensor(YTestPatients.values.astype(int))

        class_sample_count = np.array([len(np.where(YTrainPatients.values==t)[0]) for t in np.unique(YTrainPatients.values)])
        print("\nclass_sample_count: {}\n".format(class_sample_count))
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in YTrainPatients.values])

        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.reshape(-1) # Flatten out the weights so it's a 1-D tensor of weights
        # print("\nsamples_weight: {}\n".format(samples_weight))
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)

        # Apply sampler on XTrainPatients_N
        # print("\n\nsampler:\n{}\n\n".format(list(sampler)))
        PDataset = torch.utils.data.TensorDataset(torch.FloatTensor(XTrainPatients_N), torch.FloatTensor(YTrainPatients.values.astype(int)))
        # print("PDataset: {}\n".format(PDataset.tensors))
        PLoader = torch.utils.data.DataLoader(dataset = PDataset, batch_size= mbT, shuffle=False, sampler = sampler)

        CDataset = torch.utils.data.TensorDataset(torch.FloatTensor(XTrainGDSC_N), torch.FloatTensor(YTrainGDSC.values))
        CLoader = torch.utils.data.DataLoader(dataset = CDataset, batch_size= mbS, shuffle=True)

        n_sample, IE_dim = XTrainGDSC_N.shape

        AUCvals = []

        Gen = FX(dropout_gen, IE_dim, h_dim, z_dim)
        Map = MTL(dropout_mtl, h_dim, z_dim)
        DG = Discriminator(dropout_dg, h_dim, z_dim)
        DS = Discriminator(dropout_ds, h_dim, z_dim)
        DR = Discriminator(dropout_dr, h_dim, z_dim)

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


            epoch_losstotal = 0
            epoch_loss = [] 

            for i, data in enumerate(zip(CLoader, cycle(PLoader))):
                DataS = data[0]
                DataT = data[1]
                xs = DataS[0]
                ys = DataS[1].view(-1,1)
                xt = DataT[0]
                yt = DataT[1].view(-1,1)

                # Skip to next set of training batch if any of xs or xt has less
                # than a certain threshold of training examples. 
                # Let such threshold be 5 
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

                t = torch.Tensor([torch.mean(yhat_xsB0)]) 
                yhat_sxB = (yhat_xsB0 > t).float()

                Labels = torch.ones(F_xs.size(0), 1)
                Labelt = torch.zeros(F_xt.size(0), 1)
                Lst = torch.cat([Labels, Labelt],0)
                Xst = torch.cat([F_xs, F_xt], 0)    
                Yst = torch.cat([yhat_sxB, yt],0)
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


                XDS = Xst[locS]
                LabDS = Lst[locS]
                XDR = Xst[locR]
                LabDR = Lst[locR]

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
                AUC = roc_auc_score_trainval(y_true.detach().numpy(), y_pred.detach().numpy()) 
                epoch_auc.append(AUC)

                y_trueDG = Lst.view(-1,1)
                y_predDG = yhat_DG
                y_trueDR = LabDR.view(-1,1)
                y_predDR = yhat_DR
                y_trueDS = LabDS.view(-1,1)
                y_predDS = yhat_DS
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

            print("\n\nEpoch: {}".format(it))
            print("(tr) loss1 mean: {}".format(loss1tr_mean))
            print("(tr) loss2 mean: {}".format(loss2tr_mean))
            print("(tr) total loss mean: {}".format(totlosstr_mean))
            print("(tr) DG loss mean: {}".format(DGlosstr_mean))
            print("(tr) DR loss mean: {}".format(DRlosstr_mean))
            print("(tr) DS loss mean: {}".format(DSlosstr_mean))
            print("(tr) AUC mean: {}".format(AUCtr_mean))
            # Write to file
            # Take avg
            with open(trace_file_txt, 'a') as f: 
                f.write("\nepoch: {}\ttrain_loss1: {}\ttrain_loss2: {}\ttrain_losstotal: {}\ttrain_regloss: {}\ttrain_closs\ttrain_DGloss: {}\ttrain_DRloss: {}\ttrain_DSloss: {}\ttrain_AUC: {}\ttrain_DGauc: {}\ttrain_DRauc: {}\ttrain_DSauc: {}\n".format(it,
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
                                                                                DSauctr_mean))
            with open(trace_file_tsv, 'a') as f: 
                f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(it,
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
                                                                                DSauctr_mean)) 
                                                                                            
            save_model_to = SAVE_TRACE_TO + batch_sizes + '/' + model_params + '/model/'
            if not os.path.exists(save_model_to):
                os.makedirs(save_model_to)
                print("Directory ", save_model_to, " Created ")
#             else:
#                 print("Directory ", save_model_to, " already exists")
            save_best_model_to = os.path.join(save_model_to, split + '_best_model.pt')
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
        save_best_model_to = os.path.join(save_model_to, split + '_best_model.pt')
        test_loss, test_auc, test_apr = evaluate_model(TXTestPatients_N, TYTestPatients, Gen, Map)
        print("\n\n-- Test Results --\n\n")
        print("test loss: {}".format(test_loss))
        print("test auc: {}".format(test_auc))
        print("\n ----------------- \n\n\n")
        with open(test_results_file, 'a') as f: 
            f.write("-- Split {} --\n".format(split))
            f.write("Test loss: {}\t Test AUC: {}\t Test APR: {}\t\n\n\n".format(test_loss, test_auc, test_apr))
        AUCtest_splits_total.append(test_auc)
        APRtest_splits_total.append(test_apr)

            
    ## Calculate Test set's avg AUC across different splits
    AUCtest_splits_total = np.array(AUCtest_splits_total)
    APRtest_splits_total = np.array(APRtest_splits_total)
    avgAUC = np.mean(AUCtest_splits_total)
    stdAUC = np.std(AUCtest_splits_total)
    avgAPR = np.mean(APRtest_splits_total)
    stdAPR = np.std(APRtest_splits_total)
    with open(test_results_file, 'a') as f: 
        f.write("\n\n-- Average Test AUC --\n\n")
        f.write("Mean: {}\tStandard Deviation: {}\n".format(avgAUC, stdAUC))
        f.write("\n-- Average Test Precision Recall --\n\n")
        f.write("Mean: {}\tStandard Deviation: {}\n".format(avgAPR, stdAPR))


