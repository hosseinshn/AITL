import os.path
import math
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


import pandas as pd

DRUG = "Paclitaxel"  # Set to the name of the drug being split
MODEL = 'AITL'
PDX_PATIENT_COMBINED = True

if PDX_PATIENT_COMBINED:
    SAVE_RESULT_TO = './data/split/' + DRUG + '/' + MODEL + '/PDXPatientCombined/R_S_split/'
    SAVE_RESULT_TO_STRAT = './data/split/' + DRUG + '/' + MODEL + '/PDXPatientCombined/stratified/'
    TARGET_DIR = 'target_3_folds'
    SOURCE_DIR = 'source_3_folds'
    LOAD_TARGET_TRAIN_FROM = './data/split/' + DRUG + '/' + MODEL + '/PDXPatientCombined/R_S_split/'+TARGET_DIR
else: 
    SAVE_RESULT_TO = './data/split/' + DRUG + '/' + MODEL + '/R_S_split/'
    SAVE_RESULT_TO_STRAT = './data/split/' + DRUG + '/' + MODEL +'/stratified/'
    TARGET_DIR = 'target_3_folds'
    SOURCE_DIR = 'source_3_folds'
    LOAD_TARGET_TRAIN_FROM = './data/split/' + DRUG + '/' + MODEL + '/R_S_split/'+TARGET_DIR


###########################################################################################
## For AITL only ## 
# NOTE: leave the first two options as 'False' and keep 'STRATIFIED_AITL_SPLITS' as 'True'
#        when splitting the data for AITL
if MODEL == 'AITL':
    STRATIFIED_AITL_SPLITS = True
else:
    STRATIFIED_AITL_SPLITS = False

STRAT_SPLITS_TYPE = "traintest" # either 'traintest' or 'trainvaltest'

FINETUNE = True
NSPLITS = 3
## End of for AITL only ## 
###########################################################################################

if PDX_PATIENT_COMBINED:
    # Source data #
    source_exprs_resp_z = pd.read_csv('./data/split/'+DRUG + '/' + MODEL +'/PDXPatientCombined/Source_exprs_resp_z.'+DRUG+'.tsv',
                                sep='\t', index_col=0)

    # Target data #
    target_combined_exprs_resp_z = pd.read_csv('./data/split/'+DRUG + '/' + MODEL +'/PDXPatientCombined/Target_combined_expr_resp_z.'+DRUG+'.tsv',
                                            sep='\t', index_col=0)
else:
    # Source data #
    source_exprs_resp_z = pd.read_csv('./data/split/'+ DRUG + '/' + MODEL + '/Source_exprs_resp_z.'+DRUG+'.tsv',
                                sep='\t', index_col=0)

    # Target data #
    target_combined_exprs_resp_z = pd.read_csv('./data/split/'+ DRUG + '/' + MODEL +'/Target_combined_expr_resp_z.'+DRUG+'.tsv',
                                            sep='\t', index_col=0)



kf = KFold(n_splits=3, random_state=42, shuffle=True) # Define the split - into 2 folds



def create_splits_stratified(orig_df, skf, splits_dict, numfolds, dataset, splitstype="traintest"):
    """
    == For creating stratified splits for datasets ==
    :param orig_df - original dataframe (dataset) to be split
    :param skf - stratified kfold split function
    :param splits_dict - dictionary that holds the splits
    :param numfolds - number of folds
    :param dataset - source or target
    :param splitstype - indicates what kind of splits to perform (i.e. traintest or trainvaltest)
    """


    
    if dataset == "target":
        x_expression = orig_df.iloc[:, 1:]  # gene expressions (features)
        y_response = orig_df.iloc[:, 0]  # binary class labels (0 or 1)
        splitstypeds = ["traintest", "train", "test"]
        counter = 1
        for train_index, test_index in skf.split(x_expression, y_response):
            x_train, x_test = x_expression.iloc[train_index], x_expression.iloc[test_index]
            y_train, y_test = y_response.iloc[train_index], y_response.iloc[test_index]
            splits_dict["split" + str(counter)] = {}
            splits_dict["split" + str(counter)][splitstypeds[0]] = {}
            splits_dict["split" + str(counter)][splitstypeds[0]][splitstypeds[1]] = {}
            splits_dict["split" + str(counter)][splitstypeds[0]][splitstypeds[1]]["X"] = x_train
            splits_dict["split" + str(counter)][splitstypeds[0]][splitstypeds[1]]["Y"] = y_train
            splits_dict["split" + str(counter)][splitstypeds[0]][splitstypeds[2]] = {}
            splits_dict["split" + str(counter)][splitstypeds[0]][splitstypeds[2]]["X"] = x_test
            splits_dict["split" + str(counter)][splitstypeds[0]][splitstypeds[2]]["Y"] = y_test
            counter += 1

    elif dataset == "source":
        x_expression = orig_df.iloc[:, 2:]  # gene expressions (features) 
        y_logIC50 = orig_df.iloc[:, 1] # col index 1 of the source df is logIC50
        y_response = orig_df.iloc[:, 0]  # binary class labels (0 or 1)
        print("# of class 0 examples in original (unsplit) source data: {}".format(len(y_response[y_response == 0])))
        print("# of class 1 examples in original (unsplit) source data: {}".format(len(y_response[y_response == 1])))
        splitstypeds = ["train", "val"]
        counter = 1
        for train_index, val_index in skf.split(x_expression, y_response):
            x_train, x_val = x_expression.iloc[train_index], x_expression.iloc[val_index]
            y_trainB, y_valB = y_response.iloc[train_index], y_response.iloc[val_index]
            y_trainIC50, y_valIC50 = y_logIC50.iloc[train_index], y_logIC50.iloc[val_index]
            splits_dict["split" + str(counter)] = {}
            splits_dict["split" + str(counter)][splitstypeds[0]] = {}
            splits_dict["split" + str(counter)][splitstypeds[0]]["X"] = x_train
            splits_dict["split" + str(counter)][splitstypeds[0]]["Y_response"] = y_trainB
            splits_dict["split" + str(counter)][splitstypeds[0]]["Y_logIC50"] = y_trainIC50           
            splits_dict["split" + str(counter)][splitstypeds[1]] = {}
            splits_dict["split" + str(counter)][splitstypeds[1]]["X"] = x_val
            splits_dict["split" + str(counter)][splitstypeds[1]]["Y_response"] = y_valB
            splits_dict["split" + str(counter)][splitstypeds[1]]["Y_logIC50"] = y_valIC50
            counter += 1




    if splitstype == "trainvaltest":
        # further split train into train and val

        for split in splits_dict:
            x_expression = splits_dict[split]["traintest"]["train"]["X"]
            y_response = splits_dict[split]["traintest"]["train"]["Y"]
            counter = 1
            splits_dict[split][splitstype] = {}
            for train_index, val_index in skf.split(x_expression, y_response):
                x_train, x_val = x_expression.iloc[train_index], x_expression.iloc[val_index]
                y_train, y_val = y_response.iloc[train_index], y_response.iloc[val_index]
                splits_dict[split][splitstype]["ftsplit" + str(counter)] = {}
                splits_dict[split][splitstype]["ftsplit" + str(counter)]["train"] = {}
                splits_dict[split][splitstype]["ftsplit" + str(counter)]["train"]["X"] = x_train
                splits_dict[split][splitstype]["ftsplit" + str(counter)]["train"]["Y"] = y_train
                splits_dict[split][splitstype]["ftsplit" + str(counter)]["val"] = {}
                splits_dict[split][splitstype]["ftsplit" + str(counter)]["val"]["X"] = x_val
                splits_dict[split][splitstype]["ftsplit" + str(counter)]["val"]["Y"] = y_val
                counter += 1



def create_splits_sourceIC50(orig_df, kf, splits_dict):
    """
    == For creating numfolds-fold splits for source data with logIC50 as labels ==
    :param orig_df - original dataframe (dataset) to be split
    :param kf - KFold split function
    :param splits_dict - dictionary that holds the splits
    """
    # NOTE: for preprocessed source,
    #   - col index 0 is the binary class response (labels)
    #   - col index 1 is logIC50 (labels)
    #   - col index 2 and beyond is the gene expression (features)
    counter = 1
    for train_index, val_index in kf.split(orig_df):
        x_train, x_val = orig_df.iloc[train_index, 2:], orig_df.iloc[val_index, 2:] # gene expressions (features)
        y_train, y_val = orig_df.iloc[train_index, 1], orig_df.iloc[val_index, 1] # logIC50
        splits_dict["split" + str(counter)] = {}
        splits_dict["split" + str(counter)]["train"] = {}
        splits_dict["split" + str(counter)]["train"]["X"] = x_train
        splits_dict["split" + str(counter)]["train"]["Y"] = y_train
        splits_dict["split" + str(counter)]["val"] = {}
        splits_dict["split" + str(counter)]["val"]["X"] = x_val
        splits_dict["split" + str(counter)]["val"]["Y"] = y_val
        counter += 1


if STRATIFIED_AITL_SPLITS:
    skf = StratifiedKFold(n_splits=NSPLITS, random_state=42, shuffle=False)
    splits_dict_target = {}
    splits_dict_source = {} # perform regular kfold split on source (do not use binary labels)
    create_splits_stratified(target_combined_exprs_resp_z, skf, splits_dict_target, NSPLITS, "target", splitstype="trainvaltest")
    create_splits_stratified(source_exprs_resp_z, skf, splits_dict_source, NSPLITS, "source", splitstype = "trainval")
    # create_splits_sourceIC50(source_exprs_resp_z, kf, splits_dict_source)
    # print("\n\n-- Stratified Target Splits --\n{}".format(splits_dict_target))
    # print("\n\n-- Source Splits --\n{}".format(splits_dict_source))

    print("\n-- Counting the number of samples for each class from each fold of the stratified kfold split (TARGET)-- \n")
    for split in splits_dict_target:
        print("\n{}".format(split))
        print("# training examples (class 0): {}".format(len(splits_dict_target[split]["trainvaltest"]["ftsplit1"]["train"]["Y"][splits_dict_target[split]["trainvaltest"]["ftsplit1"]["train"]["Y"] == 0])))
        print("# training examples (class 1): {}".format(len(splits_dict_target[split]["trainvaltest"]["ftsplit1"]["train"]["Y"][splits_dict_target[split]["trainvaltest"]["ftsplit1"]["train"]["Y"] == 1])))
        print("# validation examples (class 0): {}".format(len(splits_dict_target[split]["trainvaltest"]["ftsplit1"]["val"]["Y"][splits_dict_target[split]["trainvaltest"]["ftsplit1"]["val"]["Y"] == 0])))
        print("# validation examples (class 1): {}".format(len(splits_dict_target[split]["trainvaltest"]["ftsplit1"]["val"]["Y"][splits_dict_target[split]["trainvaltest"]["ftsplit1"]["val"]["Y"] == 1])))
        print("# test examples (class 0): {}".format(len(splits_dict_target[split]["traintest"]["test"]["Y"][splits_dict_target[split]["traintest"]["test"]["Y"] == 0])))
        print("# test examples (class 1): {}".format(len(splits_dict_target[split]["traintest"]["test"]["Y"][splits_dict_target[split]["traintest"]["test"]["Y"] == 1])))


    print("\n-- Counting the number of samples from each fold of the 3-fold split (SOURCE)-- \n")
    for split in splits_dict_source:
        print("\n{}".format(split))
        print("# training examples: {}, # features: {}".format(splits_dict_source[split]["train"]["X"].shape[0],
                                                                splits_dict_source[split]["train"]["X"].shape[1]))
        print("# class 0 training examples: {}".format(len(splits_dict_source[split]["train"]["Y_response"][splits_dict_source[split]["train"]["Y_response"] == 0])))
        print("# class 1 training examples: {}".format(len(splits_dict_source[split]["train"]["Y_response"][splits_dict_source[split]["train"]["Y_response"] == 1])))

        print("# validation examples: {}, # features: {}".format(splits_dict_source[split]["val"]["X"].shape[0],
                                                                    splits_dict_source[split]["val"]["X"].shape[1]))
        print("# class 0 validation examples: {}".format(len(splits_dict_source[split]["val"]["Y_response"][splits_dict_source[split]["val"]["Y_response"] == 0])))
        print("# class 1 validation examples: {}".format(len(splits_dict_source[split]["val"]["Y_response"][splits_dict_source[split]["val"]["Y_response"] == 1])))
                                                            



if STRATIFIED_AITL_SPLITS:
    # Saving Source splits #
    print("Saving source splits ...")
    for split in splits_dict_source:
        # splits_dict_source[split]["train"]["X"]
        dirName = SAVE_RESULT_TO_STRAT + SOURCE_DIR + '/' + split + '/'
        if not os.path.exists(dirName):
            os.makedirs(dirName)
            print("Directory ", dirName, " Created ")
        else:
            print("Directory ", dirName, " already exists")
        for train_val in splits_dict_source[split]:
            splits_dict_source[split][train_val]['X'].to_csv(path_or_buf=os.path.join(dirName, 'X_' + train_val + '_source.tsv'),
                                                                sep='\t', index=True)
            splits_dict_source[split][train_val]['Y_response'].to_csv(path_or_buf=os.path.join(dirName, 'Y_' + train_val + '_source.tsv'),
                                                                sep='\t', index=True, header=True) # note: single col. pandas df is treated as Series
            splits_dict_source[split][train_val]['Y_logIC50'].to_csv(path_or_buf=os.path.join(dirName, 'Y_logIC50' + train_val + '_source.tsv'),
                                                                sep='\t', index=True, header=True) # note: single col. pandas df is treated as Series

    print("Successfully saved source splits.\n")

    print("Saving target splits ...")
    # Saving stratified Target splits #
    for split in splits_dict_target:
        dirName = SAVE_RESULT_TO_STRAT + TARGET_DIR + '/' + split + '/'
        if not os.path.exists(dirName):
            os.makedirs(dirName)
            print("Directory ", dirName, " Created ")
        else:
            print("Directory ", dirName, " already exists")
        for tvt in splits_dict_target[split]:
            if tvt == "trainvaltest":
                for ftsplit in splits_dict_target[split][tvt]:
                    dirNameft = SAVE_RESULT_TO_STRAT + TARGET_DIR + '/' + split + '/' + ftsplit + '/'
                    if not os.path.exists(dirNameft):
                        os.makedirs(dirNameft)
                        print("Directory ", dirNameft, " Created ")
                    else:
                        print("Directory ", dirNameft, " already exists")
                    for trainval in splits_dict_target[split][tvt][ftsplit]:
                        splits_dict_target[split][tvt][ftsplit][trainval]["X"].to_csv(path_or_buf=os.path.join(dirNameft, 'X_' + trainval + '_ft_target.tsv'),
                                                                                        sep='\t', index=True)
                        splits_dict_target[split][tvt][ftsplit][trainval]["Y"].to_csv(path_or_buf=os.path.join(dirNameft, 'Y_' + trainval + '_ft_target.tsv'),
                                                                                        sep='\t', index=True, header=True)
            if tvt == 'traintest':
                for traintest in splits_dict_target[split][tvt]:
                    splits_dict_target[split][tvt][traintest]["X"].to_csv(path_or_buf=os.path.join(dirName, 'X_' + traintest + '_target.tsv'),
                                                                            sep='\t', index=True)
                    splits_dict_target[split][tvt][traintest]["Y"].to_csv(path_or_buf=os.path.join(dirName, 'Y_' + traintest + '_target.tsv'),
                                                                            sep='\t', index=True, header=True)
    print("Successfully saved target splits.\n")

