from sklearn import preprocessing

import os.path
import math
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler


import pandas as pd


DRUG = 'Paclitaxel'
PDX = False
PDX_PATIENT_COMBINED = True

MODEL = 'AITL' 

SAVE_RESULT_TO = './data/split/'+DRUG+'/' + MODEL + '/'
LOAD_DATA_FROM = './data/original/'+DRUG+'/'    # where the original (pre-split) data is stored


if PDX: 
    SAVE_RESULT_TO = SAVE_RESULT_TO + 'PDX/'
elif PDX_PATIENT_COMBINED:
    SAVE_RESULT_TO = SAVE_RESULT_TO + 'PDXPatientCombined/'

dirName = SAVE_RESULT_TO
if not os.path.exists(dirName):
    os.makedirs(dirName)
    print("Directory ", dirName, " Created ")
else:
    print("Directory ", dirName, " already exists")


if DRUG == 'Docetaxel':
    GSEls = ['GSE6434', 'GSE25065', 'GSE28796', 'TCGA']
elif DRUG == 'Bortezomib':
    GSEls = ['GSE9782-GPL96', 'GSE55145']
elif DRUG == 'Cisplatin':
    GSEls = ['GSE18864', 'GSE23554', 'TCGA']
elif DRUG == 'Paclitaxel':
    GSEls = ['GSE15622', 'GSE22513', 'GSE25065', 'PDX', 'TCGA']


# Normalized data:
GDSC_exprs_z = pd.read_csv(LOAD_DATA_FROM+'exprs/GDSC_exprs.z.'+DRUG+'.tsv',
                                sep='\t', index_col=0, decimal='.')

GSE1_exprs_z = pd.read_csv(LOAD_DATA_FROM+'exprs/'+GSEls[0]+'_exprs.z.'+DRUG+'.tsv',
                                    sep='\t', index_col=0, decimal='.')

GSE2_exprs_z = pd.read_csv(LOAD_DATA_FROM+'exprs/'+GSEls[1]+'_exprs.z.'+DRUG+'.tsv',
                                sep='\t', index_col=0, decimal='.')

GSE3_exprs_z = pd.read_csv(LOAD_DATA_FROM+'exprs/'+GSEls[2]+'_exprs.z.'+DRUG+'.tsv',
                                sep='\t', index_col=0, decimal='.')

GSE4_exprs_z = pd.read_csv(LOAD_DATA_FROM+'exprs/'+GSEls[3]+'_exprs.z.'+DRUG+'.tsv',
                                sep='\t', index_col=0, decimal='.')

TCGA_exprs_z = pd.read_csv(LOAD_DATA_FROM+'exprs/TCGA_exprs.z.'+DRUG+'.tsv',
                           sep='\t', index_col=0, decimal='.')


# Load drug response data for drug Bortezomib
GDSC_resp = pd.read_csv(LOAD_DATA_FROM+'response/GDSC_response.'+DRUG+'.tsv',
                            sep='\t', index_col=0, decimal='.')

GSE1_resp = pd.read_csv(LOAD_DATA_FROM+'response/'+GSEls[0]+'_response.'+DRUG+'.tsv',
                                sep='\t', index_col=0, decimal='.')

GSE2_resp = pd.read_csv(LOAD_DATA_FROM+'response/'+GSEls[1]+'_response.'+DRUG+'.tsv',
                                sep='\t', index_col=0, decimal='.')

GSE3_resp = pd.read_csv(LOAD_DATA_FROM+'response/'+GSEls[2]+'_response.'+DRUG+'.tsv',
                                sep='\t', index_col=0, decimal='.')

GSE4_resp = pd.read_csv(LOAD_DATA_FROM+'response/'+GSEls[3]+'_response.'+DRUG+'.tsv',
                                sep='\t', index_col=0, decimal='.')

TCGA_resp = pd.read_csv(LOAD_DATA_FROM+'response/TCGA_response.'+DRUG+'.tsv',
                            sep='\t', index_col=0, decimal='.')


## note: ENTREZID column is the genes (features)


# Transpose the pandas data frame for expression so that the columns = genes (features) and rows = examples
# Note that response data does not need transposing
# GDSC_exprs_B = pd.DataFrame.transpose(GDSC_exprs_B)
# GSE55145_exprs_B = pd.DataFrame.transpose(GSE55145_exprs_B)
# GSE9782_exprs_B = pd.DataFrame.transpose(GSE9782_exprs_B)

GDSC_exprs_z = pd.DataFrame.transpose(GDSC_exprs_z)
GSE1_exprs_z = pd.DataFrame.transpose(GSE1_exprs_z)
GSE2_exprs_z = pd.DataFrame.transpose(GSE2_exprs_z)
GSE3_exprs_z = pd.DataFrame.transpose(GSE3_exprs_z)
GSE4_exprs_z = pd.DataFrame.transpose(GSE4_exprs_z)
TCGA_exprs_z = pd.DataFrame.transpose(TCGA_exprs_z)
#


# Remove genes with low signal (i.e. below the variance threshold) from expression data

selector = VarianceThreshold(0.05)
selector.fit_transform(GDSC_exprs_z)
GDSC_exprs_z = GDSC_exprs_z[GDSC_exprs_z.columns[selector.get_support(indices=True)]]
ls = GSE1_exprs_z.columns.intersection(GDSC_exprs_z.columns)
ls = ls.intersection(GSE2_exprs_z.columns)
ls = ls.intersection(GSE3_exprs_z.columns)
ls = ls.intersection(GSE4_exprs_z.columns)
ls = ls.intersection(TCGA_exprs_z.columns)
GSE1_exprs_z = GSE1_exprs_z.loc[:,ls]
GSE2_exprs_z = GSE2_exprs_z.loc[:,ls]
GSE3_exprs_z = GSE3_exprs_z.loc[:,ls]
GSE4_exprs_z = GSE4_exprs_z.loc[:,ls]
TCGA_exprs_z = TCGA_exprs_z.loc[:,ls]

# Obtain selected genes
GDSC_exprs_z_genes = list(GDSC_exprs_z.columns.values)
GSE1_exprs_z_genes = list(GSE1_exprs_z.columns.values)
GSE2_exprs_z_genes = list(GSE2_exprs_z.columns.values)
GSE3_exprs_z_genes = list(GSE3_exprs_z.columns.values)
GSE4_exprs_z_genes = list(GSE4_exprs_z.columns.values)
TCGA_exprs_z_genes = list(TCGA_exprs_z.columns.values)


# if not ((TCGA_exprs_z_genes == GSE1_exprs_z_genes) and (TCGA_exprs_z_genes == GSE2_exprs_z_genes) \
#     and (TCGA_exprs_z_genes == GDSC_exprs_z_genes)):
#     print("\nWARNING: genes do not have the same order\n")

## For when there are 5 sets of data (i.e. GDSC, GSE1, GSE2, GSE3, TCGA) ## 
# if not ((TCGA_exprs_z_genes == GSE1_exprs_z_genes) and (TCGA_exprs_z_genes == GSE2_exprs_z_genes) \
#     and (TCGA_exprs_z_genes == GDSC_exprs_z_genes) and (TCGA_exprs_z_genes == GSE3_exprs_z_genes)):
#     print("\nWARNING: genes do not have the same order\n")

## For when there are 6 sets of data (i.e. GDSC, GSE1, GSE2, GSE3, PDX, TCGA) ##
if not ((TCGA_exprs_z_genes == GSE1_exprs_z_genes) and (TCGA_exprs_z_genes == GSE2_exprs_z_genes) \
    and (TCGA_exprs_z_genes == GDSC_exprs_z_genes) and (TCGA_exprs_z_genes == GSE3_exprs_z_genes) \
    and (TCGA_exprs_z_genes == GSE4_exprs_z_genes)):
    print("\nWARNING: genes do not have the same order\n")

## For when there are 3 sets of data (i.e. GDSC, GSE1, GSE2) ## 
# if not ((GDSC_exprs_z_genes == GSE1_exprs_z_genes) and (GDSC_exprs_z_genes == GSE2_exprs_z_genes)):
#     print("\nWARNING: genes do not have the same order\n")

## For when there are 2 sets of data (i.e. GDSC, GSE1) ##
# if not (GDSC_exprs_z_genes == GSE1_exprs_z_genes):
#     print("\nWARNING: genes do not have the same order\n")


## For when there are 4 sets of data (i.e. GDSC, GSE1, GSE2, GSE3) ## 
# if not ((GDSC_exprs_z_genes == GSE1_exprs_z_genes) and (GDSC_exprs_z_genes == GSE2_exprs_z_genes)\
#     and (GDSC_exprs_z_genes == GSE3_exprs_z_genes)):
#     print("\nWARNING: genes do not have the same order\n")


## Add response column from GDSC resp and logIC50 to GDSC exprs as the first two columns  ##

# Convert all R to 0 and S to 1 #
GDSC_resp['response'].values[GDSC_resp['response'].values == 'R'] = 0
GDSC_resp['response'].values[GDSC_resp['response'].values == 'S'] = 1

if MODEL == 'AITL':
    GDSC_exprs_z_resp = GDSC_exprs_z.copy(deep=True)
    # print(GDSC_exprs_B_z_resp)
    # print(list(GDSC_resp_B['response']))
    GDSC_exprs_z_resp.insert(0, "response", list(GDSC_resp['response']), allow_duplicates=False)
    print("\nAfter inserting GDSC response ... \n")
    print(GDSC_exprs_z_resp)
    GDSC_exprs_z_resp.insert(1, "logIC50", list(GDSC_resp['logIC50']), allow_duplicates=False)
    print("\nAfter inserting GDSC logIC50 ...\n")
    print(GDSC_exprs_z_resp)

if MODEL == 'Protonet':
    GDSC_exprs_z_resp = GDSC_exprs_z.copy(deep=True)
    # print(GDSC_exprs_B_z_resp)
    # print(list(GDSC_resp_B['response']))
    GDSC_exprs_z_resp.insert(0, "response", list(GDSC_resp['response']), allow_duplicates=False)
    print("\nAfter inserting GDSC response ... \n")
    print(GDSC_exprs_z_resp)

# Save new DataFrame (GDSC expression + response) to a file ##
GDSC_exprs_z_resp.to_csv(path_or_buf=os.path.join(SAVE_RESULT_TO, 'Source_exprs_resp_z.'+DRUG+'.tsv'),
                            sep='\t', index=True)
print(" - - successfully saved combined preprocessed Source data - - \n")


GSE1_resp['response'].values[GSE1_resp['response'].values == 'R'] = 0
GSE1_resp['response'].values[GSE1_resp['response'].values == 'S'] = 1

GSE2_resp['response'].values[GSE2_resp['response'].values == 'R'] = 0
GSE2_resp['response'].values[GSE2_resp['response'].values == 'S'] = 1

GSE3_resp['response'].values[GSE3_resp['response'].values == 'R'] = 0
GSE3_resp['response'].values[GSE3_resp['response'].values == 'S'] = 1

GSE4_resp['response'].values[GSE4_resp['response'].values == 'R'] = 0
GSE4_resp['response'].values[GSE4_resp['response'].values == 'S'] = 1

TCGA_resp['response'].values[TCGA_resp['response'].values == 'R'] = 0
TCGA_resp['response'].values[TCGA_resp['response'].values == 'S'] = 1


## Add response column from GSE* resp to GSE* exprs as the first column ##

GSE1_exprs_z_resp = GSE1_exprs_z.copy(deep=True)
GSE1_exprs_z_resp.insert(0, "response", list(GSE1_resp['response']), allow_duplicates=False)

GSE2_exprs_z_resp = GSE2_exprs_z.copy(deep=True)
GSE2_exprs_z_resp.insert(0, "response", list(GSE2_resp['response']), allow_duplicates=False)

GSE3_exprs_z_resp = GSE3_exprs_z.copy(deep=True)
GSE3_exprs_z_resp.insert(0, "response", list(GSE3_resp['response']), allow_duplicates=False)

GSE4_exprs_z_resp = GSE4_exprs_z.copy(deep=True)
GSE4_exprs_z_resp.insert(0, "response", list(GSE4_resp['response']), allow_duplicates=False)

TCGA_exprs_z_resp = TCGA_exprs_z.copy(deep=True)
TCGA_exprs_z_resp.insert(0, "response", list(TCGA_resp['response']), allow_duplicates=False)



## Combining patient (target) data ##

## 5 sets of patient data - i.e. GSE1, GSE2, GSE3, GSE4, TCGA ##
GSE_combined_expr_resp = pd.concat([GSE1_exprs_z_resp, GSE2_exprs_z_resp, GSE3_exprs_z_resp, GSE4_exprs_z_resp, TCGA_exprs_z_resp])

## 4 sets of patient data - i.e. GSE1, GSE2, GSE3, TCGA ##
# GSE_combined_expr_resp = pd.concat([GSE1_exprs_z_resp, GSE2_exprs_z_resp, GSE3_exprs_z_resp, TCGA_exprs_z_resp])

## 2 sets of patient data - i.e. GSE1, GSE2 ##
# GSE_combined_expr_resp = pd.concat([GSE1_exprs_z_resp, GSE2_exprs_z_resp])

## 3 sets of patient data - i.e. GSE1, GSE2, GSE3 ##
# GSE_combined_expr_resp = pd.concat([GSE1_exprs_z_resp, GSE2_exprs_z_resp, GSE3_exprs_z_resp])

## 1 set of 'patient' data - i.e PDX ##
# GSE_combined_expr_resp = GSE1_exprs_z_resp

print(GSE_combined_expr_resp)

## Shuffle combined patient data dataframe (GSE*) ##

GSE_combined_expr_resp = shuffle(GSE_combined_expr_resp, random_state=42)
print(GSE_combined_expr_resp)

## Save new combined patient (target) dataframe (GSE*) ##

GSE_combined_expr_resp.to_csv(path_or_buf=os.path.join(SAVE_RESULT_TO, 'Target_combined_expr_resp_z.'+DRUG+'.tsv'),
                            sep='\t', index=True)
print(" - - successfully saved preprocessed combined Target data - - \n")


