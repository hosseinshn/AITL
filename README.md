# AITL: Adversarial Inductive Transfer Learning with input and output space adaptation for pharmacogenomics

Repository for "AITL: Adversarial Inductive Transfer Learning with input and output space adaptation for pharmacogenomics" [[paper]](https://academic.oup.com/bioinformatics/article/36/Supplement_1/i380/5870479?login=true)

Authors : Hossein Sharifi Noghabi, Shuman Peng, Olga Zolotareva, Colin Collins and Martin Ester

# Installation
## Requirements
* Python 3.6
* Conda
* PyTorch 1.1

### Conda environment
Use the provided `environment.yml` in `AITL/AITL_submit/` to create a conda environment with the required packages.
```
cd AITL_submit/
conda env create -f environment.yml
```
Running the command above will create the `py36aitl` environment. To activate the `py36aitl` environment, use:
```
conda activate py36aitl
```

# Datasets
## Downloading our pre-processed data

You can download the pre-processed data that we used to train and evaluate our AITL models [here](https://drive.google.com/drive/folders/1r4iw-qp6gep5XzYlQzaxI22cGgqI0MRG?usp=sharing).

## Structure of the data
The data contains pharmacogenomics datasets for 4 different cancer treatment drugs, namely **Bortezomib**, **Cisplatin**, **Docetaxel**, and **Paclitaxel**. The source and target pharmacogenomics datasets for each drug are included in the folders `data/split/[drug]`, where `drug=[Bortezomib, Cisplatin, Docetaxel, Paclitaxel]`. A detailed structure of the pre-processed data downloaded using our provided link above is shown below.

```
data
├── split
|   ├── Bortezomib
|       ├── stratified                              # for 3-fold cross validation
|           ├── source_3_folds
|               ├── split1
|                   ├── X_train_source.tsv          # for AITL training (inputs)
|                   ├── X_val_source.tsv            # for AITL validation (inputs)
|                   ├── Y_logIC50train_source.tsv   # for AITL training (continuous outputs)
|                   ├── Y_logIC50val_source.tsv     # for AITL validation (continuous outputs)
|                   ├── Y_train_source.tsv          # binarized outputs
|                   ├── Y_val_source.tsv            # binarized outputs
|               ├── split2
|                   ├── ...     # contains the same files as split1
|               ├── split3
|                   ├── ...     # contains the same files as split1
|           ├── target_3_folds
|               ├── split1
|                   ├── X_train_target.tsv          # for AITL training (inputs)
|                   ├── X_test_target.tsv           # for AITL testing (inputs)
|                   ├── Y_train_target.tsv          # for AITL training (outputs)
|                   ├── Y_test_target.tsv           # for AITL testing (outputs)
|               ├── split2
|                   ├── ...    # contains the same files as split1
|               ├── split3
|                   ├── ...    # contains the same files as split1
|       ├── Source_exprs_resp_z.Bortezomib.tsv              # original source data (not stratified for 3-fold cross validation)
|       ├── Target_combined_expr_resp_z.Bortezomib.tsv      # original target data (not stratified for 3-fold cross validation)
|   ├── Cisplatin
|       ├── ...   # follows the same pattern as Bortezomib
|   ├── Docetaxel  
|       ├── ...   # follows the same pattern as Bortezomib
|   ├── Paclitaxel
|       ├── ...   # follows the same pattern as Bortezomib
└──
```

## Training an AITL model using specified hyper-parameters
**Step 1:** update the data directory on `line 41` of `aitl_train_best_model.py`. Suppose you downloaded and saved the required data in the `AITL/AITL_submit/data/` directory, then replace `line 41` in `aitl_train_best_model.py` with the following line:
```
LOAD_DATA_FROM = './data/data/split/' + DRUG + '/stratified/'
```

**Step 2:** update the hyper-parameter values in `aitl_train_best_model`. You may use the hyper-parameters we provided in our [supplementary materials](https://github.com/hosseinshn/AITL/blob/master/AITL_Supp%20Material.pdf) (page 3).

**Step 3:** train an AITL model using the following command. Note: make sure you are at `AITL/AITL_submit/`
```
python aitl_train_best_model.py
```

## Trained AITL Models

We have provided access to our trained AITL models. Please find the our trained AITL models below:
* [Bortezomib](https://drive.google.com/drive/folders/1-rot1vtAQGB42EpSjFU77uD98OF3jxBg?usp=sharing)
* [Cisplatin](https://drive.google.com/drive/folders/1E6fzL53YhWWFSpeOt1ZXH9RqJgIUkGsu?usp=sharing)
* [Docetaxel](https://drive.google.com/drive/folders/12WV7RaR3mPA48YBq_UCEy2FauUiMNrrm?usp=sharing)
* [Paclitaxel](https://drive.google.com/drive/folders/1x88dHCP1mKHU_1E-5iYXTbl2oYYYvE00?usp=sharing)
