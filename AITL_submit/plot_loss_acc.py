import matplotlib.pyplot as plt
import os 
import pandas as pd 


def plot_learning_curve(opt): 

    ## Reading options ## 
    model_params = opt['model_params']
    base_trace_dir = opt['base_trace_dir']
    trace_dir = base_trace_dir + '/' + model_params + '/'
    save_plots_to = trace_dir + 'plots/'
    split = opt['split']
    ftsplit = opt['ftsplit']

    dirName = save_plots_to
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")

    trace_file = trace_dir + split + '_' + ftsplit + '_trace.tsv'
    ## start reading the file after '#' sign  (indicates end of params info) ## 
    # count number of lines preceeding the '#' sign
    # open a file using with statement
    with open(trace_file,'r') as f:
        num_lines_to_skip = 1   # also skip the line with '#' 
        for line in f:
            # check if the '#' line has been encountered
            if line.startswith("#"):
                break
            else:
                num_lines_to_skip += 1
                # print(line)
    print("num lines to skip: {}\n".format(num_lines_to_skip))


    trace_df = pd.read_csv(trace_file, sep='\t', index_col=False, skiprows=num_lines_to_skip)
    # print("len(trace_df['epoch']): {}\n".format(len(trace_df['epoch'])))
    # print("column names: {}\n\n".format(list(trace_df.columns)))

    plt.figure()
    plt.plot(trace_df['train_losstotal'][1:], label='training loss')
    plt.plot(trace_df['val_losstotal'][1:], label='validation loss')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.xlim([1, len(trace_df['epoch'])])
    # plt.ylim([-0.5, max(trace_df['train_losstotal'][1],trace_df['val_losstotal'][1])+15])
    plt.title("Loss over training epochs ({} {})".format(split, ftsplit))
    plt.savefig("{}losses_{}_{}_epochs{}.png".format(save_plots_to, split, ftsplit, len(trace_df['epoch'])))
    plt.close()
    # plt.show()

    plt.figure()
    plt.plot(trace_df['train_AUC'], label='training AUC')
    plt.plot(trace_df['val_AUC'], label='validation AUC')
    plt.legend()
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.title("Accuracy (AUC) over training epochs ({} {})".format(split, ftsplit))
    plt.savefig("{}acc_{}_{}_epochs{}.png".format(save_plots_to, split, ftsplit, len(trace_df['epoch'])))
    plt.close()
    # plt.show()


# ================================ # 
#      Regression Performance      # 
# ================================ # 

    plt.figure()
    plt.plot(trace_df['train_regloss'][1:], label='training')
    plt.plot(trace_df['val_regloss'][1:], label='validation')
    plt.legend()
    plt.ylabel('MSE Loss')
    plt.xlabel('Epochs')
    # plt.xlim([1, len(trace_df['epoch'])])
    # plt.ylim([-0.5, max(trace_df['train_losstotal'][1],trace_df['val_losstotal'][1])+15])
    plt.title("Regression MSE Loss over training epochs ({} {})".format(split, ftsplit))
    plt.savefig("{}reg_losses_{}_{}_epochs{}.png".format(save_plots_to, split, ftsplit, len(trace_df['epoch'])))
    plt.close()
    # plt.show()


# ================================ # 
#      Classification Loss         # 
# ================================ # 

    plt.figure()
    plt.plot(trace_df['train_closs'], label='training')
    plt.plot(trace_df['val_closs'], label='validation')
    plt.legend()
    plt.ylabel('BCE Loss')
    plt.xlabel('Epochs')
    # plt.xlim([1, len(trace_df['epoch'])])
    # plt.ylim([-0.5, max(trace_df['train_losstotal'][1],trace_df['val_losstotal'][1])+15])
    plt.title("Classification Loss over training epochs ({} {})".format(split, ftsplit))
    plt.savefig("{}clas_losses_{}_{}_epochs{}.png".format(save_plots_to, split, ftsplit, len(trace_df['epoch'])))
    plt.close()
    # plt.show()


# ================================ # 
#   Loss Plots for Discriminators  # 
# ================================ # 


    plt.figure()
    plt.plot(trace_df['train_DGloss'], label='DG training loss')
    plt.plot(trace_df['val_DGloss'], label='DG validation loss')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title("DG Losses over training epochs ({} {})".format(split, ftsplit))
    plt.savefig("{}DG_losses_{}_{}_epochs{}.png".format(save_plots_to, split, ftsplit, len(trace_df['epoch'])))
    plt.close()
    # plt.show()


    plt.figure()
    plt.plot(trace_df['train_DRloss'], label='DR training loss')
    plt.plot(trace_df['val_DRloss'], label='DR validation loss')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title("DR Losses over training epochs ({} {})".format(split, ftsplit))
    plt.savefig("{}DR_losses_{}_{}_epochs{}.png".format(save_plots_to, split, ftsplit, len(trace_df['epoch'])))
    plt.close()
    # plt.show()

    plt.figure()
    plt.plot(trace_df['train_DSloss'], label='DS training loss')
    plt.plot(trace_df['val_DSloss'], label='DS validation loss')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title("DS Losses over training epochs ({} {})".format(split, ftsplit))
    plt.savefig("{}DS_losses_{}_{}_epochs{}.png".format(save_plots_to, split, ftsplit, len(trace_df['epoch'])))
    plt.close()
    # plt.show()

# ================================ # 
#    AUC Plots for Discriminators  # 
# ================================ # 


    plt.figure()
    plt.plot(trace_df['train_DGauc'], label='DG training auc')
    plt.plot(trace_df['val_DGauc'], label='DG validation auc')
    plt.legend()
    plt.ylabel('AUC')
    plt.xlabel('Epochs')
    plt.xlim([1, len(trace_df['epoch'])])
    plt.title("DG AUC over training epochs ({} {})".format(split, ftsplit))
    plt.savefig("{}DG_auc_{}_{}_epochs{}.png".format(save_plots_to, split, ftsplit, len(trace_df['epoch'])))
    plt.close()
    # plt.show()


    plt.figure()
    plt.plot(trace_df['train_DRauc'], label='DR training auc')
    plt.plot(trace_df['val_DRauc'], label='DR validation auc')
    plt.legend()
    plt.ylabel('AUC')
    plt.xlabel('Epochs')
    plt.xlim([1, len(trace_df['epoch'])])
    plt.title("DR AUC over training epochs ({} {})".format(split, ftsplit))
    plt.savefig("{}DR_auc_{}_{}_epochs{}.png".format(save_plots_to, split, ftsplit, len(trace_df['epoch'])))
    plt.close()
    # plt.show()

    plt.figure()
    plt.plot(trace_df['train_DSauc'], label='DS training auc')
    plt.plot(trace_df['val_DSauc'], label='DS validation auc')
    plt.legend()
    plt.ylabel('AUC')
    plt.xlabel('Epochs')
    plt.xlim([1, len(trace_df['epoch'])])
    plt.title("DS AUC over training epochs ({} {})".format(split, ftsplit))
    plt.savefig("{}DS_auc_{}_{}_epochs{}.png".format(save_plots_to, split, ftsplit, len(trace_df['epoch'])))
    plt.close()
    # plt.show()