import pickle
import torch
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.append(str(Path().absolute().parent))

number_tags = []
for argument in sys.argv[1:]:
    number_tags.append(argument.zfill(4))
exp_path = Path().absolute().parent.parent.joinpath(f'experiments')
experiment_dirs = os.listdir(exp_path)

if sys.argv[1]=='all':
    experiments_to_calculate = experiment_dirs
else:
    experiments_to_calculate = []
    for number_tag in number_tags:
        print(number_tag)
        experiments_to_use = None
        for experiment in experiment_dirs:
            if number_tag in experiment:
                experiments_to_use = experiment
                break
        if experiments_to_use== None:
            print("Experiment {number_tag} not found")
            continue
        experiments_to_calculate.append(experiment)

def probabilities_to_labels(predictions):
    if len(predictions.shape)==1: # if not one hot encoding
        return np.round(predictions) #sigmoid outputs
    predictions_as_labels = []
    for row in predictions:
        predictions_as_labels.append(np.argmax(row))
    return np.array(predictions_as_labels)

e,d = {},{}
for experiment in experiments_to_calculate:
    if not experiment[0:4].isdigit():
        continue

    pickle_file = exp_path.joinpath(experiment).joinpath('variables/log.pkl')
    with open(pickle_file, 'rb') as handle:
        metrics = pickle.load(handle)

    e[experiment] = {}
    total_size = 0
    if "LCO" in experiment: # Leave Center Out has a different structure
        if 'FL' in experiment:
            targets, preds = pd.DataFrame(), pd.DataFrame()
            for x in metrics.keys():
                if 'test_predictions_' in x: # in
                    results = list(metrics[x].values())[0]
                    if targets.empty:
                        targets = results['targets']
                        preds = results['predictions']
                    else:
                        targets = targets.append(results['targets'])
                        preds = preds.append(results['predictions'])
                    total_size+=len(results)
        else:
            results = metrics['test_predictions']
            total_size+=len(results)
            targets = results['targets'].to_numpy()
            preds = results['predictions'].to_numpy()
    else:
        num_folds = 5
        if 'FL' in experiment:
            targets, preds = pd.DataFrame(), pd.DataFrame()
            for x in metrics.keys():
                if 'test_predictions_' in x: # in the federated case, the test predictions are separate for each center
                    for fold in range(num_folds):
                        results = metrics[x][fold]
                        if targets.empty:
                            targets = results['targets']
                            preds = results['predictions']
                        else:
                            targets = targets.append(results['targets'])
                            preds = preds.append(results['predictions'])
                        total_size+=len(results)
        elif 'CDS' in experiment:
            results = metrics['test_predictions']
            targets = results['targets']
            preds = results['predictions']
            total_size+=len(results)
        
        print(f'Total_size is {total_size}')
        targets, preds = targets.to_numpy(), preds.to_numpy()

        AUC = roc_auc_score(targets, preds)
        accuracy = sum(np.array(targets) == probabilities_to_labels(preds))/len(targets)

        # acc_per_fold, f1_per_fold = [], []
        # counter=0
        # for fold_cms in confusion_matrices:
        #     counter+=1
        #     acc, _, _, f1 = metrics_from_confusion_matrices(fold_cms)
        #     a,f = acc[-1], f1[-1]
            
        #     acc_per_fold.append(a)
        #     f1_per_fold.append(f)

        e[experiment]['Complete AUC'] = AUC
        e[experiment]['Complete Accuracy'] = accuracy
        # e[experiment] = {'accuracy': acc_per_fold, 'f1_score': f1_per_fold}
        # d[experiment] = {'accuracy': acc_per_fold, 'f1_score': f1_per_fold, 'num_folds': counter}


# for key in d.keys():
#     for subkey in d[key].keys():
#         if subkey != 'num_folds':
#             d[key][subkey] = np.mean(d[key][subkey])
print(pd.DataFrame.from_dict(e).transpose())
results = pd.DataFrame.from_dict(e)
results.to_pickle('../visualizations/all_results.pkl')
