# This script will store the split of folds as done in FL mapped to CDS
seed = 42  # for reproducibility

import pickle
import numpy as np
import os
import sys
from pathlib import Path
sys.path.append(str(Path().absolute().parent))
import argparse
parser = argparse.ArgumentParser(description='This script stores a dictionary of fold indices to be used for CDS. Use a federated experiment as input')
parser.add_argument("-n", "--name", required=True, help='Input name of output file')
parser.add_argument("-p", "--path", required=True, help='Input federated experiment name to get fold_index split')
args = parser.parse_args()

# FL_path = '0006_3D_centersVSDSUA_FL_noAugs_alldata'
full_path = Path().absolute().parent.parent.joinpath(f'experiments/{args.path}')

pickle_file = 'variables/log.pkl'
with open(full_path.joinpath(pickle_file), 'rb') as handle:
    metrics = pickle.load(handle)

d = {}
for k,v in metrics.items():
    if 'folds_indices' not in k:
        continue
    else:
        d[k]=v

offset_list = []
offset_counter = 0
print("Order is:")
for k,v in d.items():
    print(k)
    offset_counter+=np.concatenate(v[0]).shape[0]
    offset_list.append(offset_counter) # len of each center
offset_list.pop() # we don't need the last one

fold = [[] for i in range(5)]
print("Order is:")
for k,v in d.items():
    print(k)
    for i in range(5):
        fold[i].append(v[i])

final_indices = []
for j in range(5):
    train_indices = fold[j][0][0]
    val_indices = fold[j][0][1]
    test_indices = fold[j][0][2]
    for i in range(1,len(offset_list)+1):
        train_indices = np.concatenate((train_indices, fold[j][i][0]+offset_list[i-1]))
        val_indices = np.concatenate((val_indices, fold[j][i][1]+offset_list[i-1]))
        test_indices = np.concatenate((test_indices, fold[j][i][2]+offset_list[i-1]))
    final_indices.append((train_indices, val_indices, test_indices))

HOME_PATH = Path.home()
misc = Path('workenv/mnm/experiments/misc')
# Enter fold_indices name (pkl will be appended to it)
if 'pkl' in args.name:
    args.name = args.name.split('.')[0]
with open(HOME_PATH / misc.joinpath(f'{args.name}.pkl'), 'wb') as handle:
    pickle.dump(final_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
