import sys
import shutil
import os
import pickle
import yaml
import importlib
import enum
import copy
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='This script generates a new experiment inside the experiments folder, use schedule_experiments plus the number to run it')
# parser.add_argument("-n", "--number", required=True, help='Number of duplicate experiments to generate')
args = parser.parse_args()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HOME_PATH = Path.home()

config_file = Path('config.yaml')
with open(config_file) as file:
  config = yaml.safe_load(file)
experiments_folder = HOME_PATH / Path(config['paths']['experiments'])
if not experiments_folder.exists():
    experiments_folder.mkdir()

# define experiment name
experiment_name = config['name']
for center in config['data']['centres']:
    experiment_name += center[0]
experiment_name += f"_{config['cvtype']}_"
if config['federated']['isTrue']:
    experiment_name = experiment_name+config['federated']['type']+'_'+config['federated']['averaging']+'_'
else:
    experiment_name = experiment_name+'CDS_'

transformations_d = {'S':'Shape', 'N':'None', 'I':'Intensity', 'SI':'Shape+Intensity', 'B':'Basic'}

experiment_name += transformations_d[str(config['data']['transformations'])] + '_'
experiment_name += str(config['hyperparameters']['num_epochs']) + '_'
if config['data']['triplicate']:
    experiment_name += 'Triplets' + '_'
experiment_name += 'L' + str(config['model']['arch']['args']['linear_ch'])

if isinstance(config['seed'], list):
    experiment_names = [experiment_name+'_S'+str(seed) for seed in config['seed']]
else:
    experiment_names = [experiment_name]

for experiment_name in experiment_names:
    if experiment_name[0:4].isdigit():
        # note that if you want to continue or overwrite an experiment you should use the full name including the digit
        new_experiment_name =experiment_name 
    else:
        # add a new tag to create a new experiment
        max_tag = 0
        for file in experiments_folder.iterdir():
            if str(file.name)[0:4].isdigit():
                if max_tag < int(str(file.name)[0:4]):
                    max_tag = int(str(file.name)[0:4])
        tag = str(max_tag+1).zfill(4)
        new_experiment_name = tag + '_' +experiment_name 

        '''
        To avoid cluttering the experiments folder when dealing with errors, 
        this will make sure not to create a new tag for a duplicate file name that's been created within the last 10 minutes
        '''
        possible_last_file = str(max_tag).zfill(4) + '_' +experiment_name 
        possible_last_file = experiments_folder.joinpath(possible_last_file)
        if possible_last_file.exists():
            timestamp = datetime.fromtimestamp(possible_last_file.stat().st_mtime)
            now = datetime.now()
            if timestamp.hour == now.hour and now.minute-timestamp.minute<20:
                new_experiment_name = possible_last_file # overwrite in this case
                
        
    new_experiment = experiments_folder.joinpath(new_experiment_name)
    print(f"Generating experiment {new_experiment_name}")
    MODEL_STORAGE = new_experiment.joinpath('model_states')
    VARIABLE_STORAGE = new_experiment.joinpath('variables')
    if not MODEL_STORAGE.exists():
        MODEL_STORAGE.mkdir(parents=True)
        VARIABLE_STORAGE.mkdir()
    # else:
    #     if not config['model']['continue']:
    #         print(f"Deleting {MODEL_STORAGE}")
    #         for f in MODEL_STORAGE.iterdir():
    #             f.unlink() #empty directory for new model states. 
    #make a copy of config file in new experiment to keep track of parameters.
    shutil.copy(config_file, new_experiment)
    shutil.copy('train.py', new_experiment)
    shutil.copy('data_loader.py', new_experiment)
    shutil.copy('environment.yaml', new_experiment)
    shutil.copy('Dockerfile', new_experiment)
    if (new_experiment / 'models').exists():
        shutil.rmtree((new_experiment / 'models'))
    shutil.copytree('models', new_experiment / 'models')
    if (new_experiment / 'misc').exists():
        shutil.rmtree((new_experiment / 'misc'))
    shutil.copytree('misc', new_experiment / 'misc')
