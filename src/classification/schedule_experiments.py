import sys
import yaml
import os
from pathlib import Path

number_tags = []
for argument in sys.argv[1:]:
    number_tags.append(argument.zfill(4))

HOME_PATH = Path.home()
config_file = Path('config.yaml')
with open(config_file) as file:
  config = yaml.safe_load(file)
experiments_folder = HOME_PATH / Path(config['paths']['experiments'])
experiment_dirs = os.listdir(experiments_folder)

for number_tag in number_tags:
    print(number_tag)
    experiment_to_run = None
    for experiment in experiment_dirs:
        if number_tag in experiment:
            experiment_to_run = experiment
            break
    if experiment_to_run == None:
        print("Experiment {number_tag} not found")
        continue

    path_to_train_file = experiments_folder / experiment_to_run / 'train.py'
    print (path_to_train_file)
    # sys.path.insert(1, path_to_train_file.parent)
    os.chdir(path_to_train_file.parent) # change directory to import the local paths not the ones in /classification
    os.system(f'python {path_to_train_file}')