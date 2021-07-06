seed = 42  # for reproducibility

# Imports
import os
import yaml
import enum
import copy
import random
import tempfile
import warnings
import multiprocessing
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from collections import OrderedDict

# import visdom
from math import floor, ceil
from pathlib import Path
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
torch.manual_seed(seed)
import torchio as tio
from torchio.transforms import (
    RescaleIntensity,
    RandomElasticDeformation,
    RandomFlip,
    RandomAffine,
    # intensity
    RandomMotion,
    RandomGhosting,
    RandomSpike,
    RandomBiasField,
    RandomBlur,
    RandomNoise,
    RandomSwap,
    RandomAnisotropy,
#     RandomLabelsToImage,
    RandomGamma,
    OneOf,
    CropOrPad,
    ZNormalization,
    HistogramStandardization,
    Compose,
)


HOME_PATH = Path.home()
# Constants
config_file = Path('config.yaml')
with open(config_file) as file:
  config = yaml.safe_load(file)

np.random.seed(config['seed'])
data_path = HOME_PATH / Path(config['paths']['dataset'])
TRAIN_FOLDER = data_path / 'Training'
VAL_FOLDER = data_path / 'test'
TEST_FOLDER = data_path / 'Testing'
# INFO_FILE = data_path.parent / '[outdated]201103_M&Ms Dataset Information - diagnosis - opendataset.csv'
INFO_FILE = data_path.parent / '201207_M&Ms Dataset information - diagnosis - opendataset.csv'
# INFO_FILE = data_path.parent / '210219_M&Ms Dataset information - diagnosis - opendataset.csv'
RESULTS_FOLDER = Path(config['paths']['misc'])

ACDC_TRAIN_PATH = HOME_PATH / Path(config['paths']['ACDC']['processed']).joinpath('Training')
# csv columns:
CODE = 'External code'
AGE = 'Age'
SEX = 'Sex'
PATHOLOGY = 'Pathology'
VENDOR = 'Vendor'
CENTRE = 'Centre'
ED = 'ED'
ES = 'ES'

# working with the pre-processed data
ED_SUFFIX = '_ed.nii.gz'
ED_GT_SUFFIX = '_ed_gt.nii.gz'
ES_SUFFIX = '_es.nii.gz'
ES_GT_SUFFIX = '_es_gt.nii.gz'

# def import_class(name):
#     module_name, class_name = name.rsplit('.', 1)
#     module = importlib.import_module(module_name)
#     return getattr(module, class_name)

class MultiDataLoader():
    def __init__(self, data_owners=None, images_dir=TRAIN_FOLDER, split_ratio=0.1, no_split=False):
        # get all subjects from one folder
        print(f"Initializing Data Loader for {data_owners}") if data_owners is not None else print("Initialized Data Loader")
        print(f"Loading {images_dir}")

        self.data_owners = data_owners
        self.split_ratio = split_ratio
        self.subjects = []
        self.pathologies = []
        self.subjects = OrderedDict()
        for owner in data_owners:
            self.subjects[owner] = []
        # self.subjects = dict.fromkeys(data_owners, []) # initializes a dictionary with data owners as keys and empty list
        self.mri_paths, self.gts_paths = {}, {}
        for center in data_owners:
            self.mri_paths[center], self.gts_paths[center] = [], []
        excluded_cases = []
        if 'ACDC' in data_owners:
            acdc_train = ACDC_TRAIN_PATH
            acdc_subject_names = os.listdir(acdc_train)
            for subject_name in acdc_subject_names:
                # Refer to jupyter notebook visualizations/InspectingACDC for clarity.
                info_path = acdc_train.joinpath(subject_name).joinpath('Info.cfg')
                f = open(info_path, "r")
                text = f.read()
                temp = [list(x.split(':')) for x in text.split('\n') if len(x)>0]
                infoCFG = {k:v.strip() for k,v in temp}
                if infoCFG['Group'] not in config['data']['pathologies']:
                    continue
                
                #Healthy vs abnormal
                if len(config['data']['pathologies']) > 2 and isinstance(config['data']['binary_labels'], list):
                    if infoCFG['Group'] == 'NOR':
                        label = 'NOR'
                    else:
                        label = 'ABNOR'
                else:
                    label = infoCFG['Group']

                ESind = infoCFG['ES']
                EDind = infoCFG['ED']
                ED_suffix = f'_frame{EDind.zfill(2)}.nii.gz'
                ED_GT_suffix = f'_frame{EDind.zfill(2)}_gt.nii.gz'
                ES_suffix = f'_frame{ESind.zfill(2)}.nii.gz'
                ES_GT_suffix = f'_frame{ESind.zfill(2)}_gt.nii.gz'
                
                mris = [acdc_train / subject_name / str(subject_name + suffix) for suffix in [ED_suffix, ES_suffix]]
                gts = [acdc_train / subject_name / str(subject_name + suffix) for suffix in [ED_GT_suffix, ES_GT_suffix]]
                self.mri_paths['ACDC']+=mris
                self.gts_paths['ACDC']+=gts
                
                temp_sub = []
                for mri, gt in zip(mris, gts):
                    subject = tio.Subject(
                        mri=tio.ScalarImage(mri),
                        gt=tio.LabelMap(gt),
                        code=subject_name,
                        pathology=label,
                        centre='ACDC'
                        )
                    temp_sub.append(subject)
                self.pathologies.append(label)
                self.subjects['ACDC'].append(list(temp_sub))
                # self.subjects.append(list(temp_sub))

        #### MNM
        info = pd.read_csv(INFO_FILE)
        subject_names = sorted([child.name for child in Path.iterdir(images_dir)
                                if Path.is_dir(child)])

        for subject_name in subject_names:
            case = info.loc[info[CODE] == subject_name]
            if len(case) == 0:
                excluded_cases.append(subject_name)
                continue
            # load this image
            case_code = case[CODE].item()
            # if data_owners!=None and config['centralized']['centre']!='MNM': # handle different data owners
            if str(case[CENTRE].item()) not in data_owners:
                continue
            if case[PATHOLOGY].item() not in config['data']['pathologies']:
                continue

            #Healthy vs abnormal
            if len(config['data']['pathologies']) > 2 and isinstance(config['data']['binary_labels'], list):
                if case[PATHOLOGY].item() == 'NOR':
                    label = 'NOR' # healthy
                else:
                    label = 'ABNOR' # abnormal
            else:
                label = case[PATHOLOGY].item()

            mris = [images_dir / case_code / str(case_code + suffix) for suffix in [ED_SUFFIX, ES_SUFFIX]]
            gts = [images_dir / case_code / str(case_code + suffix) for suffix in [ED_GT_SUFFIX, ES_GT_SUFFIX]]
            self.mri_paths[str(case[CENTRE].item())]+=mris
            self.gts_paths[str(case[CENTRE].item())]+=gts

            if config['data']['concatenateEDES']:
                subject = tio.Subject(
                    mri=tio.ScalarImage(mris),
                    gt=tio.LabelMap(gts),
                    code=case_code,
                    # age=case[AGE].item(),
                    # sex=case[SEX].item(),
                    pathology=label,
                    # # vendor=case[VENDOR].item(),
                    centre=case[CENTRE].item(),
                    ed=case[ED].item(),
                    es=case[ES].item(),
                    )
                self.subjects.append(subject)
                self.pathologies.append(label)

            else:
                temp_sub = []
                for mri, gt in zip(mris, gts):
                    subject = tio.Subject(
                        mri=tio.ScalarImage(mri),
                        gt=tio.LabelMap(gt),
                        code=case_code,
                        pathology=label,
                        # vendor=case[VENDOR].item(),
                        centre=case[CENTRE].item(),
                        # ed=case[ED].item(),
                        # es=case[ES].item(),
                        )
                    temp_sub.append(subject)
                
                self.pathologies.append(label) # this is just to fit the label encoder. The pathology is loaded from the subject class at train time.
                #Save them as tuples so that they will be together in training or test set.
                self.subjects[str(case[CENTRE].item())].append(list(temp_sub))

        normal_count = len([i for i in self.pathologies if i=='NOR'])
        
        if not RESULTS_FOLDER.exists():
            RESULTS_FOLDER.mkdir(parents=True)
        # fit encoders to load on runtime. It's necessary to do it here because it's not guaranteed that the batch will load all classes
        print(set(self.pathologies))
        le = LabelEncoder()
        # le.fit(np.array(self.pathologies))
        le.fit(config['data']['binary_labels'])
        with open(Path(config['paths']['misc']).joinpath(config['names']['labelencoder']), 'wb') as handle:
            # If we were to do this on runtime, the encoding could change from epoch to epoch
            pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # set folds for crossvalidation
        self.fold_indices = []
        if config['federated']['isTrue']:
            self.subjects = [j for i in self.subjects.values() for j in i] #list(self.subjects.values())
            if config['cvtype'] == 'LCO':
                train_size = int(0.9*len(self.subjects))
                train, val = self.subjects[:train_size], self.subjects[train_size:]
                self.fold_indices = (train, val)
            else: 
                x,y = np.array(self.subjects), np.array(self.pathologies)
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1) 
                for train_index, test in skf.split(x, y):
                    train_size = int(0.9*len(train_index))
                    train, val = train_index[:train_size], train_index[train_size:]
                    self.fold_indices.append((train, val, test))
        else:
            if config['cvtype'] == 'LCO':
                landmarks_paths = {}
                names_to_store_for_FL = [] 
                for i in range(len(data_owners)):
                    mri_to_train_hist, gt_to_train_hist = [], []
                    test = data_owners[i]
                    train = [x for j,x in enumerate(data_owners) if j!=i] # The validation comes out of the centers we train on
                    self.fold_indices.append((train, test))
                    # For histogram matching, we need 4 different sets:
                    hist_name = str()
                    for site in train:
                        hist_name+=site[0]#Take the first letter
                        mri_to_train_hist += self.mri_paths[site]
                        gt_to_train_hist += self.gts_paths[site]
                    landmarks_paths[test] = [RESULTS_FOLDER / f"histogramLandmarks_{hist_name}-ROI.npy", mri_to_train_hist, gt_to_train_hist] # Save this list, load in federated as well
                    names_to_store_for_FL.append(hist_name)
                with open(RESULTS_FOLDER.joinpath('LCO_names.pkl'), 'wb') as handle:
                    # If we were to do this on runtime, the encoding could change from epoch to epoch
                    pickle.dump(names_to_store_for_FL, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    # print(test)
                    # print(f'train, {train}')
            else:
                self.subjects = [j for i in self.subjects.values() for j in i] #list(self.subjects.values())
                with open(RESULTS_FOLDER.joinpath(config['names']['fold_split']), 'rb') as handle: 
                    # Use the visualizations/folds_fixing.ipynb to generate the file based on a federated experiment and utils/folds_fixing.py to map it to CDS
                    self.fold_indices = pickle.load(handle)

        if config['test_unseen'] or no_split: # This will override any previous splitting
            # Testing on the entirety of an unseen center:
            print(f"You are loading the whole dataset for center {data_owners[0]} as test set:")
            test = np.array(list(range(len(self.subjects))))
            train = val = test
            for i in range(5):
                self.fold_indices.append((train, val, test)) # dummy, all indices are equal, the entire set is used for every fold's model
        subject_instances_count = len(self.subjects)
        print('Total number of subject instances is', subject_instances_count)
        print(f'Normal count is {normal_count}')
        print(f'Rest is {subject_instances_count-normal_count}')
        
        if config['cvtype'] == 'LCO':

            self.landmarks = []
            if config['federated']['isTrue']:
                # Load file generated from CDS
                with open(RESULTS_FOLDER.joinpath('LCO_names.pkl'), 'rb') as handle:
                    hist_names = pickle.load(handle)
                for hist_name in hist_names:
                    landmarks_for_fold = RESULTS_FOLDER / f"histogramLandmarks_{hist_name}-ROI.npy" # This should  be equal to L311 in CDS
                    self.landmarks.append(landmarks_for_fold)
                # IMPORTANT NOTE: for this to work as intended the compared CDS experiment from which these files originated
                # needs to be given the center list in the exact same order.
            else:
                for key in landmarks_paths:
                    print(f"Training landmarks {landmarks_paths[key][0]}...")
                    landmarks_for_fold = (
                        landmarks_paths[key][0]
                        if landmarks_paths[key][0].is_file() #if it's federated load roi calculated from centralized.
                        else tio.HistogramStandardization.train(landmarks_paths[key][1],
                                                                mask_path=landmarks_paths[key][2], # if then None else paths list.
                                                                output_path=landmarks_paths[key][0])
                    )
                    self.landmarks.append(landmarks_for_fold)

        else:

            if config['model']['arch']['args']['in_ch'] == 1 and not config['data']['multiply_by_mask']:
                self.gts_paths = None # Dont use masks for ROI norm
                hist_names = str()
                for i in config['data']['centres']:
                    hist_names+=i[0]
                landmarks_path = RESULTS_FOLDER / f'histogramLandmarks_{hist_names}_wholeArea.npy'
            else: # heart_area works nicely
                print("Using histograms based on ROIs")
                self.gts_paths = [j for i in self.gts_paths.values() for j in i]
                landmarks_path = RESULTS_FOLDER / f"histogramLandmarks_{config['names']['histogram']}.npy"

            mri_paths = [j for i in self.mri_paths.values() for j in i]

            if not landmarks_path.is_file() and config['federated']['isTrue']:
                raise NameError(f"{landmarks_path} was not found, to train one first run your script in Centralized mode.")
            
            self.landmarks = (
                landmarks_path
                if landmarks_path.is_file() #if it's federated load roi calculated from centralized.
                else tio.HistogramStandardization.train(mri_paths,
                                                        mask_path=self.gts_paths, # if then None else paths list.
                                                        output_path=landmarks_path)
            )
            np.set_printoptions(suppress=True, precision=3)
            print('\nTrained landmarks:', self.landmarks)
    
    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index):
        return self.subjects[index], self.pathologies[index]
    
    def get_paths(self):
        return self.mri_paths, self.gts_paths

    def get_fold_splits(self):
        return self.fold_indices

    def get_whole_data(self):
        return np.array(self.subjects), np.array(self.pathologies)

    def get_dataset(self, fold_index, transformations=config['data']['transformations'], LCO_FL_test=False): #cross validation hardcoded to 10 splits
    # Transform and define dataset object
        if config['cvtype'] == 'LCO':
            landmarks = self.landmarks[fold_index]
        else:
            landmarks = self.landmarks
        print(f"Using landmarks {landmarks}")

        if transformations == 'B':
            training_transform = Compose([
                HistogramStandardization({'mri': landmarks}), #correct this mean
                CropOrPad((150,150,10), mask_name='gt'), #sum crop areas to ascertain
                RescaleIntensity((0,1)),
                OneOf([
                    ### Shape (Basic)
                    RandomFlip(axes=('LR', 'SI')),
                    RandomAffine(scales=(0.9, 1.2),degrees=10,isotropic=True,image_interpolation='nearest'), # PROBLEMATIC? TypeError: not a sequence  
                ], p=0.5)

            ])
        elif transformations == 'S':
            training_transform = Compose([
                HistogramStandardization({'mri': landmarks}), #correct this mean
                CropOrPad((150,150,10), mask_name='gt'), #sum crop areas to ascertain
                RescaleIntensity((0,1)),
                OneOf([
                    ### Shape (Basic)
                    RandomFlip(axes=('LR', 'SI')),
                    RandomAffine(scales=(0.9, 1.2),degrees=10,isotropic=True,image_interpolation='nearest'), # PROBLEMATIC? TypeError: not a sequence  
                    # Shape deformations
                    RandomElasticDeformation(num_control_points=5, max_displacement=2), # If you change parameters a warning about possible folding will pop up
                ], p=1)

            ])
        elif transformations == 'I':
            training_transform = Compose([
                HistogramStandardization({'mri': landmarks}), #correct this mean
                CropOrPad((150,150,10), mask_name='gt'), #sum crop areas to ascertain
                RescaleIntensity((0,1)),
                OneOf([
                    RandomSpike(),
                    RandomBiasField(),
                    RandomNoise(),
                    RandomGamma()
                ], p=0.5)

            ])
        elif transformations == 'SI':
            training_transform = Compose([
                HistogramStandardization({'mri': landmarks}), #correct this mean
                CropOrPad((150,150,10), mask_name='gt'), #sum crop areas to ascertain
                RescaleIntensity((0,1)),
                OneOf([
                    ### Shape (Basic)
                    RandomFlip(axes=('LR', 'SI')),
                    RandomAffine(scales=(0.9, 1.2),degrees=10,isotropic=True,image_interpolation='nearest'), # PROBLEMATIC? TypeError: not a sequence  
                    # Shape deformations
                    RandomElasticDeformation(num_control_points=5, max_displacement=2), # If you change parameters a warning about possible folding will pop up
                #     # # # intensity
                    RandomSpike(),
                    RandomBiasField(),
                    RandomNoise(),
                    RandomGamma(), # non linear pertubation of the image contrast. Good to have we know the contrast is changing from scanner to scanner
                # #     # # # The following augmentations make the images non-readable:
                #     # RandomSwap(patch_size=(16, 16, 1)),
                #     # RandomBlur(), # Maybe if you reduce the blurring
                #     # RandomMotion(),
                #     # RandomGhosting()
                ], p=0.5)

            ])
        elif transformations == 'N':
            training_transform = Compose([
                HistogramStandardization({'mri': landmarks}), #correct this mean
                CropOrPad((150,150,10), mask_name='gt'), #sum crop areas to ascertain
                RescaleIntensity((0,1))
            ])

        if config['test_without_hist']:
            print("WARNING: Test set is NOT using Histogram Standardization")
            test_transform = Compose([
                CropOrPad((150,150,10), mask_name='gt'),
                RescaleIntensity((0,1))
            ])
        else:
            test_transform = Compose([
                HistogramStandardization({'mri': landmarks}),
                CropOrPad((150,150,10), mask_name='gt'),
                RescaleIntensity((0,1))
            ])

        # training_split_ratio = 0.9
        # num_subjects = self.__len__()
        # print('Dataset size:', num_subjects, 'images')
        # num_training_subjects = int(training_split_ratio * num_subjects)
        if config['cvtype'] == 'LCO':
            if config['federated']['isTrue']:
                # assert isinstance(self.fold_indices, tuple)
                training_subjects, validation_subjects = self.fold_indices
                test_subjects = []
                if LCO_FL_test:
                    test_subjects = training_subjects + validation_subjects 
                    print(f"Test center has size {len(test_subjects)}")
            else:
                indices = self.fold_indices[fold_index]
                training_subjects = []
                for center in indices[0]: # train set
                    training_subjects = training_subjects + self.subjects[center]
                random.shuffle(training_subjects)
                train_size = int(0.9*len(training_subjects))
                training_subjects, validation_subjects = training_subjects[:train_size], training_subjects[train_size:]
                test_subjects = self.subjects[indices[1]]
                # unfold:
                training_subjects = [j for i in training_subjects for j in i]
                validation_subjects = [j for i in validation_subjects for j in i]
                test_subjects = [j for i in test_subjects for j in i]
                print(f"Fold {fold_index} (test center {indices[1]})")
        else:
            indices = self.fold_indices[fold_index]
            training_subjects = np.array(self.subjects)[indices[0]].tolist()
            self.train_set_length = len(training_subjects)
            validation_subjects = np.array(self.subjects)[indices[1]].tolist()
            test_subjects = np.array(self.subjects)[indices[2]].tolist()
            print(f"Fold {fold_index} (test indices {indices[2].tolist()})")

        if isinstance(training_subjects[0], list):
            training_subjects = [item for sublist in training_subjects for item in sublist]
            validation_subjects = [item for sublist in validation_subjects for item in sublist]
            test_subjects = [item for sublist in test_subjects for item in sublist]
        
        training_set = tio.SubjectsDataset(
            training_subjects, transform=training_transform)

        validation_set = tio.SubjectsDataset(
            validation_subjects, transform=test_transform)
        
        if not len(test_subjects) < len(training_subjects): # LCO testing phase will not meet this requirement
            print('Training set:', len(training_set)/2, 'subjects')
            print('Validation set:', len(validation_set)/2, 'subjects')

        if not test_subjects:
            return training_set, validation_set, None
            
        test_set = tio.SubjectsDataset(
            test_subjects, transform=test_transform)

        print('Test set:', len(test_set)/2, 'subjects')
        return training_set, validation_set, test_set 

    def load(self, fold_index, transformations=config['data']['transformations'], LCO_FL_test=False):
        training_set, validation_set, test_set = self.get_dataset(fold_index, transformations, LCO_FL_test)
        training_batch_size = (
            int(np.ceil(len(training_set) / config['hyperparameters']['iteration_number']))
            if config['federated']['isTrue']
            else config['hyperparameters']['training_batch_size'] )
        # print(len(training_set))
        # print(len(training_set/training_batch_size))) 
        test_batch_size = config['hyperparameters']['test_batch_size']
        print(f"Training with a batch size of: {training_batch_size} and validation/test batch size: {test_batch_size}")
        training_loader = torch.utils.data.DataLoader(
            training_set, batch_size=training_batch_size)
        validation_loader = torch.utils.data.DataLoader(
            validation_set, batch_size=test_batch_size)
        if not test_set:
            return training_loader, validation_loader, None 
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=test_batch_size)
        return training_loader, validation_loader, test_loader


if __name__ == '__main__':
    # if config['federated']['isTrue']:
    #     dl = MultiDataLoader(["Vall d'Hebron", "Sagrada Familia"])
    #     t, v = dl.load(0)
    #     training_loader, test_loader = dl.load(fold_index=0)
    #     # batch = next(iter(training_loader))
    #     print(len(training_loader))
    if config['federated']['isTrue']:
        print("Testing federated loading ...")
        data = {} 
        for centre in config['data']['centres']:
            dl = MultiDataLoader([centre])
            fold_splits = dl.get_fold_splits()
            data[centre]=(centre, dl)
        
        dl,_ = data[centre][1].load(0)
        print(len(dl))
    else:
        print("Testing centralized loading ...")
        data = MultiDataLoader(data_owners=config['data']['centres'])
        fold_splits = data.get_fold_splits()
        next(iter(data))
