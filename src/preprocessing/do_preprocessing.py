import os
import copy
import tempfile
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from tqdm import tqdm
import sys
from bias_correction import n4_bias_correction
from resampling import resample

with open('../classification/config.yaml') as file:
    config = yaml.safe_load(file)

DATA_PATH = config['paths']['raw_dataset']
DESTINATION_PATH = config['paths']['dataset']

data_path = Path(DATA_PATH)
dest_path = Path(DESTINATION_PATH)

TRAIN_FOLDER = data_path / 'Training' / 'Labeled'
TRAIN_FOLDER_DEST = dest_path / 'Training'

VAL_FOLDER = data_path / 'Validation'

TEST_FOLDER = data_path / 'Testing'
TEST_FOLDER_DEST = dest_path / 'Testing'

INFO_FILE = data_path.parent / '201207_M&Ms Dataset information - diagnosis - opendataset.csv'

# csv columns:
CODE = 'External code'
ED = 'ED'
ES = 'ES'

#
IMG_SUFFIX = '_sa.nii.gz'
LBL_SUFFIX = '_sa_gt.nii.gz'
ED_SUFFIX = '_ed.nii.gz'
ED_GT_SUFFIX = '_ed_gt.nii.gz'
ES_SUFFIX = '_es.nii.gz'
ES_GT_SUFFIX = '_es_gt.nii.gz'
info = pd.read_csv(INFO_FILE)

def process_case(info, subject, images_dir, images_dir_dest):
    '''Extracts ED and ES time points from 4D MRI and applies bias
    field correction
    '''
    case = info.loc[info[CODE] == subject]
    # load this image
    case_code = case[CODE].item()
    img_path = images_dir / case_code / str(case_code + IMG_SUFFIX)
    label_path = images_dir / case_code / str(case_code + LBL_SUFFIX)
    img = nib.load(img_path)
    gt = nib.load(label_path)
    ed_id = int(case[ED].item())
    es_id = int(case[ES].item())

    # create save dir if doesn't exist
    save_path = images_dir_dest / case_code
    if not save_path.exists():
        save_path.mkdir(parents=True)  # also create missing parents
    # for both ED and ES
    for idx, suffix, gt_suffix in zip([ed_id, es_id],
                                      [ED_SUFFIX, ES_SUFFIX],
                                      [ED_GT_SUFFIX, ES_GT_SUFFIX]):
        # extract time point
        data = img.get_fdata()[..., idx]
        lbl = gt.get_fdata()[..., idx]
        # save the ground truth right away
        lbl = nib.Nifti1Image(lbl, gt.affine)
        new_header = lbl.header
        lbl, new_spacing = resample(lbl)
        new_header['pixdim'][1:4] = new_spacing #change spacing info
        nib.save(nib.Nifti1Image(lbl, gt.affine, new_header),
                 save_path / str(case_code + gt_suffix))
        # correct bias field for data and save
        with tempfile.NamedTemporaryFile(suffix='.nii') as f:
            data = nib.Nifti1Image(data, img.affine)
            new_header = data.header
            data, new_spacing = resample(data)
            new_header['pixdim'][1:4] = new_spacing #change spacing info
            nib.save(nib.Nifti1Image(data, img.affine, new_header), f.name)
            n4_bias_correction(f.name, save_path / str(case_code + suffix))

# get all subjects from one folder
images_dir = TRAIN_FOLDER
images_dir_dest = TRAIN_FOLDER_DEST
subjects = sorted([child.name for child in Path.iterdir(images_dir) if Path.is_dir(child)])
for subject in tqdm(subjects):
    process_case(info, subject, images_dir, images_dir_dest)

images_dir = VAL_FOLDER
images_dir_dest = TRAIN_FOLDER_DEST 
subjects = sorted([child.name for child in Path.iterdir(images_dir) if Path.is_dir(child)])
for subject in tqdm(subjects):
    process_case(info, subject, images_dir, images_dir_dest)

images_dir = TEST_FOLDER
images_dir_dest = TEST_FOLDER_DEST
subjects = sorted([child.name for child in Path.iterdir(images_dir) if Path.is_dir(child)])
for subject in tqdm(subjects):
    process_case(info, subject, images_dir, images_dir_dest)
