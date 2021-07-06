import os
import copy
import tempfile
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
import yaml
import shutil
from pathlib import Path
from tqdm import tqdm
import sys
from bias_correction import n4_bias_correction
from resampling import resample
with open('../classification/config.yaml') as file:
    config = yaml.safe_load(file)

DATA_PATH = config['paths']['ACDC']['raw']
DESTINATION_PATH = config['paths']['ACDC']['processed']

data_path = Path(DATA_PATH)
dest_path = Path(DESTINATION_PATH)

TRAIN_FOLDER = data_path / 'training' 
TRAIN_FOLDER_DEST = dest_path / 'Training'

TEST_FOLDER = data_path / 'testing'
TEST_FOLDER_DEST = dest_path / 'Testing'


def process_case(subject, images_dir, images_dir_dest):
    '''Extracts ED and ES time points from 4D MRI and applies bias
    field correction
    '''
    info_path = images_dir.joinpath(subject).joinpath('Info.cfg')
    f = open(info_path, "r")
    text = f.read()
    temp = [tuple(x.split(':')) for x in text.split('\n') if len(x)>0]
    infoCFG = {k:v.strip() for k,v in temp}

    ESind = infoCFG['ES']
    EDind = infoCFG['ED']
    ED_suffix = f'_frame{EDind.zfill(2)}.nii.gz'
    ED_GT_suffix = f'_frame{EDind.zfill(2)}_gt.nii.gz'
    ES_suffix = f'_frame{ESind.zfill(2)}.nii.gz'
    ES_GT_suffix = f'_frame{ESind.zfill(2)}_gt.nii.gz'

    # create save dir if doesn't exist
    save_path = images_dir_dest / subject
    if not save_path.exists():
        save_path.mkdir(parents=True)  # also create missing parents
    shutil.copy(info_path, save_path)
    # for both ED and ES
    for suffix, gt_suffix in zip([ED_suffix, ES_suffix],
                                 [ED_GT_suffix, ES_GT_suffix]):

        img_path = images_dir / subject / str(subject + suffix)
        label_path = images_dir / subject / str(subject + gt_suffix)
        # load this image
        img = nib.load(img_path)
        gt = nib.load(label_path)
        data = img.get_fdata()
        lbl = gt.get_fdata()
        # save the ground truth right away
        lbl = nib.Nifti1Image(lbl, gt.affine)
        new_header = lbl.header
        lbl, new_spacing = resample(lbl)
        new_header['pixdim'][1:4] = new_spacing #change spacing info
        nib.save(nib.Nifti1Image(lbl, gt.affine, new_header),
                 save_path / str(subject + gt_suffix))
        # correct bias field for data and save
        with tempfile.NamedTemporaryFile(suffix='.nii') as f:
            data = nib.Nifti1Image(data, img.affine)
            new_header = data.header
            data, new_spacing = resample(data)
            new_header['pixdim'][1:4] = new_spacing #change spacing info
            nib.save(nib.Nifti1Image(data, img.affine, new_header), f.name)
            n4_bias_correction(f.name, save_path / str(subject + suffix))

# get all subjects from one folder
images_dir = TRAIN_FOLDER
images_dir_dest = TRAIN_FOLDER_DEST
subjects = sorted([child.name for child in Path.iterdir(images_dir) if Path.is_dir(child)])
for subject in tqdm(subjects):
    process_case(subject, images_dir, images_dir_dest)

images_dir = TEST_FOLDER
images_dir_dest = TEST_FOLDER_DEST
subjects = sorted([child.name for child in Path.iterdir(images_dir) if Path.is_dir(child)])
for subject in tqdm(subjects):
    process_case(subject, images_dir, images_dir_dest)
