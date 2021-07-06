# Perform normalization on data based on a reference image
#
# Input:
#           - path to images (it must include a sub-folder with name MRI where all images to be normalized are expected
#           to be)
#           - path include filename to the reference image
#           - output path where to save the normalized images
#
# Output:
#           - normalized images in format .nii.gz, one per input image
#
# Example call: python histogram_normalization.py D:\Xenia\euCanSHare\Datasets\UKBB_MeatProject\ D:\Xenia\euCanSHare\Datasets\UKBB_MeatProject\reference\1095416.nii.gz D:\Xenia\euC
# anSHare\Datasets\UKBB_MeatProject\images
#
# Author: Xenia Gkontra, UB, 2020


import os, sys
from src.data_handling.mnm_dataset import *

import pdb

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn import linear_model
import argparse
import pdb
import nibabel as nib
import pickle
import pandas as pd
from pathlib import Path
from os.path import splitext, join as pjoin, split as psplit
import pdb
import matplotlib.pyplot as plt


# Carlos' function from UKBB, based initially at https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
def hist_norm(source, template):
    olddtype = source.dtype
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    # print('len svalues',len(s_values))
    # print('s_counts',s_counts)
    t_values, t_counts = np.unique(template, return_counts=True)
    # print('len tvalues',len(t_values))
    # plt.hist(t_counts, normed=False, bins=range(479))
    # plt.xlabel("x")
    # plt.ylabel("f(x)")
    # plt.show()
    # print('t_values',t_values)
    # print('t_counts',t_counts)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    interp_t_values = interp_t_values.astype(olddtype)
    return interp_t_values[bin_idx].reshape(oldshape)

def ecdf(x):
    """convenience function for computing the empirical CDF"""
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return vals, ecdf


def hist_norm_ROI(source_init, template, ROI_source, ROI_template):

    olddtype = source_init.dtype
    oldshape = source_init.shape

    source_img = source_init.copy()
    # Change min-max range of source image to be same as template
    min_s = np.min(source_img[ROI_source > 0])
    max_s = np.max(source_img[ROI_source > 0])
    min_t = np.min(template[ROI_template > 0])
    max_t = np.max(template[ROI_template > 0])
    source_img[ROI_source > 0] = (source_img[ROI_source > 0] - min_s)*(max_t - min_t)/(max_s - min_s) + min_t

    # If size source and ROI is not the same delete the last slice of the img
    if source_img.shape[2] != ROI_source.shape[2]:
        print("Empty slice deleted")
        source_img = source_img[:, :, :-1]

    index_source = np.where(ROI_source > 0)
    index_template = np.where(ROI_template > 0)
    values_source = source_img[index_source]
    values_template = template[index_template]
    s_values, bin_idx, s_counts = np.unique(values_source, return_inverse=True,
                                            return_counts=True)
    # print('len svalues',len(s_values))
    # print('s_counts',s_counts)
    t_values, t_counts = np.unique(values_template, return_counts=True)
    # print('len tvalues',len(t_values))
    # plt.hist(t_counts, normed=False, bins=range(479))
    # plt.xlabel("x")
    # plt.ylabel("f(x)")
    # plt.show()
    # print('t_values',t_values)
    # print('t_counts',t_counts)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    interp_t_values = interp_t_values.astype(olddtype)
    normalized_img = source_img.copy()
    normalized_img[index_source] = interp_t_values[bin_idx].astype(olddtype)
    return normalized_img


def preprocess_image(file_img, file_img_ref, frames, output_file):
    img_new = nib.load(file_img)
    imgs_temp = nib.four_to_three(img_new)
    img_normalized = np.zeros(img_new.shape)
    for phase in range(0, img_new.shape[3]):
        desired_3d_img = imgs_temp[phase].get_fdata()
        # Call function to normalize
        img_norm = hist_norm(desired_3d_img, file_img_ref)
        img_normalized[:, :, :, phase] = img_norm
        # Save image
    img_norm_niftii = nib.Nifti1Image(img_normalized, img_new.affine, img_new.header)
    nib.save(img_norm_niftii, output_file)
    return img_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Normalize images using histogram matching.")
    parser.add_argument("input_path", type=str, default='input_path', help="Path to images to normalize.")
    parser.add_argument("input_path_ref", type=str, default='input_path_ref', help="Path to reference image.")
    parser.add_argument("output_path", type=str, default='output_path', help="Path to save normalized images.")
    args = vars(parser.parse_args())

    input_path = args["input_path"]
    ref_path = args["input_path_ref"]
    output_path = args["output_path"]

    dataset = Dataset(input_path)
    with open('dataset.pickle', 'wb') as f:
        pickle.dump(dataset, f)

    # with open('dataset.pickle', 'rb') as f:
    #     dataset = pickle.load(f)
    #
    imgs = dataset.dataset['img_filenames']
    msks_ed = dataset.dataset['msk_ed']
    msks_es = dataset.dataset['msk_es']
    es_frames = dataset.es_frames

    # Read reference image

    img_ref_all = nib.load(ref_path).get_fdata()
    img_ref = img_ref_all[:, :, :, 0]
    phases_names = ["ED", "ES"]
    counter = 0
    exceptions = []

    for img, msk_ed, msk_es in zip(imgs, msks_ed, msks_es):
        # Get only name
        name_img = os.path.basename(img)

        if not os.path.exists(os.path.join(output_path, name_img)):
            try:
                ed_frame = 0
                es_frame = int(es_frames.loc[es_frames['feid'] == int(dataset.patient_ids[counter])]['es_frame'])
                counter += 1
                # Create numpy with the phases
                phases = [ed_frame, es_frame]
                img_new = preprocess_image(img, img_ref, phases, os.path.join(output_path, name_img))
            except:
                exceptions.append(name_img)
                print("****Cannot unzip file: " + name_img)

    pd.DataFrame(exceptions).to_csv(os.path.join("exceptions.csv"))
