import os
import tempfile
import SimpleITK as sitk


def n4_bias_correction(load_path, save_path=None, num_iterations=1, num_fitting_levels=4):
    image = sitk.ReadImage(load_path, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([num_iterations]
                                           * num_fitting_levels)
    output = corrector.Execute(image)
    if save_path is not None:
        sitk.WriteImage(output, str(save_path))
    return output