import numpy as np
from scipy import ndimage
def resample(nifti_image, new_spacing=[1,1,1]):
    # Test what happens if new spacing is the same as old spacing. Bug still there?
    # Determine current pixel spacing
    spacing = np.array(nifti_image.header.get_zooms(), dtype=np.float32)
    image = np.array(nifti_image.dataobj)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor # to check?

    image = ndimage.zoom(image, real_resize_factor, order=0, mode='constant') # cubic
#     cv2.resize()
    return image, new_spacing