import os
import json
import shutil
import numpy as np
import pandas as pd
import pydicom as pdcm
import SimpleITK as sitk

from PIL import Image
from glob import glob
from tqdm import tqdm
from monai.transforms import Rotate

# Custom functions
from utils import create_folder_if_not_exists, sort_human

# Extract metadata
def load_images(path_images):
    """
    Load DICOM files in the path as an SITK Image.
    """
    files = glob(path_images + '/*')
    files = sort_human(files)
    frames = [pdcm.read_file(dcm) for dcm in files]
    arr = np.stack([s.pixel_array for s in frames], axis=0)  # axis = 0: image.shape = (z, y, x)

    # Construct metadata
    metadata = frames[0]
    metadata_json = json.loads(metadata.to_json())  # frames[0]: first frame

    # Fetch metadata
    direction = metadata_json['00200037']['Value'] + [0, 0, 1]  # Image Orientation (Patient)
    origin = metadata_json['00200032']['Value']  # Image Position (Patient)

    # Get (x, y, z) spacing
    xy_spacing = metadata_json['00280030']['Value']
    # Set z-spacing to 1 for all patients, since we only have one slice per time point
    z_spacing = 1

    if not z_spacing > 0:
        raise ValueError('z_spacing = {} <= 0'.format(z_spacing))

    spacing = xy_spacing + [z_spacing]

    # Convert to Image
    image = sitk.GetImageFromArray(arr)
    image.SetOrigin(origin)
    image.SetDirection(direction)
    image.SetSpacing(spacing)

    return arr, metadata, image

def adjust_voxel_spacing(image, out_spacing, is_label):
    """
    Change the voxel spacing of arr to out_spacing. Use the relevant metadata for this resampling.
    """
    # Initialize variables
    original_origin = image.GetOrigin()
    original_direction = image.GetDirection()
    original_spacing = image.GetSpacing()

    # Determine output shape after
    original_size = image.GetSize()
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(original_direction)
    resample.SetOutputOrigin(original_origin)
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    # Perform spacing conversion
    resampled_image = resample.Execute(image)

    return resampled_image

# Data preprocessing
def n4_bias_field_correction(input_image):
    """
    The N4 bias field correction algorithm is a popular method for correcting low frequency intensity non-uniformity
    present in MRI image data known as a bias or gain field.
    """
    # Use Otsu's threshold estimator to separate background and foreground.
    mask = sitk.OtsuThreshold(input_image, 0, 1, 200)

    # Cast data type of input image to float32
    input_image = sitk.Cast(input_image, sitk.sitkFloat32)

    # Perform N4 bias field correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    output_image = corrector.Execute(input_image, mask)
    log_bias_field = corrector.GetLogBiasFieldAsImage(input_image)

    return output_image, mask, log_bias_field

def rotate_arr(arr, orientation, threshold=10):
    """
    Positive rotation angle = rotating clockwise
    Negative rotation angle = rotating anti-clockwise
    """
    # Select middle frames to determine rotation angle
    arr_i = arr[arr.shape[0] // 2]
    arr_i = np.expand_dims(arr_i, axis=0)

    # Determine rotation angle based on patient's orientation
    if orientation == 'rightup':
        print('\tRight-up patient')
        angle_list = [x * np.pi / 180 for x in range(-45, 45, 2)]
    elif orientation == 'laying':
        angle_list = [x * np.pi / 180 for x in range(-90, 0, 2)]
        print('\tLaying patient')
    else:
        raise ValueError('Orientation = {} is invalid'.format(orientation))

    # Add angle = 0
    angle_list = angle_list + [0]
    angle_list = list(set(angle_list))
    angle_list.sort()

    best_col = None
    best_max_colsum = None
    best_angle = None
    for angle in angle_list:
        rotator = Rotate(angle=angle)

        # Rotate
        arr_i_rot = rotator(arr_i)

        # Determine number of rows with value > threshold
        arr_i_rot_bool = arr_i_rot > threshold
        arr_i_rot_colsum = np.sum(arr_i_rot_bool, axis=1)
        max_colsum = arr_i_rot_colsum.max()

        # Find column with maximum number of rows with value > threshold
        _, j = np.where(arr_i_rot_colsum == max_colsum)

        if best_col is None or best_max_colsum < max_colsum:
            best_max_colsum = max_colsum
            best_col = j.max()
            best_angle = angle

        # Determine original max_colsum (i.e. for angle = 0)
        if angle == 0:
            original_max_colsum = max_colsum
            original_angle = angle

    if best_max_colsum <= 1.1 * original_max_colsum:
        best_angle = original_angle

    # Rotate original input array
    rotator = Rotate(angle=best_angle)
    arr_rot = rotator(arr)

    return arr_rot

def save_image_slice(image, filename, slice_nr=0):
    array = sitk.GetArrayFromImage(image)
    save_array_slice(array=array, filename=filename, slice_nr=slice_nr)

def save_array_slice(array, filename, slice_nr=0):
    array = (array - array.min()) / (array.max() - array.min()) * (2 ** 8 - 1)  # map to [0, 255]
    im = Image.fromarray(array[slice_nr])
    im = im.convert('L')
    im.save(filename)

# (row, column)
bb_centre_dict = {
    '0051SR6E0NUF': [180, 130],
    '03R8YVC7ZVXI': [180, 100],
    '1OLZ4RLHV1XW': [160, 120],
    '3JJRLKGTZ10K': [170, 130],
    '4L7MCUWASUER': [160, 130],
    '563HN7HC2C82': [155, 110],
    '5B85Y43KUS9X': [180, 120],
    '76I6SB9NTZTU': [165, 140],
    '7KYZD793DA6Q': [165, 100],
    '7OQIYMDL11OD': [175, 170],
    '8FZZRBNNVUEO': [150, 140],
    '92XW0PYU73UZ': [175, 170],
    '9YO4QA00QMTG': [135, 180],
    'BVH0U6Z96GUS': [110, 150],
    'CCDCYHH361VW': [145, 130],
    'EF22XY81ANL3': [175, 170],
    'ELNUKJTYY6ZV': [145, 100],
    'F6KX04GHOLJ7': [145, 160],
    'F9J38DOC5VHP': [155, 150],
    'GBKYTP1FY65M': [150, 135],
    'I7J3GMNY7ECN': [180, 185],
    'I7W3ZF2N7BRL': [120, 190],
    'KI5KXA3PH7L8': [160, 105],
    'KUHBI3WP0QUD': [200, 85],
    'N82YKZA572PA': [140, 130],
    'OZJVQFBP91TX': [135, 180],
    'OZU2SBL5PVIX': [180, 90],
    'PTSD2ADT8YYH': [130, 210],
    'Q11UDKKVQRWR': [170, 140],
    'QM8PYWARGV1T': [170, 130],
    'R1UZ0UD5HU0L': [155, 190],
    'REKEW7UPRYDM': [150, 180],
    'SKSNLLQHF04M': [145, 150],
    'SSBS71N9NCUS': [180, 120],
    'T7X0406U82GK': [80, 155],
    'UZFDCW9RDRSP': [155, 170],
    'VBX4717SYIYN': [130, 170],
    'VI2NW0AVD9JV': [170, 160],
    'WODRGHG00L87': [160, 120],
    'ZN5X5SUK5GOQ': [150, 160],
}

path = os.getcwd()
path_mri = os.path.join(path, 'mri')
path_figures_tmp = 'figures/tmp/'  # Temporary figures folder
path_arrays = 'dataset'

# Load data
df = pd.read_excel('Dataset.xlsx')
# NOTE: .xlsx may change the order if we sort the rows inside Excel (e.g., MPAP from lowest to highest)
df = df.sort_values('Unnamed: 0')
del df['Unnamed: 0']

# Make sure that patient_ids in df and in the MRI folder are identical
patient_ids_df = df['MRI_ID']
patient_ids_mri = os.listdir(path_mri)

patient_ids_overlap = [x for x in patient_ids_df if x in patient_ids_mri]
assert len(patient_ids_df) == len(patient_ids_mri) == len(patient_ids_overlap)

patient_ids = patient_ids_df.tolist()
patient_ids.sort()
print('Total number of patients: {}'.format(len(patient_ids)))

for p in tqdm(patient_ids):
    print('PatientID:', p)
    # Empty tmp folder (for creating GIFs)
    if os.path.isdir(path_figures_tmp):
        shutil.rmtree(path_figures_tmp)
    create_folder_if_not_exists(path_figures_tmp)

    print('Loading image...')
    _, metadata_i, image_i = load_images(path_images=os.path.join(path_mri, p))

    # Perform N4 bias field correction
    print('Performing N4 bias field correction...')
    image_i, mask_i, log_bias_field_i = n4_bias_field_correction(input_image=image_i)

    # Set isotropic voxel spacing
    print('Isostropic voxel spacing...')
    # RECALL: We set z-spacing to 1 for all patients, since we only have 25 times 'one slice per time point' (i.e. 25 frames)
    image_i = adjust_voxel_spacing(image_i, out_spacing=(1, 1, 1), is_label=False)

    # Convert image to Numpy array
    arr_i = sitk.GetArrayFromImage(image_i)

    # Image position
    if float(metadata_i['00200032'].value[0]) >= 0:
        orientation = 'rightup'
    else:
        orientation = 'laying'

    # Rotate image to 'upright' position
    print('Rotating images...')
    arr_i = rotate_arr(arr=arr_i,
                       orientation=orientation,
                       threshold=np.quantile(a=arr_i[arr_i > 10], q=0.25))

    # Extract bounding box centre
    row_center, col_center = bb_centre_dict[p]

    # Create cropped array
    print('Extracting bounding box coordinates and cropping bounding box...')
    bb_size = 128 // 2
    x_min, x_max = col_center - bb_size, col_center + bb_size
    y_min, y_max = row_center - bb_size, row_center + bb_size

    # Cropping array with shape (z, y, x) to (z, y', x')
    arr_cropped_i = arr_i[:,
                    max(0, y_min):y_max,
                    max(0, x_min):x_max]
    save_array_slice(array=arr_cropped_i, filename='figures/data_preprocessing/{}_cropped.png'.format(p), slice_nr=0)

    # Save cropped array as .npy file
    print('Saving cropped array...')
    create_folder_if_not_exists(path_arrays)
    filename_i = os.path.join(path_arrays, '{}.npy'.format(p))
    np.save(file=filename_i, arr=arr_cropped_i)
    print('')

