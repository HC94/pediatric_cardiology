import random
import numpy as np
import torch
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    CenterSpatialCropd,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    RandSpatialCropd,
    RandAffined,
    RandFlipd,
    RandRotated,
    Rotate,
    ScaleIntensityRanged,
    SpatialPadd,
    ToDeviced,
    # AugMix
    Rand3DElastic,
    RandAdjustContrast,
    RandFlip,
    RandGaussianNoise,
)

def clip(x, lower_limit, upper_limit): return max(lower_limit, min(x, upper_limit))

def normalizer(df, column, x_min, x_max):
    assert x_min < x_max
    # Normalize to [0, 1]
    df[column] = (df[column] - x_min) / (x_max - x_min)
    return df

def standardizer(df, column):
    mean = df[column].mean()
    std = df[column].std()
    df[column] = (df[column] - mean) / std
    return df

def get_transforms(perform_data_aug, modes_2d, data_aug_p, data_aug_strength, rand_cropping_size, input_size,
                   to_device, device):
    """
    Transforms for training and internal validation data.
    """
    # Define variables for exceptions
    align_corners_exception_2d = [True if mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None for mode in modes_2d]

    # Define generic transforms
    generic_transforms = Compose([
        LoadImaged(keys=['image']),
        EnsureChannelFirstd(keys=['image']),
        EnsureTyped(keys=['image', 'features', 'PVRi_raw_label', 'PVRi_label', 'MPAP_raw_label', 'MPAP_label'], data_type='tensor', dtype=torch.float32),
        SpatialPadd(keys=['image'], spatial_size=input_size)
    ])

    # Define training transforms
    train_transforms = generic_transforms

    # Define internal validation and test transforms
    val_transforms = generic_transforms

    # Data augmentation
    if perform_data_aug:
        train_transforms = Compose([
            train_transforms,
            RandSpatialCropd(keys=['image'], roi_size=rand_cropping_size, random_center=True, random_size=False),
            RandFlipd(keys=['image'], prob=data_aug_p, spatial_axis=-1),  # 3D: (num_channels, H[, W, …, ])
            RandAffined(keys=['image'], prob=data_aug_p,
                        translate_range=(7 * data_aug_strength, 7 * data_aug_strength, 7 * data_aug_strength),
                        padding_mode='border', mode=modes_2d),  # 3D: (num_channels, H, W[, D])
            RandAffined(keys=['image'], prob=data_aug_p, scale_range=(0.07 * data_aug_strength, 0.07 * data_aug_strength, 0.07 * data_aug_strength),
                        padding_mode='border', mode=modes_2d),  # 3D: (num_channels, H, W[, D])
            RandRotated(keys=['image'], prob=data_aug_p, range_x=(np.pi / 24) * data_aug_strength,
                        align_corners=align_corners_exception_2d, padding_mode='border', mode=modes_2d),
        ])

    # Resize images
    if list(rand_cropping_size) != list(input_size) or not perform_data_aug:
        train_transforms = Compose([
            train_transforms,
            CenterSpatialCropd(keys=['image'], roi_size=input_size),
        ])

    val_transforms = Compose([
        val_transforms,
        CenterSpatialCropd(keys=['image'], roi_size=input_size),
    ])

    # To device
    if to_device:
        train_transforms = Compose([
            train_transforms,
            ToDeviced(keys=['image'], device=device),
        ])

        val_transforms = Compose([
            val_transforms,
            ToDeviced(keys=['image'], device=device),
        ])

    # Flatten transforms
    train_transforms = train_transforms.flatten()
    val_transforms = val_transforms.flatten()

    # Print transforms
    for mode, t in zip(['Train', 'Validation'], [train_transforms, val_transforms]):
        for i in t.transforms:
            print('\t{}, keys={}'.format(i.__class__, i.keys))

    return train_transforms, val_transforms

def preprocess_inputs(inputs):
    """
    Preprocess input images.
    """
    # ZScoreNormalize: (x - mean) / std
    mean, std = inputs.mean(), inputs.std()
    inputs = (inputs - mean) / std

    return inputs

def preprocess_features(features):
    """
    Preprocess features.
    """
    return features

def preprocess_labels(labels, scale_raw_labels):
    return labels * scale_raw_labels

def translate(arr, mode, strength, seed):
    augmenter = Rand3DElastic(prob=1.0, sigma_range=(5, 8), magnitude_range=(0, 1),
                              translate_range=(round(7 * strength), round(7 * strength), round(7 * strength)),
                              padding_mode='border', mode=mode)
    augmenter.set_random_state(seed=seed)
    return augmenter(arr)

def rotate(arr, mode, strength, seed):
    augmenter = Rand3DElastic(prob=1.0, sigma_range=(5, 8), magnitude_range=(0, 1),
                              rotate_range=((np.pi / 24) * strength, (np.pi / 24) * strength, (np.pi / 24) * strength),
                              padding_mode='border', mode=mode)
    augmenter.set_random_state(seed=seed)
    return augmenter(arr)

def scale(arr, mode, strength, seed):
    augmenter = Rand3DElastic(prob=1.0, sigma_range=(5, 8), magnitude_range=(0, 1),
                              scale_range=(0.07 * strength, 0.07 * strength, 0.07 * strength),
                              padding_mode='border', mode=mode)
    augmenter.set_random_state(seed=seed)
    return augmenter(arr)

def aug_mix(arr, aug_list, mixture_width, mixture_depth, augmix_strength, mode, device):
    """
    Perform AugMix augmentations and compute mixture.
    """
    ws = torch.tensor(np.random.dirichlet([1] * mixture_width), dtype=torch.float32)
    m = torch.tensor(np.random.beta(1, 1), dtype=torch.float32)

    mix = torch.zeros_like(arr, device=device)
    for i in range(mixture_width):
        image_aug = arr.clone()

        depth = np.random.randint(mixture_depth[0], mixture_depth[1] + 1)
        for _ in range(depth):
            # op = np.random.choice(aug_list)
            idx = random.randint(0, len(aug_list) - 1)
            op = aug_list[idx]
            seed_i = random.getrandbits(32)

            image_aug = op(arr=image_aug, mode=mode, strength=augmix_strength, seed=seed_i)

        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * image_aug

    mixed = (1 - m) * arr + m * mix
    return mixed

