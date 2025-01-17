import os
import time
import math
import shutil
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
from statistics import mean
from IPython.display import clear_output

# Modelling
import torch
import optuna
import joblib
import optimizers
from models.resnet_dcrnn_lrelu import ResNet_DCRNN_LReLU
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
from monai.data import Dataset, CacheDataset, PersistentDataset, DataLoader, ThreadDataLoader
from monai.data.utils import pad_list_data_collate
from monai.utils import set_determinism
from monai.transforms import Activations, AsDiscrete

# Custom functions
from utils import (Logger, create_folder_if_not_exists, copy_file, findCorrelation,
                   torch_array_to_list, get_model_summary, weights_init, results_summary)
from losses import FocalLoss, CustomLoss
from metrics import Metrics
from transforms import (clip, normalizer, get_transforms,
                        preprocess_inputs, preprocess_features, preprocess_labels,
                        translate, rotate, scale, aug_mix)
from plotting import plot_values, plot_confusion_matrix


##### CONFIGS #####
# Define paths
path_arrays = 'dataset'

# General config
label_raw_col = 'PVRi'  # 'MPAP' | 'PVRi' | 'PAP/SAP-ratio' | 'PVR/SVR-ratio'
plot_model_name = 'dl'  # 'dl' | 'rf'

# Note: if `path_experiments` is an already-existing path, then the Optuna study will be continued
# (if an optuna study file exists)
path_experiments = datetime.now().strftime("%Y%m%d_%H%M%S") + '_optuna' + '_{}'.format(label_raw_col)
path_experiments_src = os.path.join(path_experiments, 'src')
exp_name = datetime.now().strftime("%Y%m%d_%H%M%S") + '_{seed}' + '_{fold}' + ''  # Can add extension
path_exp = os.path.join('experiments', path_experiments, exp_name)
filename_best_model_pth = 'best_model.pt'
filename_rf_model_pkl = 'rf_model.pkl'
filename_lr_model_pkl = 'lr_model.pkl'
gpu_condition = torch.cuda.is_available()
device = torch.device('cuda') if gpu_condition else torch.device('cpu')
seed = 84
cv_folds = 4

# Label variables
label_name_list = ['PVRi', 'MPAP']
label_threshold_list_dict = {'PVRi': [10, 30], 'MPAP': [40, 70]}

# Plotting config
plot_interval = 9999
plot_nr_images_multiple = 8
figsize_multiple = (18, 9)
max_nr_images_per_interval = 1
figsize = (12, 12)
conf_matrix_cmap = 'crest'
conf_matrix_fontsize = 20

# Deep Learning
# Transform config
perform_data_aug = True
modes_2d = ['bilinear']
interpol_mode_2d = 'bilinear'
data_aug_p = 0.5
# AugMix
mixture_width = 3  # (default)
mixture_depth = [1, 3]  # (default)
aug_list = [translate, rotate, scale]

rand_cropping_size = (25, 128, 128)  # Cropping size prior to data aug
input_size = (25, 128, 128)  # Model's input size
resize_mode = 'area'
to_device = False

# Dataloader config
dataset_type = 'cache'  # 'cache' | 'persistent'
cache_rate = 1.0
num_workers = 0
cache_dir = 'persistent_cache'
dataloader_type = 'standard'
drop_last = False
update_dict = None

# Model config
# Note: 'resnet_dcrnn_lrelu' with `lstm_num_layers` = 0 acts as 'resnet_dcnn_lrelu'
channels = 1
height = input_size[-2]
width = input_size[-1]
depth = 25
num_classes = 3
num_ohe_classes = 3

# Evaluation config
label_col_name = '{}_label'.format(label_raw_col)  # label data to fetch (e.g., raw_label: [0 - 100], label: [0, 1, 2])
ml_label_col_name = '{}_Label'.format(label_raw_col)
scale_raw_labels = 1
auc_average = 'weighted'  # {'micro', 'macro', 'weighted'} or None, default='macro'
f1_average = 'weighted'  # {'micro', 'macro', 'samples', 'weighted', 'binary'} or None, default='binary'
precision_average = 'weighted'  # {'micro', 'macro', 'samples', 'weighted', 'binary'} or None, default='binary'
recall_average = 'weighted'  # {'micro', 'macro', 'samples', 'weighted', 'binary'} or None, default='binary'
f1_zero_division = 0.0  # {'warn', 0.0, 1.0}, default='warn'
precision_zero_division = 0.0  # {'warn', 0.0, 1.0}, default='warn'
recall_zero_division = 0.0  # {'warn', 0.0, 1.0, np.nan}, default='warn'
pad_value = 0
pooling_conv_filters = None
perform_pooling = False

# Optimization config
weight_init_name = 'kaiming_uniform'  # [None, 'kaiming_uniform', 'uniform', 'xavier_uniform', 'kaiming_normal',
# 'normal', 'xavier_normal', 'orthogonal']. If None, then PyTorch's default (i.e. 'kaiming_uniform', but with
# a = math.sqrt(5)).
kaiming_a = math.sqrt(5)  # [math.sqrt(5) (default), lrelu_alpha]. Only used when kaiming_nonlinearity = 'leaky_relu'.
kaiming_mode = 'fan_in'  # ['fan_in' (default), 'fan_out'].
kaiming_nonlinearity = 'leaky_relu'  # ['leaky_relu' (default), 'relu', 'selu', 'linear']. When using
# weight_init_name = kaiming_normal for initialisation with SELU activations, then nonlinearity='linear' should be used
# instead of nonlinearity='selu' in order to get Self-Normalizing Neural Networks.
gain = 0  # [0 (default), torch.nn.init.calculate_gain('leaky_relu', lrelu_alpha)].
loss_function_name = 'custom'  # (regression) 'hubert' | 'mse'; (classification) 'custom' | 'ce' | 'f1' | 'focal' |
# 'l1' | 'ranking' | 'softauc'
label_weights = 'auto'  # [1] * num_ohe_classes  (if loss_function_name = 'ce', and therefore also if 'custom')
# list of length num_ohe_classes | 'auto'. If 'auto', then label_weights will be computed by sklearn.utils.class_weight.compute_class_weight() in each fold.
loss_reduction = 'mean'
label_smoothing = 0.0
T_mult = 1  # (CosineAnnealingWarmRestarts)
eta_min = 1e-8  # (CosineAnnealingWarmRestarts) Minimum learning rate. Default: 0.
grad_max_norm = None

# Training config
max_epochs = 1000
eval_interval = 1
patience = 25  # (EarlyStopping): stop training after this number of consecutive epochs without
# internal validation improvements.

# Optuna
optuna_study_name = 'optuna_dl_study.pkl'
optuna_sampler_name = 'optuna_dl_sampler.pkl'
optuna_path_pickles = os.path.join('experiments', path_experiments, 'pickles')
optuna_path_figures = os.path.join('experiments', path_experiments, 'figures')
# IMPORTANT: `optuna_main_metric_name` will be used for RF hyperparameter tuning, but only 'maximize' is allowed
# (for OptunaSearchCV)
optuna_main_metric_optimization = 'maximize'
optuna_main_metric_name = 'f1'
# Deep learning
optuna_sampler_n_startup_trials = 100
optuna_sampler_multivariate = True  # multivariate TPE
optuna_sampler_group = True  # (only if multivariate = True) use conditional search spaces (i.e., nested search space)
optuna_batch_size_list = [1, 2, 4, 8]
optuna_filters_list = [4, 4, 8, 8, 16, 16, 32, 32]  # Start from 4 instead of 8 to prevent OOM issues
optuna_min_kernel_size = 3
optuna_max_kernel_size = 7
optuna_lstm_hidden_size_list = [8, 16, 32, 64, 128, 256, 512]
optuna_linear_units_list = [8, 16, 32, 64, 128, 256, 512]
# EfficientNetV2
optuna_effnet_out_channels_list = [80, 160, 320, 640, 1280]
# Optuna for DL models
optuna_n_trials = 100
# Random Forest
optuna_rf_study_name = 'optuna_rf_study.pkl'
optuna_rf_sampler_name = 'optuna_rf_sampler.pkl'
optuna_rf_best_params_name = 'optuna_rf_best_params.pkl'
optuna_rf_sampler_n_startup_trials = 100
optuna_rf_sampler_multivariate = True
optuna_rf_sampler_group = True
optuna_rf_n_estimators = optuna.distributions.IntDistribution(10, 100)  # Number of trees in random forest
optuna_rf_max_features = optuna.distributions.CategoricalDistribution(['sqrt', 'log2'])  # Number of features to consider at every split
optuna_rf_max_depth = optuna.distributions.IntDistribution(1, 5)  # Maximum number of levels in tree
optuna_rf_min_samples_split = optuna.distributions.IntDistribution(2, 5)  # Minimum number of samples required to split a node
optuna_rf_min_samples_leaf = optuna.distributions.IntDistribution(1, 5)  # Minimum number of samples required at each leaf node
optuna_rf_bootstrap = optuna.distributions.CategoricalDistribution([True, False])  # Method of selecting samples for training each tree
optuna_rf_n_trials = 200  # Number of grid searches


##### INITIALIZE EXPERIMENT #####
# Create folders
for p in [os.path.join('experiments', path_experiments), os.path.join('experiments', path_experiments_src), 
          optuna_path_pickles, optuna_path_figures]:
    create_folder_if_not_exists(p)

# Copy src files to path_experiments
src_files = ['main.py']
for f in src_files:
    copy_file(src=f, dst=os.path.join('experiments', path_experiments_src, f))

# Load feature data
df = pd.read_excel('Dataset.xlsx')

# NOTE: .xlsx may change the order if we sort the rows inside Excel (e.g., MPAP from lowest to highest)
df = df.sort_values('Unnamed: 0')
del df['Unnamed: 0']

# Clip numerical variables between range
x_clip_dict = {
    'MPAP': [0, 100],
    'PVRi': [0, 50],
    'Age': [0, 20],
    'BSA': [0, 2.5],
    'RVEDVi': [0, 200],
    'RVESVi': [0, 200],
    'RVSVi': [0, 100],
    'RVMi': [0, 150],
    'RVEF': [0, 75],
    'LVEDVi': [0, 110],
    'LVESVi': [0, 40],
    'LVSVi': [0, 75],
    'LVMi': [0, 70],
    'LVEF': [0, 75],
    'RVLVr_vol': [0, 5],
    'RVLVr_mass': [0, 3],
    'EI_ED': [0, 2.5],
    'EI_ES': [0, 3.5]
}
for k, v in x_clip_dict.items():
    df[k] = df[k].apply(lambda x: clip(x, v[0], v[1])) 


##### Multicollinearity study #####
patient_id_col = 'MRI_ID'
group_id_col = 'PT_ID'
numerical_cols = [x for x in list(x_clip_dict.keys()) if x not in label_name_list]
categorical_cols = ['Female']
ml_features_cols = categorical_cols + numerical_cols
label_col = 'Label'  # [0, 1, 2]

# Remove features with high-correlation, keep the one feature that has highest correlation with the label
corr_threshold = 0.6
print('ml_features_cols (before): {}'.format(ml_features_cols))
corr = df[numerical_cols].corr()
print('Correlation matrix: {}'.format(corr))
hc = findCorrelation(corr, cutoff=corr_threshold)
df_trimmed = df[ml_features_cols].drop(columns=hc)
ml_features_cols = df_trimmed.columns
print('ml_features_cols (after): {}'.format(ml_features_cols.tolist()))


##### Preprocess DL Dataset #####
dl_features_cols = ml_features_cols

# Normalize features to [0, 1]
for c in numerical_cols:
    df = normalizer(df=df, column=c, x_min=x_clip_dict[c][0], x_max=x_clip_dict[c][1])

# Map label to [0, 1, 2]
for label_name, label_threshold_list in label_threshold_list_dict.items():
    df[label_name + '_' + label_col] = df[label_name].apply(
        lambda x: 0 if x <= label_threshold_list[0] else (1 if x <= label_threshold_list[1] else 2))
    
    label_dist = df[label_name + '_' + label_col].value_counts()
    label_dist_dict = label_dist.to_dict()
    for k, v in label_dist_dict.items():
        label_dist_dict[k] = '{} ({}%)'.format(label_dist_dict[k], round( label_dist_dict[k] / label_dist.values.sum() * 100 ))
    assert len(df[label_name + '_' + label_col].unique()) == num_ohe_classes

# IMPORTANT: For the CV splits, it is important that `patient_id_list` (and `group_id_list`) uses the same order of 
# the patient_ids as `Dataset.xlsx`! 
patient_id_list = df[patient_id_col].tolist()  # DO NOT CHANGE THE ORDER OF THE PATIENT_IDS
group_id_list = df[group_id_col].tolist()  # DO NOT CHANGE THE ORDER OF THE PATIENT_IDS
features_list = []
labels_list_dict = {label_name: list() for label_name in label_name_list}  
raw_labels_list_dict = {label_name: list() for label_name in label_name_list}  

for p in patient_id_list:
    df_i = df[df[patient_id_col] == p]
    assert len(df_i) == 1
    
    # Convert pd.DataFrame to pd.Series
    df_i = df_i.squeeze()
    
    features_list.append(df_i[dl_features_cols].tolist())
    for label_name, labels_list in labels_list_dict.items():
        labels_list.append(df_i[label_name + '_' + label_col])
    for label_name, raw_labels_list in raw_labels_list_dict.items(): 
        raw_labels_list.append(df_i[label_name])
    
assert (len(patient_id_list) == len(group_id_list) == len(features_list) == 
        len(labels_list_dict['PVRi']) == len(raw_labels_list_dict['PVRi']) == 
        len(labels_list_dict['MPAP']) == len(raw_labels_list_dict['MPAP'])
       )

assert (ml_features_cols == dl_features_cols).all()
assert len(dl_features_cols) > 0

# Rename columns for plotting
rename_dict = {'Female': 'Sex', 
               'RVLVr_vol': 'RVLVVr',
               'RVLVr_mass': 'RVLVMr',
               'EI_ED': 'EDEI',
               'EI_ES': 'ESEI'}
corr = corr.rename(columns=rename_dict, index=rename_dict)

# Plot correlation matrix
columns_sorted = list(corr.columns)
columns_sorted.sort()
corr = corr[columns_sorted]
corr = corr.sort_index()
cmap = 'vlag_r'
fontsize = 8
fig = plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, annot_kws={'size': fontsize}, cmap=cmap, vmin=-1, vmax=1, fmt='.2f')
plt.ylabel('True class', fontsize=fontsize)
plt.xlabel('Predicted class', fontsize=fontsize)
plt.savefig(os.path.join('paper', 'figs', 'correlation_matrix_multicollinearity.png'))
plt.close(fig)


##### OPTUNA #####
# (Optuna) 1. Define an objective function to be maximized.
globals()['optuna_study_trial_number'] = 0
def optuna_objective(trial):
    # (Optuna) 2. Suggest values of the hyperparameters using a trial object.
    # DCNN
    n_layers = trial.suggest_int('n_layers', 5, 7)
    block_name = trial.suggest_categorical('block_name', ['BasicResBlock', 'InvertedResidual', 'ConvNextV2Block'])
    filters = optuna_filters_list[:n_layers]
    kernel_size_time_dim = trial.suggest_int('kernel_size_time_dim', 1, 3, 1)
    if kernel_size_time_dim > 1:
        pad_time_dim = trial.suggest_categorical('pad_time_dim', [True, False])
    else:
        pad_time_dim = False  # Note: if kernel_size_time_dim == 1, then `pad_time_dim` has no effect
    kernel_sizes_i_tmp = [optuna_max_kernel_size]
    kernel_sizes = list()
    strides = list()
    for i in range(n_layers):
        # Note: `min(kernel_sizes_i_tmp)` is used to make sure that kernel_size does not become larger for later layers
        # min(..., 128 / (2**(i+1)) is used to make sure that kernel_size is not larger than feature_map size
        if round(128 / (2**(i+1))) >= optuna_min_kernel_size:
            kernel_sizes_i = trial.suggest_int('kernel_sizes_{}'.format(i),
                                               optuna_min_kernel_size,
                                               min(kernel_sizes_i_tmp))
        # If the feature map size is smaller than `optuna_min_kernel_size` (e.g.,: 2x2 vs. 3), then use
        # kernel_size smaller than `optuna_min_kernel_size`.
        else:
            kernel_sizes_i = trial.suggest_int('kernel_sizes_{}'.format(i),
                                               1, min(min(kernel_sizes_i_tmp), 128 / (2**(i+1))))

        kernel_sizes_i_tmp.append(kernel_sizes_i)
        kernel_sizes.append([kernel_size_time_dim, kernel_sizes_i, kernel_sizes_i])
        strides.append([1, 2, 2])
    lrelu_alpha = trial.suggest_float('lrelu_alpha', 0.0, 0.3)
    use_bias = trial.suggest_categorical('use_bias', [True, False])
    use_activation = trial.suggest_categorical('use_activation', [True, False])

    # LSTM
    lstm_num_layers = trial.suggest_int('lstm_num_layers', 0, 3)
    if lstm_num_layers > 0:
        lstm_hidden_size_idx = trial.suggest_int('lstm_hidden_size', 0, len(optuna_lstm_hidden_size_list) - 1, 1)
        lstm_hidden_size = optuna_lstm_hidden_size_list[lstm_hidden_size_idx]
        lstm_bidirectional = trial.suggest_categorical('lstm_bidirectional', [True, False])
        lstm_dropout_p = 0
        if lstm_num_layers > 1:
            lstm_dropout_p = trial.suggest_float('lstm_dropout_p', 0.0, 0.5)
    else:
        lstm_hidden_size = None
        lstm_dropout_p = None
        lstm_bidirectional = None

    # FC
    n_linear_layers = trial.suggest_int('n_linear_layers', 0, 1)
    linear_units = list()
    dropout_p = list()
    for i in range(n_linear_layers):
        linear_units_idx = trial.suggest_int('linear_units_idx_{}'.format(i), 0, len(optuna_linear_units_list) - 1, 1)
        linear_units_i = optuna_linear_units_list[linear_units_idx]
        linear_units.append(linear_units_i)
        dropout_p_i = trial.suggest_float('dropout_p_{}'.format(i), 0.0, 0.5)
        dropout_p.append(dropout_p_i)

    # SHARED HYPERPARAMETERS
    use_features = trial.suggest_categorical('use_features', [True, False])
        
    # TRAINING SETTINGS
    data_aug_strength = trial.suggest_float('data_aug_strength', 0.0, 3.0)
    augmix_strength = trial.suggest_float('augmix_strength', 0.0, 3.0)
    focal_loss_gamma = trial.suggest_float('focal_loss_gamma', 0.5, 10.0)
    batch_size_idx = trial.suggest_int('batch_size_idx', 0, len(optuna_batch_size_list) - 1, 1)
    batch_size = optuna_batch_size_list[batch_size_idx]
    
    # OPTIMIZATIONS
    loss_function_weights_ce = 0
    loss_function_weights_f1 = trial.suggest_float('loss_function_weights_f1', 0.0, 1.0)
    loss_function_weights_focal = trial.suggest_float('loss_function_weights_focal', 0.0, 1.0)
    loss_function_weights_l1 = trial.suggest_float('loss_function_weights_l1', 0.0, 1.0)
    loss_function_weights_ranking = trial.suggest_float('loss_function_weights_ranking', 0.0, 1.0)
    loss_function_weights_softauc = trial.suggest_float('loss_function_weights_softauc', 0.0, 1.0)
    loss_function_weights = [loss_function_weights_ce, loss_function_weights_f1, loss_function_weights_focal, 
                             loss_function_weights_l1, loss_function_weights_ranking, loss_function_weights_softauc]
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical('optimizer_name', ['ada_belief', 'ada_bound', 'adam', 'madgrad', 'sgd'])
    use_momentum = trial.suggest_categorical('use_momentum', [True, False])
    momentum = trial.suggest_float('momentum', 0.8, 1.0) if use_momentum else 0
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.25)
    scheduler_name = trial.suggest_categorical('scheduler_name', ['cosine', 'exponential'])
    if scheduler_name == 'cosine':
        T_0 = trial.suggest_int('T_0', 4, 20)
    elif scheduler_name == 'exponential':
        gamma = trial.suggest_float('gamma', 0.9, 1.0)
    else:
        raise ValueError('Scheduler_name = {} is not available'.format(scheduler_name))
    

    ##### START EXPERIMENT #####
    print('Seed: {}'.format(seed))
    exp_name = datetime.now().strftime("%Y%m%d_%H%M%S") + '_{seed}' + '_{fold}' + ''  # Can add extension 
    path_exp = os.path.join('experiments', path_experiments, exp_name)  

    # Initialize variables
    softmax_act = Activations(softmax=True)
    to_onehot = AsDiscrete(to_onehot=num_ohe_classes)
    metrics = Metrics(num_ohe_classes=num_ohe_classes, auc_average=auc_average, 
                      f1_average=f1_average, precision_average=precision_average,
                      recall_average=recall_average, f1_zero_division=f1_zero_division,
                      precision_zero_division=precision_zero_division, 
                      recall_zero_division=recall_zero_division)
    dl_train_folder_metric_value_list, rf_train_folder_metric_value_list, lr_train_folder_metric_value_list = [], [], []
    dl_val_folder_metric_value_list, rf_val_folder_metric_value_list, lr_val_folder_metric_value_list = [], [], []    
    dl_conf_matrix_all, rf_conf_matrix_all, lr_conf_matrix_all = None, None, None
    # Optuna
    optuna_dl_train_main_metric_value_list, optuna_dl_val_main_metric_value_list = [], []


    ##### PERFORM CROSS-VALIDATION #####
    for fold in range(cv_folds):
        ##### INITIALIZE EXPERIMENT ####
        clear_output(wait=False)
        path_exp_fold = path_exp.format(seed=seed, fold=fold) 
        path_figures = os.path.join(path_exp_fold, 'figures')
        path_outputs = os.path.join(path_exp_fold, 'outputs')
        for p in [path_exp_fold, path_figures, path_outputs]:
            create_folder_if_not_exists(p)

        # Set seed for reproducibility
        torch.manual_seed(seed=seed)     
        set_determinism(seed=seed)
        random.seed(a=seed)
        np.random.seed(seed=seed)
        torch.backends.cudnn.benchmark = False  # `True` will be faster, but potentially at cost of reproducibility

        # Initialize logger
        logger = Logger(output_filename=os.path.join(path_exp_fold, 'log.txt'))
        start = time.time()

        ##### CREATE TRAINING AND VALIDATION DATASET #####
        if fold == 0:
            # Create data dictionary
            data_dicts = [
                {'patient_id': patient_id,
                 'group_id': group_id,
                 'features': feature_name,
                 'image': os.path.join(path_arrays, patient_id + '.npy'),
                 'PVRi_raw_label': pvri_raw_label,
                 'PVRi_label': pvri_label,
                 'MPAP_raw_label': mpap_raw_label,
                 'MPAP_label': mpap_label
                 }
                for patient_id, group_id, feature_name, pvri_raw_label, pvri_label, mpap_raw_label, mpap_label in   
                zip(patient_id_list, group_id_list, features_list, 
                    raw_labels_list_dict['PVRi'], labels_list_dict['PVRi'],
                    raw_labels_list_dict['MPAP'], labels_list_dict['MPAP'])
            ]
            logger.my_print('Data_dicts: {}'.format(data_dicts))
            # Determine label value to stratify on
            cv_y = [[x['PVRi_label'], x['MPAP_label']] for x in data_dicts]
            # Determine group of the same patient, to make sure that each duplicated patients is in the same fold
            cv_group_id = [x['group_id'] for x in data_dicts]

            # Perform stratified N-fold cross-validation
            logger.my_print('Performing stratified {}-fold CV...'.format(cv_folds))
            cv_object = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

            # For each fold: determine train and val indices
            cv_idx_list = list()
            # Note that providing y is sufficient to generate the splits and hence np.zeros(len(cv_y)) may be used as 
            # a placeholder for X instead of actual training data.
            # IMPORTANT NOTE: StratifiedGroupKFold ONLY SUPPORTS ONE TARGET, SO ONLY STRATIFY ON PVRi
            for idx in cv_object.split(X=data_dicts, y=[x[0] for x in cv_y], groups=cv_group_id):
                cv_idx_list.append(idx)

            # Make sure that each validation folds contain each class at least once, else break the whole run
            has_invalid_fold = False
            for i, x in enumerate(cv_idx_list):
                tmp_val_idx_i = x[1]
                tmp_val_labels_i = [data_dicts[j]['{}_label'.format(label_raw_col)] for j in tmp_val_idx_i]
                tmp_val_classes_i = len(set(tmp_val_labels_i))
                # Create dict
                tmp_keys, tmp_counts = np.unique(tmp_val_labels_i, return_counts=True)
                tmp_val_classes_dict_i = {}
                for k, v in zip(tmp_keys, tmp_counts):
                    tmp_val_classes_dict_i[k] = str(v) + ' ({}%)'.format( round(v / sum(tmp_counts) * 100) )
                logger.my_print('Fold {}: {}'.format(i, tmp_val_classes_dict_i))
                if tmp_val_classes_i != num_ohe_classes: 
                    has_invalid_fold = True
                    logger.my_print('Seed = {} has a validation fold with {} classes, ' 
                                    'but it should contain {} classes'.format(seed, tmp_val_classes_i, num_ohe_classes))
                    break

            if has_invalid_fold:
                # Close logger, remove folder and break the run
                logger.close()
                del logger
                shutil.rmtree(path_exp_fold)
                break

        # Check: make sure that each duplicated patients is in the same fold
        train_group_ids = df.iloc[cv_idx_list[fold][0]][group_id_col].tolist()
        val_group_ids = df.iloc[cv_idx_list[fold][1]][group_id_col].tolist()
        assert sum([x for x in train_group_ids if x in val_group_ids]) == 0

        # For this fold: select training and internal validation indices
        train_idx, valid_idx = cv_idx_list[fold]

        # Fetch training set using cv indices in 'cv_idx_list'
        train_dict = list()
        for i in train_idx:
            train_dict.append(data_dicts[i])

        # Fetch internal validation set using cv indices in 'cv_idx_list'
        val_dict = list()
        for i in valid_idx:
            val_dict.append(data_dicts[i])

        assert len(train_dict) + len(val_dict) == len(data_dicts)


        ##### INITIALIZE DATALOADER ####
        if dataset_type in ['standard', None]:
            ds_class = Dataset
        elif dataset_type == 'cache':
            ds_class = CacheDataset
            update_dict = {'cache_rate': cache_rate, 'num_workers': num_workers}
        elif dataset_type == 'persistent':
            ds_class = PersistentDataset
            update_dict = {'cache_dir': cache_dir}
            create_folder_if_not_exists(cache_dir)
        else:
            raise ValueError('Invalid dataset_type: {}.'.format(dataset_type))

        # Define Dataset function arguments
        train_transforms, val_transforms = get_transforms(
            perform_data_aug=perform_data_aug, modes_2d=modes_2d, data_aug_p=data_aug_p,
            data_aug_strength=data_aug_strength, rand_cropping_size=rand_cropping_size, input_size=input_size,
            to_device=to_device, device=device)
        train_ds_args_dict = {'data': train_dict, 'transform': train_transforms}
        val_ds_args_dict = {'data': val_dict, 'transform': val_transforms}

        # Update Dataset function arguments based on type of Dataset class
        if update_dict is not None:
            train_ds_args_dict.update(update_dict)
            val_ds_args_dict.update(update_dict)

        # Initialize Dataset
        train_ds = ds_class(**train_ds_args_dict)
        val_ds = ds_class(**val_ds_args_dict)

        # Define DataLoader class
        if dataloader_type in ['standard', None]:
            dl_class = DataLoader
        elif dataloader_type == 'thread':
            dl_class = ThreadDataLoader
        else:
            raise ValueError('Invalid dataloader_type: {}.'.format(dataloader_type))

        # Define Dataloader function arguments
        # Shuffle is not necessary for val_dl and test_dl, but shuffle can be useful for plotting random patients 
        # Weighted random sampler
        # Create a list of the labels: e.g., [2, 1, 1, 0, 2, 0, ...]
        y_train_list = [x[label_col_name] for x in train_dict]
        # Create a list of counts of each label (0, 1, 2, ...)
        y_train_count_list = [sum([x == i for x in y_train_list]) for i in range(num_ohe_classes)]
        shuffle = True
        sampler = None

        # Define Dataloader function arguments
        train_dl_args_dict = {'dataset': train_ds, 'batch_size': batch_size, 'shuffle': shuffle, 'sampler': sampler,
                              'num_workers': num_workers, 'drop_last': drop_last, 'collate_fn': pad_list_data_collate}
        val_dl_args_dict = {'dataset': val_ds, 'batch_size': 1, 'shuffle': False, 'num_workers': int(num_workers//2),
                            'drop_last': False, 'collate_fn': pad_list_data_collate}

        # Initialize DataLoader
        train_dl = dl_class(**train_dl_args_dict) if len(train_dict) > 0 else None
        val_dl = dl_class(**val_dl_args_dict) if len(val_dict) > 0 else None

        # Get training and validation patient ids
        train_patient_ids = [x['patient_id'] for x in train_dict]
        val_patient_ids = [x['patient_id'] for x in val_dict]

        # Make sure that train and val patient_ids do not have any overlaps
        assert sum([x in val_patient_ids for x in train_patient_ids]) == 0

        
        ##### MACHINE LEARNING MODELS #####
        # Prepare dataset
        train_df = df[df[patient_id_col].isin(train_patient_ids)]
        val_df = df[df[patient_id_col].isin(val_patient_ids)]

        train_X = train_df[ml_features_cols]
        train_y = train_df[ml_label_col_name]
        val_X = val_df[ml_features_cols]
        val_y = val_df[ml_label_col_name]

        # Initiate ML models (RF and LR)
        rf_model = RandomForestClassifier(random_state=seed)
        lr_model = LogisticRegression(random_state=seed)
        train_y_list = [to_onehot(i) for i in torch.as_tensor(train_y.tolist())]
        val_y_list = [to_onehot(i) for i in torch.as_tensor(val_y.tolist())]


        ##### DETERMINE BEST HYPERPARAMETERS FOR RANDOM FOREST MODEL #####
        # NOTE: grid search is performed in the following CV, therefore we will run the following only during 
        # the first fold (fold = 0) or if not os.path.exists(optuna_rf_out_file_best_params)
        # Tune RF once and only once per study, because the full hyperparameter tuning pipeline is ran by OptunaSearchCV
        global optuna_rf_study
        global optuna_rf_sampler
        global optuna_rf_out_file_study
        global optuna_rf_out_file_sampler
        global optuna_rf_out_file_best_params
        
        optuna_rf_out_file_study = os.path.join(optuna_path_pickles, optuna_rf_study_name)
        optuna_rf_out_file_sampler = os.path.join(optuna_path_pickles, optuna_rf_sampler_name)
        optuna_rf_out_file_best_params = os.path.join(optuna_path_pickles, optuna_rf_best_params_name)

        if (fold == 0) or not os.path.exists(optuna_rf_out_file_best_params):
            # Optuna
            optuna_rf_base = RandomForestClassifier(random_state=seed)
            optuna_rf_sampler = optuna.samplers.TPESampler(n_startup_trials=optuna_rf_sampler_n_startup_trials,
                                                           multivariate=optuna_rf_sampler_multivariate, 
                                                           group=optuna_rf_sampler_group,
                                                           seed=seed)
            optuna_rf_study = optuna.create_study(sampler=optuna_rf_sampler, direction='maximize')
            optuna_rf_random_grid = {
                'n_estimators': optuna_rf_n_estimators,
                'max_features': optuna_rf_max_features,
                'max_depth': optuna_rf_max_depth,
                'min_samples_split': optuna_rf_min_samples_split,
                'min_samples_leaf': optuna_rf_min_samples_leaf,
                'bootstrap': optuna_rf_bootstrap}
            if optuna_main_metric_name == 'r2':
                optuna_rf_scoring = 'r2'
            elif optuna_main_metric_name == 'mae':
                optuna_rf_scoring = 'neg_mean_absolute_error'
            elif optuna_main_metric_name == 'auc':
                optuna_rf_scoring = 'roc_auc_ovr_weighted'
            elif optuna_main_metric_name == 'mse':
                optuna_rf_scoring = 'neg_mean_squared_error'
            elif optuna_main_metric_name == 'f1':
                optuna_rf_scoring = 'f1_weighted'
            elif optuna_main_metric_name == 'precision':
                optuna_rf_scoring = 'precision'
            elif optuna_main_metric_name == 'recall':
                optuna_rf_scoring = 'recall'
            elif optuna_main_metric_name == 'ce':
                optuna_rf_scoring = 'roc_auc_ovr_weighted'
            else:
                raise ValueError('Optuna_main_metric_name = {} is not available'.format(optuna_main_metric_name))
            
            rf_grid = optuna.integration.OptunaSearchCV(study=optuna_rf_study,
                                                        estimator=optuna_rf_base, 
                                                        param_distributions=optuna_rf_random_grid, 
                                                        n_trials=optuna_rf_n_trials, 
                                                        cv=cv_idx_list,
                                                        scoring=optuna_rf_scoring,
                                                        random_state=seed)

            # Determine optimal hyperparameters
            logger.my_print('(RandomForest) Performing CV hyperparameter tuning...')
            rf_grid.fit(X=df[ml_features_cols], y=df[ml_label_col_name])
            optuna_rf_best_params = rf_grid.best_params_
            
            # Save RF hyperparamter tuning results
            joblib.dump(optuna_rf_study, optuna_rf_out_file_study)
            joblib.dump(optuna_rf_study.sampler, optuna_rf_out_file_sampler)
            joblib.dump(optuna_rf_best_params, optuna_rf_out_file_best_params)
        else: 
            optuna_rf_best_params = joblib.load(optuna_rf_out_file_best_params)
                
        logger.my_print('(RandomForest) Best parameters: {}'.format(optuna_rf_best_params))    
        rf_model.set_params(**optuna_rf_best_params)

        # Train ML models
        rf_model.fit(train_X, train_y)
        lr_model.fit(train_X, train_y)

        # Save fitted ML models
        joblib.dump(rf_model, os.path.join(path_exp_fold, filename_rf_model_pkl))
        joblib.dump(lr_model, os.path.join(path_exp_fold, filename_lr_model_pkl))

        # Make predictions
        rf_train_y_pred_list = [torch.as_tensor(x) for x in rf_model.predict_proba(train_X)]
        lr_train_y_pred_list = [torch.as_tensor(x) for x in lr_model.predict_proba(train_X)]
        rf_val_y_pred_list = [torch.as_tensor(x) for x in rf_model.predict_proba(val_X)]
        lr_val_y_pred_list = [torch.as_tensor(x) for x in lr_model.predict_proba(val_X)]

        # Store predictions in pd.DataFrame
        df_ml_results = pd.DataFrame({
            'PatientID': train_df[patient_id_col].tolist() + val_df[patient_id_col].tolist(),
            'RF_Pred': [torch_array_to_list(x) for x in rf_train_y_pred_list] + [torch_array_to_list(x) for x in rf_val_y_pred_list],
            'LR_Pred': [torch_array_to_list(x) for x in lr_train_y_pred_list] + [torch_array_to_list(x) for x in lr_val_y_pred_list],
            'ML_Label': [torch_array_to_list(x) for x in train_y_list] + [torch_array_to_list(x) for x in val_y_list],
            'ML_Mode': ['train'] * len(train_y_list) + ['val'] * len(val_y_list),
        })

        # (RandomForest) Plot feature importance
        features = train_X.columns.tolist()
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)

        plt.title('Feature importances')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative importance (mean decrease in impurity)')
        plt.tight_layout()
        plt.savefig(os.path.join(path_outputs, 'rf_feature_importance.png'))
        plt.close()
        

        ##### DEEP LEARNING MODEL ####
        if use_features:
            n_features = len(dl_features_cols)
        else:
            n_features = 0
            
        # Initialize DL model
        logger.my_print('(DL) Initializing model...')
        model = ResNet_DCRNN_LReLU(
            n_input_channels=channels, depth=depth, height=height, width=width, n_features=n_features,
            num_classes=num_classes, block_name=block_name, filters=filters, kernel_sizes=kernel_sizes,
            strides=strides, lstm_num_layers=lstm_num_layers, lstm_hidden_size=lstm_hidden_size,
            lstm_dropout_p=lstm_dropout_p, lstm_bidirectional=lstm_bidirectional, pad_time_dim=pad_time_dim,
            pad_value=pad_value, lrelu_alpha=lrelu_alpha, dropout_p=dropout_p,
            pooling_conv_filters=pooling_conv_filters, perform_pooling=perform_pooling,
            linear_units=linear_units, use_bias=use_bias, use_activation=use_activation)
        model.to(device=device)

        # Initialize model weights
        if weight_init_name is not None:
            model.apply(lambda m: weights_init(m=m,
                                               label_count_list=y_train_count_list,
                                               weight_init_name=weight_init_name,
                                               kaiming_a=kaiming_a,
                                               kaiming_mode=kaiming_mode,
                                               kaiming_nonlinearity=kaiming_nonlinearity,
                                               gain=gain,
                                               logger=logger))

        total_params = get_model_summary(
            model=model,
            input_size=[(batch_size, channels, depth, height, width), (batch_size, max(n_features, 1))],
            path=path_exp_fold,
            device=device
        )
        logger.my_print('Number of model parameters: {}'.format(total_params))

        # Loss function
        if label_weights == 'auto':
            label_weights_values = compute_class_weight('balanced',
                                                        classes=np.unique(y_train_list),
                                                        y=y_train_list)
            label_weights_values = torch.as_tensor(label_weights_values, dtype=torch.float32, device=device)
        else:
            label_weights_values = torch.as_tensor(label_weights, dtype=torch.float32, device=device)
        logger.my_print('Label_weights_values: {}'.format(label_weights_values))

        if loss_function_name == 'ce':
            loss_function = torch.nn.CrossEntropyLoss(weight=label_weights_values,
                                                      reduction=loss_reduction,
                                                      label_smoothing=label_smoothing)
        elif loss_function_name == 'focal':
            loss_function = FocalLoss(num_ohe_classes=num_ohe_classes, gamma=focal_loss_gamma)
        elif loss_function_name == 'custom':
            loss_function = CustomLoss(loss_function_weights=loss_function_weights,
                                       label_weights=label_weights_values,
                                       num_ohe_classes=num_ohe_classes,
                                       reduction=loss_reduction,
                                       label_smoothing=label_smoothing,
                                       focal_loss_gamma=focal_loss_gamma)
        else:
            raise ValueError('Loss_function_name = {} is not available'.format(loss_function_name))

        # Optimizer
        if optimizer_name == 'ada_belief':
            optimizer = optimizers.AdaBelief(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'ada_bound':
            optimizer = optimizers.AdaBound(model.parameters(), lr=lr, final_lr=lr * 100, weight_decay=weight_decay)
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'madgrad':
            optimizer = optimizers.MADGRAD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError('Invalid optimizer_name: {}.'.format(optimizer_name))

        # Scheduler
        if scheduler_name is not None:
            if scheduler_name == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=T_0, T_mult=T_mult, 
                                                                                 eta_min=eta_min)
            elif scheduler_name == 'exponential':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
            else:
                raise ValueError('Invalid scheduler_name: {}.'.format(scheduler_name))


        ##### MODEL TRAINING ####
        # Initiate variables
        best_val_loss_value = np.inf
        if optuna_main_metric_optimization == 'minimize':
            best_val_main_metric_value = np.inf
        elif optuna_main_metric_optimization == 'maximize':
            best_val_main_metric_value = -np.inf
        else:
            raise ValueError('Main metric optimization = {} is not available'.format(optuna_main_metric_optimization))

        train_loss_values_list = list() if max_epochs > 0 else [None]
        train_main_metric_values_list = list() if max_epochs > 0 else [None]
        val_loss_values_list = list() if max_epochs > 0 else [None]
        val_main_metric_values_list = list() if max_epochs > 0 else [None]
        lr_values_list = list() if max_epochs > 0 else [None]
        nr_epochs_not_improved = 0  # (EarlyStopping)

        train_num_iterations = len(train_dl)
        logger.my_print('Number of training iterations per epoch: {}.'.format(train_num_iterations))
        best_epoch = 0
        for epoch in range(max_epochs):
            # START OF TRAINING
            logger.my_print(f'Epoch {epoch + 1}/{max_epochs}...')
            for param_group in optimizer.param_groups:
                logger.my_print('Learning rate: {}.'.format(param_group['lr']))
                lr_values_list.append(param_group['lr'])
            cur_batch_size = train_dl.batch_size
            logger.my_print('Batch size: {}.'.format(cur_batch_size))

            # Initiate variables
            model.train()
            train_loss_value = 0
            train_y_pred = torch.as_tensor([], dtype=torch.float32, device=device)
            train_y = torch.as_tensor([], device=device)                

            # Train on batches
            for i, batch_data in tqdm(enumerate(train_dl)):
                # Load data
                train_inputs, train_features, train_labels = (
                    batch_data['image'].to(device),
                    batch_data['features'].to(device),
                    batch_data[label_col_name].to(device)
                )

                # AugMix
                if augmix_strength != 0:
                    for b in range(len(train_inputs)):
                        train_inputs[b] = aug_mix(arr=train_inputs[b], aug_list=aug_list, mixture_width=mixture_width,
                                                  mixture_depth=mixture_depth, augmix_strength=augmix_strength,
                                                  mode=interpol_mode_2d, device=device)

                # Preprocess inputs, features, and labels
                train_inputs = preprocess_inputs(inputs=train_inputs)
                train_features = preprocess_features(features=train_features)
                train_labels = preprocess_labels(labels=train_labels, scale_raw_labels=scale_raw_labels)

                # Zero the parameter gradients, plot DL input at the end of the epoch, and make predictions
                optimizer.zero_grad(set_to_none=True)
                if (epoch == 0) or (epoch % plot_interval == 0):
                    # Plot for random patients
                    plot_inputs = train_inputs.cpu().numpy().copy()
                    for patient_idx in range(min(max_nr_images_per_interval, len(plot_inputs))):
                        arr_i = plot_inputs[patient_idx][0]

                if not use_features:
                    train_features = torch.tensor([], dtype=train_features.dtype)
                train_outputs = model(x=train_inputs, features=train_features)

                # Calculate loss
                train_loss = loss_function(train_outputs, train_labels)
                train_loss.backward()

                # Perform gradient clipping
                if grad_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=grad_max_norm)

                # Update model weights
                optimizer.step()

                # Scheduler: step() called after every batch update
                # Schedulers: 'cosine', 'exponential', 'step'
                if scheduler_name in ['cosine', 'exponential', 'step']:
                    scheduler.step(epoch + (i + 1) / train_num_iterations)

                # Store predictions and labels
                train_loss_value += train_loss.item()
                train_y_pred = torch.cat([train_y_pred, train_outputs], dim=0)
                train_y = torch.cat([train_y, train_labels], dim=0)

            # Averaging training loss
            train_loss_value /= train_num_iterations

            # Evaluate model
            # Compute training R-squared, MAE, or MSE (regression) or AUC (classification)
            if optuna_main_metric_name == 'ce':
                train_y_pred_list = [x for x in train_y_pred]
            else:
                train_y_pred_list = [softmax_act(i) for i in train_y_pred]
            train_y_list = [to_onehot(i) for i in train_y]
            train_main_metric_value = metrics.compute_metric(
                y_pred_list=train_y_pred_list, y_true_list=train_y_list, name=optuna_main_metric_name)

            logger.my_print(f'Training loss: {train_loss_value:.3f}.')
            logger.my_print(f'Training {optuna_main_metric_name.upper()}: {train_main_metric_value:.3f}.')

            train_loss_values_list.append(train_loss_value)
            train_main_metric_values_list.append(train_main_metric_value)
            # END OF TRAINING 

            # START OF VALIDATION
            if (epoch + 1) % eval_interval == 0:        
                # Initialize variable
                model.eval()
                val_loss_value = 0
                val_num_iterations = len(val_dl)
                logger.my_print('Number of validation iterations: {}.'.format(val_num_iterations))

                with torch.no_grad():
                    val_y_pred = torch.as_tensor([], dtype=torch.float32, device=device)
                    val_y = torch.as_tensor([], device=device)
                    for val_data in val_dl:
                        # Load data
                        val_inputs, val_features, val_labels = (
                            val_data['image'].to(device),
                            val_data['features'].to(device),
                            val_data[label_col_name].to(device)
                        )

                        # Preprocess inputs, features, and labels
                        val_inputs = preprocess_inputs(inputs=val_inputs)
                        val_features = preprocess_features(features=val_features)
                        val_labels = preprocess_labels(labels=val_labels, scale_raw_labels=scale_raw_labels)

                        # Make predictions
                        if not use_features:
                            val_features = torch.tensor([], dtype=val_features.dtype)
                        val_outputs = model(x=val_inputs, features=val_features)

                        # Calculate loss
                        val_loss = loss_function(val_outputs, val_labels)

                        # Store predictions and labels
                        val_loss_value += val_loss.item()
                        val_y_pred = torch.cat([val_y_pred, val_outputs], dim=0)
                        val_y = torch.cat([val_y, val_labels], dim=0)

                    # Averaging internal validation loss
                    val_loss_value /= val_num_iterations

                    # Evaluate model
                    # Compute internal validation R-squared, MAE, or MSE (regression) or AUC (classification)
                    if optuna_main_metric_name == 'ce':
                        val_y_pred_list = [x for x in val_y_pred]
                    else:
                        val_y_pred_list = [softmax_act(i) for i in val_y_pred]
                    val_y_list = [to_onehot(i) for i in val_y]
                    val_main_metric_value = metrics.compute_metric(
                        y_pred_list=val_y_pred_list, y_true_list=val_y_list, name=optuna_main_metric_name)

                    logger.my_print(f'Validation loss: {val_loss_value:.3f}.')
                    logger.my_print(f'Validation {optuna_main_metric_name.upper()}: {val_main_metric_value:.3f}.')

                    val_loss_values_list.append(val_loss_value)
                    val_main_metric_values_list.append(val_main_metric_value)

                # Determine best model so far
                if optuna_main_metric_optimization == 'minimize':
                    if val_main_metric_value < best_val_main_metric_value:
                        best_val_loss_value = val_loss_value
                        best_val_main_metric_value = val_main_metric_value
                        best_epoch = epoch + 1
                        nr_epochs_not_improved = 0
                        torch.save(model.state_dict(), os.path.join(path_exp_fold, filename_best_model_pth))
                        logger.my_print('Saved new best metric model.')
                    else:
                        nr_epochs_not_improved += 1
                elif optuna_main_metric_optimization == 'maximize':
                    if val_main_metric_value > best_val_main_metric_value:
                            best_val_loss_value = val_loss_value
                            best_val_main_metric_value = val_main_metric_value
                            best_epoch = epoch + 1
                            nr_epochs_not_improved = 0
                            torch.save(model.state_dict(), os.path.join(path_exp_fold, filename_best_model_pth))
                            logger.my_print('Saved new best metric model.')
                    else:
                        nr_epochs_not_improved += 1
                else:
                    raise ValueError('Main metric optimization = {} is not available'.format(optuna_main_metric_optimization))

                logger.my_print(
                    'Best internal validation {}: {:.3f} at epoch {}'.format(
                        optuna_main_metric_name.upper(), best_val_main_metric_value, best_epoch))
                logger.my_print(
                    'Corresponding internal validation loss: {:.3f} at epoch {}'.format(
                        best_val_loss_value, best_epoch))

                # EarlyStopping
                if nr_epochs_not_improved >= patience:
                    logger.my_print('No internal validation improvement during the last {} consecutive epochs. '
                                    'Stop training.'.format(nr_epochs_not_improved))
                    break
                # END OF VALIDATION


        ##### EVALUATION #####   
        # Plot training and internal validation losses
        y_list = [
            [train_loss_values_list, val_loss_values_list],
            [train_main_metric_values_list, val_main_metric_values_list],
            [lr_values_list]
        ]
        y_label_list = ['Loss', optuna_main_metric_name.upper(), 'LR']
        legend_list = [['Training', 'Internal validation']] * (len(y_list) - 1) + [None]
        plot_values(y_list=y_list, y_label_list=y_label_list,
                    best_epoch=best_epoch, legend_list=legend_list,
                    figsize=figsize, save_filename=os.path.join(path_exp_fold, 'graphs.png'))

        # Make predictions for all patients
        # Load best model
        logger.my_print('Loading best model...')
        model.load_state_dict(torch.load(os.path.join(path_exp_fold, filename_best_model_pth)))

        logger.my_print('Making predictions...')
        all_patient_ids = list()
        all_mode = list()
        all_y_pred = torch.as_tensor([], dtype=torch.float32, device=device)
        all_y = torch.as_tensor([], device=device)

        for dl_i, mode in zip([train_dl, val_dl], ['train', 'val']):
            for all_data in tqdm(dl_i):
                # Load data
                all_inputs, all_features, all_labels, all_p = (
                    all_data['image'].to(device),
                    all_data['features'].to(device),
                    all_data[label_col_name].to(device),
                    all_data['patient_id']
                )
                batch_size_i = all_inputs.shape[0]

                # Preprocess inputs, features, and labels
                all_inputs = preprocess_inputs(inputs=all_inputs)
                all_features = preprocess_features(features=all_features)
                all_labels = preprocess_labels(labels=all_labels, scale_raw_labels=scale_raw_labels)

                # Make predictions
                if not use_features:
                    all_features = torch.tensor([], dtype=all_features.dtype)
                all_outputs = model(x=all_inputs, features=all_features)

                # Store predictions and labels
                all_y_pred = torch.cat([all_y_pred, all_outputs], dim=0)
                all_y = torch.cat([all_y, all_labels], dim=0)

                all_patient_ids += all_p
                all_mode += [mode] * batch_size_i

        # Compute internal validation R-squared and MAE (regression) or AUC and MSE (classification)
        if optuna_main_metric_name == 'ce':
            all_y_pred_list = [x for x in all_y_pred]
        else:
            all_y_pred_list = [softmax_act(i) for i in all_y_pred]
        all_y_list = [to_onehot(i) for i in all_y]

        # Store predictions in pd.DataFrame
        df_dl_results = pd.DataFrame({
            'PatientID': all_patient_ids,
            'DL_Pred': [torch_array_to_list(x) for x in all_y_pred_list],
            'DL_Label': [torch_array_to_list(x) for x in all_y_list],
            'DL_Mode': all_mode,
        })

        # Merge DL and ML predictions
        df_results = df_dl_results.merge(df_ml_results, how='inner', on=['PatientID'])

        # Final print performance of all models and store these in the folder name
        train_df_results = df_results[df_results['ML_Mode'] == 'train']
        val_df_results = df_results[df_results['ML_Mode'] == 'val']

        # Training
        dl_train_folder_metric_value = metrics.compute_metric(
            y_pred_list=torch.as_tensor(np.array(train_df_results['DL_Pred'].tolist())), 
            y_true_list=torch.as_tensor(np.array(train_df_results['DL_Label'].tolist())),
            name=optuna_main_metric_name
        )
        dl_train_folder_metric_value_list.append(dl_train_folder_metric_value)

        rf_train_folder_metric_value = metrics.compute_metric(
            y_pred_list=torch.as_tensor(np.array(train_df_results['RF_Pred'].tolist())), 
            y_true_list=torch.as_tensor(np.array(train_df_results['ML_Label'].tolist())),
            name=optuna_main_metric_name
        )
        rf_train_folder_metric_value_list.append(rf_train_folder_metric_value)

        lr_train_folder_metric_value = metrics.compute_metric(
            y_pred_list=torch.as_tensor(np.array(train_df_results['LR_Pred'].tolist())), 
            y_true_list=torch.as_tensor(np.array(train_df_results['ML_Label'].tolist())),
            name=optuna_main_metric_name
        )
        lr_train_folder_metric_value_list.append(lr_train_folder_metric_value)

        logger.my_print('Train {}s:'.format(optuna_main_metric_name.upper()))
        logger.my_print('\tDL: {}'.format(dl_train_folder_metric_value))
        logger.my_print('\tRF: {}'.format(rf_train_folder_metric_value))
        logger.my_print('\tLR: {}'.format(lr_train_folder_metric_value))

        # Validation
        dl_val_folder_metric_value = metrics.compute_metric(
            y_pred_list=torch.as_tensor(np.array(val_df_results['DL_Pred'].tolist())), 
            y_true_list=torch.as_tensor(np.array(val_df_results['DL_Label'].tolist())),
            name=optuna_main_metric_name
        )
        dl_val_folder_metric_value_list.append(dl_val_folder_metric_value)

        rf_val_folder_metric_value = metrics.compute_metric(
            y_pred_list=torch.as_tensor(np.array(val_df_results['RF_Pred'].tolist())), 
            y_true_list=torch.as_tensor(np.array(val_df_results['ML_Label'].tolist())),
            name=optuna_main_metric_name
        )
        rf_val_folder_metric_value_list.append(rf_val_folder_metric_value)

        lr_val_folder_metric_value = metrics.compute_metric(
            y_pred_list=torch.as_tensor(np.array(val_df_results['LR_Pred'].tolist())), 
            y_true_list=torch.as_tensor(np.array(val_df_results['ML_Label'].tolist())),
            name=optuna_main_metric_name
        )
        lr_val_folder_metric_value_list.append(lr_val_folder_metric_value)

        logger.my_print('Val {}s:'.format(optuna_main_metric_name.upper()))
        logger.my_print('\tDL: {}'.format(dl_val_folder_metric_value))
        logger.my_print('\tRF: {}'.format(rf_val_folder_metric_value))
        logger.my_print('\tLR: {}'.format(lr_val_folder_metric_value))

        # Main metric
        optuna_dl_train_main_metric_value = metrics.compute_metric(
            y_pred_list=torch.as_tensor(np.array(train_df_results['DL_Pred'].tolist())), 
            y_true_list=torch.as_tensor(np.array(train_df_results['DL_Label'].tolist())),
            name=optuna_main_metric_name
        )
        optuna_dl_train_main_metric_value_list.append(optuna_dl_train_main_metric_value)
        
        optuna_dl_val_main_metric_value = metrics.compute_metric(
            y_pred_list=torch.as_tensor(np.array(val_df_results['DL_Pred'].tolist())), 
            y_true_list=torch.as_tensor(np.array(val_df_results['DL_Label'].tolist())),
            name=optuna_main_metric_name
        )
        optuna_dl_val_main_metric_value_list.append(optuna_dl_val_main_metric_value)

        # Save predictions
        logger.my_print('Saving predictions...')
        df_results.to_excel(os.path.join(path_exp_fold, 'predictions.xlsx'), index=False)

        # Save variables
        cfg_variables_dict = {}
        cfg_variables_dict['optuna_rf_best_params'] = optuna_rf_best_params
        results_summary(path=path_exp_fold, **cfg_variables_dict)

        # Create and save validation confusion matrix
        # Deep Learning
        dl_conf_matrix_i = metrics.compute_metric(
            y_pred_list=torch.as_tensor(np.array(val_df_results['DL_Pred'].tolist())), 
            y_true_list=torch.as_tensor(np.array(val_df_results['DL_Label'].tolist())),
            name='confusion_matrix'
        )
        plot_confusion_matrix(cm=dl_conf_matrix_i, filename=os.path.join(path_outputs, 'dl_confusion_matrix.png'))
        np.save(file=os.path.join(path_outputs, 'dl_confusion_matrix.npy'), arr=dl_conf_matrix_i)
        if dl_conf_matrix_all is None:  # i.e., fold == 0
            dl_conf_matrix_all = dl_conf_matrix_i.copy()
        else:
            dl_conf_matrix_all = dl_conf_matrix_all + dl_conf_matrix_i.copy()
            
        # Random Forest
        rf_conf_matrix_i = metrics.compute_metric(
            y_pred_list=torch.as_tensor(np.array(val_df_results['RF_Pred'].tolist())), 
            y_true_list=torch.as_tensor(np.array(val_df_results['ML_Label'].tolist())),
            name='confusion_matrix'
        )
        plot_confusion_matrix(cm=rf_conf_matrix_i, filename=os.path.join(path_outputs, 'rf_confusion_matrix.png'))
        np.save(file=os.path.join(path_outputs, 'rf_confusion_matrix.npy'), arr=rf_conf_matrix_i)
        if rf_conf_matrix_all is None:  # i.e., fold == 0
            rf_conf_matrix_all = rf_conf_matrix_i.copy()
        else:
            rf_conf_matrix_all = rf_conf_matrix_all + rf_conf_matrix_i.copy()
            
        # Logistic Regression
        lr_conf_matrix_i = metrics.compute_metric(
            y_pred_list=torch.as_tensor(np.array(val_df_results['LR_Pred'].tolist())), 
            y_true_list=torch.as_tensor(np.array(val_df_results['ML_Label'].tolist())),
            name='confusion_matrix'
        )
        plot_confusion_matrix(cm=lr_conf_matrix_i, filename=os.path.join(path_outputs, 'lr_confusion_matrix.png'))
        np.save(file=os.path.join(path_outputs, 'lr_confusion_matrix.npy'), arr=lr_conf_matrix_i)
        if lr_conf_matrix_all is None:  # i.e., fold == 0
            lr_conf_matrix_all = lr_conf_matrix_i.copy()
        else:
            lr_conf_matrix_all = lr_conf_matrix_all + lr_conf_matrix_i.copy()

        end = time.time()
        logger.my_print('Elapsed time: {time} seconds'.format(time=end - start))
        logger.my_print('DONE!')
        logger.close()
        del logger

        # Rename folder
        src_folder_name = path_exp_fold
        dst_folder_name = os.path.join(path_exp_fold + 
                                       '_{}'.format(optuna_study_run_nr) + 
                                       '_{}'.format(globals()['optuna_study_trial_number'])
                                       )
        # Last fold
        if fold == cv_folds - 1:
            # Plot and save combined confusion matrix
            plot_confusion_matrix(cm=dl_conf_matrix_all, filename=os.path.join(path_outputs, 'dl_confusion_matrix_all.png'))
            np.save(file=os.path.join(path_outputs, 'dl_confusion_matrix_all.npy'), arr=dl_conf_matrix_all)
            
            plot_confusion_matrix(cm=rf_conf_matrix_all, filename=os.path.join(path_outputs, 'rf_confusion_matrix_all.png'))
            np.save(file=os.path.join(path_outputs, 'rf_confusion_matrix_all.npy'), arr=rf_conf_matrix_all)
            
            plot_confusion_matrix(cm=lr_conf_matrix_all, filename=os.path.join(path_outputs, 'lr_confusion_matrix_all.png'))
            np.save(file=os.path.join(path_outputs, 'lr_confusion_matrix_all.npy'), arr=lr_conf_matrix_all)
            
            dst_folder_name = os.path.join(path_exp_fold + 
                                           '_{}'.format(optuna_study_run_nr) + 
                                           '_{}'.format(globals()['optuna_study_trial_number'])
                                           )

        shutil.move(src_folder_name, dst_folder_name)
        globals()['optuna_study_trial_number'] += 1

    # Save optuna objects
    joblib.dump(optuna_study, optuna_out_file_study)
    # Save the sampler for reproducibility after resuming study
    joblib.dump(optuna_study.sampler, optuna_out_file_sampler)
    
    # Optuna objective to maximize
    return mean(optuna_dl_val_main_metric_value_list)


# (Optuna) 3. Create a study object and optimize the objective function.
# Resume study if study and sampler files exist
optuna_file_study_list = [x for x in os.listdir(optuna_path_pickles) if optuna_study_name in x]
optuna_file_sampler_list = [x for x in os.listdir(optuna_path_pickles) if optuna_sampler_name in x]
if len(optuna_file_study_list) > 0 and len(optuna_file_sampler_list) > 0:
    # Find last study, and add 1 for the next study run
    optuna_study_run_nr = max([int(x.split('_')[0]) for x in optuna_file_study_list]) + 1
    optuna_sampler_run_nr = max([int(x.split('_')[0]) for x in optuna_file_sampler_list]) + 1
    assert optuna_study_run_nr == optuna_sampler_run_nr
    # Load last sampler and study
    optuna_in_file_sampler = os.path.join(optuna_path_pickles, '{}_'.format(optuna_sampler_run_nr - 1) + optuna_sampler_name)
    optuna_in_file_study = os.path.join(optuna_path_pickles, '{}_'.format(optuna_study_run_nr - 1) + optuna_study_name)
    print('Resuming previous sampler and study: {} and {}'.format(optuna_in_file_sampler, optuna_in_file_study))
    optuna_sampler = joblib.load(optuna_in_file_sampler)
    optuna_study = joblib.load(optuna_in_file_study)
else:
    optuna_sampler_run_nr = 0
    optuna_study_run_nr = 0
    # Create new sampler and study
    optuna_sampler = optuna.samplers.TPESampler(n_startup_trials=optuna_sampler_n_startup_trials,
                                                multivariate=optuna_sampler_multivariate, 
                                                group=optuna_sampler_group,
                                                seed=seed)
    optuna_study = optuna.create_study(sampler=optuna_sampler, direction=optuna_main_metric_optimization)

# Run hyperparameter tuning using Optuna
optuna_start = time.time()
optuna_study.optimize(optuna_objective, n_trials=optuna_n_trials)

# Save study
optuna_out_file_study = os.path.join(optuna_path_pickles, '{}_'.format(optuna_study_run_nr) + optuna_study_name)
optuna_out_file_sampler = os.path.join(optuna_path_pickles, '{}_'.format(optuna_sampler_run_nr) + optuna_sampler_name)
joblib.dump(optuna_study, optuna_out_file_study)
joblib.dump(optuna_study.sampler, optuna_out_file_sampler)

optuna_end = time.time()
print('Elapsed time: {time} seconds'.format(time=optuna_end - optuna_start))

