import os
import re
import math
import shutil
import logging
import numpy as np
import pandas as pd

import torch
from torchinfo import summary


class Logger:
    def __init__(self, output_filename=None):
        logging.basicConfig(filename=output_filename,
                            format='%(asctime)s - %(message)s',
                            level=logging.INFO,
                            filemode='w',
                            force=True)

    def my_print(self, message, level='info'):
        """
        Manual print operation.
        """
        if level == 'info':
            print_message = 'INFO: {}'.format(message)
            logging.info(print_message)
        elif level == 'exception':
            print_message = 'EXCEPTION: {}'.format(message)
            logging.exception(print_message)
        elif level == 'warning':
            print_message = 'WARNING: {}'.format(message)
            logging.warning(print_message)
        else:
            print_message = 'INFO: {}'.format(message)
            logging.info(print_message)
        print(print_message)

    def close(self):
        logging.shutdown()

def create_folder_if_not_exists(folder):
    """
    Create folder if it does not exist yet.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

def copy_file(src, dst):
    """
    Copy source (src) file to destination (dst) file. Note that renaming is possible.
    """
    shutil.copy(src, dst)

def sort_human(l):
    """
    Sort the input list. However normally with l.sort(), e.g., l = ['1', '2', '10', '4'] would be sorted as
    l = ['1', '10', '2', '4']. The sort_human() function makes sure that l will be sorted properly,
    i.e.: l = ['1', '2', '4', '10'].
    """
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
    l.sort(key=alphanum)
    return l

def findCorrelation(corr, cutoff, exact=None):
    """
    This function is the Python implementation of the R function `findCorrelation()`.
    """
    def _findCorrelation_fast(corr, avg, cutoff):
        combsAboveCutoff = corr.where(lambda x: (np.tril(x) == 0) & (x > cutoff)).stack().index
        rowsToCheck = combsAboveCutoff.get_level_values(0)
        colsToCheck = combsAboveCutoff.get_level_values(1)

        msk = avg[colsToCheck] > avg[rowsToCheck].values
        deletecol = pd.unique(np.r_[colsToCheck[msk], rowsToCheck[~msk]]).tolist()

        return deletecol

    def _findCorrelation_exact(corr, avg, cutoff):
        x = corr.loc[(*[avg.sort_values(ascending=False).index] * 2,)]
        if (x.dtypes.values[:, None] == ['int64', 'int32', 'int16', 'int8']).any():
            x = x.astype(float)
        x.values[(*[np.arange(len(x))] * 2,)] = np.nan

        deletecol = []
        for ix, i in enumerate(x.columns[:-1]):
            for j in x.columns[ix + 1:]:
                if x.loc[i, j] > cutoff:
                    if x[i].mean() > np.nanmean(x.drop(j)):
                        deletecol.append(i)
                        x.loc[i] = x[i] = np.nan
                    else:
                        deletecol.append(j)
                        x.loc[j] = x[j] = np.nan
        return deletecol

    if not np.allclose(corr, corr.T) or any(corr.columns != corr.index):
        raise ValueError("correlation matrix is not symmetric.")

    acorr = corr.abs()
    avg = acorr.mean()

    if exact or exact is None and corr.shape[1] < 100:
        return _findCorrelation_exact(acorr, avg, cutoff)
    else:
        return _findCorrelation_fast(acorr, avg, cutoff)

def torch_array_to_list(torch_array):
    return torch_array.cpu().detach().numpy().ravel().tolist()

def list_to_torch_array(input_list):
    return torch.stack(input_list)

def get_model_summary(model, input_size, path, device):
    """
    Get model summary and number of trainable parameters.
    """
    # Get and save summary
    txt = str(summary(model=model, input_size=input_size, device=device))
    file = open(os.path.join(path, 'model.txt'), 'a+', encoding='utf-8')
    file.write(txt)
    file.close()

    # Determine number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def weights_init(m, label_count_list, weight_init_name, kaiming_a, kaiming_mode, kaiming_nonlinearity, gain, logger):
    """
    Custom weights initialization.
    """
    # Initialize variables
    classname = m.__class__.__name__

    # Initialize output layer, for which a sigmoid-function will be preceded.
    if 'Output' in classname:
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)

        # If we have an imbalanced dataset of a ratio 1:10 of positives:negatives, set the bias on our logits
        # such that our network predicts probability of 0.1 at initialization
        if m.bias is not None:
            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(m.bias, -bound, bound)

            # Freqs should sum to 1. (Theortical) If the data is balanced, then frequencies are 1/num_ohe_classes.
            # Note: log_label_freq.softmax(-1) == label_freq
            label_freq_sum = sum(label_count_list)
            label_freq = torch.as_tensor([x / label_freq_sum for x in label_count_list], dtype=m.bias.data.dtype)
            assert label_freq.sum() == 1
            log_label_freq = label_freq.log()
            m.bias.data = log_label_freq
        logger.my_print('Weights init of output layer: Xavier uniform.')

    elif ('Conv2d' in classname) or ('Conv3d' in classname) or ('Linear' in classname):  # or ('Output' in classname):
        if weight_init_name == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(m.weight, a=kaiming_a, mode=kaiming_mode, nonlinearity=kaiming_nonlinearity)
        elif weight_init_name == 'uniform':
            torch.nn.init.uniform_(m.weight)
        elif weight_init_name == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        elif weight_init_name == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(m.weight, a=kaiming_a, mode=kaiming_mode, nonlinearity=kaiming_nonlinearity)
        elif weight_init_name == 'normal':
            torch.nn.init.normal_(m.weight)
        elif weight_init_name == 'xavier_normal':
            torch.nn.init.xavier_normal_(m.weight, gain=gain)
        elif weight_init_name == 'orthogonal':
            torch.nn.init.orthogonal_(m.weight, gain=gain)
        else:
            raise ValueError('Invalid weight_init_name: {}.'.format(weight_init_name))

        if m.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)  # if fan_in > 0 else 0
            torch.nn.init.uniform_(m.bias, -bound, bound)

def results_summary(path, **kwargs):
    """
    Create results_summary.xlsx.
    """
    df = pd.DataFrame([x for x in kwargs.values()],
                      index=[x for x in kwargs.keys()],
                      columns=[path])
    df.to_excel(os.path.join(path, 'results_summary.xlsx'), index=True)
