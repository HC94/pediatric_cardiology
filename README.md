<!---
Markdown syntax: https://www.markdownguide.org/basic-syntax
-->

# Predicting Hemodynamics in Pediatric Pulmonary Arterial Hypertension Using Cardiac Magnetic Resonance Imaging and Machine Learning: An Exploratory Pilot Study

<b>Authors: Hung Chu; Rosaria J. Ferreira; Chantal Lokhorst; Johannes M. Douwes; Meindina G. Haarman; Tineke P. Willems; 
Rolf M.F. Berger; Mark-Jan Ploegstra.</b>

This repository provides the code for preprocessing cardiac magnetic resonance (CMR) cine imaging data, as well as 
training (including hyperparameter tuning) and evaluating machine learning models to predict hemodynamics in pediatric 
pulmonary arterial hypertension (PAH).

---

<span style="font-size: 25px;">[Pytorch](https://pytorch.org/) | [MONAI](https://monai.io/) | [Optuna](https://optuna.org/)</span>

---

## Table of Contents
1. [Data preprocessing](#data-preprocessing)
2. [Model training, hyperparameter tuning, and evaluation](#model-training-hyperparameter-tuning-and-evaluation)
3. [Notes](#notes)

---

## Data preprocessing

This step preprocesses MRI cine frames for use in the deep learning models. A flowchart outlining the data preprocessing 
steps is shown below:


<img src="images/data_preproc.png" alt="Data preprocessing" title="Data preprocessing" width="1200" height="250">

#### Requirements
- **Input Data**:
  - A file named `Dataset.xlsx` containing at least the following column:
    - `MRI_ID`: Unique identifier for each subfolder of MRI frames.
  - A folder named `mri` containing subfolders of MRI frames. 
    - Each subfolder must include the patient's MRI cine frames and have the same name as the corresponding `MRI_ID` in 
    `Dataset.xlsx`.

#### Output
- A new folder named `dataset` will be created.
  - This folder will contain `.npy` files for each `MRI_ID`.
  - Each `.npy` file represents the cropped MRI cine frames with dimensions `25 × 128 × 128` (time × height × width).

#### How to run
Run the following command to run the data preprocessing:
```
$ python3 data_preproc.py
```

---

## Model training, hyperparameter tuning, and evaluation

This step trains, performs hyperparameter tuning using [Optuna](https://optuna.org/), and evaluates machine learning models: 
logistic regression, random forest, and deep learning.
The flowchart below summarizes the training process:

<img src="images/flowchart.png" alt="Flowchart" title="Flowchart" width=auto height="250">

#### Requirements
- **Input Data**:
  - A file named `Dataset.xlsx` containing:
    - `MRI_ID`: Unique identifier for each subfolder of MRI frames.
    - `PT_ID`: Unique patient identifier. Generally, having either `MRI_ID` and `PT_ID` is sufficient. 
    However, we use both to ensure that different MRI subfolders from the same patient are grouped into the same 
    cross-validation fold.
    Observe that `MRI_ID` values are unique, while `PT_ID` values can be duplicated.
    - Additional columns for predictors as needed by the model.
  - A folder named `dataset` containing `.npy` files generated during the data preprocessing step. 
    - Each `.npy` file should match the corresponding `MRI_ID` in `Dataset.xlsx` and have dimensions `25 × 128 × 128`.

#### Output
- The code will train and tune hyperparameters for the machine learning models and evaluates the models' 
cross-validation performance.

#### How to run
Run the following command to start model training, hyperparameter tuning, and evaluation:
```
$ python3 train.py
```

---

## Notes
- Configuration settings can be found and adjusted in the `train.py` file.
- Ensure all required files and folders are correctly named and placed in the working directory.
- The scripts are tested for Python 3.8.13. 

