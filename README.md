# AISAFE-BACKEND

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A backend leveraging gradient boosting machines and multimodal neural networks for early detection of child abuse. 

## Installation

```bash
pip install aisafe_xgboost
```

## Usage

`aisafe_xgboost` can be used as either a CLI or a Python package. 
To train the model, run the following command:

```bash
safe-train
```

To use the model in python, run the following command:

```python
>>> from aisafe_xgboost import model
>>> prediction, explainability = model(
    info = info, # Iterable of Length 4 [Patient Age, Sex, Height Percentile, Weight Percentile]
    bruise = bruse, # Iterable of Length 11 [float]
    response = response, # Iterable of Length 9 [int]
    lab = lab, # Iterable of Length 19 [float]
    xray = xray, # Iterable of Length 9 [float]
    video = video, # Iterable of Length 30 [float]
)
>>> prediction 
0.75
>>> explainability 
[("문진정보", 0.5), ("Lab 수치", 0.3), ("X-ray 영상", 0.1), ("진료 영상", 0.1), ("신체 계측치", 0.0), ("멍 정보", 0.0)]
```



## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. 
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         aisafe and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
│
└── aisafe_xgboost   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes aisafe a Python module
    │
    ├── training.py             <- For training the xgboost model
    │
    ├── utils.py                <- General utility functions for parsing data
    │
    ├── inference.py            <- Code to run model inference with trained models
    │
    ├── models                
    │   ├── *.ubj               <- XGBoost models, stored as .ubj files (universal binary format) 
    │   └── *.npy               <- Weights for combinining the xgboost model as an ensemble        
    │
    ├── growth data                 
    │   └── *.csv               <- Growth Data for Korean male/female children
    └── mock_data
         └── *.csv              <- Mock data for training the model
```
--------

