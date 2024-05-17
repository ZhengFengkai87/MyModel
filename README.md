## STCAN: A Traffic Forecasting Method Based on Spatial-Temporal Cross-Attention Network

This repository is the original pytorch implementation for STCAN.

### Project Structure

* `data`: datasets for experiments
    + PEMS04

    + PEMS07

    + PEMS08
    
    + NYCBike
    
    + NYCTaxi

* `lib`: library of training utils

* `logs`: directory for training logs

* `model`: implementation of STCAN, hyperparameter settings for STCAN and the training framework

* `saved_models`: state_dict of trained models

### Requirements

#### requirements for STCAN:

```
python=3.9.18
pytorch=1.11.0+cu113
numpy
pandas
matplotlib
pyyaml
pickle
torchinfo
```

### Get Started

```bash
cd model/
python train.py -d <dataset> -g <gpu_id>
```

For example: `python train.py -d PEMS04 -g 0`
