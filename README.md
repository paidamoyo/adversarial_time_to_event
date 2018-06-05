# Adversarial Time-to-Event Modeling (ICML 2018)

This repository contains the TensorFlow code to replicate experiments in our paper [Adversarial Time-to-Event Modeling](https://arxiv.org/pdf/1804.03184.pdf) (ICML 2018):
```latex
@inproceedings{chapfuwa2018adversarial, 
  title={Adversarial Time-to-Event Modeling},
  author={Chapfuwa, Paidamoyo and Tao, Chenyang and Li, Chunyuan and Page, Courtney and Goldstein, Benjamin and Carin, Lawrence and Henao, Ricardo},
  booktitle={ICML},
  year={2018}
}
```
 
This project is maintained by [Paidamoyo Chapfuwa](https://github.com/paidamoyo). Please contact <paidamoyo.chapfuwa@duke.edu> for any relevant issues.


### Prerequisites
The code is implemented with following dependencies:

- [Python 3.5.1](https://github.com/pyenv/pyenv)
- [TensorFlow 1.5]( https://www.tensorflow.org/)
- Additional python packages can be installed by running:   

```
pip install -r requirements.txt
```

### Data
We consider the following datasets:

- [SUPPORT](http://biostat.mc.vanderbilt.edu/wiki/Main/DataSets)
- [Flchain](https://vincentarelbundock.github.io/Rdatasets/doc/survival/flchain.html)
- [SEER](https://seer.cancer.gov/)
- EHR (a large study from Duke Hospital centered around impatient visits due to comorbidities in patients with Type-2 diabetes)

 For convenience, we provide pre-processing scripts of all datasets (except EHR). In addition, the *data* directory contains downloaded Flchain and SUPPORT datasets.

### Model Training

The code consists of 3 models: DATE, DATE-AE and DRAFT. For each model first modify the train scripts with the chosen datasets.

* To train DATE or DATE_AE model (modify train script to choose the model)

```
 python train_date.py
 ```
 

* To train DRAFT model

```
 python train_draft.py
 ```

* The hyper-parameters settings can be found at *flags_parameters.py*


## Metrics and Visualizations

Once the networks are trained and the results are saved we extract the following key results: 

* Training and evaluation metrics are logged in *model.log*
* Epoch based cost function plots can be found in the *plot* directory 
* To evaluate and plot generated time-to-event distribution we provide raw results in the  *matrix* directory
