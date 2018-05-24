# Adversarial Time-to-Event Modeling (ICML 2018)

This repository contains the TensorFlow code to replicate experiments in our paper:
 * [Adversarial Time-to-Event Modeling](https://arxiv.org/pdf/1804.03184.pdf) (ICML 2018)
 
This project is maintained by [Paidamoyo Chapfuwa](https://github.com/paidamoyo). Please contact <paidamoyo.chapfuwa@duke.edu> for any relevant issues.


### Prerequisites
You will need to install the following:

- Python 3.5.1: https://github.com/pyenv/pyenv
- tensorflow-gpu 1.5: https://www.tensorflow.org/

Once the above dependencies are installed run

```
pip install -r requirements.txt
```

### Data
The following datesets can be found in the *data* folder:

- SUPPORT: http://biostat.mc.vanderbilt.edu/wiki/Main/DataSets
- Flchain: https://vincentarelbundock.github.io/Rdatasets/doc/survival/flchain.html

We provide a preprocessing script for the SEER dataset which can downloaded from <https://seer.cancer.gov/>

### Training the model

* To train DATE or DATE_AE model

```
 python train_date.py
 ```
 

* To train DRAFT model

```
 python train_draft.py
 ```

* The hyper-parameters settings are in the file: *flags_parameters.py*


* The training log file: *model.log*

* Results will be saved in the folders: *plot, matrix* folders

## Citation 
Please cite our ICML paper in your publications if it helps your research:

```latex
@inproceedings{chapfuwa2018adversarial, 
  title={Adversarial Time-to-Event Modeling},
  author={Chapfuwa, Paidamoyo and Tao, Chenyang and Li, Chunyuan and Page, Courtney and Goldstein, Benjamin and Carin, Lawrence and Henao, Ricardo},
  booktitle={ICML},
  year={2018}
}
