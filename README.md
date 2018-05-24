# Adversarial Time-to-Event Modeling

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

* Data can be found in the *data* folder

### Prerequisites
You will need to install the following:

- Python 3.5.1: https://github.com/pyenv/pyenv
- tensorflow-gpu 1.5: https://www.tensorflow.org/

Once the above dependencies are installed run

```
pip install -r requirements.txt
```