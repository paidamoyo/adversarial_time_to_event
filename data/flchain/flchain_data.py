import os

import numpy as np
import pandas

from utils.pre_processing import one_hot_encoder, formatted_data, missing_proportion, \
    one_hot_indices, get_train_median_mode


# age: age in years
# sex: F=female, M=male
# sample.yr: the calendar year in which a blood sample was obtained
# kappa: serum free light chain, kappa portion
# lambda: serum free light chain, lambda portion
# flc.grp: the FLC group for the subject, as used in the original analysis
# creatinine: serum creatinine
# mgus: 1 if the subject had been diagnosed with monoclonal gammapothy (MGUS)
# futime: days from enrollment until death. Note that there are 3 subjects whose sample was obtained on their death date.
# death 0=alive at last contact date, 1=dead
# chapter: for those who died, a grouping of their primary cause of death by chapter headings of
# the International Code of Diseases ICD-9


def generate_data():
    np.random.seed(31415)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.abspath(os.path.join(dir_path, '', 'flchain.csv'))
    print("path:{}".format(path))
    data_frame = pandas.read_csv(path, index_col=0)
    print("head of data:{}, data shape:{}".format(data_frame.head(), data_frame.shape))
    # x_data = data_frame[['age', 'sex', 'kappa', 'lambda', 'flc.grp', 'creatinine', 'mgus']]
    # Preprocess
    to_drop = ['futime', 'death', 'chapter']
    print("missing:{}".format(missing_proportion(data_frame.drop(labels=to_drop, axis=1))))
    one_hot_encoder_list = ['sex', 'flc.grp', 'sample.yr']
    data_frame = one_hot_encoder(data_frame, encode=one_hot_encoder_list)
    t_data = data_frame[['futime']]
    e_data = data_frame[['death']]
    dataset = data_frame.drop(labels=to_drop, axis=1)
    print("head of dataset data:{}, data shape:{}".format(dataset.head(), dataset.shape))
    encoded_indices = one_hot_indices(dataset, one_hot_encoder_list)
    print("data description:{}".format(dataset.describe()))
    covariates = np.array(dataset.columns.values)
    print("columns:{}".format(covariates))
    x = np.array(dataset).reshape(dataset.shape)
    t = np.array(t_data).reshape(len(t_data))
    e = np.array(e_data).reshape(len(e_data))

    print("x:{}, t:{}, e:{}, len:{}".format(x[0], t[0], e[0], len(t)))
    idx = np.arange(0, x.shape[0])
    print("x_shape:{}".format(x.shape))

    np.random.shuffle(idx)
    x = x[idx]
    t = t[idx]
    e = e[idx]
    end_time = max(t)
    print("end_time:{}".format(end_time))
    print("observed percent:{}".format(sum(e) / len(e)))
    print("shuffled x:{}, t:{}, e:{}, len:{}".format(x[0], t[0], e[0], len(t)))
    num_examples = int(0.80 * len(e))
    print("num_examples:{}".format(num_examples))
    train_idx = idx[0: num_examples]
    split = int((len(t) - num_examples) / 2)

    test_idx = idx[num_examples: num_examples + split]
    valid_idx = idx[num_examples + split: len(t)]
    print("test:{}, valid:{}, train:{}, all: {}".format(len(test_idx), len(valid_idx), num_examples,
                                                        len(test_idx) + len(valid_idx) + num_examples))
    # print("test_idx:{}, valid_idx:{},train_idx:{} ".format(test_idx, valid_idx, train_idx))

    imputation_values = get_train_median_mode(x=np.array(x[train_idx]), categorial=encoded_indices)
    print("imputation_values:{}".format(imputation_values))
    preprocessed = {
        'train': formatted_data(x=x, t=t, e=e, idx=train_idx),
        'test': formatted_data(x=x, t=t, e=e, idx=test_idx),
        'valid': formatted_data(x=x, t=t, e=e, idx=valid_idx),
        'end_t': end_time,
        'covariates': covariates,
        'one_hot_indices': encoded_indices,
        'imputation_values': imputation_values
    }
    return preprocessed


if __name__ == '__main__':
    generate_data()
