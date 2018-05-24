import os

import numpy as np
import pandas

from utils.pre_processing import one_hot_encoder, formatted_data, missing_proportion, \
    one_hot_indices, get_train_median_mode, log_transform


# TODO Check scoma attribute
# TODO You may want to use the following normal values that have been found to
# be satisfactory in imputing missing baseline physiologic data:
# http://biostat.mc.vanderbilt.edu/wiki/Main/DataSets
# http://annals.org/aim/article/708396/support-prognostic-model-objective-estimates-survival-seriously-ill-hospitalized-adults
# To develop models without using findings from previous models, be sure not to use aps, sps, surv2m, surv6m as predictors.
#  You also will probably not want to use prg2m, prg6m, dnr, dnrday.
# Columns with na
#   "edu"     "scoma"   "charges" "totcst"  "totmcst" "avtisst" "sps"     "aps"     "surv2m"  "surv6m"
# "prg2m"   "prg6m"   "dnrday"  "meanbp"  "wblc"    "hrt"     "resp"    "temp"    "pafi"    "alb"
#  "bili"    "crea"    "sod"     "ph"      "glucose" "bun"     "urine"   "adlp"    "adls"

# # Log transform: totmcst, totcst, charges, pafi, sod

# TODO CHECK: avtisst, wblc
def generate_data():
    np.random.seed(31415)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.abspath(os.path.join(dir_path, '', 'support2.csv'))
    print("path:{}".format(path))
    data_frame = pandas.read_csv(path, index_col=0)
    to_drop = ['hospdead', 'death', 'prg2m', 'prg6m', 'dnr', 'dnrday', 'd.time', 'aps', 'sps', 'surv2m', 'surv6m',
               'totmcst']
    print("head of data:{}, data shape:{}".format(data_frame.head(), data_frame.shape))
    print("missing:{}".format(missing_proportion(data_frame.drop(labels=to_drop, axis=1))))
    # Preprocess
    one_hot_encoder_list = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca', 'sfdm2']
    data_frame = one_hot_encoder(data=data_frame, encode=one_hot_encoder_list)
    # data_frame = replace_na_with_median(data=data_frame,
    #                                     replace_ls=['totmcst', 'totcst', 'wblc', 'alb', 'adlp', 'adls',
    #                                                 'charges', 'scoma', 'hrt', 'resp', 'temp', 'pafi', 'sod', 'ph',
    #                                                 'glucose', 'edu', 'avtisst', 'meanbp'])
    # data_frame = replace_na_with_vitals_normal(data_frame,
    #                                            replace_dic={'urine': 2502, 'crea': 1.01, 'bili': 1.01, 'bun': 6.51})
    data_frame = log_transform(data_frame, transform_ls=['totmcst', 'totcst', 'charges', 'pafi', 'sod'])
    print("na columns:{}".format(data_frame.columns[data_frame.isnull().any()].tolist()))
    t_data = data_frame[['d.time']]
    e_data = data_frame[['death']]
    x_data = data_frame.drop(labels=to_drop, axis=1)
    encoded_indices = one_hot_indices(x_data, one_hot_encoder_list)
    print("head of x data:{}, data shape:{}".format(x_data.head(), x_data.shape))
    print("data description:{}".format(x_data.describe()))
    covariates = np.array(x_data.columns.values)
    x = np.array(x_data).reshape(x_data.shape)
    t = np.array(t_data).reshape(len(t_data))
    e = np.array(e_data).reshape(len(e_data))
    # assert_nan([x, t, e])
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

    imputation_values = get_train_median_mode(x=x[train_idx], categorial=encoded_indices)
    print("imputation_values:{}".format(imputation_values))

    dataset = {
        'train': formatted_data(x=x, t=t, e=e, idx=train_idx),
        'test': formatted_data(x=x, t=t, e=e, idx=test_idx),
        'valid': formatted_data(x=x, t=t, e=e, idx=valid_idx),
        'end_t': end_time,
        'covariates': covariates,
        'one_hot_indices': encoded_indices,
        'imputation_values': imputation_values
    }
    return dataset
