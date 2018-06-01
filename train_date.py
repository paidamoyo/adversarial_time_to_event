import os
import pprint
import sys

import data.flchain.flchain_data as flchain_data
import data.seer.seer_data as seer_data
import data.support.support_data as support_data
from flags_parameters import set_params
from model.date import DATE
from model.date_ae import DATE_AE

if __name__ == '__main__':
    GPUID = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)

    flchain = {"path": '', "preprocess": flchain_data, "epochs": 600}
    support = {"path": '', "preprocess": support_data, "epochs": 400}
    seer = {"path": '/data/ash/seer/', "preprocess": seer_data, "epochs": 120}  # TODO replace with your path
    # TODO choose data and model use simple for now
    dataset = support
    simple = True
    if simple:
        model = DATE
    else:
        model = DATE_AE

    flags = set_params()
    flags.DEFINE_string("path_large_data", dataset['path'], "path to save folder")
    FLAGS = flags.FLAGS
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS.__flags)

    args = sys.argv[1:]
    print("args:{}".format(args))
    if args:
        vm = float(args[0])
    else:
        vm = 1.0
    print("gpu_memory_fraction:{}".format(vm))
    data_set = dataset['preprocess'].generate_data()
    train_data, valid_data, test_data, end_t, covariates, one_hot_indices, imputation_values \
        = data_set['train'], \
          data_set['valid'], \
          data_set['test'], \
          data_set['end_t'], \
          data_set['covariates'], \
          data_set[
              'one_hot_indices'], \
          data_set[
              'imputation_values']

    print("imputation_values:{}, one_hot_indices:{}".format(imputation_values, one_hot_indices))
    print("end_t:{}".format(end_t))
    train = {'x': train_data['x'], 'e': train_data['e'], 't': train_data['t']}
    valid = {'x': valid_data['x'], 'e': valid_data['e'], 't': valid_data['t']}
    test = {'x': test_data['x'], 'e': test_data['e'], 't': test_data['t']}

    perfomance_record = []

    date = model(batch_size=FLAGS.batch_size,
                 learning_rate=FLAGS.learning_rate,
                 beta1=FLAGS.beta1,
                 beta2=FLAGS.beta2,
                 require_improvement=FLAGS.require_improvement,
                 num_iterations=FLAGS.num_iterations, seed=FLAGS.seed,
                 l2_reg=FLAGS.l2_reg,
                 hidden_dim=FLAGS.hidden_dim,
                 train_data=train, test_data=test, valid_data=valid,
                 input_dim=train['x'].shape[1],
                 num_examples=train['x'].shape[0], keep_prob=FLAGS.keep_prob,
                 latent_dim=FLAGS.latent_dim, end_t=end_t,
                 path_large_data=FLAGS.path_large_data,
                 covariates=covariates,
                 categorical_indices=one_hot_indices,
                 disc_updates=FLAGS.disc_updates,
                 sample_size=FLAGS.sample_size, imputation_values=imputation_values,
                 max_epochs=dataset['epochs'])

    with date.session:
        date.train_test()
