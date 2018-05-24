import logging
import math
import os
import threading
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from lifelines.utils import concordance_index
from scipy.stats.stats import spearmanr

from model.aft_decoder import pt_log_normal_given_x
from utils.cost import log_normal_q_function, batch_metrics, log_normal_neg_log_lik, l1_loss, l2_loss
from utils.distributions import sample_log_normal
from utils.generated_times import plot_predicted_distribution
from utils.metrics import plot_cost
from utils.pre_processing import risk_set, get_missing_mask, flatten_nested
from utils.tf_helpers import show_all_variables


class DeepRegularizedAFT(object):
    def __init__(self,
                 batch_size,
                 learning_rate,
                 beta1,
                 beta2,
                 require_improvement,
                 seed,
                 num_iterations,
                 hidden_dim,
                 latent_dim,
                 risk_function,
                 input_dim,
                 num_examples,
                 keep_prob,
                 train_data,
                 valid_data,
                 test_data,
                 end_t,
                 covariates,
                 imputation_values,
                 categorical_indices,
                 l2_reg,
                 dropout_rate,
                 sample_size,
                 max_epochs,
                 path_large_data=""
                 ):

        self.max_epochs = max_epochs
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.latent_dim = latent_dim
        self.sample_size = sample_size
        self.path_large_data = path_large_data
        self.seed = seed
        self.require_improvement = require_improvement
        self.num_iterations = num_iterations
        self.learning_rate, self.beta1, self.beta2 = learning_rate, beta1, beta2
        self.l2_reg = l2_reg
        self.log_file = 'model.log'
        logging.basicConfig(filename=self.log_file, filemode='w', level=logging.DEBUG)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.batch_norm = True
        self.risk_function = risk_function
        self.covariates = covariates

        self.config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        self.config.gpu_options.allow_growth = True
        # self.config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        # Load Data
        self.train_x, self.train_t, self.train_e = train_data['x'], train_data['t'], train_data['e']

        self.valid_x, self.valid_t, self.valid_e = valid_data['x'], valid_data['t'], valid_data['e']

        self.test_x, self.test_t, self.test_e = test_data['x'], test_data['t'], test_data['e']
        self.end_t = end_t
        self.keep_prob = keep_prob
        self.input_dim = input_dim
        self.imputation_values = imputation_values
        # self.imputation_values = np.zeros(shape=self.input_dim)
        self.num_examples = num_examples
        self.categorical_indices = categorical_indices
        self.continuous_indices = np.setdiff1d(np.arange(input_dim), flatten_nested(categorical_indices))
        print_features = "input_dim:{}, continuous:{}, size:{}, categorical:{}, size{}".format(self.input_dim,
                                                                                               self.continuous_indices,
                                                                                               len(
                                                                                                   self.continuous_indices),
                                                                                               self.categorical_indices,
                                                                                               len(
                                                                                                   self.categorical_indices))
        print(print_features)
        logging.debug(print_features)

        self._build_graph()
        self.train_cost, self.train_ci, self.train_t_rae, self.train_log_lik, self.train_ranking, \
        self.train_recon = [], [], [], [], [], []
        self.valid_cost, self.valid_ci, self.valid_t_rae, self.valid_log_lik, self.valid_ranking, \
        self.valid_recon = [], [], [], [], [], []

    def _build_graph(self):
        self.G = tf.Graph()
        with self.G.as_default():
            self.x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='x')
            self.x_lab = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='x_lab')
            self.x_unlab = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='x_unlab')
            self.e = tf.placeholder(tf.float32, shape=[None], name='e')
            self.t = tf.placeholder(tf.float32, shape=[None], name='t')
            self.t_lab = tf.placeholder(tf.float32, shape=[None], name='t_lab')
            self.t_unlab = tf.placeholder(tf.float32, shape=[None], name='t_unlab')
            # are used to feed data into our queue
            self.batch_size_tensor = tf.placeholder(tf.int32, shape=[], name='batch_size')
            self.risk_set = tf.placeholder(tf.float32, shape=[None, None])
            self.impute_mask = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='impute_mask')
            self.is_training = tf.placeholder(tf.bool)

            self._objective()
            self.session = tf.Session(config=self.config)

            self.capacity = 1400
            self.coord = tf.train.Coordinator()
            enqueue_thread = threading.Thread(target=self.enqueue, daemon=True)
            self.queue = tf.RandomShuffleQueue(capacity=self.capacity, dtypes=[tf.float32, tf.float32, tf.float32],
                                               shapes=[[self.input_dim], [], []], min_after_dequeue=self.batch_size)
            # self.queue = tf.FIFOQueue(capacity=self.capacity, dtypes=[tf.float32, tf.float32, tf.float32],
            #                           shapes=[[self.input_dim], [], []])
            self.enqueue_op = self.queue.enqueue_many([self.x, self.t, self.e])
            # enqueue_thread.isDaemon()
            enqueue_thread.start()
            dequeue_op = self.queue.dequeue()
            self.x_batch, self.t_batch, self.e_batch = tf.train.batch(dequeue_op, batch_size=self.batch_size,
                                                                      capacity=self.capacity)
            self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.session)

            self.saver = tf.train.Saver()
            self.merged = tf.summary.merge_all()
            self.current_dir = os.getcwd()
            self.save_path = self.current_dir + "/summaries/deep_regularized_aft_model"
            self.train_writer = tf.summary.FileWriter(self.save_path, self.session.graph)

    def _objective(self):
        self.num_batches = self.num_examples / self.batch_size
        logging.debug("num batches:{}, batch_size:{} epochs:{}".format(self.num_batches, self.batch_size,
                                                                       int(self.num_iterations / self.num_batches)))
        self._build_model()
        self.reg_loss = l2_loss(self.l2_reg) + l1_loss(self.l2_reg)
        self.cost = self.neg_log_lik + self.ranking_partial_lik
        # self.cost = self.neg_log_lik
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1,
                                                beta2=self.beta2).minimize(self.cost)

    def _build_model(self):
        self._risk_model()

    def _risk_model(self):
        # t_mu = shape, t_log_var = rate
        self.t_mu, self.t_log_var = pt_log_normal_given_x(x=self.x,
                                                          hidden_dim=self.hidden_dim,
                                                          is_training=self.is_training,
                                                          batch_norm=self.batch_norm, keep_prob=self.keep_prob)
        self.predicted_time = tf.exp(self.t_mu)
        # Only for training
        indices_lab = tf.where(tf.equal(tf.constant(1.0, dtype=tf.float32), self.e))
        indices_unlab = tf.where(tf.equal(tf.constant(0.0, dtype=tf.float32), self.e))

        mu_lab = tf.squeeze(tf.gather(self.t_mu, indices_lab))
        log_var_lab = tf.squeeze(tf.gather(self.t_log_var, indices_lab))

        mu_unlab = tf.squeeze(tf.gather(self.t_mu, indices_unlab))
        log_var_unlab = tf.squeeze(tf.gather(self.t_log_var, indices_unlab))
        # Cost Functions
        self.observed_neg_lik = log_normal_neg_log_lik(mu=mu_lab, log_var=log_var_lab, t=self.t_lab)
        self.censored_neg_lik = log_normal_q_function(mu=mu_unlab, log_var=log_var_unlab, t=self.t_unlab)
        # censored_neg_lik = tf.constant(0.0)
        self.neg_log_lik = tf.add(self.observed_neg_lik, self.censored_neg_lik)
        self.ranking_partial_lik, self.total_rae, self.total_t_recon_loss = batch_metrics(e=self.e,
                                                                                          risk_set=self.risk_set,
                                                                                          predicted=self.predicted_time,
                                                                                          batch_size=self.batch_size_tensor,
                                                                                          empirical=self.t)

    def predict_concordance_index(self, x, t, e, outcomes=None):
        input_size = x.shape[0]
        i = 0
        num_batches = input_size / self.batch_size
        predicted_time = np.zeros(shape=input_size, dtype=np.int)
        log_var = np.zeros(shape=input_size, dtype=np.int)
        total_ranking = 0.0
        total_rae = 0.0
        total_cost = 0.0
        total_log_lik = 0.0
        total_recon = 0.0
        total_reg = 0.0
        while i < input_size:
            # The ending index for the next batch is denoted j.
            j = min(i + self.batch_size, input_size)
            feed_dict = self.batch_feed_dict(e=e, i=i, j=j, t=t, x=x, outcomes=outcomes)
            predicted_time[i:j], cost, ranking, lik, rae, reg, log_var[i:j], recon = self.session.run(
                [self.predicted_time, self.cost, self.ranking_partial_lik, self.neg_log_lik, self.total_rae,
                 self.reg_loss,
                 self.t_log_var, self.total_t_recon_loss],
                feed_dict=feed_dict)

            total_ranking += ranking
            total_cost += cost
            total_rae += rae
            total_log_lik += lik
            total_reg += reg
            total_recon += recon
            i = j

        predicted_event_times = predicted_time.reshape(input_size)
        ci_index = concordance_index(event_times=t, predicted_event_times=predicted_event_times.tolist(),
                                     event_observed=e)

        def batch_average(total):
            return total / num_batches

        return ci_index, batch_average(total_cost), batch_average(total_rae), batch_average(
            total_ranking), batch_average(
            total_log_lik), batch_average(total_reg), log_var, batch_average(total_recon)

    def batch_feed_dict(self, e, i, j, t, x, outcomes):
        batch_x = x[i:j, :]
        batch_t = t[i:j]
        batch_risk = risk_set(batch_t)
        batch_impute_mask = get_missing_mask(batch_x, self.imputation_values)
        batch_e = e[i:j]
        idx_observed = batch_e == 1
        feed_dict = {self.x: batch_x,
                     self.x_lab: batch_x[idx_observed],
                     self.x_unlab: batch_x[np.logical_not(idx_observed)],
                     self.impute_mask: batch_impute_mask,
                     self.t: batch_t,
                     self.t_lab: batch_t[idx_observed],
                     self.t_unlab: batch_t[np.logical_not(idx_observed)],
                     self.e: batch_e,
                     self.risk_set: batch_risk,
                     self.batch_size_tensor: len(batch_t),
                     self.is_training: False}
        # TODO replace with abstract methods

        updated_feed_dic = self.outcomes_function(idx=i, j=j, feed_dict=feed_dict, outcomes=outcomes)
        return updated_feed_dic

    def outcomes_function(self, idx, j, feed_dict, outcomes):
        return feed_dict

    def train_neural_network(self):
        train_print = "Training Deep Regularized AFT Model:"
        params_print = "Parameters: risk:{}, l2_reg:{}, learning_rate:{}," \
                       " momentum: beta1={} beta2={}, batch_size:{}, batch_norm:{}," \
                       " hidden_dim:{}, latent_dim:{}, num_of_batches:{}, keep_prob:{}" \
            .format(self.risk_function, self.l2_reg, self.learning_rate, self.beta1, self.beta2, self.batch_size,
                    self.batch_norm, self.hidden_dim, self.latent_dim, self.num_batches, self.keep_prob)
        print(train_print)
        print(params_print)
        logging.debug(train_print)
        logging.debug(params_print)
        self.session.run(tf.global_variables_initializer())

        best_ci = 0
        best_validation_epoch = 0
        last_improvement = 0

        start_time = time.time()
        epochs = 0
        show_all_variables()
        j = 0

        for i in range(self.num_iterations):
            # Batch Training
            run_options = tf.RunOptions(timeout_in_ms=4000)
            x_batch, t_batch, e_batch = self.session.run([self.x_batch, self.t_batch, self.e_batch],
                                                         options=run_options)
            risk_batch = risk_set(data_t=t_batch)
            batch_impute_mask = get_missing_mask(x_batch, self.imputation_values)
            batch_size = len(t_batch)
            idx_observed = e_batch == 1
            # TODO simplify batch processing
            feed_dict_train = {self.x: x_batch,
                               self.x_lab: x_batch[idx_observed],
                               self.x_unlab: x_batch[np.logical_not(idx_observed)],
                               self.impute_mask: batch_impute_mask,
                               self.t: t_batch,
                               self.t_lab: t_batch[idx_observed],
                               self.t_unlab: t_batch[np.logical_not(idx_observed)],
                               self.e: e_batch,
                               self.risk_set: risk_batch, self.batch_size_tensor: batch_size, self.is_training: True}
            summary, train_time, train_cost, train_ranking, train_rae, train_reg, train_lik, train_recon, \
            train_obs_lik, train_censo_lik, _ = self.session.run(
                [self.merged, self.predicted_time, self.cost, self.ranking_partial_lik, self.total_rae,
                 self.reg_loss, self.neg_log_lik, self.total_t_recon_loss, self.observed_neg_lik, self.censored_neg_lik,
                 self.optimizer],
                feed_dict=feed_dict_train)
            train_ci = concordance_index(event_times=t_batch,
                                         predicted_event_times=train_time.reshape(t_batch.shape),
                                         event_observed=e_batch)
            tf.verify_tensor_all_finite(train_cost, "Training Cost has Nan or Infinite")
            if j >= self.num_examples:
                epochs += 1
                is_epoch = True
                # idx = 0
                j = 0
            else:
                # idx = j
                j += self.batch_size
                is_epoch = False

            if i % 100 == 0:
                train_print = "it:{}, trainCI:{}, train_ranking:{}, train_RAE:{},  train_lik:{}, train_obs_lik:{}, " \
                              "train_cens_lik:{}, train_reg:{}".format(i, train_ci, train_ranking, train_rae, train_lik,
                                                                       train_obs_lik, train_censo_lik, train_reg)
                print(train_print)
                logging.debug(train_print)

            if is_epoch or (i == (self.num_iterations - 1)):
                improved_str = ''
                # Calculate  Vaid CI the CI
                self.train_ci.append(train_ci)
                self.train_cost.append(train_cost)
                self.train_t_rae.append(train_rae)
                self.train_log_lik.append(train_lik)
                self.train_ranking.append(train_ranking)
                self.train_recon.append(train_recon)

                self.train_writer.add_summary(summary, i)
                valid_ci, valid_cost, valid_rae, valid_ranking, valid_lik, valid_reg, valid_log_var, valid_recon = self.predict_concordance_index(
                    x=self.valid_x,
                    e=self.valid_e,
                    t=self.valid_t)
                self.valid_cost.append(valid_cost)
                self.valid_ci.append(valid_ci)
                self.valid_t_rae.append(valid_rae)
                self.valid_log_lik.append(valid_lik)
                self.valid_ranking.append(valid_ranking)
                self.valid_recon.append(valid_recon)
                tf.verify_tensor_all_finite(valid_cost, "Validation Cost has Nan or Infinite")

                if valid_ci > best_ci:
                    self.saver.save(sess=self.session, save_path=self.save_path)
                    best_validation_epoch = epochs
                    best_ci = valid_ci
                    print("valid_ci:{}".format(valid_ci))
                    last_improvement = i
                    improved_str = '*'
                    # Save  Best Perfoming all variables of the TensorFlow graph to file.
                # update best validation accuracy
                optimization_print = "Iteration: {} epochs:{}, Training: RAE:{}, Loss: {}," \
                                     " Ranking:{}, Reg:{}, Lik:{}, T_Recon:{}, CI:{}" \
                                     " Validation RAE:{} Loss:{}, Ranking:{}, Reg:{}, Lik:{}, T_Recon:{}, CI:{}, {}" \
                    .format(i + 1, epochs, train_rae, train_cost, train_ranking, train_reg, train_lik,
                            train_recon,
                            train_ci, valid_rae, valid_cost, valid_ranking, valid_reg, valid_lik, valid_recon,
                            valid_ci, improved_str)

                print(optimization_print)
                logging.debug(optimization_print)
                if i - last_improvement > self.require_improvement or math.isnan(
                        valid_cost) or epochs >= self.max_epochs:
                    print("No improvement found in a while, stopping optimization.")
                    # Break out from the for-loop.
                    break
        # Ending time.

        end_time = time.time()
        time_dif = end_time - start_time
        time_dif_print = "Time usage: " + str(timedelta(seconds=int(round(time_dif))))
        print(time_dif_print)
        logging.debug(time_dif_print)
        # shutdown everything to avoid zombies
        self.session.run(self.queue.close(cancel_pending_enqueues=True))
        self.coord.request_stop()
        self.coord.join(self.threads)
        return best_validation_epoch, epochs

    def train_test(self):

        def get_dict(x, t, e):
            observed_idx = e == 1
            feed_dict = {self.x: x,
                         self.x_lab: x[observed_idx],
                         self.x_unlab: x[np.logical_not(observed_idx)],
                         self.impute_mask: get_missing_mask(x, self.imputation_values),
                         self.t: t,
                         self.t_lab: t[observed_idx],
                         self.t_unlab: t[np.logical_not(observed_idx)],
                         self.e: e,
                         self.batch_size_tensor: len(t),
                         self.is_training: False}
            return {'feed_dict': feed_dict, 'outcomes': {}}

        best_epoch, epochs = self.train_neural_network()

        session_dict = {'Test': get_dict(x=self.test_x, t=self.test_t, e=self.test_e),
                        'Train': get_dict(x=self.train_x, t=self.train_t, e=self.train_e),
                        'Valid': get_dict(x=self.valid_x, t=self.valid_t, e=self.valid_e)}
        self.time_related_metrics(best_epoch, epochs, session_dict=session_dict)

        self.session.close()

    def time_related_metrics(self, best_epoch, epochs, session_dict):
        plot_cost(training=self.train_cost, validation=self.valid_cost, model='Deep AFT', name="Cost",
                  epochs=epochs,
                  best_epoch=best_epoch)
        plot_cost(training=self.train_ci, validation=self.valid_ci, model='Deep AFT', name="CI",
                  epochs=epochs,
                  best_epoch=best_epoch)
        plot_cost(training=self.train_t_rae, validation=self.valid_t_rae, model='Deep AFT', name="RAE",
                  epochs=epochs,
                  best_epoch=best_epoch)
        plot_cost(training=self.train_ranking, validation=self.valid_ranking, model='Deep AFT', name="Rank",
                  epochs=epochs,
                  best_epoch=best_epoch)
        plot_cost(training=self.train_log_lik, validation=self.valid_log_lik, model='Deep AFT', name="Neg Log Lik",
                  epochs=epochs, best_epoch=best_epoch)

        plot_cost(training=self.train_recon, validation=self.valid_recon, model='Deep AFT', name="Recon",
                  epochs=epochs, best_epoch=best_epoch)
        # TEST
        self.generate_plots(data_x=self.test_x, data_e=self.test_e, data_t=self.test_t, name='Test',
                            session_dict=session_dict['Test'])
        # VALID
        self.generate_plots(data_x=self.valid_x, data_e=self.valid_e, data_t=self.valid_t, name='Valid',
                            session_dict=session_dict['Valid'])
        # TRAIN
        self.generate_plots(data_x=self.train_x, data_e=self.train_e, data_t=self.train_t, name='Train',
                            session_dict=session_dict['Train'])

    def generate_plots(self, data_x, data_e, data_t, name, session_dict, save=True):
        self.saver.restore(sess=self.session, save_path=self.save_path)
        ci, cost, rae, ranking, lik, reg, log_var, recon = self.predict_concordance_index(x=data_x, e=data_e,
                                                                                          t=data_t,
                                                                                          outcomes=session_dict[
                                                                                              'outcomes'])

        observed_idx = self.extract_observed_death(name=name, observed_e=data_e, observed_t=data_t, save=save)
        predicted_time = self.session.run(self.predicted_time, feed_dict=session_dict['feed_dict'])

        observed_empirical = data_t[observed_idx]
        print("predicted_time:{}".format(predicted_time.shape))
        observed_predicted = predicted_time[observed_idx]
        observed_ci = concordance_index(event_times=observed_empirical, predicted_event_times=observed_predicted,
                                        event_observed=data_e[observed_idx])

        corr = spearmanr(observed_empirical, observed_predicted)
        results = ":{} RAE:{}, Loss:{}, Lik:{}, Reg:{}, Ranking{}, Total_T_Recon:{}, CI:{}, Observed: CI:{}, " \
                  "Correlation:{}".format(name, rae, cost, lik, reg, ranking, recon, ci, observed_ci, corr)
        logging.debug(results)
        print(results)

        if name == 'Test':
            self.save_time_samples(x=data_x[observed_idx], e=data_e[observed_idx],
                                   t=data_t[observed_idx], name='obs', cens=False)

            self.save_time_samples(x=data_x[np.logical_not(observed_idx)], e=data_e[np.logical_not(observed_idx)],
                                   t=data_t[np.logical_not(observed_idx)], name='cen', cens=True)

            np.save('matrix/{}_predicted_time'.format(name), predicted_time)
            np.save('matrix/{}_empirical_time'.format(name), data_t)
            np.save('matrix/{}_data_e'.format(name), data_e)
            np.savetxt('matrix/{}_log_var'.format(name), log_var)

    def enqueue(self):
        """ Iterates over our data puts small junks into our queue."""
        # TensorFlow Input Pipelines for Large Data Sets
        # ischlag.github.io
        # http://ischlag.github.io/2016/11/07/tensorflow-input-pipeline-for-large-datasets/
        # http://web.stanford.edu/class/cs20si/lectures/slides_09.pdf
        under = 0
        max = len(self.train_x)
        try:
            while not self.coord.should_stop():
                # print("starting to write into queue")
                upper = under + self.capacity
                # print("try to enqueue ", under, " to ", upper)
                if upper <= max:
                    curr_x = self.train_x[under:upper]
                    curr_t = self.train_t[under:upper]
                    curr_e = self.train_e[under:upper]
                    under = upper
                else:
                    rest = upper - max
                    curr_x = np.concatenate((self.train_x[under:max], self.train_x[0:rest]))
                    curr_t = np.concatenate((self.train_t[under:max], self.train_t[0:rest]))
                    curr_e = np.concatenate((self.train_e[under:max], self.train_e[0:rest]))
                    under = rest

                self.session.run(self.enqueue_op,
                                 feed_dict={self.x: curr_x, self.t: curr_t, self.e: curr_e})
        except tf.errors.CancelledError:
            print("finished enqueueing")

    @staticmethod
    def extract_observed_death(name, observed_e, observed_t, save=False):
        idx_observed = observed_e == 1
        observed_death = observed_t[idx_observed]
        if save:
            death_observed_print = "{} observed_death:{}, percentage:{}".format(name, observed_death.shape, float(
                len(observed_death) / len(observed_t)))
            logging.debug(death_observed_print)
            print(death_observed_print)
        return idx_observed

    def save_time_samples(self, x, t, e, name, cens=False):
        observed = e == 1
        feed_dict = {self.x: x,
                     self.impute_mask: get_missing_mask(x, self.imputation_values),
                     self.t: t,
                     self.t_lab: t[observed],
                     self.e: e,
                     self.risk_set: risk_set(t),
                     self.batch_size_tensor: len(t),
                     self.is_training: False}

        mean, log_var = self.session.run([self.t_mu, self.t_log_var], feed_dict=feed_dict)
        predicted_time = sample_log_normal(log_var=log_var, mean=mean, sample_size=self.sample_size)
        print("predicted_time_samples:{}".format(predicted_time.shape))
        np.save('matrix/{}_{}_samples_predicted_time'.format('Test', name), predicted_time)
        plot_predicted_distribution(predicted=predicted_time, empirical=t, data='Test_' + name, cens=cens)
        return
