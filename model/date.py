import logging

import tensorflow as tf

from model.date_ae import DATE_AE
from model.risk_network import pt_given_x, discriminator_one
from utils.cost import batch_metrics, l2_loss, l1_loss


class DATE(DATE_AE):
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
                 input_dim,
                 num_examples,
                 keep_prob,
                 train_data,
                 valid_data,
                 test_data,
                 end_t,
                 covariates,
                 imputation_values,
                 sample_size,
                 disc_updates,
                 categorical_indices,
                 l2_reg,
                 max_epochs,
                 path_large_data=""
                 ):
        DATE_AE.__init__(self, batch_size=batch_size,
                         learning_rate=learning_rate,
                         beta1=beta1,
                         beta2=beta2,
                         require_improvement=require_improvement,
                         num_iterations=num_iterations, seed=seed,
                         l2_reg=l2_reg,
                         hidden_dim=hidden_dim,
                         train_data=train_data, test_data=test_data, valid_data=valid_data,
                         input_dim=input_dim,
                         num_examples=num_examples, keep_prob=keep_prob,
                         latent_dim=latent_dim, end_t=end_t,
                         path_large_data=path_large_data,
                         covariates=covariates,
                         categorical_indices=categorical_indices,
                         disc_updates=disc_updates,
                         sample_size=sample_size, imputation_values=imputation_values,
                         max_epochs=max_epochs)

        print_model = "model is DATE"
        print(print_model)
        logging.debug(print_model)
        self.model = 'DATE'
        self.imputation_values = imputation_values

    def _objective(self):
        self.num_batches = self.num_examples / self.batch_size
        logging.debug("num batches:{}, batch_size:{} epochs:{}".format(self.num_batches, self.batch_size,
                                                                       int(self.num_iterations / self.num_batches)))
        self._build_model()
        self.reg_loss = l2_loss(self.l2_reg) + l1_loss(self.l2_reg)
        self.layer_one_recon = tf.constant(0.0)
        self.cost = self.t_regularization_loss + self.disc_one_loss + self.gen_one_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1,
                                           beta2=self.beta2)

        dvars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Discriminator_one")
        self.disc_solver = optimizer.minimize(self.disc_one_loss, var_list=dvars1)
        self.gen_solver = optimizer.minimize(self.gen_one_loss + self.t_regularization_loss)

    def _build_model(self):
        self._risk_date()

    @staticmethod
    def log(x):
        return tf.log(x + 1e-8)

    def _risk_date(self):
        def expand_t_dim(t):
            return tf.expand_dims(t, axis=1)

        indices_lab = tf.where(tf.equal(tf.constant(1.0, dtype=tf.float32), self.e))
        x_lab = tf.squeeze(tf.gather(self.x, indices_lab), axis=[1])
        t_lab_exp = expand_t_dim(self.t_lab)

        t_gen = pt_given_x(x=self.x, hidden_dim=self.hidden_dim, is_training=self.is_training,
                           batch_norm=self.batch_norm, keep_prob=self.keep_prob, batch_size=self.batch_size_tensor,
                           input_dim=self.input_dim, noise_alpha=self.noise_alpha)

        # Discriminator B
        d_one_real, f_one_real = discriminator_one(pair_one=x_lab, pair_two=t_lab_exp, hidden_dim=self.hidden_dim,
                                                   is_training=self.is_training, batch_norm=self.batch_norm,
                                                   keep_prob=self.keep_prob)  # (x_nc, t_nc)
        d_one_fake, f_one_fake = discriminator_one(pair_one=self.x, pair_two=t_gen, hidden_dim=self.hidden_dim,
                                                   is_training=self.is_training, batch_norm=self.batch_norm,
                                                   reuse=True, keep_prob=self.keep_prob)  # (x, t_gen)

        # Discriminator loss
        self.disc_one_loss = -tf.reduce_mean(self.log(d_one_real)) - tf.reduce_mean(self.log(1 - d_one_fake))

        # Generator loss
        self.gen_one_loss = tf.reduce_mean(f_one_real) - tf.reduce_mean(f_one_fake)
        self.predicted_time = tf.squeeze(t_gen)
        self.ranking_partial_lik, self.total_rae, self.total_t_recon_loss = \
            batch_metrics(e=self.e,
                          risk_set=self.risk_set,
                          predicted=self.predicted_time,
                          batch_size=self.batch_size_tensor,
                          empirical=self.t)

        self.t_regularization_loss = self.total_t_recon_loss
        self.t_mse = tf.losses.mean_squared_error(labels=self.t_lab,
                                                  predictions=tf.gather(self.predicted_time, indices_lab))
