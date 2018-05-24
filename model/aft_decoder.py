import tensorflow as tf

from utils.tf_helpers import create_nn_weights, mlp_neuron, hidden_mlp_layers


def pt_log_normal_given_x(x, hidden_dim, is_training, batch_norm, keep_prob=1, reuse=False,
                          scope='generate_t'):
    size = len(hidden_dim)
    with tf.variable_scope(scope, reuse=reuse):
        # Variables
        layer_input = hidden_mlp_layers(layer_input=x,
                                        is_training=is_training,
                                        batch_norm=batch_norm, keep_prob=keep_prob,
                                        size=len(hidden_dim), hidden_dim=hidden_dim)

        w_mu, b_mu = create_nn_weights('mu_t', 'decoder', [hidden_dim[size - 1], 1])
        w_logvar, b_logvar = create_nn_weights('var_t', 'decoder', [hidden_dim[size - 1], 1])
        # Model
        # Reconstruction layer
        t_mu = mlp_neuron(layer_input, w_mu, b_mu, activation=False)
        t_logvar = mlp_neuron(layer_input, w_logvar, b_logvar, activation=False)
        squeezed_t_mu = tf.squeeze(t_mu)
        squeeze_t_logvar = tf.squeeze(t_logvar)
        return squeezed_t_mu, squeeze_t_logvar
