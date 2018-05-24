import tensorflow as tf

from utils.distributions import uniform
from utils.tf_helpers import create_nn_weights, mlp_neuron, hidden_mlp_layers, hidden_mlp_layers_noise


def pt_given_z(z, hidden_dim, is_training, batch_norm, batch_size, latent_dim, noise_alpha, keep_prob=0.9, reuse=False):
    size = len(hidden_dim)
    with tf.variable_scope('generate_t_given_z', reuse=reuse):
        # Variables
        noise = uniform(dim=latent_dim, batch_size=batch_size) * tf.gather(noise_alpha, 0)
        z_plus_noise = tf.concat([z, noise], axis=1)

        hidden_z = hidden_mlp_layers_noise(batch_norm=batch_norm, hidden_dim=hidden_dim,
                                           is_training=is_training, keep_prob=keep_prob,
                                           layer_input=z_plus_noise, size=size, batch_size=batch_size,
                                           noise_alpha=noise_alpha)

        w_t, b_t = create_nn_weights('t', 'encoder', [hidden_z.get_shape().as_list()[1], 1])
        t_mu = mlp_neuron(hidden_z, w_t, b_t, activation=False)
        return tf.exp(t_mu)


def pt_given_x(x, hidden_dim, is_training, batch_norm, batch_size, input_dim, noise_alpha, keep_prob=0.9, reuse=False):
    size = len(hidden_dim)
    with tf.variable_scope('generate_t_given_x', reuse=reuse):
        # Variables
        noise = uniform(dim=input_dim, batch_size=batch_size) * tf.gather(noise_alpha, 0)
        x_plus_noise = tf.concat([x, noise], axis=1)
        hidden_x = hidden_mlp_layers_noise(batch_norm=batch_norm, hidden_dim=hidden_dim,
                                           is_training=is_training, keep_prob=keep_prob,
                                           layer_input=x_plus_noise, size=size, batch_size=batch_size,
                                           noise_alpha=noise_alpha)

        w_t, b_t = create_nn_weights('t', 'encoder', [hidden_x.get_shape().as_list()[1], 1])
        t_mu = mlp_neuron(hidden_x, w_t, b_t, activation=False)
        return tf.exp(t_mu)


def discriminator(pair_one, pair_two, hidden_dim, is_training, batch_norm, scope, keep_prob=1, reuse=False):
    size = len(hidden_dim)
    with tf.variable_scope(scope, reuse=reuse):
        # Variables
        print("scope:{}, pair_one:{}, pair_two:{}".format(scope, pair_one.shape, pair_two.shape))
        hidden_pair_one = hidden_mlp_layers(batch_norm=batch_norm, hidden_dim=hidden_dim,
                                            is_training=is_training, keep_prob=keep_prob,
                                            layer_input=pair_one, size=size)

        hidden_pair_two = hidden_mlp_layers(batch_norm=batch_norm, hidden_dim=hidden_dim,
                                            is_training=is_training, keep_prob=keep_prob,
                                            layer_input=pair_two, size=size)
        hidden_pairs = tf.concat([hidden_pair_one, hidden_pair_two], axis=1)
        print("hidden_pairs:{}".format(hidden_pairs.get_shape()))
        w_logit, b_logit = create_nn_weights('logits', 'discriminator', [hidden_dim[size - 1] * 2, 1])
        f = mlp_neuron(layer_input=hidden_pairs, weights=w_logit, biases=b_logit, activation=False)
        logit = tf.nn.sigmoid(f)

        return tf.squeeze(logit), tf.squeeze(f)


def discriminator_one(pair_one, pair_two, hidden_dim, is_training, batch_norm, keep_prob=1, reuse=False):
    score, f = discriminator(pair_one=pair_one, pair_two=pair_two, scope='Discriminator_one', batch_norm=batch_norm,
                             is_training=is_training,
                             keep_prob=keep_prob, reuse=reuse, hidden_dim=hidden_dim)
    return score, f


def discriminator_two(pair_one, pair_two, hidden_dim, is_training, batch_norm, keep_prob=1, reuse=False):
    score, f = discriminator(pair_one=pair_one, pair_two=pair_two, scope='Discriminator_two', batch_norm=batch_norm,
                             is_training=is_training,
                             keep_prob=keep_prob, reuse=reuse, hidden_dim=hidden_dim)
    return score, f
