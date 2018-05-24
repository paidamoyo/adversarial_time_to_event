import numpy as np
import tensorflow as tf

from utils.distributions import uniform
from utils.tf_helpers import create_nn_weights, mlp_neuron, hidden_mlp_layers


def generate_x_given_z(z, hidden_dim, latent_dim, is_training, batch_norm, batch_size, input_dim, keep_prob=1,
                       reuse=False):
    # Generative p(x|z)
    size = len(hidden_dim)
    with tf.variable_scope("decoder", reuse=reuse):
        noise = uniform(dim=latent_dim, batch_size=batch_size)
        z_plus_noise = tf.concat([z, noise], axis=1)
        layer_input = hidden_mlp_layers(batch_norm=batch_norm, hidden_dim=hidden_dim, is_training=is_training,
                                        keep_prob=keep_prob, layer_input=z_plus_noise, size=size)

        w_x, b_x = create_nn_weights('x', 'decoder', [hidden_dim[size - 1], input_dim])
        # Model
        # Reconstruction layer
        x = mlp_neuron(layer_input, w_x, b_x, activation=False)
        return x


def generate_z_given_x(x, hidden_dim, latent_dim, is_training, batch_norm, input_dim, batch_size,
                       keep_prob=1,
                       reuse=False, sample_size=200):
    size = len(hidden_dim)
    z = tf.zeros(shape=(batch_size, latent_dim))

    for _ in np.arange(sample_size):
        print("hidden_dim:{}, size:{}".format(hidden_dim, size))
        with tf.variable_scope("encoder_given_x", reuse=reuse):
            # Variables
            sample = sample_z(batch_norm, batch_size, hidden_dim, input_dim, is_training, keep_prob, latent_dim, size,
                              x)
            z = tf.add(z, sample)

    return tf.scalar_mul(scalar=1 / sample_size, x=z)


def sample_z(batch_norm, batch_size, hidden_dim, input_dim, is_training, keep_prob, latent_dim, size, x):
    noise = uniform(dim=input_dim, batch_size=batch_size)
    x_plus_noise = tf.concat([x, noise], axis=1)
    layer_input = hidden_mlp_layers(batch_norm=batch_norm, hidden_dim=hidden_dim,
                                    is_training=is_training, keep_prob=keep_prob,
                                    layer_input=x_plus_noise, size=size)
    w_z, b_z = create_nn_weights('z', 'encoder', [hidden_dim[size - 1], latent_dim])
    z = mlp_neuron(layer_input, w_z, b_z, activation=False)
    return z
