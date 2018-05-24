import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from utils.distributions import uniform


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def mlp_neuron(layer_input, weights, biases, activation=True):
    mlp = tf.add(tf.matmul(layer_input, weights), biases)
    if activation:
        return tf.nn.relu(mlp)
    else:
        return mlp


def normalized_mlp(layer_input, weights, biases, is_training, batch_norm, layer, activation=tf.nn.relu):
    mlp = tf.add(tf.matmul(layer_input, weights), biases)
    if batch_norm:
        norm = batch_norm_wrapper(mlp, is_training, layer=layer)
        # norm = tf_batch_norm(is_training=is_training, inputs=mlp, layer=layer)
        return activation(norm)
    else:
        return activation(mlp)


def dropout_normalised_mlp(layer_input, weights, biases, is_training, batch_norm, layer, keep_prob=1,
                           activation=tf.nn.relu):
    mlp = normalized_mlp(layer_input, weights, biases, is_training, batch_norm,
                         layer=layer, activation=activation)  # apply DropOut to hidden layer
    drop_out = tf.cond(is_training, lambda: tf.nn.dropout(mlp, keep_prob), lambda: mlp)
    return drop_out


def create_nn_weights(layer, network, shape):
    h_vars = {}
    w_h = 'W_' + network + '_' + layer
    b_h = 'b_' + network + '_' + layer
    h_vars[w_h] = create_weights(shape=shape, name=w_h)
    h_vars[b_h] = create_biases([shape[1]], name=b_h)
    variable_summaries(h_vars[w_h], w_h)
    variable_summaries(h_vars[b_h], b_h)

    return h_vars[w_h], h_vars[b_h]


def create_biases(shape, name):
    print("name:{}, shape{}".format(name, shape))
    return tf.Variable(tf.constant(shape=shape, value=0.0), name=name)


def create_weights(shape, name):
    print("name:{}, shape{}".format(name, shape))
    # initialize weights using Glorot and Bengio(2010) scheme
    a = tf.sqrt(6.0 / (shape[0] + shape[1]))
    # return tf.Variable(tf.random_normal(shape, stddev=tf.square(0.0001)), name=name)
    return tf.Variable(tf.random_uniform(shape, minval=-a, maxval=a, dtype=tf.float32), name=name)


def variable_summaries(var, summary_name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(summary_name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))


def batch_norm_wrapper(inputs, is_training, layer):
    # http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False, name='{}_batch_norm_mean'.format(layer))
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False, name='{}_batch_norm_var'.format(layer))
    print("batch inputs {}, shape for var{}".format(inputs.get_shape(), inputs.get_shape()[-1]))

    offset = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), name='{}_batch_norm_offset'.format(layer))
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]), name='{}_batch_norm_scale'.format(layer))
    epsilon = 1e-5
    alpha = 0.9  # use numbers closer to 1 if you have more data

    def batch_norm():
        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        print("batch mean {}, var {}".format(batch_mean.shape, batch_var.shape))
        train_mean = tf.assign(pop_mean,
                               pop_mean * alpha + batch_mean * (1 - alpha))
        train_var = tf.assign(pop_var,
                              pop_var * alpha + batch_var * (1 - alpha))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, mean=batch_mean, variance=batch_var, offset=offset, scale=scale,
                                             variance_epsilon=epsilon)

    def pop_norm():
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, offset=offset, scale=scale,
                                         variance_epsilon=epsilon)

    return tf.cond(is_training, batch_norm, pop_norm)


def hidden_mlp_layers(batch_norm, hidden_dim, is_training, keep_prob, layer_input, size):
    tmp = layer_input
    for i in np.arange(size):
        input_shape = tmp.get_shape().as_list()[1]
        print("layer input shape:{}".format(input_shape))
        w_hi, b_hi = create_nn_weights('h{}_z'.format(i), 'decoder', [input_shape, hidden_dim[i]])
        h_i = dropout_normalised_mlp(layer_input=tmp, weights=w_hi, biases=b_hi,
                                     is_training=is_training,
                                     batch_norm=batch_norm, keep_prob=keep_prob,
                                     layer='h{}_z_decoder'.format(i))

        tmp = h_i
    return tmp


def hidden_mlp_layers_noise(batch_norm, hidden_dim, is_training, keep_prob, layer_input, noise_alpha, size,
                            batch_size):
    tmp = layer_input
    for i in np.arange(size):
        input_shape = tmp.get_shape().as_list()[1]
        print("layer input shape:{}".format(input_shape))
        w_hi, b_hi = create_nn_weights('h{}_z'.format(i), 'decoder', [input_shape, hidden_dim[i]])
        h_i = dropout_normalised_mlp(layer_input=tmp, weights=w_hi, biases=b_hi,
                                     is_training=is_training,
                                     batch_norm=batch_norm, keep_prob=keep_prob,
                                     layer='h{}_z_decoder'.format(i))

        # noise = standard_gaussian(dim=hidden_dim[i], batch_size=batch_size) * tf.gather(noise_alpha, i + 1)
        noise = uniform(dim=hidden_dim[i], batch_size=batch_size) * tf.gather(noise_alpha, i + 1)
        tmp = tf.concat([h_i, noise], axis=1)
    return tmp
