import numpy as np
import tensorflow as tf


def standard_gaussian(dim, batch_size):
    ones = np.ones(shape=dim, dtype=np.float32)
    noise = tf.distributions.Normal(loc=0 * ones, scale=ones).sample(sample_shape=[batch_size])
    return noise


def uniform(dim, batch_size):
    ones = np.ones(shape=dim, dtype=np.float32)
    noise = tf.distributions.Uniform(low=0 * ones, high=ones).sample(sample_shape=[batch_size])
    return noise


def sample_log_normal(log_var, mean, sample_size, multi=0.5):
    log_predicted_time = []
    for loc, scale in zip(mean, log_var):
        log_predicted_time.append(
            np.random.normal(loc=loc, scale=np.exp(scale * multi), size=sample_size))
    predicted_time = np.exp(log_predicted_time)
    predicted_time = np.transpose(predicted_time)
    return predicted_time


def tf_normal_log_qfunction(x, mu, log_var):
    constant = 1e-8
    cdf = tf.distributions.Normal(loc=mu, scale=tf.exp(log_var * 0.5)).cdf(x)
    return tf.log(1 - cdf + constant)


def tf_normal_logpdf(x, mu, log_var):
    constant = 1e-8
    pdf = tf.distributions.Normal(loc=mu, scale=tf.exp(log_var * 0.5)).prob(x)
    return tf.log(pdf + constant)
