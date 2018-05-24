import tensorflow as tf


def set_params():
    flags = tf.app.flags
    flags.DEFINE_integer("num_iterations", 40000, "DASA number of iterations")
    flags.DEFINE_integer("batch_size", 350, "Batch size")
    flags.DEFINE_float("dropout_rate", 0.0, "denoising input dropout rate")
    flags.DEFINE_integer("seed", 31415, "random seed")
    flags.DEFINE_integer("require_improvement", 10000, "num of iterations before early stopping")
    flags.DEFINE_float("learning_rate", 3e-4, "optimizer learning rate")
    flags.DEFINE_float("beta1", 0.9, "optimizer beta 1")
    flags.DEFINE_float("beta2", 0.999, "optimizer beta 2")
    flags.DEFINE_list("hidden_dim", [50, 50], "hidden layer dimensions and size")
    flags.DEFINE_string("risk_function", 'NA', "risk function is not simulated [linear, gaussian, NA]")
    flags.DEFINE_integer("latent_dim", 50, "latent dimensions of z")
    flags.DEFINE_float("l2_reg", 0.001, "l2 regularization weight multiplier (just for debugging not optimization)")
    flags.DEFINE_float("l1_reg", 0.001, "l1 regularization weight multiplier (just for debugging not optimization)")
    flags.DEFINE_float("keep_prob", 0.8, "keep prob for weights implementation in layers")
    flags.DEFINE_integer("sample_size", 200, "number of samples of generated time")
    flags.DEFINE_integer("disc_updates", 1, "number of discriminator updates before generator update")
    return flags
