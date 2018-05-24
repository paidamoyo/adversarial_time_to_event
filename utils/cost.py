import tensorflow as tf

from utils.distributions import tf_normal_log_qfunction, tf_normal_logpdf


def x_reconstruction(x_recon, x, categorical_indices, continuous_indices, batch_size):
    print("x_reconstruction categorical_indices:{}".format(categorical_indices))

    def condition(i, recon):
        return i < batch_size

    def body(i, recon):
        # get edges for observation i
        x_i = tf.gather(x, i)
        x_recon_i = tf.gather(x_recon, i)
        # print("ae x_i:{}".format(x_i.shape))
        x_masked = tf.where(tf.is_nan(x_i), tf.zeros_like(x_i), x_i)
        x_recon_masked = tf.where(tf.is_nan(x_i), tf.zeros_like(x_i), x_recon_i)
        if len(categorical_indices) == 0:
            total_recon = tf.losses.mean_squared_error(x_masked, x_recon_masked)
        else:
            continous_x = tf.gather(x_masked, continuous_indices)
            continous_x_recon = tf.gather(x_recon_masked, continuous_indices)
            # print("ae continous x_i:{}".format(continous_x.shape))
            l2_recon = tf.losses.mean_squared_error(continous_x, continous_x_recon)
            cross_entropy = tf.constant(0.0)
            for category in categorical_indices:
                labels = tf.gather(x_masked, category)
                logits = tf.gather(x_recon_masked, category)
                # print("ae categorial x_i:{}".format(labels.shape))
                cross_entropy += tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
            total_recon = tf.add(l2_recon, cross_entropy)

        cumulative_recon = tf.add(total_recon, recon)
        return [i + 1, tf.reshape(cumulative_recon, shape=())]

    # Relevant Functions
    idx = tf.constant(0, shape=())
    initial_recon = tf.constant(0.0, shape=())
    _, total_recon, = tf.while_loop(condition, body, loop_vars=[idx, initial_recon], shape_invariants=[idx.get_shape(),
                                                                                                       initial_recon.get_shape(),
                                                                                                       ])
    return tf.div(total_recon, tf.cast(batch_size, dtype=tf.float32))


def batch_metrics(e, risk_set, predicted, batch_size, empirical):
    partial_likelihood = tf.constant(0.0, shape=())
    rel_abs_err = tf.constant(0.0, shape=())
    total_cens_loss = tf.constant(0.0, shape=())
    total_obs_loss = tf.constant(0.0, shape=())
    predicted = tf.squeeze(predicted)
    observed = tf.reduce_sum(e)
    censored = tf.subtract(tf.cast(batch_size, dtype=tf.float32), observed)

    def condition(i, likelihood, rae, recon_loss, obs_recon_loss):
        return i < batch_size

    def body(i, likelihood, rae, cens_recon_loss, obs_recon_loss):
        # get edges for observation i
        pred_t_i = tf.gather(predicted, i)
        emp_t_i = tf.gather(empirical, i)
        e_i = tf.gather(e, i)
        censored = tf.equal(e_i, 0)
        obs_at_risk = tf.gather(risk_set, i)
        print("obs_at_risk:{}, g_theta:{}".format(obs_at_risk.shape, predicted.shape))
        risk_hazard_list = tf.multiply(predicted, obs_at_risk)
        num_adjacent = tf.reduce_sum(obs_at_risk)
        # calculate partial likelihood
        risk = tf.subtract(pred_t_i, risk_hazard_list)
        activated_risk = tf.nn.sigmoid(risk)
        # logistic = map((lambda ele: log(1 + exp(ele * -1)) * 1 / log(2)), x)
        constant = 1e-8
        log_activated_risk = tf.div(tf.log(activated_risk + constant), tf.log(2.0))
        obs_likelihood = tf.add(log_activated_risk, num_adjacent)
        uncensored_likelihood = tf.cond(censored, lambda: tf.constant(0.0), lambda: obs_likelihood)
        cumulative_likelihood = tf.reduce_sum(uncensored_likelihood)
        updated_likelihood = tf.add(cumulative_likelihood, likelihood)

        # RElative absolute error
        abs_error_i = tf.abs(tf.subtract(pred_t_i, emp_t_i))
        pred_great_empirical = tf.greater(pred_t_i, emp_t_i)
        min_rea_i = tf.minimum(tf.div(abs_error_i, pred_t_i), tf.constant(1.0))
        rea_i = tf.cond(tf.logical_and(censored, pred_great_empirical), lambda: tf.constant(0.0), lambda: min_rea_i)
        cumulative_rae = tf.add(rea_i, rae)

        # Censored generated t loss
        diff_time = tf.subtract(pred_t_i, emp_t_i)
        # logistic = map((lambda ele: log(1 + exp(ele * -1)) * 1 / log(2)), x)
        # logistic = tf.div(tf.nn.sigmoid(diff_time) + constant, tf.log(2.0))
        # hinge = map(lambda ele: max(0, 1 - ele), x)
        hinge = tf.nn.relu(1.0 - diff_time)
        censored_loss_i = tf.cond(censored, lambda: hinge, lambda: tf.constant(0.0))
        # Sum over all edges and normalize by number of edges
        # L1 recon
        observed_loss_i = tf.cond(censored, lambda: tf.constant(0.0),
                                  lambda: tf.losses.absolute_difference(labels=emp_t_i, predictions=pred_t_i))
        # add observation risk to total risk
        cum_cens_loss = tf.add(cens_recon_loss, censored_loss_i)
        cum_obs_loss = tf.add(obs_recon_loss, observed_loss_i)
        return [i + 1, tf.reshape(updated_likelihood, shape=()), tf.reshape(cumulative_rae, shape=()),
                tf.reshape(cum_cens_loss, shape=()), tf.reshape(cum_obs_loss, shape=())]

    # Relevant Functions
    idx = tf.constant(0, shape=())
    _, total_likelihood, total_rel_abs_err, batch_cens_loss, batch_obs_loss = \
        tf.while_loop(condition, body,
                      loop_vars=[idx,
                                 partial_likelihood,
                                 rel_abs_err,
                                 total_cens_loss,
                                 total_obs_loss],
                      shape_invariants=[
                          idx.get_shape(),
                          partial_likelihood.get_shape(),
                          rel_abs_err.get_shape(),
                          total_cens_loss.get_shape(),
                          total_obs_loss.get_shape()])
    square_batch_size = tf.pow(batch_size, tf.constant(2))

    def normarlize_loss(cost, size):
        return tf.div(cost, tf.cast(size, dtype=tf.float32))

    total_recon_loss = tf.add(normarlize_loss(batch_cens_loss, size=censored),
                              normarlize_loss(batch_obs_loss, size=observed))
    normalized_log_likelihood = normarlize_loss(total_likelihood, size=square_batch_size)
    return normalized_log_likelihood, normarlize_loss(total_rel_abs_err, size=batch_size), total_recon_loss


def l2_loss(scale):
    l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    return l2 * scale


def l1_loss(scale):
    l1_regularizer = tf.contrib.layers.l1_regularizer(
        scale=scale, scope=None
    )
    weights = tf.trainable_variables()  # all vars of your graph
    l1 = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
    return l1


def log_normal_q_function(log_var, mu, t):
    constant = 1e-8
    lik = tf_normal_log_qfunction(x=tf.log(t + constant), mu=mu, log_var=log_var)
    print("log_lik:{}".format(lik))
    return -tf.reduce_mean(lik)


def log_normal_neg_log_lik(log_var, mu, t):
    constant = 1e-8
    lik = tf_normal_logpdf(x=tf.log(t + constant), mu=mu, log_var=log_var)
    print("log_lik:{}".format(lik))
    return -tf.reduce_mean(lik)
