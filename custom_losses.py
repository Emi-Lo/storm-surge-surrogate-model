import tensorflow as tf


def quantile_loss_huber(y_true, y_pred, quantile, alpha=0.1):
    errors = y_true - y_pred
    abs_errors = tf.abs(errors)
    huber_errors = tf.where(abs_errors <= alpha, 0.5 * tf.square(errors)/alpha, (abs_errors - 0.5 * alpha))
    quantile_loss = tf.maximum(quantile * huber_errors * (y_true - y_pred), (1-quantile) * huber_errors * (y_pred - y_true))
    return tf.reduce_mean(quantile_loss)

def quantile_loss(y_true, y_pred):
    composite_loss = tf.reduce_mean([quantile_loss_huber(y_true, y_pred, 0.90)])
    return composite_loss
