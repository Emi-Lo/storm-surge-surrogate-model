import tensorflow as tf

def quantile_loss(y_true, y_pred, quantile=0.25):
    quantile_loss = tf.maximum(quantile * (y_true - y_pred), (1-quantile) * (y_pred - y_true))
    return tf.reduce_mean(quantile_loss)
    
def expectile_loss(y_true, y_pred, expectile=0.82):
    errors = y_true - y_pred
    quantile_loss = tf.where(errors >=0, expectile * tf.square(errors), (1-expectile) * tf.square(errors))
    return tf.reduce_mean(quantile_loss)
    
def weighted_quantile_expectile(y_true, y_pred):
    composite_loss = tf.add((1/6)*quantile_loss(y_true, y_pred, 0.25),
                            (5/6)*expectile_loss(y_true, y_pred, 0.82))
    return composite_loss
    
