import tensorflow as tf
from tensorflow.python.keras.layers import Layer


class BiInteractionPooling(Layer):

    def __init__(self, **kwargs):
        super(BiInteractionPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BiInteractionPooling, self).build(input_shape)

    def call(self, inputs, **kwargs):
        square_of_sum = tf.square(tf.reduce_sum(inputs, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(tf.square(inputs), axis=1, keepdims=True)
        outputs = 0.5 * (square_of_sum - sum_of_square)
        return tf.reshape(outputs, (-1, outputs.shape.dims[-1]))


