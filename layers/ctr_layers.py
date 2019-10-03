import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.initializers import Zeros


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


class Cross(Layer):

    def __init__(self, cross_num, **kwargs):
        self.cross_num = cross_num
        super().__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[-1].value
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(dim, 1),
                                        trainable=True) for i in range(self.cross_num)]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(dim,),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(self.cross_num)]
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x0 = inputs
        xl = x0
        for i in range(self.cross_num):
            feature_cross = tf.tensordot(xl, self.kernels[i], axes=(-1, 0))
            feature_cross = feature_cross * x0
            xl = feature_cross + self.bias[i] + xl
        return xl


