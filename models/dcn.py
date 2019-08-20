import tensorflow as tf
from .lr import LR
from ..layers.ctr_layers import Cross


class DCN(LR):
    def __init__(self, scheme_dict, cross_num, embedding_size=None, steps=None):
        super(DCN, self).__init__(scheme_dict, embedding_size, steps)
        self.cross_num = cross_num

    def network(self, input):
        embedding, numerical = self.feature_engineer(input)
        feature_concat = tf.concat(embedding + numerical, -1)
        cross_net = Cross(self.cross_num)
        cross_net_out = cross_net(feature_concat)
        return cross_net_out
