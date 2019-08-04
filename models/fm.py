import tensorflow as tf
from .lr import LR
from layers.ctr_layers import BiInteractionPooling


class FM(LR):

    def network(self, input):
        embedding, numerical = self.feature_engineer(input)
        bi_interaction_pooling = BiInteractionPooling()
        embedding_bi = tf.stack(embedding, axis=1)
        bi_out = bi_interaction_pooling(embedding_bi)
        network_out = tf.concat([bi_out] + numerical + embedding, -1)
        return network_out
