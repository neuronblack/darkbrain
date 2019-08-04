import tensorflow as tf
from .model import Model


class LR(Model):
    def __init__(self, scheme_dict, embedding_size=None, steps=None):
        super(LR, self).__init__(steps)
        self._scheme_dict = scheme_dict
        self._embedding_size = embedding_size

    def network(self, input):
        embedding, numerical = self.feature_engineer(input)
        network_out = tf.concat(embedding + numerical, -1)
        return network_out

    def feature_engineer(self, features):
        embedding_layers = {k: tf.keras.layers.Embedding(v,self.get_embedding_size(v)) for k,v in self._scheme_dict['sparse_feature'].items()}
        embedding = [tf.squeeze(v(features[k]), 1) for k,v in embedding_layers.items()]
        numerical = [features[d] for d in self._scheme_dict['dense_feature']]
        numerical = tf.keras.layers.concatenate(numerical)
        return embedding, [tf.keras.layers.BatchNormalization()(numerical)]

    def get_embedding_size(self,vocab_size):
        return self._embedding_size if self._embedding_size else int(vocab_size ** 0.25 * 6)
    @staticmethod
    def model_metric(labels, predictions):
        return {'auc': tf.metrics.auc(labels, predictions)}

    @staticmethod
    def model_loss(labels, predictions):
        return tf.losses.log_loss(labels, predictions)

    @staticmethod
    def model_optimizer():
        return tf.train.AdamOptimizer()

    @staticmethod
    def model_predictions(network_out):
        return tf.layers.dense(network_out, 1, activation='sigmoid')
