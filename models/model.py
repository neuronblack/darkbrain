import tensorflow as tf


class Model(object):
    def __init__(self, steps=None):
        self._estimator = tf.estimator.Estimator(self.build_model())
        self._steps = steps
        self._optimizer = self.model_optimizer()

    def build_model(self):
        def model_fn(features, labels, mode, params):
            predictions = self.network(features)
            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)
            loss = self.model_loss(labels, predictions)
            metrics = self.model_metric(labels, predictions)
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics)
            assert mode == tf.estimator.ModeKeys.TRAIN
            train_op = self.model_train_op(loss)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        return model_fn

    def network(self, x):
        raise NotImplementedError

    def fit(self, data_set):
        self._estimator.train(data_set.parsed_data, steps=self._steps)

    def predict(self, data_set):
        return self._estimator.predict(data_set.parsed_data)

    def eval(self, data_set):
        return self._estimator.evaluate(data_set.parsed_data)

    @staticmethod
    def feature_engineer(self, features):
        raise NotImplementedError

    @staticmethod
    def model_loss(labels, predictions):
        raise NotImplementedError

    @staticmethod
    def model_metric(label, predictions):
        raise NotImplementedError

    @staticmethod
    def model_optimizer():
        raise NotImplementedError

    def model_train_op(self, loss):
        train_op = self._optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return train_op
