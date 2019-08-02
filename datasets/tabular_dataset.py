import tensorflow as tf


class Pd2TfrecordSet(object):
    def __init__(self, data, dense_feature, sparse_feature,
                 tfrecord_path, label=None, epochs=5,batch_size=32, buffer_size=50,
                 only_read=False):
        self.data = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.dense_feature = dense_feature
        self.sparse_feature = sparse_feature
        self.label = label
        self.tfrecord_path = tfrecord_path
        if not only_read:
            self._write_tfrecord()
        self.feature_description = self._make_feature()

    def _write_tfrecord(self):
        with tf.io.TFRecordWriter(self.tfrecord_path) as writer:
            for _, row in self.data.iterrows():
                example = self._serialize_example(row)
                writer.write(example)

    def _make_feature(self):
        feature_description = {}
        for d in self.dense_feature + [self.label]:
            feature_description[d] = tf.FixedLenFeature((1,), tf.float32, default_value=0)
        for s in self.sparse_feature:
            feature_description[s] = tf.FixedLenFeature((1,), tf.float32, default_value=0)
        return feature_description

    def _parse_function(self, example_proto):
        feature = tf.io.parse_single_example(example_proto, self.feature_description)
        if self.label:
            label = feature.pop(self.label)
            return feature, label
        return feature

    def parsed_data(self):
        filenames = self.tfrecord_path
        parsed_dataset = tf.data.TFRecordDataset(filenames)
        parsed_dataset = parsed_dataset.repeat(self.epochs)
        parsed_dataset = parsed_dataset.map(self._parse_function)
        parsed_dataset = parsed_dataset.prefetch(buffer_size=self.buffer_size)
        return parsed_dataset.batch(self.batch_size)

    def _serialize_example(self, row):
        feature = {}
        for d in self.dense_feature + [self.label]:
            feature[d] = tf.train.Feature(float_list=tf.train.FloatList(value=[row[d]]))
        for s in self.sparse_feature:
            feature[s] = tf.train.Feature(float_list=tf.train.FloatList(value=[row[s]]))
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

