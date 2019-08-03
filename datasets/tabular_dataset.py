import tensorflow as tf
import json


class TabularDataSet(object):
    def __init__(self, scheme_dict, tfrecord_path, is_train=False, epochs=5,batch_size=32, buffer_size=50):
        self._epochs = epochs
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        if isinstance(scheme_dict, dict):
            self._scheme_dict = scheme_dict
        elif isinstance(scheme_dict,str):
            with open(scheme_dict, "r") as f:
                self._scheme_dict = json.load(f)
        else:
            raise TypeError
        self._tfrecord_path = tfrecord_path
        self._is_train = is_train
        if self._is_train:
            assert 'label' in self._scheme_dict.keys()
        self._feature_description = self._make_feature()

    def _make_feature(self):
        feature_description = {}
        for d in self._scheme_dict['dense_feature']:
            feature_description[d] = tf.FixedLenFeature((1,), tf.float32, default_value=0)
        for s in self._scheme_dict['sparse_feature'].keys():
            feature_description[s] = tf.FixedLenFeature((1,), tf.int64, default_value=0)
        if self._is_train:
            feature_description[self._scheme_dict['label']] = tf.FixedLenFeature((1,), tf.float32, default_value=0)
        return feature_description

    def _parse_function(self, example_proto):
        feature = tf.io.parse_single_example(example_proto, self._feature_description)
        if self._is_train:
            label = feature.pop(self._scheme_dict['label'])
            return feature, label
        return feature

    def parsed_data(self):
        filenames = self._tfrecord_path
        parsed_dataset = tf.data.TFRecordDataset(filenames)
        if self._is_train:
            parsed_dataset = parsed_dataset.shuffle(buffer_size=self._buffer_size)
        parsed_dataset = parsed_dataset.repeat(self._epochs)
        parsed_dataset = parsed_dataset.map(self._parse_function)
        parsed_dataset = parsed_dataset.prefetch(buffer_size=self._buffer_size)
        return parsed_dataset.batch(self._batch_size)


class TfrecordBuilder(object):
    def __init__(self, data, scheme_dict, tfrecord_path):
        self._data = data
        self._tfrecord_path = tfrecord_path
        if isinstance(scheme_dict, dict):
            self._scheme_dict = scheme_dict
        elif isinstance(scheme_dict,str):
            with open(scheme_dict, "r") as f:
                self._scheme_dict = json.load(f)
        else:
            raise TypeError
        self._write_tfrecord()

    def _write_tfrecord(self):
        with tf.io.TFRecordWriter(self._tfrecord_path) as writer:
            for _, row in self._data.iterrows():
                example = self._serialize_example(row)
                writer.write(example)

    def _serialize_example(self, row):
        feature = {}
        for d in self._scheme_dict['dense_feature']:
            feature[d] = tf.train.Feature(float_list=tf.train.FloatList(value=[row[d]]))
        for s in self._scheme_dict['sparse_feature'].keys():
            feature[s] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(row[s])]))
        if 'label' in self._scheme_dict:
            feature[self._scheme_dict['label']] = tf.train.Feature(float_list=tf.train.FloatList(value=[row[self._scheme_dict['label']]]))
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
