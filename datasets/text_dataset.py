import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
import json


class TextDataSet(object):
    def __init__(self, tfrecord_path, vocab_dict, is_train=False, epochs=5,batch_size=32, buffer_size=50):
        self._tfrecord_path = tfrecord_path
        if isinstance(vocab_dict, dict):
            self._scheme_dict = vocab_dict
        elif isinstance(vocab_dict, str):
            with open(vocab_dict, "r") as f:
                self._scheme_dict = json.load(f)
        else:
            raise TypeError
        self._is_train = is_train
        self._epochs = epochs
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._feature_description = self._make_feature()

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

    @staticmethod
    def _make_feature():
        feature_description = {}
        feature_description['sequence'] = tf.VarLenFeature(tf.int64)
        return feature_description


class SeqTfrecordBuilder(object):
    def __init__(self, tfrecord_path, texts, label=None,vocab_dict_path=None):
        self._texts = texts
        self._tfrecord_path = tfrecord_path
        self._word_index, self._sequences = self.tokenizer_text(texts)
        self._vocab_dict_path = vocab_dict_path
        self.label = label
        if self._vocab_dict_path:
            with open(self._vocab_dict_path, "w") as f:
                json.dump(self._word_index, f)

    def _write_tfrecord(self):
        with tf.io.TFRecordWriter(self._tfrecord_path) as writer:
            for s in self._sequences:
                example = self._serialize_example(s)
                writer.write(example)

    def _serialize_example(self, sequence):
        feature = {}
        feature['sequence'] = tf.train.Feature(int64_list=sequence)
        if self.label:
            feature['label'] = tf.train.Feature(int64_list=sequence)
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    @staticmethod
    def tokenizer_text(text):
        tok = Tokenizer()
        tok.fit_on_texts(text)
        word_index = tok.word_index
        return word_index, tok.texts_to_sequences(text)

