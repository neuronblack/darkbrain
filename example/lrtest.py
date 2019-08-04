import tensorflow as tf
from datasets.tabular_dataset import TfrecordBuilder, TabularDataSet
import pandas as pd
from models.lr import LR
from sklearn.preprocessing import LabelEncoder
from utils.toolbox import build_scheme_dict
from sklearn.preprocessing import MinMaxScaler


def main(_):
    data = pd.read_csv('train10w.csv')
    feature = data.columns.tolist()
    feature.remove('finish')
    feature.remove('like')
    sparse_feature = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'music_id', 'did']
    dense_feature = ['video_duration']
    label = 'finish'
    lbl = LabelEncoder()
    mms = MinMaxScaler()
    for s in sparse_feature:
        data[s] = lbl.fit_transform(data[s])
    for d in dense_feature:
        data[d] = mms.fit_transform(data[d].values.reshape(-1, 1))
    vaild_split = int(data.shape[0]*0.7)
    train_data = data[:vaild_split]
    vaild_data = data[vaild_split:]
    scheme_dict = build_scheme_dict(data, dense_feature, sparse_feature, label, 'sc.json')
    TfrecordBuilder(train_data, scheme_dict, 'train.tfrecord')
    TfrecordBuilder(vaild_data, scheme_dict, 'vaild.tfrecord')
    train_dataset = TabularDataSet(scheme_dict, 'train.tfrecord', is_train=True, epochs=1, batch_size=1000)
    vaild_dataset = TabularDataSet(scheme_dict, 'vaild.tfrecord', is_train=True, epochs=1, batch_size=1000)
    model = LR(scheme_dict)
    model.fit(train_dataset)
    result = model.eval(vaild_dataset)
    print(result)


tf.logging.set_verbosity(tf.logging.INFO)
tf.app.run(main)
