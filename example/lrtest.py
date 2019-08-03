import tensorflow as tf
from datasets.tabular_dataset import TfrecordBuilder, TabularDataSet
import pandas as pd
from models.lr import LR
from sklearn.preprocessing import LabelEncoder
from utils.toolbox import build_scheme_dict
from sklearn.preprocessing import MinMaxScaler


def main(_):
    data = pd.read_csv('train_set.csv')
    feature = data.columns.tolist()
    feature.remove('ID')
    feature.remove('y')
    sparse_feature = ['campaign','contact','default','education','housing','job','loan','marital','month','poutcome']
    dense_feature = list(set(feature)-set(sparse_feature))
    label = 'y'
    lbl = LabelEncoder()
    mms = MinMaxScaler()
    for s in sparse_feature:
        data[s] = lbl.fit_transform(data[s])
    for d in dense_feature:
        data[d] = mms.fit_transform(data[d].values.reshape(-1, 1))
    scheme_dict = build_scheme_dict(data, dense_feature, sparse_feature, label, 'sc.json')
    TfrecordBuilder(data, scheme_dict, 'test.tfrecord')
    train_dataset = TabularDataSet(scheme_dict, 'test.tfrecord', is_train=True, epochs=1, batch_size=1000)
    model = LR(scheme_dict, 5)
    model.fit(train_dataset)
    train_dataset = TabularDataSet(scheme_dict, 'test.tfrecord', is_train=True, epochs=1, batch_size=1000)
    result = model.eval(train_dataset)
    print(result)


tf.logging.set_verbosity(tf.logging.INFO)
tf.app.run(main)
