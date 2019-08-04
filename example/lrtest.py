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
    train_data = data[:20000]
    vaild_data = data[20000:]
    scheme_dict = build_scheme_dict(data, dense_feature, sparse_feature, label, 'sc.json')
    TfrecordBuilder(train_data, scheme_dict, 'train.tfrecord')
    TfrecordBuilder(vaild_data, scheme_dict, 'vaild.tfrecord')
    train_dataset = TabularDataSet(scheme_dict, 'train.tfrecord', is_train=True, epochs=100, batch_size=8000)
    vaild_dataset = TabularDataSet(scheme_dict, 'vaild.tfrecord', is_train=True, epochs=1, batch_size=8000)
    model = LR(scheme_dict, 5)
    model.fit(train_dataset, eval_set=vaild_dataset, patient=1, metric='roc_auc')
    result = model.eval(train_dataset)
    print(result)


tf.logging.set_verbosity(tf.logging.INFO)
tf.app.run(main)
