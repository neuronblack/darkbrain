import json


def build_scheme_dict(data, dense_feature, sparse_feature, label=None, scheme_path=None):
    scheme_dict = {}
    scheme_dict['sparse_feature'] = {}
    for s in sparse_feature:
        scheme_dict['sparse_feature'][s] = data[s].nunique()
    scheme_dict['dense_feature'] = []
    for d in dense_feature:
        scheme_dict['dense_feature'].append(d)
    if label:
        scheme_dict['label'] = label
    if scheme_path:
        with open(scheme_path, "w") as f:
            json.dump(scheme_dict, f)
    return scheme_dict
