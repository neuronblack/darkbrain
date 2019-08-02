def build_scheme_dict(data, dense_feature, sparse_feature):
    scheme_dict = {}
    scheme_dict['sparse_feature'] = {}
    for s in sparse_feature:
        scheme_dict['sparse_feature'][s] = data[s].nunique()
    scheme_dict['dense_feature']=[]
    for d in dense_feature:
        scheme_dict['dense_feature'].append(d)
    return scheme_dict