def join_dicts(dict1, dict2):
    for k, v in dict2.items():
        if k in dict1:
            continue
        else:
            dict1[k] = v
    return dict1
