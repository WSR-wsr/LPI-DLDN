# ======================================================================
#
# -*- coding: utf-8 -*-
#
# ======================================================================

# 'data' is the sample to be tested, 'row' is the number of rows, 'col' is the number of columns
# CV1 represents rows, CV2 represents columns

import numpy as np
import random

def kfold(data, k, row=0, col=0, cv=3):
    dlen = len(data)
    if cv == 1:
        lens = row
    elif cv == 2:
        lens = col
    else:
        lens = dlen
    d = list(range(lens))
    random.shuffle(d)
    test_n = lens // k
    n = lens % k
    test_res = []
    train_res = []
    for i in range(n):
        test = d[i * (test_n + 1):(i + 1) * (test_n + 1)]
        train = list(set(d) - set(test))
        test_res.append(test)
        train_res.append(train)
    for i in range(n, k):
        test = d[i * test_n + n:(i + 1) * test_n + n]
        train = list(set(d) - set(test))
        test_res.append(test)
        train_res.append(train)
    if cv == 3:
        return train_res, test_res
    else:
        train_s = []
        test_s = []
        for i in range(k):
            train_ = []
            test_ = []
            for j in range(dlen):
                if data[j][cv - 1] in test_res[i]:
                    test_.append(j)
                else:
                    train_.append(j)
            train_s.append(train_)
            test_s.append(test_)
        return train_s, test_s


def get_one_hot(targets, nb_classes) -> object:
    """
    :rtype: object
    """
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])
