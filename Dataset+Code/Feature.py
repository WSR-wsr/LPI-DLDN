# ======================================================================
#
# -*- coding: utf-8 -*-
# Author        : Chang Wang
# Date          : 3/22/2021
# Environment   : python-3.7.9, numpy-1.19.2, pandas-1.1.3, scikit-learn-0.23.2
# File          : Feature.py
#
# ======================================================================


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def del_0(file):
    pro = pd.read_csv(file, header=None, index_col=None).to_numpy()
    idx = np.argwhere(np.all(pro[..., :] == 0, axis=0))
    a2 = np.delete(pro, idx, axis=1)
    res = pd.DataFrame(a2)
    res.to_csv('Pro_feat.csv', header=None, index=None)


def norm(file):
    print('norm')
    data = pd.read_csv(file, header=None, index_col=None).to_numpy()
    data = data - data.mean(axis=0)
    data = data / np.max(np.abs(data), axis=0)
    return data


def dim(file, n):
    print('dim')
    data = pd.read_csv(file, header=None, index_col=None).to_numpy()
    l = len(data)
    if len(data) < n:
        data = np.vstack([data, data])
        if len(data) < n:
            data = np.vstack([data, data])
            if len(data) < n:
                data = np.vstack([data, data])
    pca = PCA(n_components=n)
    res = pca.fit_transform(data)
    return res[:l]


def connect(rna, pro):
    print('connect')
    rna = pd.read_csv(rna, header=None, index_col=None).to_numpy()
    pro = pd.read_csv(pro, header=None, index_col=None).to_numpy()
    feat_num = rna.shape[1] + pro.shape[1]
    feat = np.zeros((1, feat_num))
    k = 0
    for i in rna:
        temp = np.array([])
        for j in pro:
            temp = np.hstack([i, j])
            feat = np.vstack([feat, temp])
            k += 1
            if k % 10000 == 0:
                print('waiting connect , now in {}'.format(k))
    feat = feat[1:]
    feat = pd.DataFrame(feat)
    print('wait to write')
    feat.to_csv('feat.csv', header=None, index=None)


def extract_label(land):
    land = pd.read_csv(land, header=None, index_col=None).to_numpy()
    label = np.zeros((885, 84))
    print('waiting extract label')
    land = land - 1
    label[land[:, 0], land[:, 1]] = 1
    # label = label.flatten()
    label = pd.DataFrame(label)
    label.to_csv('label.csv', header=None, index=None)


def con_feat_label(feat, label):
    feat = pd.read_csv(feat, header=None, index_col=None).to_numpy()
    label = pd.read_csv(label, header=None, index_col=None).to_numpy()
    label = label.flatten().reshape((-1, 1))
    data = pd.DataFrame(np.hstack([feat, label]))
    data.to_csv('./data/data_con.csv', header=None, index=None)


def extract_test(data_con, n):
    print('waiting extract test')
    data = pd.read_csv(data_con, header=None, index_col=None).to_numpy()
    res = np.zeros(len(data[0]))
    np.random.shuffle(data)
    p_n = 0
    n_n = 0
    for i in data:
        if i[-1] and p_n < 100:
            p_n += 1
            res = np.vstack([res, i])
        elif n_n < 100:
            res = np.vstack([res, i])
            n_n += 1
        if p_n == 100 and n_n == 100:
            break

    res = res[1:]
    res = pd.DataFrame(res)
    res.to_csv('./data/test.csv', header=None, index=None)


def extract_train(data_con, n):
    print('waiting extract test')
    data = pd.read_csv(data_con, header=None, index_col=None).to_numpy()
    res = np.zeros(len(data[0]))
    np.random.shuffle(data)
    p_n = 0
    n_n = 0
    for i in data:
        if i[-1] and p_n < n:
            p_n += 1
            res = np.vstack([res, i])
        elif n_n < n:
            res = np.vstack([res, i])
            n_n += 1
        if p_n == n and n_n == n:
            break

    res = res[1:]
    res = pd.DataFrame(res)
    res.to_csv('./data/train.csv', header=None, index=None)


if __name__ == '__main__':
    # extract_label('land.csv')
    print('Run')
    rna = 'RNA_feat.csv'
    pro = 'Pro_feat.csv'
    # land = 'land.csv'
    del_0(pro)
    rna_n = pd.DataFrame(norm(rna))
    pro_n = pd.DataFrame(norm(pro))
    rna_n.to_csv('./data/rna_n.csv', header=None, index=None)
    pro_n.to_csv('./data/pro_n.csv', header=None, index=None)

    rna_d = pd.DataFrame(dim('./data/rna_n.csv', 100))
    pro_d = pd.DataFrame(dim('./data/pro_n.csv', 100))
    rna_d.to_csv('./data/rna_d.csv', header=None, index=None)
    pro_d.to_csv('./data/pro_d.csv', header=None, index=None)

    connect('./data/rna_d.csv', './data/pro_d.csv')
    feat = 'feat.csv'
    label = 'label.csv'
    con_feat_label(feat, label)
    # data = './data/data_con.csv'
    # extract_train(data, n=3265)
    # extract_test(data, n=100)
