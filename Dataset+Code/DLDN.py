# ======================================================================
#
# -*- coding: utf-8 -*-
# Author        : Chang Wang
# Date          : 3/22/2021
# Environment   : python-3.7.9, numpy-1.19.2, tensorflow-2.1.0, pandas-1.1.3, scikit-learn-0.23.2
# File          : DLDN.py
#
# ======================================================================


import numpy as np
import os
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from src.FIR import FeatureImportanceRank
import pandas as pd
from CV import get_one_hot, kfold
from sklearn.metrics import auc, roc_curve, precision_recall_curve, accuracy_score, recall_score, f1_score, \
    precision_score
import time as TIM

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
time = 2
k = 5

s = 25
N_FEATURES = 200
FEATURE_SHAPE = (200,)
dataset_label = "RNA_PRO"
data_batch_size = 32
mask_batch_size = 32
s_p = 2
phase_2_start = 200
early_stopping_patience = 200

# main program
for mm in range(1, 6):
    for cv in range(1, 5):
        data_file = './data' + str(mm) + '/data.csv'
        label_file = './data' + str(mm) + '/label.csv'
        data = pd.read_csv(data_file, header=None, index_col=None).to_numpy()
        label = pd.read_csv(label_file, index_col=None, header=None).to_numpy()
        label_copy = label.copy()
        row, col = label.shape
        if cv == 4:
            c = np.array([(i, j) for i in range(row) for j in range(col)])
        else:
            a = np.array([(i, j) for i in range(row) for j in range(col) if label[i][j]])
            b = np.array([(i, j) for i in range(row) for j in range(col) if label[i][j] == 0])
            np.random.shuffle(b)
            sample = len(a)
            b = b[:sample]
        mPREs = np.array([])
        mACCs = np.array([])
        mRECs = np.array([])
        mAUCs = np.array([])
        mAUPRs = np.array([])
        mF1 = np.array([])

        for j in range(time):
            if cv == 4:
                c_tr, c_te = np.array(kfold(c, k=k, row=row, col=col, cv=cv))
            elif cv == 3:
                a_tr, a_te = np.array(kfold(a, k=k, row=row, col=col, cv=cv))
                b_tr, b_te = np.array(kfold(b, k=k, row=row, col=col, cv=cv))
            else:
                c = np.vstack([a, b])
                c_tr, c_te = np.array(kfold(c, k=k, row=row, col=col, cv=cv))
            for i in range(k):
                if cv == 4:
                    b_tr = []
                    a_tr = []
                    # print(c_tr[i])
                    for ep in c_tr[i]:
                        if label_copy[c[ep][0]][c[ep][1]]:
                            b_tr.append(c[ep])
                        else:
                            a_tr.append(c[ep])
                    b_te = []
                    a_te = []
                    for ep in c_te[i]:
                        if label_copy[c[ep][0]][c[ep][1]]:
                            b_te.append(c[ep])
                        else:
                            a_te.append(c[ep])
                    b_te = np.array(b_te)
                    b_tr = np.array(b_tr)
                    a_te = np.array(a_te)
                    a_tr = np.array(a_tr)
                    np.random.shuffle(b_te)
                    np.random.shuffle(a_te)
                    np.random.shuffle(b_tr)
                    np.random.shuffle(a_tr)
                    a_tr = a_tr[:len(b_tr)]
                    a_te = a_te[:len(b_te)]
                    train_sample = np.vstack([a_tr, b_tr])
                    test_sample = np.vstack([a_te, b_te])
                elif cv == 3:
                    train_sample = np.vstack([np.array(a[a_tr[i]]), np.array(b[b_tr[i]])])
                    test_sample = np.vstack([np.array(a[a_te[i]]), np.array(b[b_te[i]])])
                else:
                    train_sample = np.array(c[c_tr[i]])
                    test_sample = np.array(c[c_te[i]])
                train_land = train_sample[:, 0] * col + train_sample[:, 1]
                test_land = test_sample[:, 0] * col + test_sample[:, 1]
                np.random.shuffle(train_land)
                np.random.shuffle(test_land)

                X_tr = data[train_land][:, :-1]
                y_tr = data[train_land][:, -1]
                X_te = data[test_land][:, :-1]
                y_te = data[test_land][:, -1]

                # max_batches = int(len(y_tr) / 2)
                max_batches = 2500
                y_tr = get_one_hot(y_tr.astype(np.int8), 2)
                y_te = get_one_hot(y_te.astype(np.int8), 2)
                fir = FeatureImportanceRank(FEATURE_SHAPE, s, data_batch_size, mask_batch_size,
                                            str_id=dataset_label)
                fir.create_dense_MLP([60, 30, 20, 2], "softmax", metrics=[keras.metrics.CategoricalAccuracy()],
                                     error_func=K.binary_crossentropy)
                fir.MLP.set_early_stopping_params(phase_2_start, patience_batches=early_stopping_patience,
                                                  minimize=True)
                fir.create_dense_FIR([100, 50, 10, 1])
                fir.create_mask_optimizer(epoch_condition=phase_2_start, perturbation_size=s_p)
                fir.train_networks_on_data(X_tr, y_tr, max_batches)

                importances, optimal_mask = fir.get_importances(return_chosen_features=True)
                optimal_subset = np.nonzero(optimal_mask)
                predict = fir.MLP.test_one(X_te, optimal_mask[None, :], y_te)
                label = y_te[:, 1]
                score = predict[:, 1]
                pre_label = np.argmax(predict, axis=1)
                fpr, tpr, threshold = roc_curve(label, score)
                pre, rec_, _ = precision_recall_curve(label, score)

                acc = accuracy_score(label, pre_label)
                rec = recall_score(label, pre_label)
                f1 = f1_score(label, pre_label)
                Pre = precision_score(label, pre_label)
                au = auc(fpr, tpr)
                apr = auc(rec_, pre)
                mPREs = np.append(mPREs, Pre)
                mACCs = np.append(mACCs, acc)
                mRECs = np.append(mRECs, rec)
                mAUCs = np.append(mAUCs, au)
                mAUPRs = np.append(mAUPRs, apr)
                mF1 = np.append(mF1, f1)

                curve_1 = np.vstack([fpr, tpr])
                curve_1 = pd.DataFrame(curve_1.T)
                curve_1.to_csv('./co_s/d' + str(mm) + 'c' + str(cv) + '_c' + str(au) + '.csv', header=None, index=None)

                curve_2 = np.vstack([rec_, pre])
                curve_2 = pd.DataFrame(curve_2.T)
                curve_2.to_csv('./co_s/d' + str(mm) + 'c' + str(cv) + '_r' + str(apr) + '.csv', header=None, index=None)

                # print('In time {}, k = {}:'.format(j + 1, i + 1))
                # print('Precision is :{}'.format(Pre))
                # print('Recall is :{}'.format(rec))
                # print("ACC is: {}".format(acc))
                # print("F1 is: {}".format(f1))
                print("AUC is: {}".format(au))
                print('AUPR is :{}'.format(apr))

        toa = np.vstack([mAUCs, mAUPRs])
        toa = pd.DataFrame(toa)
        toa.to_csv('res_data/' + str(s) + '/cv' + str(cv) + '_data' + str(mm) + '.csv', header=None, index=None)
        PRE = mPREs.mean()
        ACC = mACCs.mean()
        REC = mRECs.mean()
        AUC = mAUCs.mean()
        AUPR = mAUPRs.mean()
        F1 = mF1.mean()

        PRE_err = np.std(mPREs)
        ACC_err = np.std(mACCs)
        REC_err = np.std(mRECs)
        AUC_err = np.std(mAUCs)
        AUPR_err = np.std(mAUPRs)
        F1_err = np.std(mF1)

        print('\n')
        print("PRE is:{}±{}".format(round(PRE, 4), round(PRE_err, 4)))
        print("REC is:{}±{}".format(round(REC, 4), round(REC_err, 4)))
        print("ACC is:{}±{}".format(round(ACC, 4), round(ACC_err, 4)))
        print("F1 is:{}±{}".format(round(F1, 4), round(F1_err, 4)))
        print('AUC is :{}±{}'.format(round(AUC, 4), round(AUC_err, 4)))
        print('AUPR is :{}±{}'.format(round(AUPR, 4), round(AUPR_err, 4)))

        f_1 = open('./co_s/' + str(s) + 'res_cv.txt', 'a')
        f_1.write('data' + str(mm) + 'cv' + str(cv) + ':\n')
        f_1.write(str(round(PRE, 4)) + '±' + str(round(PRE_err, 4)) + '\t')
        f_1.write(str(round(REC, 4)) + '±' + str(round(REC_err, 4)) + '\t')
        f_1.write(str(round(ACC, 4)) + '±' + str(round(ACC_err, 4)) + '\t')
        f_1.write(str(round(F1, 4)) + '±' + str(round(F1_err, 4)) + '\t')
        f_1.write(str(round(AUC, 4)) + '±' + str(round(AUC_err, 4)) + '\t')
        f_1.write(str(round(AUPR, 4)) + '±' + str(round(AUPR_err, 4)) + '\t\n\n')
        f_1.close()
        print('\nwrite\n')
