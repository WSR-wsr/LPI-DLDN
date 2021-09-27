import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import auc, roc_curve, precision_recall_curve, accuracy_score, recall_score, f1_score, \
    precision_score
from tensorflow.keras import layers, models
from CV import get_one_hot, kfold



for mm in range(1, 2):
    for cv in range(3, 4):
        f_1 = open('pli_res_cv.txt', 'a')
        time = 20
        k = 5
        data_file = './data' + str(mm) + '/data.csv'
        label_file = './data' + str(mm) + '/label.csv'
        data = pd.read_csv(data_file, header=None, index_col=None).to_numpy()
        label = pd.read_csv(label_file, header=None).to_numpy()
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
                label = y_te
                y_tr = get_one_hot(y_tr.astype(np.int8), 2)
                y_te = get_one_hot(y_te.astype(np.int8), 2)
                model = models.Sequential()
                model.add(layers.Dense(60, activation='sigmoid', input_shape=(200,)))
                model.add(layers.Dense(30, activation='sigmoid'))
                model.add(layers.Dense(20, activation='sigmoid'))
                model.add(layers.Dense(2, activation='sigmoid'))
                # compile the model
                model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

                model.fit(X_tr, y_tr, epochs=1, batch_size=32, verbose=2)
                proba = model.predict_proba(X_te)
                score = proba[:, 1]
                pre_label = np.argmax(proba, axis=1)

                fpr, tpr, threshold = roc_curve(label, score)
                pre, rec_, _ = precision_recall_curve(label, score)

                curve_1 = np.vstack([fpr, tpr])
                curve_1 = pd.DataFrame(curve_1.T)
                curve_1.to_csv('curve_auc.csv', header=None, index=None)

                curve_2 = np.vstack([rec_, pre])
                curve_2 = pd.DataFrame(curve_2.T)
                curve_2.to_csv('curve_aupr.csv', header=None, index=None)

                # plt.plot(fpr, tpr)
                # plt.title('AUC')
                # plt.show()
                # plt.plot(rec_, pre)
                # plt.title('AUPR')
                # plt.show()

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

                print('In time {}, k = {}:'.format(j + 1, i + 1))
                # print('Precision is :{}'.format(Pre))
                # print('Recall is :{}'.format(rec))
                # print("ACC is: {}".format(acc))
                # print("F1 is: {}".format(f1))
                print("AUC is: {}".format(au))
                print('AUPR is :{}'.format(apr))

        # toa = np.vstack([mPREs, mACCs, mRECs, mAUCs, mAUPRs, mF1])
        # toa = pd.DataFrame(toa)
        # toa.to_csv('cv2_data3.csv', header=None, index=None)
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

        f_1.write('data' + str(mm) + 'cv' + str(cv) + ':\n')
        f_1.write(str(round(PRE, 4)) + '±' + str(round(PRE_err, 4)) + '\t')
        f_1.write(str(round(REC, 4)) + '±' + str(round(REC_err, 4)) + '\t')
        f_1.write(str(round(ACC, 4)) + '±' + str(round(ACC_err, 4)) + '\t')
        f_1.write(str(round(F1, 4)) + '±' + str(round(F1_err, 4)) + '\t')
        f_1.write(str(round(AUC, 4)) + '±' + str(round(AUC_err, 4)) + '\t')
        f_1.write(str(round(AUPR, 4)) + '±' + str(round(AUPR_err, 4)) + '\t\n\n')
        f_1.close()
