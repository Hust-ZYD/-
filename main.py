#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import re
from sklearn.metrics import mean_squared_error


# 时间处理
def timeTranSecond(t):
    """
    把时间转成小时为单位
    :param t:
    :return:
    """
    if t != t:
        return -1
    try:
        t, m, s = t.split(":")
    except:
        if t == '1900/1/9 7:00':
            return 7 * 3600 / 3600
        elif t == '1900/1/1 2:30':
            return (2 * 3600 + 30 * 60) / 3600
        elif t == '700':
            return 7
        elif t == ':30:00':
            return (12 * 3600 + 30 * 60) / 3600
        elif t == -1:
            return -1
        else:
            return 0

    try:
        tm = (int(t) * 3600 + int(m) * 60 + int(s)) / 3600
    except:
        return (30 * 60) / 3600

    return tm


def getDuration(se):
    """
    时间区间转换
    :param se:
    :return:
    """
    if se != se:
        return -1
    try:
        sh, sm, eh, em = re.findall(r"\d+\.?\d*", se)
    except:
        if se == -1:
            return -1

    try:
        if int(sh) > int(eh):
            tm = (int(eh) * 3600 + int(em) * 60 - int(sm) * 60 - int(sh) * 3600) / 3600 + 24
        else:
            tm = (int(eh) * 3600 + int(em) * 60 - int(sm) * 60 - int(sh) * 3600) / 3600
    except:
        if se == '19:-20:05':
            return 1
        elif se == '15:00-1600':
            return 1

    return tm


def run(train, test, res_path):
    train = train.copy()
    test_id = test['样本id']

    # 去除异常点
    train = train[(train['收率'] >= 0.85) & (train['收率'] <= 1)]
    # 新数据B14列的取值范围在350-460之间（不一定包含两端点）
    train = train[(train['B14'] >= 350) & (train['B14'] <= 460)]
    train = train.reset_index(drop=True)

    # 删除类别唯一的特征

    for df in [train, test]:
        df.drop(['B3', 'B13', 'A13', 'A18', 'A23'], axis=1, inplace=True)

    # 删除训练集数据单一值率超过90%的列
    good_cols = list(train.columns)
    for col in train.columns:
        # normalize=True:返回频率
        rate = train[col].value_counts(normalize=True, dropna=False).values[0]
        if rate > 0.9:
            good_cols.remove(col)
    # 添加A1，A2,A3,A4
    good_cols.extend(['A1', 'A2', 'A3', 'A4'])
    train = train[good_cols]
    good_cols.remove('收率')
    test = test[good_cols]

    # 合并数据集
    target = train['收率']
    train.drop(['收率'], axis=1, inplace=True)
    all_data = pd.concat([train, test], axis=0, ignore_index=True)

    all_data.loc[all_data['A3'].isnull(), 'A3'] = all_data.loc[all_data['A3'].isnull(), 'A2']

    # 缺失值填充
    all_data = all_data.fillna(-1)

    # 时间转换
    for f in ['A5', 'A7', 'A9', 'A11', 'A14', 'A24', 'A26', 'B5', 'B7', 'A16']:
        all_data[f] = all_data[f].apply(timeTranSecond)

    for f in ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']:
        all_data[f] = all_data.apply(lambda df: getDuration(df[f]), axis=1)

    # 特征进行分类
    del all_data['样本id']
    categorical_columns = [f for f in all_data.columns]
    numerical_columns = []

    # 强特 添加了一个特征，B14除以其余原料总和
    all_data['b14/a1_a3_a4_a19_b1_b12'] = all_data['B14'] / (
            all_data['A1'] + all_data['A3'] + all_data['A4'] + all_data['A19'] + all_data['B1'] + all_data['B12'])
    numerical_columns.append('b14/a1_a3_a4_a19_b1_b12')
    del all_data['A1']
    del all_data['A2']
    del all_data['A3']
    del all_data['A4']
    categorical_columns.remove('A1')
    categorical_columns.remove('A2')
    categorical_columns.remove('A3')
    categorical_columns.remove('A4')

    # label encoder
    for f in categorical_columns:
        all_data[f] = all_data[f].map(dict(zip(all_data[f].unique(), range(0, all_data[f].nunique()))))

    all_data_ = all_data[categorical_columns]
    all_data_ = pd.get_dummies(all_data_, columns=categorical_columns).astype('float')
    for f in all_data_.columns:
        all_data[f] = all_data_[f]
    train = all_data[:train.shape[0]]
    test = all_data[train.shape[0]:]

    # 添加分箱特征   有修改空间
    train['target'] = list(target)
    train['intTarget'] = pd.cut(train['target'], 5, labels=False)
    train = pd.get_dummies(train, columns=['intTarget']).astype('float')
    # li = ['intTarget_0.0','intTarget_1.0','intTarget_2.0','intTarget_3.0','intTarget_4.0']
    li = train.columns[-5:]
    mean_columns = []
    for f1 in categorical_columns:
        cate_rate = train[f1].value_counts(normalize=True, dropna=False).values[0]
        if cate_rate < 0.90:
            for f2 in li:
                col_name = 'B14_to_' + f1 + "_" + f2 + '_mean'
                mean_columns.append(col_name)
                order_label = train.groupby([f1])[f2].mean()
                #            train['B14']的值去map order_label的index
                train[col_name] = train['B14'].map(order_label)
                miss_rate = train[col_name].isnull().sum() * 100 / train[col_name].shape[0]
                if miss_rate > 0:
                    #                 如果map的结果有缺失值 就drop掉
                    train = train.drop([col_name], axis=1)
                    mean_columns.remove(col_name)
                else:
                    test[col_name] = test['B14'].map(order_label)

    train.drop(li, axis=1, inplace=True)
    train.drop(['target'], axis=1, inplace=True)

    select_columns = ['A16_6',
                      'A10_2',
                      'A15_1',
                      'A24_10',
                      'A5_5',
                      'B14_to_A7_intTarget_3_mean',
                      'B14_to_A25_intTarget_0_mean',
                      'A10_1',
                      'A25_4',
                      'A5_9',
                      'B14_to_B7_intTarget_2_mean',
                      'B14_to_B14_intTarget_4_mean',
                      'A26_3',
                      'A6_7',
                      'B9_0',
                      'B14_to_A25_intTarget_2_mean',
                      'b14/a1_a3_a4_a19_b1_b12',
                      'B12_1',
                      'B6_1',
                      'B5_3',
                      'A26_1',
                      'A25_15',
                      'B7_2',
                      'A26_26',
                      'B14_to_B10_intTarget_2_mean']

    train = train.drop(categorical_columns, axis=1)
    test = test.drop(categorical_columns, axis=1)

    X_train = train[select_columns]
    X_test = test[select_columns]
    Y_train = target.values
   

    def lgb_cross_valid(X_train, Y_train, X_test):
        param = {'num_leaves': 64,
                 'min_data_in_leaf': 30,
                 'objective': 'regression',
                 'max_depth': 7,
                 'learning_rate': 0.01,
                 "boosting": "gbdt",
                 "feature_fraction": 0.8,
                 "bagging_freq": 1,
                 "bagging_fraction": 0.8,
                 "bagging_seed": 11,
                 "metric": 'mse',
                 "lambda_l2": 5,
                 "verbosity": -1}

        kfold = KFold(n_splits=5, shuffle=True, random_state=2018)
        oof_lgb = np.zeros(len(train))
        predictions_lgb = np.zeros(len(test))

        for fold_, (trn_idx, val_idx) in enumerate(kfold.split(X_train, Y_train)):
            print('第 %d 折交叉验证开始' % (fold_ + 1))
            trn_data = lgb.Dataset(X_train.iloc[trn_idx], Y_train[trn_idx])
            val_data = lgb.Dataset(X_train.iloc[val_idx], Y_train[val_idx])

            num_round = 10000
            model = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=400,
                              early_stopping_rounds=200)
            oof_lgb[val_idx] = model.predict(X_train.iloc[val_idx], num_iteration=model.best_iteration)
            predictions_lgb += model.predict(X_test, num_iteration=model.best_iteration) / kfold.n_splits

        print('LGBM交叉验证结果:%.8f' % mean_squared_error(oof_lgb, Y_train))
        return oof_lgb, predictions_lgb


    # 运行lgb
    oof_lgb, predictions_lgb = lgb_cross_valid(X_train, Y_train, X_test)


    def xgb_cross_valid(X_train, Y_train, test):
        xgb_params = {'eta': 0.005, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8,
                      'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 4,
                      'min_child_weight': 6
                      }

        folds = KFold(n_splits=5, shuffle=True, random_state=2018)
        oof_xgb = np.zeros(len(X_train))
        predictions_xgb = np.zeros(len(test))

        for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, Y_train)):
            print('第 %d 折交叉验证开始' % (fold_ + 1))
            trn_data = xgb.DMatrix(X_train.iloc[trn_idx], Y_train[trn_idx])
            val_data = xgb.DMatrix(X_train.iloc[val_idx], Y_train[val_idx])

            watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
            clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                            verbose_eval=400, params=xgb_params)
            oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train.iloc[val_idx]), ntree_limit=clf.best_ntree_limit)
            predictions_xgb += clf.predict(xgb.DMatrix(test), ntree_limit=clf.best_ntree_limit) / folds.n_splits

        print('XGB交叉验证结果:%.8f' % mean_squared_error(oof_xgb, Y_train))
        return oof_xgb, predictions_xgb


    # 运行xgb
    oof_xgb, predictions_xgb = xgb_cross_valid(X_train, Y_train, X_test)

    # 将lgb和xgb的结果进行stacking
    train_stack = np.vstack([oof_lgb, oof_xgb]).transpose()
    test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()

    folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)
    oof_stack = np.zeros(train_stack.shape[0])
    predictions = np.zeros(test_stack.shape[0])

    for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, Y_train)):
        print("fold {}".format(fold_))
        trn_data, trn_y = train_stack[trn_idx], Y_train[trn_idx]
        val_data, val_y = train_stack[val_idx], Y_train[val_idx]

        clf_3 = BayesianRidge()
        clf_3.fit(trn_data, trn_y)

        oof_stack[val_idx] = clf_3.predict(val_data)
        predictions += clf_3.predict(test_stack) / 10

    print('stack——mse:', mean_squared_error(target.values, oof_stack))
    
    # 保存结果
    sub_df =test_id.to_frame()
    sub_df[1] = predictions
    # 保留3位小数
    sub_df[1] = sub_df[1].apply(lambda x: round(x, 3))
    sub_df.iloc[:,:2].to_csv(res_path, index=False, header=None)


if __name__ == '__main__':
    # 读取数据
    train = pd.read_csv('data/jinnan_round1_train_20181227.csv', encoding='gbk')
    trainA = pd.read_csv('data/jinnan_round1_testA_20181227.csv', encoding='gbk')
    trainB = pd.read_csv('data/jinnan_round1_testB_20190121.csv', encoding='gbk')
    trainC = pd.read_csv('data/jinnan_round1_test_20190201.csv', encoding='gbk')
    ansA = pd.read_csv('data/jinnan_round1_ansA_20190125.csv', encoding='gbk', header=None)
    ansB = pd.read_csv('data/jinnan_round1_ansB_20190125.csv', encoding='gbk', header=None)
    ansC = pd.read_csv('data/jinnan_round1_ans_20190201.csv', encoding='gbk', header=None)

    trainA['收率'] = ansA.iloc[:, 1].values
    trainB['收率'] = ansB.iloc[:, 1].values
    trainC['收率'] = ansC.iloc[:, 1].values
    train = pd.concat([train, trainA, trainB, trainC], axis=0, ignore_index=True)

    optimize = pd.read_csv('data/optimize.csv', encoding='gbk')
    test = pd.read_csv('data/FuSai.csv', encoding='gbk')

    # 生成submit_optimize
    run(train, optimize, 'submit_optimize.csv')
	# 生成submit_FuSai
    run(train,test,'submit_FuSai.csv')
