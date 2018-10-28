import argparse
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import f1_score
from gensim.models import KeyedVectors
import jieba


def get_max_similarity(model, text1, text2):
    if not text1 or not text2:
        return 0
    words1 = [w for w in jieba.cut(text1, cut_call=False) if len(w) > 1 and w in model]
    words2 = [w for w in jieba.cut(text2, cut_call=False) if len(w) > 1 and w in model]
    if not words1 or not words2:
        return 0
    return max([model.similarity(w1, w2) for w1 in words1 for w2 in words2])


def train(train_file, valid_file, test_file, model_file):
    train_data = pd.read_table(train_file, names=['prefix', 'query_prediction', 'title', 'tag', 'label']
                               , header=None, encoding='utf-8').astype(str)
    valid_data = pd.read_table(valid_file, names=['prefix', 'query_prediction', 'title', 'tag', 'label']
                               , header=None, encoding='utf-8').astype(str)
    test_data = pd.read_table(test_file, names=['prefix', 'query_prediction', 'title', 'tag', 'label']
                               , header=None, encoding='utf-8').astype(str)
    model = KeyedVectors.load_word2vec_format(model_file, binary=True)

    train_data = pd.concat([train_data, valid_data])
    train_data['prefix-title'] = get_max_similarity(model, train_data['prefix'], train_data['title'])
    train_data['prefix-tag'] = get_max_similarity(model, train_data['prefix'], train_data['tag'])
    test_data['prefix-title'] = get_max_similarity(model, test_data['prefix'], test_data['title'])
    test_data['prefix-tag'] = get_max_similarity(model, test_data['prefix'], test_data['tag'])
    for i in range(10):
        train_data['query_prediction-title-{}'.format(i)] = get_max_similarity(model, train_data['query_prediction'][i],
                                                                               train_data['title'])
        train_data['query_prediction-tag-{}'.format(i)] = get_max_similarity(model, train_data['query_prediction'][i],
                                                                             train_data['tag'])

        test_data['query_prediction-title-{}'.format(i)] = get_max_similarity(model, test_data['query_prediction'][i],
                                                                              test_data['title'])
        test_data['query_prediction-tag-{}'.format(i)] = get_max_similarity(model, test_data['query_prediction'][i],
                                                                            train_data['tag'])

    train_data = train_data.drop(['prefix', 'query_prediction', 'title', 'tag'], axis=1)

    xx_logloss = []
    xx_submit = []
    train_data_x = np.array(train_data.drop(['label'], axis=1))
    train_data_y = np.array(train_data['label'])
    # valid_data_x = np.array(valid_data.drop(['label'], axis=1))
    # valid_data_y = np.array(valid_data['label'])
    test_data_x = np.array(test_data.drop(['label'], axis=1))
    n = 5
    train_fold = np.zeros(train_data.shape[0])
    train_fold[:-valid_data.shape[0]] = -1
    pkf = PredefinedSplit(test_fold=train_fold)
    for train_in, test_in in pkf:
        x_train, x_test, y_train, y_test = train_data_x[train_in], train_data_x[test_in], train_data_y[train_data]\
            , train_data_y[test_in]
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 32,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 1,
        }
        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=5000,
            valid_sets=lgb_eval,
            early_stopping_rounds=50,
            verbose_eval=50,
        )
        print(f1_score(y_test, np.where(gbm.predict(x_test, num_iteration=gbm.best_iteration))))
        xx_logloss.append(gbm.best_score['valid_0']['binary_logloss'])
        xx_submit.append(gbm.predict(test_data_x, num_iteration=gbm.best_iteration))

    print('train_logloss:', np.mean(xx_logloss))
    s = sum(xx_submit)

    test_data['label'] = list(s/n)
    test_data['label'] = test_data['label'].apply(lambda v: round(v))
    print('test_logloss', np.mean(test_data.label))
    test_data['label'].to_csv("result.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--valid_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--model_file', type=str, required=True)
    args = parser.parse_args()
    train(args.train_file, args.valid_file, args.test_file, args.model_file)

