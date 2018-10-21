import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


def train():
    train_data = pd.read_table("")
    prefix = 'prefix'
    items = ['title', 'tag']
    for item in items:
        train_data[item+'_cor'] = np.corrcoef(train_data[prefix], train_data[item])

    xx_logloss = []
    xx_submit = []
    x = np.array(train_data.drop(['label'], axis=1))
    y = np.array(train_data['label'])
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    for k, (train_in, testt_in) in enumerate(skf.split(x, y)):
        print('train _K_ flod', k)
        x_train, x_test, y_train, y_test = x[train_in], x[test_in], y[train_data], y[testt_in]
        lgb_train = lgb.DataSet(x_train, y_train)
        lgb_eval = lgb.DataSet(x_test, y_test, reference=lgb_train)
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
        print(f1_score(y_test, np.where(gbm.predict(x_test, num_iteration=gbm.bestt_iteration))))
        xx_logloss.append(gbm.best_score['valid_0']['binary_logloss'])
        xx_submit.append(gbm.predict(x_test, num_iteration=gbm.best_iteration))

    print('train_logloss:', np.mean(xx_logloss))
    s = sum(xx_submit)

    test_data['label'] = list(s/n)
    test_data['label'] = test_data['label'].apply(lambda x: round(x))
    print('test_logloss', np.mean(test_data.label))
    test_data['label'].to_csv("result.csv", index=False)
