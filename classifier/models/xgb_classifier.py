import xgboost as xgb

NUM_CLASSES = 17


class XgbClassifier:
    def __init__(self, eta, min_child_weight, depth, num_round, threads=8, exist_prediction=0, exist_num_round=20):
        self.eta = eta
        self.min_child_weight = min_child_weight
        self.depth = depth
        self.num_round = num_round
        self.threads = threads
        self.clf = None
        self.feature_num = None
        self.feature_names = None

    def train(self, X_train, y_train):
        xgmat_train = xgb.DMatrix(X_train, label=y_train)
        param = {'objective': 'multi:softmax',
                 'num_class': NUM_CLASSES,
                 'bst:eta': self.eta,
                 'colsample_bytree': 1,
                 'min_child_weight': self.min_child_weight,
                 'bst:max_depth': self.depth,
                 'eval_metric': 'mlogloss',
                 'silent': 1, 'nthread': self.threads}

        watchlist = [(xgmat_train, 'train')]
        num_round = self.num_round

        # if self.exist_prediction:
        # train xgb with existing predictions
        # see more at https://github.com/tqchen/xgboost/blob/master/demo/guide-python/boost_from_prediction.py

        # tmp_train = bst.predict(xgmat_train, output_margin=True)
        # tmp_test = bst.predict(xgmat_test, output_margin=True)
        # xgmat_train.set_base_margin(tmp_train)
        # xgmat_test.set_base_margin(tmp_test)
        # bst = xgb.train(param, xgmat_train, self.exist_num_round, watchlist)

        self.clf = xgb.train(param, xgmat_train, num_round, watchlist)
        self.feature_num = len(xgmat_train.feature_names)
        self.feature_names = xgmat_train.feature_names

    def predict(self, x_test):
        x_test[0, self.feature_num - 1] = 1e-7
        xgmat_test = xgb.DMatrix(x_test, feature_names=self.feature_names)
        pred = self.clf.predict(xgmat_test)
        return pred
