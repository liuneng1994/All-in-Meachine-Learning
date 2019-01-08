import pandas as pd
from scipy.stats import entropy
from collections import Counter, defaultdict
import numpy as np


class C45:
    def __init__(self):
        self.root = None

    def fit(self, x: pd.DataFrame, y: pd.Series):
        """
        训练C4.5模型
        :param x: pandas.DataFrame 训练特征数据
        :param y: pandas.Series 目标数据
        :return:
        """
        self.root = C45._split_data(x, y, x.columns)

    def predict(self, x):
        assert len(x.shape) == 2
        result = []
        for i in range(x.shape[0]):
            result.append(C45._look_up(x.iloc[i, :], self.root))
        return pd.Series(result)

    @staticmethod
    def _look_up(features, node):
        label = None
        if node.is_leaf:
            label = node.label
        else:
            for item in node.nodes:
                fe_name = node.split_feature
                if node.threshold is None:
                    if features[fe_name] == item.feature_value:
                        label = C45._look_up(features, item)
                else:
                    if item.less and features[fe_name] <= node.threshold:
                        label = C45._look_up(features, item)
                    elif not item.less and features[fe_name] > node.threshold:
                        label = C45._look_up(features, item)
        if label is None:
            label = "Lookup Fail"
        return label

    @staticmethod
    def _split_data(x, y, remain_features):
        if len(x) == 0:
            return Node(True, "Fail")
        if len(remain_features) == 0:
            return Node(True, C45._most_common_label(y))
        if len(y.unique()) == 1:
            return Node(True, y.unique()[0])
        best_feature, metric, threshold = C45._best_spilt_feature(x, y, remain_features)
        node = Node(False, None, list(), split_feature=best_feature, threshold=threshold, metric=metric)
        if node.threshold is not None:
            less_mask = x[node.split_feature] <= threshold
            greater_mask = x[node.split_feature] > threshold
            less_node = C45._split_data(x[less_mask], y[less_mask],
                                        list([fe for fe in remain_features if fe != best_feature]))
            less_node.less = True
            greater_node = C45._split_data(x[greater_mask], y[greater_mask],
                                           list([fe for fe in remain_features if fe != best_feature]))
            greater_node.less = False
            node.nodes.append(less_node)
            node.nodes.append(greater_node)
        else:
            for value in x[best_feature].unique():
                examples = x[x[best_feature] == value]
                target = y[x[best_feature] == value]
                child_node = C45._split_data(examples, target,
                                             list([fe for fe in remain_features if fe != best_feature]))
                child_node.feature_value = value
                node.nodes.append(child_node)
        return node

    @staticmethod
    def _best_spilt_feature(x, y, remain_features):
        if len(remain_features) == 0:
            return remain_features[0]
        best_feature = None
        best_metric = None
        best_threshold = None
        for feature in remain_features:
            if C45._is_discrete(x[feature]):
                metric = C45._gain_ratio(x[feature], y)
                if best_metric is None:
                    best_metric = metric
                    best_feature = feature
                elif best_metric > metric:
                    best_metric = metric
                    best_feature = feature
            else:
                threshold = (np.min(x[feature]) + np.max(x[feature])) / 2
                less = x[x[feature] <= threshold][feature]
                greater = x[x[feature] > threshold][feature]
                less_y = y[less.index]
                greater_y = y[greater.index]
                metric = len(less_y) / len(y) * C45._gain_ratio(less, less_y) + len(greater_y) / len(
                    y) * C45._gain_ratio(greater, greater_y)
                if best_metric is None:
                    best_metric = metric
                    best_feature = feature
                    best_threshold = threshold
                elif best_metric > metric:
                    best_metric = metric
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_metric, best_threshold

    @staticmethod
    def _is_discrete(feature):
        if str(feature.dtype) in ["object", "str"]:
            return True
        else:
            return False

    @staticmethod
    def _gain_ratio(x, y):
        fe_group_feature = defaultdict(list)
        for fe, label in zip(x, y):
            fe_group_feature[fe].append(label)
        after_entropy = 0
        before_entropy = entropy(np.asarray(list(Counter(y).values())) / len(y))
        for fe in fe_group_feature:
            labels = fe_group_feature[fe]
            label_counter = Counter(labels)
            label_counts = np.asarray(list(label_counter.values()))
            pk = label_counts / np.sum(label_counts)
            e = entropy(pk)
            label_portion = len(fe_group_feature[fe]) / len(y)
            after_entropy += e * label_portion
        return (before_entropy - after_entropy) / (after_entropy + 1e-5)

    @staticmethod
    def _most_common_label(labels):
        label_counter = Counter(labels)
        return label_counter.most_common(1)[0][0]


class Node:
    def __init__(self, is_leaf, label, nodes=None, feature_value=None, threshold=None, metric=None,
                 split_feature=None, less=None):
        self.is_leaf = is_leaf
        self.label = label
        self.nodes = nodes
        self.threshold = threshold
        self.metric = metric
        self.split_feature = split_feature
        self.feature_value = feature_value
        self.less = less


if __name__ == '__main__':
    # dataset = pd.read_csv("xigua3.csv")
    # x = dataset[["色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "密度", "含糖率"]]
    # y = dataset["好瓜"]
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_wine

    x, y = load_wine(True)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    model = C45()
    model.fit(pd.DataFrame(X_train), pd.Series(y_train))
    print("实际值", y_test)
    pred = model.predict(pd.DataFrame(X_test))
    print("预测值", pred)
    print("acc:", np.sum(pred.values == y_test) / len(pred))
