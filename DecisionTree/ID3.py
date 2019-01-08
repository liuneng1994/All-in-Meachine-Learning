import pandas as pd
from scipy.stats import entropy
from collections import Counter, defaultdict
import numpy as np


class ID3:
    def __init__(self):
        self.root = None
        self.features = None

    def fit(self, x: pd.DataFrame, y: pd.Series):
        """
        训练ID3模型
        :param x: pandas.DataFrame 训练特征数据
        :param y: pandas.Series 目标数据
        :return:
        """
        self.features = x.columns
        self.root = ID3._split_data(x, y, x.columns)

    def predict(self, x):
        assert len(x.shape) == 2
        result = []
        for i in range(x.shape[0]):
            result.append(ID3._look_up(x.iloc[i, :], self.root))
        return pd.Series(result)

    @staticmethod
    def _look_up(features, node):
        label = None
        if "label" in node:
            label = node["label"]
        else:
            for n in node["nodes"]:
                fe_name = n["split_feature"]
                if features[fe_name] == n["feature_value"]:
                    label = ID3._look_up(features, n)
        if label is None:
            label = "Lookup Fail"
        return label

    @staticmethod
    def _most_common_label(labels):
        label_counter = Counter(labels)
        return label_counter.most_common(1)[0][0]

    @staticmethod
    def _split_data(x: pd.DataFrame, y: pd.Series, remain_features):
        node = dict()
        # 类别唯一返回叶子节点
        if len(y.unique()) == 1:
            node["label"] = y.unique()[0]
            return node

        # 无可分割属性，返回叶子节点以数量最多的类别作为分类结果
        if len(remain_features) == 0:
            node["label"] = ID3._most_common_label(y)

        best_feature, metric = ID3._best_split_feature(x, y, remain_features)
        node["nodes"] = list()

        for value in x[best_feature].unique():
            examples = x[x[best_feature] == value]
            target = y[x[best_feature] == value]
            child_node = ID3._split_data(examples, target, list([fe for fe in remain_features if fe != best_feature]))
            child_node["feature_value"] = value
            child_node["split_feature"] = best_feature
            child_node["metric"] = metric
            node["nodes"].append(child_node)
        return node

    @staticmethod
    def _gain(x, y):
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
        return before_entropy - after_entropy

    @staticmethod
    def _best_split_feature(x, y, remain_features):
        metric_map = dict()
        best_feature = None
        best_metric = None
        for feature in remain_features:
            metric = ID3._gain(x[feature], y)
            metric_map[feature] = metric
            if best_metric is None:
                best_metric = metric
                best_feature = feature
            elif best_metric < metric:
                best_metric = metric
                best_feature = feature
        return best_feature, best_metric


if __name__ == '__main__':
    dataset = pd.read_csv("xigua3.csv")
    x = dataset[["色泽", "根蒂", "敲声", "纹理", "脐部", "触感"]]
    y = dataset["好瓜"]
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    model = ID3()
    model.fit(X_train, y_train)
    print("实际值", y_test)
    pred = model.predict(X_test)
    print("预测值", pred)
    print("acc:", np.sum(pred.values == y_test.values) / len(pred))
