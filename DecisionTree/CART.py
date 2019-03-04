import pandas as pd
from collections import Counter
import numpy as np
from scipy.stats import entropy
from collections import Counter


class CART:
    def __init__(self, classfication=True, metric="gain", category_features=None):
        self.is_classification = classfication
        self.metric = metric
        self.features = None
        self.tree = None
        self.category_features = category_features
        self.splitter = self.get_splitter()

    def get_splitter(self):
        return Splitter()

    def fit(self, x, y):
        if not isinstance(x, pd.DataFrame):
            raise ValueError(f"x type should be pandas.DataFrame, but x is {type(x)}")
        assert isinstance(y, pd.Series)
        if x.shape[0] != y.shape[0]:
            raise ValueError("x numbers should equal to y")

        self.features = x.cloumns
        self.tree = self.build_tree(x, y, self.features)

    def build_tree(self, x, y, attrs):
        """
        使用深度优先算法构建决策树
        :param x:
        :param y:
        :param attrs: 属性集
        :return:
        """
        node_stack = []
        split_feature = []
        is_leaf = self.is_leaf_node(x, y, attrs)
        if is_leaf is True:
            node = Tree(None, is_leaf, self.get_major_label(y))
            root = node
        elif is_leaf == "Fail":
            node = Tree(None, True, is_leaf)
            root = node
        else:
            node = Tree(None, is_leaf)
            root = node
            data_x = x
            data_y = y
            best_feature, threshold, left_data, right_data = self.get_best_split_feature(data_x, data_y, attrs)
            node.split_feature = best_feature
            node.threshold = threshold
            node.left_values = left_data
            node.right_values = right_data
            node_stack.append(node)
            while not len(node_stack) == 0:
                if not node.is_leaf and node.left is not None and node.right is not None:
                    node = node_stack.pop()
                    split_feature.remove(node.split_feature)
                    continue
                if node.is_leaf:
                    node = node_stack.pop()
                    continue
                if node.left is None:
                    left = True
                else:
                    left = False
                split_feature.append(best_feature)
                if threshold is None:
                    if left:
                        part_data_mask = x[best_feature].isin(left_data)
                    else:
                        part_data_mask = x[best_feature].isin(right_data)
                else:
                    if left:
                        part_data_mask = x[best_feature] <= threshold
                    else:
                        part_data_mask = x[best_feature] > threshold
                part_x = x[part_data_mask]
                part_y = y[part_data_mask]

                is_leaf = self.is_leaf_node(part_x, part_y,
                                            list(feature for feature in attrs if feature not in split_feature))
                parent = node
                if is_leaf is True:
                    node = Tree(parent, is_leaf, self.get_major_label(left))
                elif is_leaf == "Fail":
                    node = Tree(parent, True, is_leaf)
                else:
                    best_feature, threshold, left_data, right_data = self.get_best_split_feature(part_x, part_y, list(
                        feature for feature in attrs if feature not in split_feature))
                    node = Tree(parent, is_leaf)
                    node.split_feature = best_feature
                    node.threshold = threshold
                    node.left_values = left_data
                    node.right_values = right_data
                node_stack.append(node)
                if left:
                    parent.left = node
                else:
                    parent.right = node
        return root

    def get_major_label(self, y):
        """
        获得主要的标签，分类问题选择数量最多的标签，回归问题使用平均值
        :param y: 标签数据
        :return: 标签
        """
        if self.is_classification:
            label_counter = Counter(y)
            return label_counter.most_common(1)[0][0]
        else:
            return np.mean(y)

    def get_best_split_feature(self, x, y, remain_attr):
        """
        获得最优的属性分割点，
        :param x:
        :param y:
        :param remain_attr:
        :return:
        """
        best_feature = None
        threshold = None
        left = None
        right = None
        for feature in remain_attr:
            is_category = feature in self.category_features
            if is_category:
                pass
            else:
                pass
        return best_feature, threshold, left, right

    def is_leaf_node(self, x, y, attrs):
        """
        判断是否为叶子节点，包括常规的结束规则，以及预减枝方法
        :param x: 特征数据
        :param y: 标签数据
        :param attrs: 特征名称
        :return: True 叶子结点 False 不是叶子节点 'Fail' 无法分类
        """
        if self.is_classification:
            if y.nunique() == 1:
                return True
            elif len(attrs) == 0:
                return True
            elif x.shape[0] == 0:
                return "Fail"
            else:
                return False
        else:
            pass


class Tree:
    def __init__(self, parent, is_leaf, label=None):
        if is_leaf and label is None:
            raise ValueError("leaf node must have label")
        self.left = None
        self.right = None
        self.parent = parent
        self.is_leaf = is_leaf
        self.label = label
        self.split_feature = None
        self.left_values = None
        self.right_values = None
        self.threshold = None


class Splitter:
    def split_category(self, feature: pd.Series, y: pd.Series):
        feature_values = list(feature.unique())
        best_metric = None
        best_left_values = None
        best_right_values = None
        for value in feature_values:
            left = [value]
            right = feature_values.copy().remove(value)
            left_data = feature[feature.isin(left)]
            right_data = feature[feature.isin(right)]
            metric = self._gain_ratio(feature, left_data, right_data, y)
            if best_metric is None or best_metric < metric:
                best_metric = metric
                best_left_values = left
                best_right_values = right
        return best_metric, best_left_values, best_right_values

    def _gain_ratio(self, feature: pd.Series, left: pd.Series, right: pd.Series, y):
        before_gain = entropy(np.asarray(Counter(feature).values()) / len(feature))
        left_gain = entropy(np.asarray(Counter(left).values()) / len(feature))
        right_gain = entropy(np.asarray(Counter(right).values()) / len(feature))
        after_gain = (len(left) / len(feature)) * left_gain + (len(right) / len(feature)) * right_gain
        return (before_gain - after_gain) / after_gain

    def split_numerical(self, feature: pd.Series, y: pd.Series):
        pass
