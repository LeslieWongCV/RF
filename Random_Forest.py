# -*- coding: utf-8 -*-
# @Time    : 2021/4/4 7:48 下午
# @Author  : Yushuo Wang
# @FileName: Random_Forest.py
# @Software: PyCharm
# @Blog    ：https://lesliewongcv.github.io/

import pandas as pd
import numpy as np
import random
import math
import collections
from joblib import Parallel, delayed
from scipy.io import loadmat
import os
from sklearn.model_selection import KFold


class Tree(object):
    """Definition of a Decision Tree"""
    def __init__(self):
        self.split_feature = None
        self.split_value = None
        self.leaf_value = None
        self.tree_left = None
        self.tree_right = None

    def calc_predict_value(self, dataset):
        """Find the leaf node of the sample through the recursive decision tree"""
        if self.leaf_value is not None:
            return self.leaf_value
        elif dataset[self.split_feature] <= self.split_value:
            return self.tree_left.calc_predict_value(dataset)
        else:
            return self.tree_right.calc_predict_value(dataset)

    def describe_tree(self):
        """Print the decision tree in json format for easy viewing of the tree structure"""
        if not self.tree_left and not self.tree_right:
            leaf_info = "{leaf_value:" + str(self.leaf_value) + "}"
            return leaf_info
        left_info = self.tree_left.describe_tree()
        right_info = self.tree_right.describe_tree()
        tree_structure = "{split_feature:" + str(self.split_feature) + \
                         ",split_value:" + str(self.split_value) + \
                         ",left_tree:" + left_info + \
                         ",right_tree:" + right_info + "}"
        return tree_structure


class RandomForestClassifier(object):
    def __init__(self, n_estimators=10, max_depth=-1, min_samples_split=2, min_samples_leaf=1,
                 min_split_gain=0.0, colsample_bytree=None, subsample=0.8, random_state=None):
        """
        Params
        ----------
        n_estimators:      No. of trees
        max_depth:         Tree depth, -1 for unlimited depth
        min_samples_split: The minimum number of samples required for node splitting
        min_samples_leaf:  The minimum sample number of leaf nodes
        min_split_gain:    The Minimal change of the Gini
        colsample_bytree:  Column sampling setting can be [sqrt, log2].
                           sqrt means randomly selecting sqrt(n_features) features,
                           log2 indicates that log(n_features) features are randomly selected,
                           and if set to other, column sampling is not performed
        subsample: line sampling ratio
        random_state:      Random seed, the set of n_estimators generated each time
                           after setting will not change to ensure that the experiment can be repeated
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth if max_depth != -1 else float('inf')
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_split_gain = min_split_gain
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.random_state = random_state
        self.trees = None  # Multiple decision trees built in parallel
        self.feature_importances_ = dict()

    def fit(self, dataset, targets):
        """Where it begins"""
        # assert targets.unique().__len__() == 2, "There must be two class for targets!"
        targets = targets.to_frame(name='label')  # label in the pd frame

        if self.random_state:
            random.seed(self.random_state)
        random_state_stages = random.sample(range(self.n_estimators), self.n_estimators)

        # Two column sampling methods
        if self.colsample_bytree == "sqrt":
            self.colsample_bytree = int(len(dataset.columns) ** 0.5)
        elif self.colsample_bytree == "log2":
            self.colsample_bytree = int(math.log(len(dataset.columns)))
        else:
            self.colsample_bytree = len(dataset.columns)

        # Build multiple decision trees in parallel
        self.trees = Parallel(n_jobs=-1, verbose=0, backend="threading")(
            delayed(self._parallel_build_trees)(dataset, targets, random_state)
            for random_state in random_state_stages)

    def _parallel_build_trees(self, dataset, targets, random_state):
        global describe
        """Bootstrap has replacement sampling to generate training sample set and establish decision tree"""
        subcol_index = random.sample(dataset.columns.tolist(), self.colsample_bytree)
        dataset_stage = dataset.sample(n=int(self.subsample * len(dataset)), replace=True,
                                       random_state=random_state).reset_index(drop=True)
        dataset_stage = dataset_stage.loc[:, subcol_index]
        targets_stage = targets.sample(n=int(self.subsample * len(dataset)), replace=True,
                                       random_state=random_state).reset_index(drop=True)

        tree = self._build_single_tree(dataset_stage, targets_stage, depth=0)
        describe += [file_name + ' : ' + tree.describe_tree()]

        return tree

    def _build_single_tree(self, dataset, targets, depth):
        """
        Recursive
        If the categories of the node are all the same/the sample is smaller than the minimum number of
        samples required for splitting, then the category with the most occurrences is selected. End split

        """

        if len(targets['label'].unique()) <= 1 or dataset.__len__() <= self.min_samples_split:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets['label'])
            return tree

        if depth < self.max_depth:
            best_split_feature, best_split_value, best_split_gain = self.choose_best_feature(dataset,
                                                                                             targets)  # choose_best_feature
            left_dataset, right_dataset, left_targets, right_targets = \
                self.split_dataset(dataset, targets, best_split_feature, best_split_value)

            tree = Tree()
            """
            If the left leaf node/right leaf node samples are less than the minimum sample number of 
            leaf nodes after the parent node is split, the parent node will terminate the split
            
            """

            if left_dataset.__len__() <= self.min_samples_leaf or \
                    right_dataset.__len__() <= self.min_samples_leaf or \
                    best_split_gain <= self.min_split_gain:
                tree.leaf_value = self.calc_leaf_value(targets['label'])
                return tree
            else:

                """
                If the feature is used during the split, the importance of the feature is increased by 1

                """

                self.feature_importances_[best_split_feature] = \
                    self.feature_importances_.get(best_split_feature, 0) + 1

                tree.split_feature = best_split_feature
                tree.split_value = best_split_value
                """递归 Recursive"""
                tree.tree_left = self._build_single_tree(left_dataset, left_targets, depth + 1)
                tree.tree_right = self._build_single_tree(right_dataset, right_targets, depth + 1)
                return tree

        else:
            '''
            If the depth of the tree exceeds the preset value, terminate the split
            '''

            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets['label'])
            return tree

    def choose_best_feature(self, dataset, targets):
        """
        Find the best way to divide the data set, find the optimal split feature, split threshold, and split gain
        """
        best_split_gain = 1
        best_split_feature = None
        best_split_value = None

        for feature in dataset.columns:
            _ = 1 + 1
            if dataset[feature].unique().__len__() <= 100:
                unique_values = sorted(dataset[feature].unique().tolist())
            else:
                unique_values = np.unique([np.percentile(dataset[feature], x)
                                           for x in np.linspace(0, 100, 100)])

            for split_value in unique_values:
                left_targets = targets[dataset[feature] <= split_value]
                right_targets = targets[dataset[feature] > split_value]
                split_gain = self.calc_gini(left_targets['label'], right_targets['label'])

                if split_gain < best_split_gain:
                    best_split_feature = feature
                    best_split_value = split_value
                    best_split_gain = split_gain
        return best_split_feature, best_split_value, best_split_gain

    @staticmethod
    def calc_leaf_value(targets):
        label_counts = collections.Counter(targets)
        major_label = max(zip(label_counts.values(), label_counts.keys()))
        return major_label[1]

    @staticmethod
    def calc_gini(left_targets, right_targets):
        split_gain = 0
        for targets in [left_targets, right_targets]:
            gini = 1
            label_counts = collections.Counter(targets)
            for key in label_counts:
                prob = label_counts[key] * 1.0 / len(targets)
                gini -= prob ** 2
            split_gain += len(targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini
        return split_gain

    @staticmethod
    def split_dataset(dataset, targets, split_feature, split_value):
        left_dataset = dataset[dataset[split_feature] <= split_value]
        left_targets = targets[dataset[split_feature] <= split_value]
        right_dataset = dataset[dataset[split_feature] > split_value]
        right_targets = targets[dataset[split_feature] > split_value]
        return left_dataset, right_dataset, left_targets, right_targets

    def predict(self, dataset):
        res = []
        for _, row in dataset.iterrows():
            pred_list = []
            for tree in self.trees:
                pred_list.append(tree.calc_predict_value(row))

            pred_label_counts = collections.Counter(pred_list)
            pred_label = max(zip(pred_label_counts.values(), pred_label_counts.keys()))
            res.append(pred_label[1])
        return np.array(res)


if __name__ == '__main__':

    PATH = '/Users/leslie/Downloads/MatDataset/'
    folders = os.listdir(PATH)
    res = []
    describe = []
    for folder_name in folders:
        file_name = folder_name
        if folder_name == '.DS_Store':
            continue
        else:
            matfn = PATH + folder_name + '/' + folder_name + '_Train.mat'
            df_data = loadmat(matfn)['Data']
            df_label = loadmat(matfn)['Label']
            df_ = np.concatenate((df_data, df_label), axis=1)
            df_f = pd.DataFrame(df_)
            # df_f = df_f[df_f.loc[:, 8].isin([0, 2])].sample(frac=1, random_state=66).reset_index(drop=True)
            kf = KFold(n_splits=4, shuffle=False)

            for ntree in range(3,7):
                acc_valid = 0
                acc_train = 0
                for train_index, test_index in kf.split(df_label):  # 4-fold

                    clf = RandomForestClassifier(n_estimators=ntree,
                                                 max_depth=5,
                                                 min_samples_split=6,
                                                 min_samples_leaf=2,
                                                 min_split_gain=0.0,
                                                 colsample_bytree="sqrt",
                                                 subsample=0.8,
                                                 random_state=66)

                    feature_list = np.arange(df_.shape[1] - 1)
                    clf.fit(df_f.loc[train_index, feature_list], df_f.loc[train_index, df_f.shape[1] - 1])

                    from sklearn import metrics

                    acc_train += metrics.accuracy_score(df_f.loc[train_index,
                                                       df_.shape[1] - 1],
                                                       clf.predict(df_f.loc[train_index, feature_list]))
                    acc_valid += metrics.accuracy_score(df_f.loc[test_index, df_.shape[1] - 1],
                                                       clf.predict(df_f.loc[test_index, feature_list]))

                    acc_train = round(10 ** 4 * acc_train) / 10 ** 4
                    acc_valid = round(10 ** 4 * acc_valid) / 10 ** 4
                acc_train /= 4
                acc_valid /= 4
                res += [folder_name + ':' + str(acc_train) + '/' + str(acc_valid) +" tree No. = " + str(ntree)]
                print(folder_name + ":" + str(round(ntree/7 * 100)) + "%")

    res = np.array(res)
    np.savetxt("res_my_method.txt", res, fmt='%s', delimiter=',')
