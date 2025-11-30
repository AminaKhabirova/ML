import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    if len(feature_vector) < 2 or np.all(feature_vector == feature_vector[0]):
        return None, None, None, None

    sorted_idx = np.argsort(feature_vector)
    sorted_feature = feature_vector[sorted_idx]
    sorted_target = target_vector[sorted_idx]

    cumsum = np.cumsum(sorted_target)
    total_sum = cumsum[-1]

    thresholds = (sorted_feature[:-1] + sorted_feature[1:]) / 2

    left_count = np.arange(1, len(feature_vector))
    right_count = len(feature_vector) - left_count

    left_sum = cumsum[:-1]
    right_sum = total_sum - left_sum

    left_gini = 1 - (left_sum/left_count)**2 - ((left_count-left_sum)/left_count)**2
    right_gini = 1 - (right_sum/right_count)**2 - ((right_count-right_sum)/right_count)**2

    ginis = -(left_count/len(feature_vector))*left_gini - (right_count/len(feature_vector))*right_gini

    if np.all(np.isnan(ginis)):
        return None, None, None, None

    best_idx = np.nanargmax(ginis)
    return thresholds, ginis, thresholds[best_idx], ginis[best_idx]


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any([x != "real" and x != "categorical" for x in feature_types]):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def get_params(self, deep=True):
       return {
           'feature_types': self._feature_types,
           'max_depth': self._max_depth,
           'min_samples_split': self._min_samples_split,
           'min_samples_leaf': self._min_samples_leaf
       }

    def set_params(self, **params):
       for key, value in params.items():
           if key == 'feature_types':
               self._feature_types = value
           elif key == 'max_depth':
               self._max_depth = value
           elif key == 'min_samples_split':
               self._min_samples_split = value
           elif key == 'min_samples_leaf':
               self._min_samples_leaf = value
       return self

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if len(sub_y) == 0:
            node["type"] = "terminal"
            node["class"] = 0
            return

        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if (self._max_depth is not None and depth >= self._max_depth) or \
           (self._min_samples_split is not None and len(sub_y) < self._min_samples_split):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                feature_vector = sub_X[:, feature].astype(int)

                unique_categories = np.unique(feature_vector)
                if len(unique_categories) < 2:
                    continue

                category_ratios = {}
                for category in unique_categories:
                    mask = feature_vector == category
                    if np.sum(mask) > 0:
                        category_ratios[category] = np.mean(sub_y[mask])

                sorted_categories = sorted(category_ratios.items(), key=lambda x: x[1])
                categories_map = {category: idx for idx, (category, _) in enumerate(sorted_categories)}

                feature_vector = np.array([categories_map[x] for x in feature_vector])

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if threshold is None:
                continue

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [category for category, idx in categories_map.items() if idx < threshold]

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_leaf is not None:
            left_count = np.sum(split)
            right_count = len(sub_y) - left_count
            if left_count < self._min_samples_leaf or right_count < self._min_samples_leaf:
                node["type"] = "terminal"
                node["class"] = Counter(sub_y).most_common(1)[0][0]
                return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best

        node["left_child"], node["right_child"] = {}, {}

        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_idx = node["feature_split"]
        feature_type = self._feature_types[feature_idx]

        if feature_type == "real":
            if x[feature_idx] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            if x[feature_idx] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)