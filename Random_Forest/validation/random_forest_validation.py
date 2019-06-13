import numpy as np
from collections import Counter
from forest.forest import Forest

def validate_random_forest(data: np.ndarray,
                           labels: np.ndarray,
                           n_trees: int = 100,
                           max_features: int = 3,
                           bootstrap_features: bool = True,
                           max_depth: int = 4,
                           min_leaf_points: int = 2,
                           train_percent: float = .9):


    n = data.shape[0]
    shuffled_indices = np.random.choice(range(n),
                                        size=n,
                                        replace=False)
    # shuffle data
    X = data[shuffled_indices[:],:]
    y = labels[shuffled_indices[:]]

    # create test train split
    X_train = X[:int(train_percent*n), :]
    X_test= X[int(train_percent*n):, :]
    y_train = y[:int(train_percent*n)]
    y_test= y[int(train_percent*n):]

    # train model on training data
    forest = Forest(data=X_train,
                    labels=y_train,
                    n_trees=n_trees,
                    max_features=max_features,
                    bootstrap_features=bootstrap_features,
                    max_depth=max_depth,
                    min_leaf_points=min_leaf_points)

    train_confusion, train_accuracy = _get_confusion(forest, X_train, y_train)
    test_confusion, test_accuracy = _get_confusion(forest, X_test, y_test)

    print("")
    print("my-forest")
    print("")
    print("Training")
    print(train_confusion)
    print("train accuracy:", train_accuracy)
    print("")
    print("Testing")
    print(test_confusion)
    print("test accuracy:", test_accuracy)


def _get_confusion(forest: Forest,
                   data: np.ndarray,
                   labels: np.ndarray):

    unique_labels = np.unique(labels)
    unique_labels = np.sort(unique_labels)
    n_labels = unique_labels.shape[0]

    n = data.shape[0]
    # key: (actual, predicted)
    confusion_dict = Counter()
    for i in range(n):
        actual = labels[i,0]
        predicted = forest.predict(data[i,:])[0]
        confusion_dict[(actual, predicted)] += 1

    confusion_matrix = np.zeros((n_labels,n_labels))
    for k, v in confusion_dict.items():
        confusion_matrix[k[1], k[0]] = v

    correct = np.sum([confusion_dict[(k,k)] for k in unique_labels])
    total = sum(confusion_dict.values())
    accuracy = correct/total

    return confusion_matrix, accuracy






